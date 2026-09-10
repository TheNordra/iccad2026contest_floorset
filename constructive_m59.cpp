// Constructive fixed-outline floorplanner (C++ port of teammate's my_optimizer
// architecture). Replaces our SA+skyline-BL placer (oracle-perm ceiling ~3.27).
//
// M1: boundary-aspect dims + anchor-guided greedy frame packing (singles).
// M2 (this file): cluster blocks become compound "items" with an internal layout
//     and are placed as a unit, so grouping violations collapse. The M1 diagnosis
//     showed hpwl/area already match the teammate's v5; the whole gap was vrel.
// M9: two-pass wire refinement (the dominant cost gap is HPWL). The greedy wire
//     term only sees already-placed neighbors, so the first blocks are placed
//     blind to HPWL. After a full pass we re-pack the same frame REFINE_ITERS
//     times, each pass adding wire pull toward not-yet-placed neighbors at their
//     previous-pass positions (force-directed coordinate descent), keeping the
//     best by layout_score per frame. Single base 1.7045->1.658, portfolio
//     1.5659->1.5375. Deterministic; ICCAD_NO_REFINE / ICCAD_REFINE_ITERS knobs.
// M11: iterative compaction. After the initial 12-candidate round in compact_layout,
//     continue packing from the current best until convergence: each Y-pack shifts
//     rows so a subsequent X-pack can reclaim new slack. csc downside-protected;
//     converges typically in 1 pass. ICCAD_COMPACT_ITERS (default 8) controls max rounds.
// M14: post-placement HPWL push (hpwl_push). hgap is the dominant cost lever; the
//     frame-face compaction can spread connected blocks apart. After compaction we
//     slide every FREE SINGLE block toward its connectivity-weighted L1-median within
//     the void inside the current bbox. Downside-free (area/bv/gf/mib unchanged), so
//     default-on for every profile. Portfolio 1.4349 -> 1.4253. ICCAD_NO_PUSH=1.
// M15: widen the HPWL push movable range to BOUNDARY SINGLE blocks. A boundary block
//     has exactly one axis pinned by its edge (LEFT/RIGHT pin x, TOP/BOTTOM pin y);
//     sliding the free axis keeps edge-contact, so bv is preserved -- still
//     downside-free. Portfolio 1.4253 -> 1.4236. (Rigid free-cluster slide was tried
//     and rejected: clusters have no slack and FP translation breaks exact abutments;
//     see hpwl_push comment.)
// M16: same-size swap inside the HPWL push. Two non-preplaced non-cluster blocks
//     with bit-identical (w,h) and equal boundary code exchange positions when that
//     strictly lowers HPWL. The geometry multiset is unchanged, so area/bv/gf/mib
//     are identical by construction -- downside-free, default-on. ICCAD_NO_SWAP=1.
//
// Input/Output format identical to optimizer_claude.cpp.
// Build: g++ -O3 -std=c++17 -o constructive.exe constructive.cpp
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <map>
#include <set>
#include <cstdlib>
#include <functional>
#include <string>
#include <tuple>
#include <chrono>
#include <unordered_set>
#include <cstring>
using namespace std;

// ─── M46 offline probe: gated stage timing (never shipped) ───────────────────
static int STAGE_TIMING = 0;      // ICCAD_STAGE_TIMING=1 -> stderr TIMING lines
// ─── M49 offline probe: gated refine trace (never shipped) ────────────────────
// ICCAD_REFINE_TRACE=1 -> stderr RTRACE lines: per refine pass sc/improved/hash
// (cycle detection), per-frame last accepted improving pass, per-case winner
// frame. Gate off = zero behavioural change; gate on writes stderr only.
static int REFINE_TRACE = 0;
// ─── M59 offline probe: gated refine state dump (never shipped) ──────────────
// ICCAD_REFINE_DUMP=<path> -> per-frame pre-refine c1 (r=-1) and per-pass c2
// states, N lines of "x y w h" (%.17g) after each STATE header. Gate off =
// zero behavioural change; gate on writes the dump file only.
static FILE* REFINE_DUMP_FP = nullptr;
static unsigned long long m49_fnv(const void* p, size_t nb){
    unsigned long long h=1469598103934665603ULL;
    const unsigned char* b=(const unsigned char*)p;
    for (size_t i=0;i<nb;i++){ h^=b[i]; h*=1099511628211ULL; }
    return h;
}
static inline double m46_now(){
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}
struct M46Acc {
    double build=0, order=0, pipeline=0, pack=0, anch=0, freep=0, generic=0,
           cand=0, nudge=0, refine=0, swapmove=0, compact=0, push=0, total=0,
           ov32=0, wire32=0;
    long long packs=0, pack_fail=0, frames=0, refine_passes=0, items=0,
              cands=0, feas=0, free_items=0, anch_members=0, refine_fixed=0;
};
static M46Acc m46;

static const int B_LEFT = 1, B_RIGHT = 2, B_TOP = 4, B_BOTTOM = 8;
static const double MARGIN = 1e-4;
static const double TOL = 1e-6;

struct Edge { int i, j; double w; };
struct Block {
    double area; bool is_fixed, is_preplaced;
    int mib, cluster, boundary; double tx, ty, tw, th;
};
struct XYWH { double x, y, w, h; };
static void m59_dump_state(int f,int r,double sc,int imp,double fw,double fh,
                           const vector<XYWH>& st){
    if (!REFINE_DUMP_FP) return;
    fprintf(REFINE_DUMP_FP,"STATE f=%d r=%d sc=%.17g imp=%d fw=%.17g fh=%.17g n=%d\n",
            f,r,sc,imp,fw,fh,(int)st.size());
    for (const XYWH& p:st)
        fprintf(REFINE_DUMP_FP,"%.17g %.17g %.17g %.17g\n",p.x,p.y,p.w,p.h);
}

static int N;
static double BP_W = 30000.0;   // boundary-miss penalty in greedy item scoring
static double WIRE_MULT = 1.0;  // extra scale on the incremental-HPWL term (env)
static double ANCHOR_W = 0.10;  // anchor pull in greedy item scoring
static bool WIRE_FOR_ALL = false; // ICCAD_WIRE_FOR_ALL: compute wire for ALL bp positions
static bool WIRE_ORDER = false;   // ICCAD_WIRE_ORDER: sort items by total_wire first (hpwl-first packing)
static bool WIRE_TIEBREAK = false; // ICCAD_WIRE_TIEBREAK: total_wire as 2nd sort key after bscore
static bool WIRE_BFS = false;      // ICCAD_WIRE_BFS: reorder inside each bscore class by greedy
                                   // connectivity attachment to already-ordered items + preplaced
static bool BFS_PIN = false;       // ICCAD_BFS_PIN: BFS seed attachment also counts p2b pin
                                   // weights (pins are fixed anchors just like preplaced blocks)
static bool BFS_NORM = false;      // ICCAD_BFS_NORM: BFS pick compares attach/sqrt(item area) so
                                   // small densely-connected items order before big sparse ones
static int ORDER_MOVE = 0;         // ICCAD_ORDER_MOVE=K: after ORDER_SWAP (if any), greedy
                                   // hill-climb relocating one top-K total_wire item to another
                                   // top-K item's order position (remove+insert; the segment
                                   // between them shifts by one). A structural jump ORDER_SWAP
                                   // can't make: swap keeps everyone else's position fixed.
static int CLUSTER_ORD = 0;        // ICCAD_CLUSTER_ORD: 1 = compound (multi-block cluster)
                                   // items first within their bscore class, 2 = last. Stable
                                   // partition applied after all ordering layers (WT/BFS), so
                                   // the class-internal relative order is otherwise kept.
static int ORDER_SWAP = 0;         // ICCAD_ORDER_SWAP=K: before refinement, greedy hill-climb on
                                   // the pack order — try swapping the top-K total_wire items
                                   // pairwise, re-pack once, keep a swap iff layout_score improves
static double LR_ASPECT = 2.50; // w/h for LEFT/RIGHT-only boundary blocks (env)
static double TB_ASPECT = 0.40; // w/h for TOP/BOTTOM-only boundary blocks (env)
static double SOFT_ASPECT = 1.0; // M29 free-aspect: w/h for INTERIOR (code==0) soft
                                 // blocks; default 1.0 = square = unchanged (env)
static int FREE_ASPECT = 0;      // M29 ICCAD_FREE_ASPECT>0: per-block aspect SEARCH for
                                 // single interior movable blocks (0=off=unchanged)
static const vector<double> FREE_RATIOS = {1.0, 1.5, 0.6667, 2.0, 0.5}; // w/h tried
static double CLUSTER_ASPECT = 1.0; // M33 ICCAD_CLUSTER_ASPECT: w/h for pure-movable
                                    // INTERIOR cluster members (1.0 = square = unchanged)
static double MIB_ASPECT = 1.0;     // M37 probe ICCAD_MIB_ASPECT: w/h for no-master square-
                                    // fallback MIB groups (shared shape, same area avg -> MIB
                                    // violation stays 0). 1.0 = square = bit-identical.
static int FREE_CLUSTER = 0;        // M34 ICCAD_FREE_CLUSTER>0: per-member aspect SEARCH
                                    // for pure-movable INTERIOR cluster members (0=off)
static vector<double> FREE_CLUSTER_RATIOS = {1.0, 1.5, 0.6667, 2.0, 0.5}; // M34 per-member
                                    // search set; ICCAD_FREE_CLUSTER_RATIOS overrides (build-
                                    // time search -> widening is wall-free, unlike FREE_ASPECT)
static int FREE_ANCHORED = 0;       // M35 PROBE ICCAD_FREE_ANCHORED>0: per-member aspect SEARCH
                                    // for movable members of ANCHORED (mixed preplaced+movable)
                                    // clusters in the wall-attach first-pass (0=off=unchanged).
                                    // Arbitrated by the packing greedy score.
static vector<double> FREE_ANCHORED_RATIOS = {1.0, 1.5, 0.6667, 2.0, 0.5}; // M36 probe: search
                                    // set for FREE_ANCHORED; default == FREE_RATIOS so off/default
                                    // = bit-identical. ICCAD_FREE_ANCHORED_RATIOS overrides. NOTE
                                    // per-frame search (unlike build-time FREE_CLUSTER) -> widening
                                    // MAY raise the wall (M31 per-frame precedent: 21->29.6s).
static int FREE_ANCHORED_BND = 0;   // M36 probe ICCAD_FREE_ANCHORED_BND>0: also let BOUNDARY
                                    // movable members of anchored clusters search aspect (default
                                    // 0 = gate them out = bit-identical). M32 FREE_BOUNDARY analogy
                                    // says DEAD, but M35 flipped the anchored analogy -> probe it.
static vector<double> FRAME_ASPECTS; // outline w:h set; empty = default (env)
static vector<double> FRAME_SCALES;  // outline size set;  empty = default (env)
static bool REFRAME = false;       // ICCAD_REFRAME: after the normal pipeline, re-seed the
                                   // frame loop from the measured compacted bbox aspect (a
                                   // shape the blocks already fit) and re-run once; the single
                                   // output is arbitrated by the wrapper proxy (M26)
static bool GUIDE_MED = false;     // ICCAD_GUIDE_MED: add the connectivity-weighted L1-median of
                                   // an item's placed/guide neighbours as an extra greedy candidate
                                   // origin (a wire-optimal seed the abutment slots miss) (M26)
static vector<double> area_targets;
static vector<Edge>   b2b_edges, p2b_edges;
static vector<pair<double,double>> pins;
static vector<Block>  blocks;

static vector<pair<double,double>> dims;   // (w,h) per block
static vector<XYWH> pos;                    // working positions
static vector<char> placed;

// Two-pass refinement: the greedy wire term only sees already-placed neighbors,
// so the first-placed blocks are positioned nearly blind to HPWL (the dominant
// cost gap). After a full pass we re-pack the same frame with the wire term also
// pulling toward not-yet-placed neighbors at their previous-pass positions
// (force-directed). prev_pos holds those guide positions; use_prev enables it.
static vector<XYWH> prev_pos;
static bool use_prev = false;
static bool REFINE = true;   // ICCAD_NO_REFINE=1 disables refinement
static int REFINE_ITERS = 12; // refinement passes per frame (ICCAD_REFINE_ITERS);
                              // single-base plateaus ~1.655 over 12-24, 0.68s/case

struct Anchor { double x, y, w; };
static vector<Anchor> anchors;

static vector<vector<pair<int,double>>> b2b_adj;  // block -> [(neighbor block, w)]
static vector<vector<pair<int,double>>> p2b_adj;  // block -> [(pin index, w)]

// A cluster that mixes preplaced (fixed-position) and movable members. Its
// movable members are first attached to the preplaced "walls" so the group stays
// connected, instead of floating off as an independent compound item.
struct AnchoredCluster { vector<int> preplaced, movable; };
static vector<AnchoredCluster> anchored_clusters;

// An item is one or more blocks placed together. offs[k] = offset of blocks[k]
// from the item origin. Singles have one block at offset (0,0).
struct Item {
    vector<int> blocks;
    vector<pair<double,double>> offs;
    double w, h;
    int bscore;
    double total_wire;   // sum of b2b edge weights over all members
    double ax, ay, aw;   // anchor
};

// ─── basic helpers ────────────────────────────────────────────────────────────
static double soft_ratio(int code) {
    bool lr = (code & (B_LEFT | B_RIGHT)) != 0;
    bool tb = (code & (B_TOP | B_BOTTOM)) != 0;
    if (lr && !tb) return LR_ASPECT;
    if (tb && !lr) return TB_ASPECT;
    return SOFT_ASPECT;
}
static pair<double,double> default_soft_dim(double area, int code) {
    double r = soft_ratio(code), w = sqrt(area * r);
    return {w, (w > 0) ? area / w : sqrt(area)};
}
static int block_boundary_score(int b) {
    int code = blocks[b].boundary; if (code == 0) return 0;
    int s = 10;
    if (code & B_LEFT) s++; if (code & B_RIGHT) s++;
    if (code & B_TOP) s++;  if (code & B_BOTTOM) s++;
    return s;
}
static int boundary_penalty_est(int b, double x, double y, double w, double h,
                                double fw, double fh) {
    int code = blocks[b].boundary; if (code == 0) return 0;
    bool bad = false;
    if ((code & B_LEFT)   && fabs(x - 0.0) > TOL)      bad = true;
    if ((code & B_RIGHT)  && fabs((x + w) - fw) > TOL) bad = true;
    if ((code & B_TOP)    && fabs((y + h) - fh) > TOL) bad = true;
    if ((code & B_BOTTOM) && fabs(y - 0.0) > TOL)      bad = true;
    return bad ? 1 : 0;
}
static bool rect_overlap(double x1,double y1,double w1,double h1,
                         double x2,double y2,double w2,double h2) {
    if (x1+w1 <= x2+TOL || x2+w2 <= x1+TOL) return false;
    if (y1+h1 <= y2+TOL || y2+h2 <= y1+TOL) return false;
    return true;
}

// ─── MIB shape unification (port of teammate _apply_safe_mib_dimensions) ───────
// Members of a MIB group must share one (w,h) or each distinct shape costs a
// violation. We only unify when it keeps every soft block within the 1% area
// hard-constraint, so feasibility is preserved.
static void apply_safe_mib_dims() {
    map<int,vector<int>> groups;
    for (int i=0;i<N;i++){ int g=blocks[i].mib; if (g>0) groups[g].push_back(i); }
    for (auto& kv:groups){
        auto& mem=kv.second; if (mem.size()<=1) continue;
        // prefer a fixed/preplaced master when its shape fits every movable area
        int master=-1;
        for (int i:mem) if (blocks[i].is_fixed||blocks[i].is_preplaced){ master=i; break; }
        if (master>=0){
            double mw=dims[master].first, mh=dims[master].second, ma=mw*mh; bool ok=true;
            for (int i:mem){ if (blocks[i].is_fixed||blocks[i].is_preplaced) continue;
                double ta=blocks[i].area; if (ta<=0||fabs(ma-ta)/ta>0.01){ ok=false; break; } }
            if (ok){ for (int i:mem) if (!blocks[i].is_fixed&&!blocks[i].is_preplaced) dims[i]={mw,mh}; continue; }
        }
        // otherwise unify movable members to a square iff their areas are mutually ≤1%
        vector<int> mov; for (int i:mem) if (!blocks[i].is_fixed&&!blocks[i].is_preplaced) mov.push_back(i);
        if (mov.size()<=1) continue;
        double sa=0; bool bad=false; for (int i:mov){ if (blocks[i].area<=0){bad=true;break;} sa+=blocks[i].area; }
        if (bad) continue;
        double avg=sa/mov.size(); bool okall=true;
        for (int i:mov) if (fabs(avg-blocks[i].area)/blocks[i].area>0.01){ okall=false; break; }
        if (okall){
            // M37 probe: reshape the shared square to a shared rectangle of the SAME area
            // (avg) when ICCAD_MIB_ASPECT!=1.0 and every movable member is interior. Keeps
            // one shared shape (MIB violation stays 0) and area avg (1% constraint preserved).
            // Boundary members would distort edge capacity AND can't be reshaped as a subset
            // of a shared shape -> gate on all-interior (mirrors the CLUSTER_ASPECT gate).
            double w=sqrt(avg), h=sqrt(avg);
            bool all_interior=true;
            for (int i:mov) if (blocks[i].boundary!=0){ all_interior=false; break; }
            if (all_interior && getenv("ICCAD_MIB_DBG"))
                fprintf(stderr, "MIB_RESHAPEABLE group=%d size=%d avg=%.6g\n", kv.first, (int)mov.size(), avg);
            if (MIB_ASPECT != 1.0 && all_interior){ w=sqrt(avg*MIB_ASPECT); h=avg/w; }
            for (int i:mov) dims[i]={w,h};
        }
    }
}

// ─── anchors ──────────────────────────────────────────────────────────────────
static void estimate_anchors() {
    anchors.assign(N, {0,0,0});
    vector<double> sx(N,0), sy(N,0), sw(N,0);
    for (const Edge& e : b2b_edges) {
        if (e.w<=0||e.i<0||e.j<0||e.i>=N||e.j>=N) continue;
        if (placed[e.j]) { sx[e.i]+=e.w*(pos[e.j].x+pos[e.j].w/2); sy[e.i]+=e.w*(pos[e.j].y+pos[e.j].h/2); sw[e.i]+=e.w; }
        if (placed[e.i]) { sx[e.j]+=e.w*(pos[e.i].x+pos[e.i].w/2); sy[e.j]+=e.w*(pos[e.i].y+pos[e.i].h/2); sw[e.j]+=e.w; }
    }
    for (const Edge& e : p2b_edges) {
        if (e.w<=0||e.j<0||e.j>=N||e.i<0||e.i>=(int)pins.size()) continue;
        sx[e.j]+=e.w*pins[e.i].first; sy[e.j]+=e.w*pins[e.i].second; sw[e.j]+=e.w;
    }
    for (int i=0;i<N;i++) if (sw[i]>0) anchors[i]={sx[i]/sw[i], sy[i]/sw[i], sw[i]};
}

// ─── cluster internal layout ──────────────────────────────────────────────────
static void finalize_item(Item& it) {
    double w=0,h=0;
    for (size_t k=0;k<it.blocks.size();k++){
        double bw=dims[it.blocks[k]].first, bh=dims[it.blocks[k]].second;
        w=max(w, it.offs[k].first+bw); h=max(h, it.offs[k].second+bh);
    }
    it.w=w; it.h=h;
    int bs=0; double tw=0;
    for (int b: it.blocks) {
        bs+=block_boundary_score(b);
        for (auto& nb:b2b_adj[b]) tw+=nb.second;
    }
    it.bscore=bs; it.total_wire=tw;
}
// fragments / boundary-exposure of a cluster's internal layout (offsets in item
// frame). These are soft-constraint signals we rank ahead of compactness, since
// vrel sits inside the cost's exp() multiplier (teammate _make_compact_group_item).
static int item_fragment_count(const Item& it){
    int n=it.blocks.size(); if (n<=1) return 0;
    vector<int> par(n); for(int i=0;i<n;i++)par[i]=i;
    function<int(int)> find=[&](int a){ while(par[a]!=a){par[a]=par[par[a]];a=par[a];} return a; };
    for (int i=0;i<n;i++) for (int j=i+1;j<n;j++){
        double ax=it.offs[i].first, ay=it.offs[i].second, aw=dims[it.blocks[i]].first, ah=dims[it.blocks[i]].second;
        double bx=it.offs[j].first, by=it.offs[j].second, bw=dims[it.blocks[j]].first, bh=dims[it.blocks[j]].second;
        bool tx=(fabs(ax+aw-bx)<=1e-3||fabs(bx+bw-ax)<=1e-3)&&(ay<by+bh-TOL&&by<ay+ah-TOL);
        bool ty=(fabs(ay+ah-by)<=1e-3||fabs(by+bh-ay)<=1e-3)&&(ax<bx+bw-TOL&&bx<ax+aw-TOL);
        if (tx||ty) par[find(i)]=find(j);
    }
    set<int> r; for(int i=0;i<n;i++) r.insert(find(i)); return (int)r.size()-1;
}
static int item_boundary_bad(const Item& it){
    int bad=0;
    for (size_t k=0;k<it.blocks.size();k++){ int code=blocks[it.blocks[k]].boundary; if(!code) continue;
        double ox=it.offs[k].first, oy=it.offs[k].second, bw=dims[it.blocks[k]].first, bh=dims[it.blocks[k]].second;
        if ((code&B_LEFT)  && fabs(ox-0.0)>TOL)        bad++;
        if ((code&B_RIGHT) && fabs((ox+bw)-it.w)>TOL)  bad++;
        if ((code&B_BOTTOM)&& fabs(oy-0.0)>TOL)        bad++;
        if ((code&B_TOP)   && fabs((oy+bh)-it.h)>TOL)  bad++;
    }
    return bad;
}
// produce a compact CONNECTED layout for a movable cluster's members. We rank
// candidates by (fragments, boundary_bad, area, aspect) so a non-fragmenting /
// boundary-exposing layout beats a smaller but fragmented one.
static Item make_group_item(const vector<int>& members) {
    auto build_shelf = [&](const vector<int>& order, double target_w)->Item{
        Item it; it.blocks=order; it.offs.resize(order.size());
        double cx=0, cy=0, rowh=0;
        for (size_t k=0;k<order.size();k++){
            double bw=dims[order[k]].first, bh=dims[order[k]].second;
            if (cx>TOL && cx+bw>target_w){ cx=0; cy+=rowh; rowh=0; }
            it.offs[k]={cx,cy}; cx+=bw; rowh=max(rowh,bh);
        }
        finalize_item(it); return it;
    };
    // balance members by width into two touching rows (teammate _layout_group_two_rows)
    auto build_two_rows = [&](const vector<int>& order)->Item{
        vector<int> r1,r2; double w1=0,w2=0;
        for (int b:order){ if (w1<=w2){ r1.push_back(b); w1+=dims[b].first; } else { r2.push_back(b); w2+=dims[b].first; } }
        double r1h=0; for(int b:r1) r1h=max(r1h,dims[b].second);
        Item it; it.blocks=order; it.offs.resize(order.size());
        map<int,pair<double,double>> off; double x=0; for(int b:r1){ off[b]={x,0}; x+=dims[b].first; }
        x=0; for(int b:r2){ off[b]={x,r1h}; x+=dims[b].first; }
        for (size_t k=0;k<order.size();k++) it.offs[k]=off[order[k]];
        finalize_item(it); return it;
    };
    // Build all candidate internal layouts for the CURRENT dims[] and return the
    // lex-best (fragments, boundary_bad, area, aspect). Orders depend on dims[], so
    // this is re-evaluated per FREE_CLUSTER trial. With FREE_CLUSTER=0 it runs once
    // on the return path and is byte-identical to the pre-M34 selection.
    auto build_best = [&]()->Item{
        vector<int> boundary_first=members;
        sort(boundary_first.begin(),boundary_first.end(),[](int a,int b){
            int ba=block_boundary_score(a), bb=block_boundary_score(b);
            if (ba!=bb) return ba>bb;
            return dims[a].first*dims[a].second > dims[b].first*dims[b].second;
        });
        vector<int> by_w=members, by_h=members;
        sort(by_w.begin(),by_w.end(),[](int a,int b){ return dims[a].first>dims[b].first; });
        sort(by_h.begin(),by_h.end(),[](int a,int b){ return dims[a].second>dims[b].second; });
        double tot=0; for(int b:members) tot+=dims[b].first*dims[b].second;
        double base=sqrt(max(tot,1.0));
        vector<Item> cands;
        for (auto& order:{boundary_first, by_w, by_h}){
            cands.push_back(build_shelf(order, 1e18));     // horizontal row
            cands.push_back(build_shelf(order, 1e-9));     // vertical column
            cands.push_back(build_shelf(order, base));     // square-ish shelf
            cands.push_back(build_shelf(order, base*1.4)); // wide-ish shelf
            if (order.size()>=3) cands.push_back(build_two_rows(order));
        }
        Item best; bool have=false;
        int bfrag=0, bbad=0; double barea=0, baspect=0;   // best key so far
        for (auto& c: cands){
            int f=item_fragment_count(c), bd=item_boundary_bad(c);
            double area=c.w*c.h, aspect=fabs(c.w-c.h);
            bool take=!have;
            if (!take){                                    // lexicographic min
                if (f!=bfrag)                 take = f<bfrag;
                else if (bd!=bbad)            take = bd<bbad;
                else if (fabs(area-barea)>TOL)take = area<barea;
                else                          take = aspect<baspect;
            }
            if (take){ bfrag=f; bbad=bd; barea=area; baspect=aspect; best=c; have=true; }
        }
        return best;
    };
    // M34 per-member free-aspect (ICCAD_FREE_CLUSTER): greedily search each pure-
    // movable interior member's aspect over FREE_CLUSTER_RATIOS, arbitrated by the
    // SAME cluster layout-key above (NOT the packing greedy score, whose local area
    // term made per-block FREE_BOUNDARY fail in M32). Each trial's key is read under
    // that trial's dims and kept as scalars: item_fragment_count re-reads global
    // dims[], so cross-trial comparisons must use stored keys, never re-evaluated
    // Items. The winning dims are committed to the global dims[] because the packing
    // output (~l.672) and finalize_item re-read dims[b]. FREE_CLUSTER=0 -> skipped
    // -> bit-identical.
    if (FREE_CLUSTER>0){
        for (int m:members){
            if (!(blocks[m].cluster>0 && blocks[m].mib==0 && blocks[m].boundary==0
                  && !blocks[m].is_preplaced && !blocks[m].is_fixed && blocks[m].area>0)) continue;
            double A=blocks[m].area;
            pair<double,double> best_dim=dims[m];
            Item c0=build_best();                          // baseline under current dims[m]
            int bf=item_fragment_count(c0), bb=item_boundary_bad(c0);
            double bar=c0.w*c0.h, bas=fabs(c0.w-c0.h);
            for (double r:FREE_CLUSTER_RATIOS){
                double w=sqrt(A*r); dims[m]={w, A/w};
                Item c=build_best();
                int f=item_fragment_count(c), bd=item_boundary_bad(c);
                double ar=c.w*c.h, as=fabs(c.w-c.h);
                bool take = (f!=bf) ? (f<bf)
                          : (bd!=bb) ? (bd<bb)
                          : (fabs(ar-bar)>TOL) ? (ar<bar)
                          : (as<bas);
                if (take){ bf=f; bb=bd; bar=ar; bas=as; best_dim=dims[m]; }
            }
            dims[m]=best_dim;                              // commit (global dims write-back)
        }
    }
    return build_best();
}

// ─── item anchor (connectivity centroid over members) ─────────────────────────
static void set_item_anchor(Item& it) {
    double sx=0,sy=0,sw=0;
    for (int b: it.blocks){
        if (anchors[b].w<=0) continue;
        double weight=anchors[b].w*max(dims[b].first*dims[b].second,1e-9);
        sx+=weight*anchors[b].x; sy+=weight*anchors[b].y; sw+=weight;
    }
    if (sw>0){ it.ax=sx/sw; it.ay=sy/sw; it.aw=sw; } else { it.ax=it.ay=it.aw=0; }
}

// ─── frame candidates ─────────────────────────────────────────────────────────
static vector<pair<double,double>> frame_candidates() {
    double total=0,max_iw=1,max_ih=1,pre_w=0,pre_h=0;
    for (int i=0;i<N;i++){
        total+=dims[i].first*dims[i].second;
        max_iw=max(max_iw,dims[i].first); max_ih=max(max_ih,dims[i].second);
        if (placed[i]){ pre_w=max(pre_w,pos[i].x+pos[i].w); pre_h=max(pre_h,pos[i].y+pos[i].h); }
    }
    double base=sqrt(max(total,1.0));
    vector<double> aspects = FRAME_ASPECTS.empty() ? vector<double>{1.0,1.35,0.75,1.8,0.55}
                                                   : FRAME_ASPECTS;
    vector<double> scales = !FRAME_SCALES.empty() ? FRAME_SCALES
        : (N>=60&&N<80)?vector<double>{1.00,1.03,1.05,1.15,1.35,1.65,2.10}
                       :vector<double>{1.05,1.15,1.35,1.65,2.10};
    set<pair<long long,long long>> seen; vector<pair<double,double>> frames;
    for (double s:scales) for (double a:aspects){
        double w=base*s*sqrt(a), h=base*s/sqrt(a);
        w=max(w,max(pre_w+MARGIN,max_iw+MARGIN)); h=max(h,max(pre_h+MARGIN,max_ih+MARGIN));
        auto key=make_pair((long long)llround(w*1e6),(long long)llround(h*1e6));
        if (seen.insert(key).second) frames.push_back({w,h});
    }
    sort(frames.begin(),frames.end(),[](const pair<double,double>&A,const pair<double,double>&B){
        double aa=A.first*A.second,ab=B.first*B.second;
        if (fabs(aa-ab)>TOL) return aa<ab; return max(A.first,A.second)<max(B.first,B.second);
    });
    return frames;
}

// running bbox for O(1) bbox_area_with
static double bminx,bminy,bmaxx,bmaxy; static int bcount;
static void bbox_reset(){ bminx=bminy=1e18; bmaxx=bmaxy=-1e18; bcount=0; }
static void bbox_add(double x,double y,double w,double h){ bminx=min(bminx,x);bminy=min(bminy,y);bmaxx=max(bmaxx,x+w);bmaxy=max(bmaxy,y+h);bcount++; }
static double bbox_area_with(double x,double y,double w,double h){
    if (bcount==0) return w*h;
    double nx0=min(bminx,x),ny0=min(bminy,y),nx1=max(bmaxx,x+w),ny1=max(bmaxy,y+h);
    return (nx1-nx0)*(ny1-ny0);
}

// candidate origins for an item of size (iw,ih)
struct M46KeyHash {
    size_t operator()(const pair<long long,long long>& p) const {
        return (size_t)((unsigned long long)p.first*1000003ULL
                        ^ (unsigned long long)p.second*7919ULL);
    }
};
static vector<pair<double,double>> item_candidates(
        const Item& it, double fw, double fh, const vector<XYWH>& rects) {
    double iw=it.w, ih=it.h, xmax=max(0.0,fw-iw), ymax=max(0.0,fh-ih);
    vector<pair<double,double>> cands={{0,0},{xmax,0},{0,ymax},{xmax,ymax}};
    cands.reserve(8*rects.size()+16);
    if (it.aw>0) cands.push_back({it.ax-iw/2, it.ay-ih/2});
    // M46 opt-A: boundary scan FIRST — xs/ys are read only when a boundary
    // member has no exact origin on that axis, so interior items (the FREE
    // hot path) skip every std::set insert. Same sets when needed (std::set
    // iteration is sorted by value, independent of insert order).
    vector<double> xv, yv; bool any=false, slide_x=true, slide_y=true;
    for (size_t k=0;k<it.blocks.size();k++){
        int code=blocks[it.blocks[k]].boundary; if (code==0) continue; any=true;
        double ox=it.offs[k].first, oy=it.offs[k].second;
        double bw=dims[it.blocks[k]].first, bh=dims[it.blocks[k]].second;
        if (code&B_LEFT){ xv.push_back(-ox); slide_x=false; }
        if (code&B_RIGHT){ xv.push_back(fw-ox-bw); slide_x=false; }
        if (code&B_BOTTOM){ yv.push_back(-oy); slide_y=false; }
        if (code&B_TOP){ yv.push_back(fh-oy-bh); slide_y=false; }
    }
    bool need_xs = any && xv.empty(), need_ys = any && yv.empty();
    set<long long> xs,ys;
    if (need_xs){ xs.insert(0);xs.insert(llround(xmax*1e6)); }
    if (need_ys){ ys.insert(0);ys.insert(llround(ymax*1e6)); }
    auto addx=[&](double v){ xs.insert(llround(min(max(0.0,v),xmax)*1e6)); };
    auto addy=[&](double v){ ys.insert(llround(min(max(0.0,v),ymax)*1e6)); };
    for (const XYWH& r:rects){
        double rx2=r.x+r.w+MARGIN, ry2=r.y+r.h+MARGIN;
        double lx=max(0.0,r.x-iw-MARGIN), by=max(0.0,r.y-ih-MARGIN);
        if (need_xs){ addx(rx2);addx(lx);addx(r.x); }
        if (need_ys){ addy(ry2);addy(by);addy(r.y); }
        cands.push_back({rx2,r.y}); cands.push_back({rx2,max(0.0,r.y+r.h-ih)});
        cands.push_back({r.x,ry2}); cands.push_back({max(0.0,r.x+r.w-iw),ry2});
        cands.push_back({lx,r.y});  cands.push_back({lx,max(0.0,r.y+r.h-ih)});
        cands.push_back({r.x,by});  cands.push_back({max(0.0,r.x+r.w-iw),by});
    }
    if (any){
        if (xv.empty()){ for(long long v:xs) xv.push_back(v/1e6); }
        if (yv.empty()){ for(long long v:ys) yv.push_back(v/1e6); }
        for (double xx:xv) for (double yy:yv) cands.push_back({xx,yy});
    }
    // M46 opt-B: hash dedup (same first-occurrence-wins semantics -> identical
    // out order -> identical sort input; equality is exact on the key pair).
    unordered_set<pair<long long,long long>,M46KeyHash> seen;
    seen.reserve(cands.size()*2);
    vector<pair<double,double>> out; out.reserve(cands.size());
    for (auto&c:cands){
        double x=min(max(0.0,c.first),xmax), y=min(max(0.0,c.second),ymax);
        auto key=make_pair((long long)llround(x*1e6),(long long)llround(y*1e6));
        if (seen.insert(key).second) out.push_back({x,y});
    }
    sort(out.begin(),out.end(),[](const pair<double,double>&A,const pair<double,double>&B){
        if (fabs(A.second-B.second)>TOL) return A.second<B.second; return A.first<B.first;
    });
    return out;
}

static double item_boundary_penalty(const Item& it, double ox, double oy, double fw, double fh) {
    double pen=0;
    for (size_t k=0;k<it.blocks.size();k++){
        int b=it.blocks[k];
        pen += boundary_penalty_est(b, ox+it.offs[k].first, oy+it.offs[k].second,
                                    dims[b].first, dims[b].second, fw, fh);
    }
    return pen;
}

// candidate origins for a single block placed adjacent to existing cluster rects
// (teammate _adjacent_candidates_for_block): 8 abutment slots per rect + exact
// frame-edge slots if the block carries a boundary constraint.
static vector<pair<double,double>> adjacent_candidates_for_block(
        double w,double h,const vector<XYWH>& cluster_rects,double fw,double fh,int code){
    double xmax=max(0.0,fw-w), ymax=max(0.0,fh-h);
    vector<pair<double,double>> cands;
    for (const XYWH& r:cluster_rects){
        cands.push_back({r.x+r.w, r.y});
        cands.push_back({r.x+r.w, max(0.0, r.y+r.h-h)});
        cands.push_back({r.x-w, r.y});
        cands.push_back({r.x-w, max(0.0, r.y+r.h-h)});
        cands.push_back({r.x, r.y+r.h});
        cands.push_back({max(0.0, r.x+r.w-w), r.y+r.h});
        cands.push_back({r.x, r.y-h});
        cands.push_back({max(0.0, r.x+r.w-w), r.y-h});
    }
    vector<double> xs, ys;
    if (code&B_LEFT)   xs.push_back(0.0);
    if (code&B_RIGHT)  xs.push_back(fw-w);
    if (code&B_BOTTOM) ys.push_back(0.0);
    if (code&B_TOP)    ys.push_back(fh-h);
    if (!xs.empty()||!ys.empty()){
        if (xs.empty()){ for(auto&c:cands) xs.push_back(c.first); xs.push_back(0.0); xs.push_back(fw-w); }
        if (ys.empty()){ for(auto&c:cands) ys.push_back(c.second); ys.push_back(0.0); ys.push_back(fh-h); }
        vector<double> xss=xs, yss=ys;
        for (double x:xss) for (double y:yss) cands.push_back({x,y});
    }
    set<pair<long long,long long>> seen; vector<pair<double,double>> out;
    for (auto&c:cands){
        double x=min(max(0.0,c.first),xmax), y=min(max(0.0,c.second),ymax);
        auto key=make_pair((long long)llround(x*1e6),(long long)llround(y*1e6));
        if (seen.insert(key).second) out.push_back({x,y});
    }
    sort(out.begin(),out.end(),[](const pair<double,double>&A,const pair<double,double>&B){
        if (fabs(A.second-B.second)>TOL) return A.second<B.second; return A.first<B.first;
    });
    return out;
}
static bool rect_touches_any(double x,double y,double w,double h,const vector<XYWH>& rects){
    for (const XYWH& r:rects){
        bool tx=(fabs(x+w-r.x)<=1e-3||fabs(r.x+r.w-x)<=1e-3)&&(y<r.y+r.h-TOL&&r.y<y+h-TOL);
        bool ty=(fabs(y+h-r.y)<=1e-3||fabs(r.y+r.h-y)<=1e-3)&&(x<r.x+r.w-TOL&&r.x<x+w-TOL);
        if (tx||ty) return true;
    }
    return false;
}

static double wmedian(vector<pair<double,double>>&);  // defined below (used by GUIDE_MED)

// M46 opt-C: uniform-grid overlap index. Returns the same boolean as the
// linear rect_overlap scan: a true overlap has interval intersection > TOL
// on both axes, so the two rects share at least one grid cell -> querying the
// candidate's cell span visits every possible overlapper (superset); the exact
// rect_overlap test then decides. Boolean-only -> bit-identical output.
struct M46Grid {
    double csx=1, csy=1; static const int G=32;
    const vector<XYWH>* rp=nullptr;
    vector<vector<int>> cells;
    vector<int> stamp; int qid=0;
    void init(double fw, double fh, const vector<XYWH>* r){
        rp=r; csx=max(fw,1e-9)/G; csy=max(fh,1e-9)/G;
        cells.assign(G*G,{}); stamp.clear(); qid=0;
    }
    inline int ixc(double v) const { int c=(int)(v/csx); return c<0?0:(c>=G?G-1:c); }
    inline int iyc(double v) const { int c=(int)(v/csy); return c<0?0:(c>=G?G-1:c); }
    void add(int i){
        const XYWH& r=(*rp)[i];
        int x0=ixc(r.x), x1=ixc(r.x+r.w), y0=iyc(r.y), y1=iyc(r.y+r.h);
        for (int a=x0;a<=x1;a++) for (int b=y0;b<=y1;b++) cells[a*G+b].push_back(i);
        stamp.push_back(0);
    }
    bool overlaps(double x,double y,double w,double h){
        ++qid;
        int x0=ixc(x), x1=ixc(x+w), y0=iyc(y), y1=iyc(y+h);
        for (int a=x0;a<=x1;a++) for (int b=y0;b<=y1;b++)
            for (int i:cells[a*G+b]){
                if (stamp[i]==qid) continue;
                stamp[i]=qid;
                const XYWH& r=(*rp)[i];
                if (rect_overlap(x,y,w,h,r.x,r.y,r.w,r.h)) return true;
            }
        return false;
    }
};

// greedy pack of items into frame
static bool pack_in_frame(double fw,double fh,const vector<Item>& items,vector<XYWH>& out){
    out=pos; vector<XYWH> rects; bbox_reset();
    M46Grid g46; g46.init(fw,fh,&rects);
    vector<char> done(N,0);
    // Wire weight calibrated high: bbox-area minimisation alone scatters connected
    // blocks; the baseline we reconstruct is wire-driven. Swept optimum ~2000-3000
    // (broad basin, area_gap stays flat) → bake the old 0.025-0.075 base ×2000.
    double ww=(N>=116)?150.0:(N>=100?70.0:50.0);
    for (int i=0;i<N;i++) if (placed[i]){
        if (pos[i].x<-TOL||pos[i].y<-TOL||pos[i].x+pos[i].w>fw+TOL||pos[i].y+pos[i].h>fh+TOL) return false;
        for (const XYWH&r:rects) if (rect_overlap(pos[i].x,pos[i].y,pos[i].w,pos[i].h,r.x,r.y,r.w,r.h)) return false;
        rects.push_back(pos[i]); g46.add((int)rects.size()-1); bbox_add(pos[i].x,pos[i].y,pos[i].w,pos[i].h); done[i]=1;
    }
    // First-pass: attach anchored-cluster movable members to their preplaced walls
    // (teammate _pack_in_frame 637-689). Placed members are skipped by the singles
    // loop below; any that don't fit here fall back to that loop.
    double t_anch0=m46_now();
    for (const AnchoredCluster& ac:anchored_clusters){
        vector<XYWH> cluster_rects;
        for (int b:ac.preplaced) if (done[b]) cluster_rects.push_back(out[b]);
        vector<int> mov=ac.movable;
        sort(mov.begin(),mov.end(),[](int a,int b){
            int ba=block_boundary_score(a), bb=block_boundary_score(b);
            if (ba!=bb) return ba>bb;
            return dims[a].first*dims[a].second > dims[b].first*dims[b].second;
        });
        for (int b:mov){
            if (done[b]) continue;
            m46.anch_members++;
            // M35 probe (ICCAD_FREE_ANCHORED): search this member's aspect over
            // FREE_RATIOS jointly with the wall-attach position, arbitrated by the
            // SAME packing greedy score below (anchored members have no cluster
            // layout-key, unlike FREE_CLUSTER). Per-frame like FREE_ASPECT: commit
            // out[] only, never dims[], so each frame re-searches from the original
            // shape. Ineligible members / FREE_ANCHORED=0 use the sentinel ratio -1
            // -> original dims[b] -> single pass -> bit-identical.
            bool elig = FREE_ANCHORED>0 && blocks[b].mib==0
                      && (blocks[b].boundary==0 || FREE_ANCHORED_BND>0)
                      && !blocks[b].is_fixed && !blocks[b].is_preplaced && blocks[b].area>0;
            double A=blocks[b].area;
            vector<double> ratios = elig ? FREE_ANCHORED_RATIOS : vector<double>{-1.0};
            double best=1e300, bx=0, by=0, bw=0, bh=0; bool found=false;
            for (double r:ratios){
                double cw,ch;
                if (r<0){ cw=dims[b].first; ch=dims[b].second; }
                else    { cw=sqrt(A*r); ch=A/cw; }
                double t_c0=m46_now();
                auto cands=adjacent_candidates_for_block(cw,ch,cluster_rects,fw,fh,blocks[b].boundary);
                m46.cand+=m46_now()-t_c0;
                for (auto&c:cands){
                    m46.cands++;
                    bool samp = STAGE_TIMING && ((m46.cands & 31)==0);
                    double x=c.first, y=c.second;
                    if (x<-TOL||y<-TOL||x+cw>fw+TOL||y+ch>fh+TOL) continue;
                    bool ov=false;
                    double t_ov=samp?m46_now():0.0;
                    ov=g46.overlaps(x,y,cw,ch);
                    if (samp) m46.ov32+=m46_now()-t_ov;
                    if (ov) continue;
                    m46.feas++;
                    double cx=x+cw/2, cy=y+ch/2;
                    double ad=anchors[b].w>0?fabs(cx-anchors[b].x)+fabs(cy-anchors[b].y):0.0;
                    int bp=boundary_penalty_est(b,x,y,cw,ch,fw,fh);
                    double area=bbox_area_with(x,y,cw,ch), wire=0.0;
                    double t_w=samp?m46_now():0.0;
                    if (bp==0){
                        for (auto& nb:b2b_adj[b]){
                            double ncx,ncy;
                            if (done[nb.first]){ ncx=out[nb.first].x+out[nb.first].w/2; ncy=out[nb.first].y+out[nb.first].h/2; }
                            else if (use_prev){ ncx=prev_pos[nb.first].x+prev_pos[nb.first].w/2; ncy=prev_pos[nb.first].y+prev_pos[nb.first].h/2; }
                            else continue;
                            wire+=nb.second*(fabs(cx-ncx)+fabs(cy-ncy));
                        }
                        for (auto& pn:p2b_adj[b])
                            wire+=pn.second*(fabs(cx-pins[pn.first].first)+fabs(cy-pins[pn.first].second));
                    }
                    if (samp) m46.wire32+=m46_now()-t_w;
                    double score=area+ANCHOR_W*ad+ww*WIRE_MULT*wire+BP_W*bp+1e-3*y+1e-4*x;
                    if (!rect_touches_any(x,y,cw,ch,cluster_rects)) score+=7000.0; // keep group connected
                    if (score<best){ best=score; bx=x; by=y; bw=cw; bh=ch; found=true; }
                }
            }
            if (found){
                out[b]={bx,by,bw,bh}; rects.push_back({bx,by,bw,bh}); g46.add((int)rects.size()-1);
                bbox_add(bx,by,bw,bh); done[b]=1; cluster_rects.push_back({bx,by,bw,bh});
            }
        }
    }
    m46.anch+=m46_now()-t_anch0;
    for (const Item& it:items){
        bool all_done=true; for (int b:it.blocks) if(!done[b]){ all_done=false; break; }
        if (all_done) continue;                       // already placed in first-pass
        bool any_done=false; for (int b:it.blocks) if(done[b]){ any_done=true; break; }
        if (any_done) continue;                       // partial: leave to other items/frames
        m46.items++;
        // M29 free-aspect: single interior movable block -> search its own aspect
        // (±exact area) jointly with position; self-contained, then continue. The
        // generic path below is untouched, so FREE_ASPECT=0 is bit-identical.
        if (FREE_ASPECT>0 && it.blocks.size()==1){
            int sb=it.blocks[0];
            if (!blocks[sb].is_fixed && !blocks[sb].is_preplaced && blocks[sb].mib==0
                && blocks[sb].boundary==0 && blocks[sb].area>0){
                double A=blocks[sb].area;
                double t_f0=m46_now(); m46.free_items++;
                double best=1e300, bx=0, by=0, bw=0, bh=0; bool found=false;
                for (double r:FREE_RATIOS){
                    double IW=sqrt(A*r), IH=A/IW;
                    Item t=it; t.w=IW; t.h=IH;        // single: offs[0]=(0,0)
                    double t_c0=m46_now();
                    auto cands=item_candidates(t,fw,fh,rects);
                    m46.cand+=m46_now()-t_c0;
                    if (GUIDE_MED){
                        vector<pair<double,double>> xs, ys;
                        for (auto& nb:b2b_adj[sb]){
                            double ncx,ncy;
                            if (done[nb.first]){ ncx=out[nb.first].x+out[nb.first].w/2; ncy=out[nb.first].y+out[nb.first].h/2; }
                            else if (use_prev){ ncx=prev_pos[nb.first].x+prev_pos[nb.first].w/2; ncy=prev_pos[nb.first].y+prev_pos[nb.first].h/2; }
                            else continue;
                            xs.push_back({ncx,nb.second}); ys.push_back({ncy,nb.second});
                        }
                        for (auto& pn:p2b_adj[sb]){ xs.push_back({pins[pn.first].first,pn.second}); ys.push_back({pins[pn.first].second,pn.second}); }
                        if (!xs.empty()){ double mx=wmedian(xs), my=wmedian(ys);
                            double gx=min(max(0.0,mx-IW/2),max(0.0,fw-IW));
                            double gy=min(max(0.0,my-IH/2),max(0.0,fh-IH));
                            cands.push_back({gx,gy}); }
                    }
                    for (auto&c:cands){
                        m46.cands++;
                        bool samp = STAGE_TIMING && ((m46.cands & 31)==0);
                        double x=c.first,y=c.second;
                        if (x<-TOL||y<-TOL||x+IW>fw+TOL||y+IH>fh+TOL) continue;
                        bool ov=false;
                        double t_ov=samp?m46_now():0.0;
                        ov=g46.overlaps(x,y,IW,IH);
                        if (samp) m46.ov32+=m46_now()-t_ov;
                        if (ov) continue;
                        m46.feas++;
                        double cx=x+IW/2, cy=y+IH/2;
                        double ad=it.aw>0?fabs(cx-it.ax)+fabs(cy-it.ay):0.0;
                        double area=bbox_area_with(x,y,IW,IH), wire=0.0;
                        double t_w=samp?m46_now():0.0;
                        for (auto& nb:b2b_adj[sb]){
                            double ncx,ncy;
                            if (done[nb.first]){ ncx=out[nb.first].x+out[nb.first].w/2; ncy=out[nb.first].y+out[nb.first].h/2; }
                            else if (use_prev){ ncx=prev_pos[nb.first].x+prev_pos[nb.first].w/2; ncy=prev_pos[nb.first].y+prev_pos[nb.first].h/2; }
                            else continue;
                            wire+=nb.second*(fabs(cx-ncx)+fabs(cy-ncy));
                        }
                        for (auto& pn:p2b_adj[sb])
                            wire+=pn.second*(fabs(cx-pins[pn.first].first)+fabs(cy-pins[pn.first].second));
                        if (samp) m46.wire32+=m46_now()-t_w;
                        double score=area+ANCHOR_W*ad+ww*WIRE_MULT*wire+1e-3*y+1e-4*x;
                        if (score<best){ best=score; bx=x; by=y; bw=IW; bh=IH; found=true; }
                    }
                }
                if (!found){ m46.freep+=m46_now()-t_f0; return false; }
                out[sb]={bx,by,bw,bh}; rects.push_back({bx,by,bw,bh}); g46.add((int)rects.size()-1); bbox_add(bx,by,bw,bh); done[sb]=1;
                m46.freep+=m46_now()-t_f0;
                continue;
            }
        }
        double t_g0=m46_now(), t_c0=m46_now();
        auto cands=item_candidates(it,fw,fh,rects);
        m46.cand+=m46_now()-t_c0;
        if (GUIDE_MED){
            // Connectivity-weighted L1-median of this item's neighbours (placed, or
            // their guide positions during refinement) as an extra candidate seed:
            // a wire-optimal origin the geometric abutment slots don't include. The
            // existing scoring overlap-checks and ranks it, so it's downside-safe.
            vector<pair<double,double>> xs, ys;
            for (size_t k=0;k<it.blocks.size();k++){
                int b=it.blocks[k];
                for (auto& nb:b2b_adj[b]){
                    double ncx,ncy;
                    if (done[nb.first]){ ncx=out[nb.first].x+out[nb.first].w/2; ncy=out[nb.first].y+out[nb.first].h/2; }
                    else if (use_prev){ ncx=prev_pos[nb.first].x+prev_pos[nb.first].w/2; ncy=prev_pos[nb.first].y+prev_pos[nb.first].h/2; }
                    else continue;
                    xs.push_back({ncx,nb.second}); ys.push_back({ncy,nb.second});
                }
                for (auto& pn:p2b_adj[b]){
                    xs.push_back({pins[pn.first].first,pn.second});
                    ys.push_back({pins[pn.first].second,pn.second});
                }
            }
            if (!xs.empty()){
                double mx=wmedian(xs), my=wmedian(ys);
                double gx=min(max(0.0,mx-it.w/2),max(0.0,fw-it.w));
                double gy=min(max(0.0,my-it.h/2),max(0.0,fh-it.h));
                cands.push_back({gx,gy});
            }
        }
        double best=1e300, bx=0, by=0; bool found=false;
        for (auto&c:cands){
            m46.cands++;
            bool samp = STAGE_TIMING && ((m46.cands & 31)==0);
            double x=c.first,y=c.second;
            if (x<-TOL||y<-TOL||x+it.w>fw+TOL||y+it.h>fh+TOL) continue;
            bool ov=false;
            double t_ov=samp?m46_now():0.0;
            for (size_t k=0;k<it.blocks.size()&&!ov;k++){
                int b=it.blocks[k]; double rx=x+it.offs[k].first, ry=y+it.offs[k].second;
                double bw=dims[b].first, bh=dims[b].second;
                if (g46.overlaps(rx,ry,bw,bh)){ ov=true; break; }
            }
            if (samp) m46.ov32+=m46_now()-t_ov;
            if (ov) continue;
            m46.feas++;
            double cx=x+it.w/2, cy=y+it.h/2;
            double ad=it.aw>0?fabs(cx-it.ax)+fabs(cy-it.ay):0.0;
            double bp=item_boundary_penalty(it,x,y,fw,fh);
            double area=bbox_area_with(x,y,it.w,it.h);
            double wire=0.0;
            double t_w=samp?m46_now():0.0;
            if (bp==0 || WIRE_FOR_ALL){
                for (size_t k=0;k<it.blocks.size();k++){
                    int b=it.blocks[k];
                    double mcx=x+it.offs[k].first+dims[b].first/2, mcy=y+it.offs[k].second+dims[b].second/2;
                    for (auto& nb:b2b_adj[b]){
                        double ncx,ncy;
                        if (done[nb.first]){ ncx=out[nb.first].x+out[nb.first].w/2; ncy=out[nb.first].y+out[nb.first].h/2; }
                        else if (use_prev){ ncx=prev_pos[nb.first].x+prev_pos[nb.first].w/2; ncy=prev_pos[nb.first].y+prev_pos[nb.first].h/2; }
                        else continue;
                        wire+=nb.second*(fabs(mcx-ncx)+fabs(mcy-ncy));
                    }
                    for (auto& pn:p2b_adj[b])
                        wire+=pn.second*(fabs(mcx-pins[pn.first].first)+fabs(mcy-pins[pn.first].second));
                }
            }
            if (samp) m46.wire32+=m46_now()-t_w;
            double score=area+ANCHOR_W*ad+ww*WIRE_MULT*wire+BP_W*bp+1e-3*y+1e-4*x;
            if (score<best){ best=score; bx=x; by=y; found=true; }
        }
        if (!found){ m46.generic+=m46_now()-t_g0; return false; }
        for (size_t k=0;k<it.blocks.size();k++){
            int b=it.blocks[k]; double rx=bx+it.offs[k].first, ry=by+it.offs[k].second;
            double bw=dims[b].first, bh=dims[b].second;
            out[b]={rx,ry,bw,bh}; rects.push_back({rx,ry,bw,bh}); g46.add((int)rects.size()-1); bbox_add(rx,ry,bw,bh); done[b]=1;
        }
        m46.generic+=m46_now()-t_g0;
    }
    return true;
}

// ─── full-layout scoring ──────────────────────────────────────────────────────
static int count_boundary_violations(const vector<XYWH>& p){
    double xmin=1e18,ymin=1e18,xmax=-1e18,ymax=-1e18;
    for (auto&q:p){xmin=min(xmin,q.x);ymin=min(ymin,q.y);xmax=max(xmax,q.x+q.w);ymax=max(ymax,q.y+q.h);}
    int bad=0;
    for (int i=0;i<N;i++){ int code=blocks[i].boundary; if(code==0)continue;
        const XYWH&q=p[i]; bool ok=true;
        if(code&B_LEFT) ok=ok&&fabs(q.x-xmin)<=TOL;
        if(code&B_RIGHT) ok=ok&&fabs(q.x+q.w-xmax)<=TOL;
        if(code&B_TOP) ok=ok&&fabs(q.y+q.h-ymax)<=TOL;
        if(code&B_BOTTOM) ok=ok&&fabs(q.y-ymin)<=TOL;
        if(!ok) bad++;
    }
    return bad;
}
static double approx_hpwl(const vector<XYWH>& p){
    double hpwl=0;
    for (const Edge&e:b2b_edges){ if(e.w<=0||e.i<0||e.j<0||e.i>=N||e.j>=N)continue;
        double cxi=p[e.i].x+p[e.i].w/2,cyi=p[e.i].y+p[e.i].h/2,cxj=p[e.j].x+p[e.j].w/2,cyj=p[e.j].y+p[e.j].h/2;
        hpwl+=e.w*(fabs(cxi-cxj)+fabs(cyi-cyj)); }
    for (const Edge&e:p2b_edges){ if(e.w<=0||e.j<0||e.j>=N||e.i<0||e.i>=(int)pins.size())continue;
        double cxb=p[e.j].x+p[e.j].w/2,cyb=p[e.j].y+p[e.j].h/2;
        hpwl+=e.w*(fabs(cxb-pins[e.i].first)+fabs(cyb-pins[e.i].second)); }
    return hpwl;
}
// count cluster members not touch-connected to their cluster (grouping fragments)
static int count_group_fragments(const vector<XYWH>& p){
    map<int,vector<int>> cl;
    for (int i=0;i<N;i++) if (blocks[i].cluster>0) cl[blocks[i].cluster].push_back(i);
    int frag=0;
    for (auto& kv:cl){
        auto& m=kv.second; int n=m.size(); if (n<=1) continue;
        // union-find by edge touching
        vector<int> par(n); for(int i=0;i<n;i++)par[i]=i;
        function<int(int)> find=[&](int a){ while(par[a]!=a){par[a]=par[par[a]];a=par[a];} return a; };
        for (int i=0;i<n;i++) for (int j=i+1;j<n;j++){
            const XYWH&A=p[m[i]],&B=p[m[j]];
            bool tx = (fabs(A.x+A.w-B.x)<=1e-3||fabs(B.x+B.w-A.x)<=1e-3) && (A.y<B.y+B.h-TOL&&B.y<A.y+A.h-TOL);
            bool ty = (fabs(A.y+A.h-B.y)<=1e-3||fabs(B.y+B.h-A.y)<=1e-3) && (A.x<B.x+B.w-TOL&&B.x<A.x+A.w-TOL);
            if (tx||ty){ par[find(i)]=find(j); }
        }
        set<int> roots; for(int i=0;i<n;i++) roots.insert(find(i));
        frag += (int)roots.size()-1;   // extra components beyond the first
    }
    return frag;
}
// ─── repair nudges (port of teammate's 3 final passes) ────────────────────────
static void get_bbox(const vector<XYWH>& p, double& xmin,double& ymin,double& xmax,double& ymax){
    xmin=ymin=1e18; xmax=ymax=-1e18;
    for(auto&q:p){xmin=min(xmin,q.x);ymin=min(ymin,q.y);xmax=max(xmax,q.x+q.w);ymax=max(ymax,q.y+q.h);}
}
static bool overlaps_others(const vector<XYWH>& p,int self,double x,double y,double w,double h){
    for(int j=0;j<N;j++){ if(j==self) continue;
        if(rect_overlap(x,y,w,h,p[j].x,p[j].y,p[j].w,p[j].h)) return true; }
    return false;
}
// snap non-cluster single boundary blocks to the bbox edge they must touch
static void final_boundary_nudge(vector<XYWH>& out){
    double xmin,ymin,xmax,ymax; get_bbox(out,xmin,ymin,xmax,ymax);
    for(int i=0;i<N;i++){
        int code=blocks[i].boundary;
        if(code==0||blocks[i].is_preplaced||blocks[i].cluster>0) continue;
        double w=out[i].w,h=out[i].h,nx=out[i].x,ny=out[i].y;
        if(code&B_LEFT) nx=xmin; if(code&B_RIGHT) nx=xmax-w;
        if(code&B_BOTTOM) ny=ymin; if(code&B_TOP) ny=ymax-h;
        if(!overlaps_others(out,i,nx,ny,w,h)) out[i]={nx,ny,w,h};
    }
}
// rigidly translate non-preplaced clusters so a boundary member reaches the edge
static void final_group_boundary_nudge(vector<XYWH>& out){
    map<int,vector<int>> groups;
    for(int i=0;i<N;i++) if(blocks[i].cluster>0) groups[blocks[i].cluster].push_back(i);
    for(auto& kv:groups){
        auto& m=kv.second; if((int)m.size()<=1) continue;
        bool has_pre=false, has_bnd=false;
        for(int b:m){ if(blocks[b].is_preplaced) has_pre=true; if(blocks[b].boundary) has_bnd=true; }
        if(has_pre||!has_bnd) continue;
        double base=count_boundary_violations(out)+2.0*count_group_fragments(out);
        double xmin,ymin,xmax,ymax; get_bbox(out,xmin,ymin,xmax,ymax);
        vector<pair<double,double>> shifts={{0,0}};
        for(int b:m){ int code=blocks[b].boundary; if(!code) continue;
            double x=out[b].x,y=out[b].y,w=out[b].w,h=out[b].h;
            vector<double> dxs={0}, dys={0};
            if(code&B_LEFT) dxs.push_back(xmin-x);
            if(code&B_RIGHT) dxs.push_back(xmax-(x+w));
            if(code&B_BOTTOM) dys.push_back(ymin-y);
            if(code&B_TOP) dys.push_back(ymax-(y+h));
            for(double dx:dxs) shifts.push_back({dx,0});
            for(double dy:dys) shifts.push_back({0,dy});
            for(double dx:dxs) for(double dy:dys) shifts.push_back({dx,dy});
        }
        sort(shifts.begin(),shifts.end(),[](const pair<double,double>&a,const pair<double,double>&b){
            return fabs(a.first)+fabs(a.second) < fabs(b.first)+fabs(b.second); });
        set<int> mset(m.begin(),m.end());
        vector<XYWH> bestlay; double bestsc=base; bool found=false;
        for(auto& s:shifts){ double dx=s.first,dy=s.second;
            if(fabs(dx)<=1e-12&&fabs(dy)<=1e-12) continue;
            vector<XYWH> temp=out; bool ok=true;
            for(int b:m){ double nx=out[b].x+dx,ny=out[b].y+dy;
                for(int j=0;j<N;j++){ if(mset.count(j)) continue;
                    if(rect_overlap(nx,ny,out[b].w,out[b].h,out[j].x,out[j].y,out[j].w,out[j].h)){ ok=false; break; } }
                if(!ok) break; temp[b]={nx,ny,out[b].w,out[b].h}; }
            if(!ok) continue;
            double tx0,ty0,tx1,ty1; get_bbox(temp,tx0,ty0,tx1,ty1);
            double sc=count_boundary_violations(temp)+2.0*count_group_fragments(temp)+1e-9*(tx1-tx0)*(ty1-ty0);
            if(sc<bestsc){ bestsc=sc; bestlay=temp; found=true; }
        }
        if(found) out=bestlay;
    }
}
// large cases: push missed single-edge blocks outside the bbox when net-positive
static void final_single_edge_escape(vector<XYWH>& out){
    if(N<80) return;
    double xmin,ymin,xmax,ymax; get_bbox(out,xmin,ymin,xmax,ymax);
    auto side=[&](int bit, vector<int>& missed, int& touch){
        touch=0;
        for(int i=0;i<N;i++){ int code=blocks[i].boundary; if(!(code&bit)) continue;
            double x=out[i].x,y=out[i].y,w=out[i].w,h=out[i].h; bool t=false;
            if(bit==B_LEFT) t=fabs(x-xmin)<=TOL; else if(bit==B_RIGHT) t=fabs(x+w-xmax)<=TOL;
            else if(bit==B_BOTTOM) t=fabs(y-ymin)<=TOL; else t=fabs(y+h-ymax)<=TOL;
            bool elig=(code==bit)&&!t&&!blocks[i].is_preplaced&&blocks[i].cluster==0;
            if(elig) missed.push_back(i); else if(t) touch++;
        }
    };
    vector<int> L,R,B,T; int lt,rt,bt,tt;
    side(B_LEFT,L,lt); side(B_RIGHT,R,rt); side(B_BOTTOM,B,bt); side(B_TOP,T,tt);
    int gain=(N<100)?4:1;
    if((int)L.size()-lt>=gain){ double mw=0; for(int i:L) mw=max(mw,out[i].w); double xe=xmin-mw-MARGIN,yc=ymin;
        sort(L.begin(),L.end(),[&](int a,int b){return out[a].y<out[b].y;}); for(int i:L){ out[i]={xe,yc,out[i].w,out[i].h}; yc+=out[i].h+MARGIN; } }
    if((int)R.size()-rt>=gain){ double mw=0; for(int i:R) mw=max(mw,out[i].w); double xe=xmax+mw+MARGIN,yc=ymin;
        sort(R.begin(),R.end(),[&](int a,int b){return out[a].y<out[b].y;}); for(int i:R){ out[i]={xe-out[i].w,yc,out[i].w,out[i].h}; yc+=out[i].h+MARGIN; } }
    if((int)B.size()-bt>=gain){ double mh=0; for(int i:B) mh=max(mh,out[i].h); double ye=ymin-mh-MARGIN,xc=xmin;
        sort(B.begin(),B.end(),[&](int a,int b){return out[a].x<out[b].x;}); for(int i:B){ out[i]={xc,ye,out[i].w,out[i].h}; xc+=out[i].w+MARGIN; } }
    if((int)T.size()-tt>=gain){ double mh=0; for(int i:T) mh=max(mh,out[i].h); double ye=ymax+mh+MARGIN,xc=xmin;
        sort(T.begin(),T.end(),[&](int a,int b){return out[a].x<out[b].x;}); for(int i:T){ out[i]={xc,ye-out[i].h,out[i].w,out[i].h}; xc+=out[i].w+MARGIN; } }
}

static double layout_score(const vector<XYWH>& p){
    double xmin=1e18,ymin=1e18,xmax=-1e18,ymax=-1e18;
    for (auto&q:p){xmin=min(xmin,q.x);ymin=min(ymin,q.y);xmax=max(xmax,q.x+q.w);ymax=max(ymax,q.y+q.h);}
    double area=(xmax-xmin)*(ymax-ymin), hpwl=approx_hpwl(p);
    int bv=count_boundary_violations(p), gf=count_group_fragments(p);
    double hw=(N>=116)?0.12:0.06;
    return area+hw*hpwl+150000.0*bv+6500.0*gf;
}

// ─── compaction (squeeze void out of a packed layout) ─────────────────────────
// The greedy packer fills its outline frame but leaves ~27% internal void
// (density 1.31 vs the baseline's 1.035); the frame-scale lever is exhausted, so
// the only remaining area lever is post-pack compaction. These directional packs
// slide every non-preplaced block toward one face as far as it can go without
// overlap, preserving the left-right / down-up order of every space-conflicting
// pair (so the result is provably overlap-free) and pinning preplaced blocks.
// Packing toward a face also pulls scattered cluster members into abutment, so it
// often REDUCES grouping fragments too. Boundary edge-contact is restored
// afterwards by the existing final_*_nudge passes, and the whole thing is
// downside-protected: compact_layout returns the best candidate by layout_score
// (which includes the original), so a pack that worsens area/hpwl/violations is
// simply not selected. ICCAD_NO_COMPACT=1 disables it.
static bool COMPACT = true;
static int COMPACT_ITERS = 8; // iterative compaction rounds (ICCAD_COMPACT_ITERS)
static vector<XYWH> pack_x(const vector<XYWH>& in, int dir, double anchor){
    int n=in.size(); vector<XYWH> out=in; vector<char> placed(n,0);
    vector<int> ord(n); for(int i=0;i<n;i++) ord[i]=i;
    if (dir<0) sort(ord.begin(),ord.end(),[&](int a,int b){return in[a].x<in[b].x;});
    else       sort(ord.begin(),ord.end(),[&](int a,int b){return in[a].x+in[a].w>in[b].x+in[b].w;});
    for (int idx=0;idx<n;idx++){ int i=ord[idx];
        if (blocks[i].is_preplaced){ placed[i]=1; continue; }   // out[i]=in[i]
        if (dir<0){
            double lb=anchor;
            for (int j=0;j<n;j++) if (placed[j] &&
                out[j].y<in[i].y+in[i].h-TOL && in[i].y<out[j].y+out[j].h-TOL)
                lb=max(lb,out[j].x+out[j].w);
            out[i].x=min(in[i].x,lb);
        } else {
            double ub=anchor;
            for (int j=0;j<n;j++) if (placed[j] &&
                out[j].y<in[i].y+in[i].h-TOL && in[i].y<out[j].y+out[j].h-TOL)
                ub=min(ub,out[j].x);
            out[i].x=max(in[i].x,ub-in[i].w);
        }
        placed[i]=1;
    }
    return out;
}
static vector<XYWH> pack_y(const vector<XYWH>& in, int dir, double anchor){
    int n=in.size(); vector<XYWH> out=in; vector<char> placed(n,0);
    vector<int> ord(n); for(int i=0;i<n;i++) ord[i]=i;
    if (dir<0) sort(ord.begin(),ord.end(),[&](int a,int b){return in[a].y<in[b].y;});
    else       sort(ord.begin(),ord.end(),[&](int a,int b){return in[a].y+in[a].h>in[b].y+in[b].h;});
    for (int idx=0;idx<n;idx++){ int i=ord[idx];
        if (blocks[i].is_preplaced){ placed[i]=1; continue; }
        if (dir<0){
            double lb=anchor;
            for (int j=0;j<n;j++) if (placed[j] &&
                out[j].x<in[i].x+in[i].w-TOL && in[i].x<out[j].x+out[j].w-TOL)
                lb=max(lb,out[j].y+out[j].h);
            out[i].y=min(in[i].y,lb);
        } else {
            double ub=anchor;
            for (int j=0;j<n;j++) if (placed[j] &&
                out[j].x<in[i].x+in[i].w-TOL && in[i].x<out[j].x+out[j].w-TOL)
                ub=min(ub,out[j].y);
            out[i].y=max(in[i].y,ub-in[i].h);
        }
        placed[i]=1;
    }
    return out;
}
// N_soft (max possible soft violations) = #boundary blocks + Σ(cluster-1) +
// Σ(mib distinct shapes-1). Constant during compaction (positions don't change
// shapes), so computed once for the candidate-selection proxy.
static double compute_nsoft(){
    int nsoft=0; map<int,int> clsz; map<int,set<pair<long long,long long>>> mibsh;
    for (int i=0;i<N;i++){
        if (blocks[i].boundary) nsoft++;
        if (blocks[i].cluster>0) clsz[blocks[i].cluster]++;
        if (blocks[i].mib>0) mibsh[blocks[i].mib].insert({llround(dims[i].first*1e4),llround(dims[i].second*1e4)});
    }
    for (auto&kv:clsz)  nsoft+=max(0,kv.second-1);
    for (auto&kv:mibsh) nsoft+=max(0,(int)kv.second.size()-1);
    return max(nsoft,1);
}
// csc = (area + hw*hpwl)*exp(2*(bv+gf)/nsoft): proxy of the TRUE cost shape.
// layout_score weights boundary 150000 but grouping only 6500, so it loves a pack
// that trades a boundary miss for a cluster fragment — yet the real cost
// normalises (bv+gf+vmb)/nsoft inside exp(), so that trade is neutral (or a loss
// once it also lifts hpwl). csc weights bv and gf equally. Used to select among
// compaction candidates (M10) and, under ICCAD_FRAME_CSC, among per-frame
// compacted+pushed finalists.
static double csc_of(const vector<XYWH>& p, double nsoft){
    double x0=1e18,y0=1e18,x1=-1e18,y1=-1e18;
    for (auto&q:p){x0=min(x0,q.x);y0=min(y0,q.y);x1=max(x1,q.x+q.w);y1=max(y1,q.y+q.h);}
    double area=(x1-x0)*(y1-y0), hpwl=approx_hpwl(p);
    int bv=count_boundary_violations(p), gf=count_group_fragments(p);
    double hw=(N>=116)?0.12:0.06;
    return (area+hw*hpwl)*exp(2.0*(bv+gf)/nsoft);
}
static vector<XYWH> compact_layout(const vector<XYWH>& base){
    if (!COMPACT) return base;
    double xmin=1e18,ymin=1e18,xmax=-1e18,ymax=-1e18;
    for (auto&q:base){xmin=min(xmin,q.x);ymin=min(ymin,q.y);xmax=max(xmax,q.x+q.w);ymax=max(ymax,q.y+q.h);}
    // Select with the true-cost proxy csc, not layout_score (see csc_of comment);
    // keeps the original whenever no pack truly wins.
    double nsoft=compute_nsoft();
    auto csc=[&](const vector<XYWH>& p)->double{ return csc_of(p,nsoft); };
    // Only the single-block boundary nudge here (final_group_boundary_nudge would
    // re-introduce the bv-for-area trade the proxy is trying to avoid).
    auto finish=[&](vector<XYWH> L)->vector<XYWH>{ final_boundary_nudge(L); return L; };
    auto xL=pack_x(base,-1,xmin), xR=pack_x(base,1,xmax);
    auto yD=pack_y(base,-1,ymin), yU=pack_y(base,1,ymax);
    vector<XYWH> best=base; double bsc=csc(base);
    auto consider=[&](vector<XYWH> c){ double sc=csc(c); if (sc<bsc){bsc=sc;best=c;} };
    consider(finish(xL)); consider(finish(xR));
    consider(finish(yD)); consider(finish(yU));
    for (const vector<XYWH>* xx:{&xL,&xR}){          // X then Y
        consider(finish(pack_y(*xx,-1,ymin)));
        consider(finish(pack_y(*xx,1,ymax)));
    }
    for (const vector<XYWH>* yy:{&yD,&yU}){          // Y then X
        consider(finish(pack_x(*yy,-1,xmin)));
        consider(finish(pack_x(*yy,1,xmax)));
    }
    // Iterative passes: after the initial 12-candidate round, continue packing
    // from the current best until convergence. Each Y-pack shifts rows so a
    // subsequent X-pack can reclaim new slack (and vice versa). Stop when no
    // single-axis pack improves the csc proxy. ICCAD_COMPACT_ITERS (default 8).
    for (int iter=0; iter<COMPACT_ITERS; iter++){
        double cx0=1e18,cy0=1e18,cx1=-1e18,cy1=-1e18;
        for (auto&q:best){cx0=min(cx0,q.x);cy0=min(cy0,q.y);cx1=max(cx1,q.x+q.w);cy1=max(cy1,q.y+q.h);}
        double prev_bsc=bsc;
        const vector<XYWH> cur=best;
        consider(finish(pack_x(cur,-1,cx0)));
        consider(finish(pack_x(cur,1,cx1)));
        consider(finish(pack_y(cur,-1,cy0)));
        consider(finish(pack_y(cur,1,cy1)));
        if (bsc>=prev_bsc) break;
    }
    return best;
}

// ─── post-placement HPWL slack push ───────────────────────────────────────────
// hgap is the dominant cost lever (weighted 0.404 vs agap 0.23 vs vrel 0.04).
// Compaction packs toward frame faces (area) and can spread connected blocks apart,
// lifting HPWL. This slides blocks toward their connectivity-weighted L1-median
// within the void available inside the CURRENT bbox (coordinate descent,
// Gauss-Seidel). Two downside-free movable categories:
//   * FREE SINGLE (boundary==0, cluster==0, not preplaced): slide BOTH x and y.
//   * BOUNDARY SINGLE (boundary!=0, cluster==0, not preplaced, not a corner):
//     slide ONLY the free axis -- LEFT/RIGHT blocks slide y (x pinned on the edge),
//     TOP/BOTTOM slide x (y pinned). The constrained-axis coord never changes, so
//     edge-contact (hence bv) is preserved. (M15)
// Downside-free by construction: free singles touch no boundary/grouping rule;
// boundary singles keep their edge contact; every move stays inside the bbox, so
// the bbox is non-increasing. A bbox-shrink can only un-satisfy a boundary block on
// the shrunk edge, but a SATISFIED boundary block lies exactly on that edge and so
// co-defines the extreme -- it pins the edge and cannot be un-satisfied. Hence
// area / bv / gf / mib are all unchanged; only HPWL drops (a move is accepted only
// if it strictly lowers it). The void interval [lo,hi] is the gap to the nearest
// blocks overlapping in the other axis, so no overlap is ever created. Validated in
// Python (dbg_hpwl_push.py): free-single -0.63%, +boundary-axis a further -0.11%,
// both 0/100 regressions, 0/100 bv-or-gf increases. ICCAD_NO_PUSH=1 disables.
//
// Rigid free-cluster slide was tried (dbg_hpwl_push.py ENABLE_CLUSTER) and dropped:
// the compound-item packer leaves clusters with no slack (only 1 cluster across all
// 100 cases could move at all), and a FP rigid translation breaks the cluster's
// exact internal abutments at the ULP level -> shapely (exact) flags a spurious
// grouping fragment (the M10 precision hazard). Net effect +0.004% with 1 regression.
//
// M16 same-size swap: swap the positions of two non-preplaced, non-cluster blocks
// with EXACTLY equal (w,h) and equal boundary code. The geometry multiset is
// unchanged -> bbox/area identical, no overlap can appear, the number of blocks
// touching each required edge is unchanged (bv identical), no cluster member moves
// (gf identical), dims unchanged (mib identical). Only HPWL changes; accepted only
// on strict decrease. Soft blocks with equal area and equal boundary code get
// bit-identical dims from default_soft_dim, so exact-equality grouping finds real
// partners. Repairing VIOLATING boundary singles by sliding them to their edge was
// also prototyped and is a DEAD END: all 34 violating singles across 100 cases are
// blocked by other blocks (123 violating cluster members and 45 preplaced are
// immovable anyway) -- see dbg_hpwl_push.py ENABLE_VIOLATING / dbg_vio_stats.py.
static bool PUSH = true;
static bool PUSH_BND = true;     // M15 boundary-axis slide (ICCAD_NO_BND_PUSH=1 -> M14 only)
static bool PUSH_SWAP = true;    // M16 same-size swap (ICCAD_NO_SWAP=1 -> M15 only)
static bool PUSH_JUMP = true;    // M24 cross-obstacle rip-and-reinsert (ICCAD_NO_JUMP=1 -> M16 only)
static int PUSH_PASSES = 8;
static double wmedian(vector<pair<double,double>>& t){   // weighted L1 median
    if (t.empty()) return 0.0;
    sort(t.begin(),t.end());
    double total=0; for(auto&q:t) total+=q.second;
    double acc=0; for(auto&q:t){ acc+=q.second; if (acc>=total/2.0) return q.first; }
    return t.back().first;
}
static void hpwl_push(vector<XYWH>& p){
    if (!PUSH) return;
    double xmin=1e18,ymin=1e18,xmax=-1e18,ymax=-1e18;
    for (auto&q:p){xmin=min(xmin,q.x);ymin=min(ymin,q.y);xmax=max(xmax,q.x+q.w);ymax=max(ymax,q.y+q.h);}
    // free singles slide on both axes; boundary singles (exactly one axis pinned by
    // their edge) slide only on the free axis. corner boundary blocks (both LR & TB
    // bits) and clustered/preplaced blocks are immovable here.
    vector<int> freeb, bndb;
    for (int i=0;i<N;i++){
        if (blocks[i].is_preplaced || blocks[i].cluster!=0) continue;
        int code=blocks[i].boundary;
        if (code==0) freeb.push_back(i);
        else if (PUSH_BND) {
            bool lr=(code&(B_LEFT|B_RIGHT))!=0, tb=(code&(B_TOP|B_BOTTOM))!=0;
            if (lr!=tb) bndb.push_back(i);   // exactly one axis constrained
        }
    }
    if (freeb.empty() && bndb.empty()) return;
    auto hwire=[&](int i,double x,double y)->double{
        double cx=x+p[i].w/2, cy=y+p[i].h/2, h=0;
        for (auto&nb:b2b_adj[i]){ double ncx=p[nb.first].x+p[nb.first].w/2, ncy=p[nb.first].y+p[nb.first].h/2;
            h+=nb.second*(fabs(cx-ncx)+fabs(cy-ncy)); }
        for (auto&pn:p2b_adj[i]) h+=pn.second*(fabs(cx-pins[pn.first].first)+fabs(cy-pins[pn.first].second));
        return h;
    };
    // slide block i along one axis to its weighted L1 median within the void; keep
    // the other axis fixed. returns true if it moved (strictly lower HPWL).
    auto slide_x=[&](int i)->bool{
        double w=p[i].w, h=p[i].h, lo=xmin, hi=xmax-w;
        for (int j=0;j<N;j++){ if (j==i) continue;
            if (p[j].y<p[i].y+h-TOL && p[i].y<p[j].y+p[j].h-TOL){      // overlap in y
                if (p[j].x+p[j].w<=p[i].x+TOL) lo=max(lo,p[j].x+p[j].w);
                else if (p[j].x>=p[i].x+w-TOL)  hi=min(hi,p[j].x-w);
            }
        }
        vector<pair<double,double>> tx;
        for (auto&nb:b2b_adj[i]) tx.push_back({p[nb.first].x+p[nb.first].w/2, nb.second});
        for (auto&pn:p2b_adj[i]) tx.push_back({pins[pn.first].first, pn.second});
        if (tx.empty() || hi<lo-TOL) return false;
        double nx=min(max(wmedian(tx)-w/2, lo), hi);
        if (fabs(nx-p[i].x)>TOL && hwire(i,nx,p[i].y)<hwire(i,p[i].x,p[i].y)-TOL){ p[i].x=nx; return true; }
        return false;
    };
    auto slide_y=[&](int i)->bool{
        double w=p[i].w, h=p[i].h, lo=ymin, hi=ymax-h;
        for (int j=0;j<N;j++){ if (j==i) continue;
            if (p[j].x<p[i].x+w-TOL && p[i].x<p[j].x+p[j].w-TOL){      // overlap in x
                if (p[j].y+p[j].h<=p[i].y+TOL) lo=max(lo,p[j].y+p[j].h);
                else if (p[j].y>=p[i].y+h-TOL)  hi=min(hi,p[j].y-h);
            }
        }
        vector<pair<double,double>> ty;
        for (auto&nb:b2b_adj[i]) ty.push_back({p[nb.first].y+p[nb.first].h/2, nb.second});
        for (auto&pn:p2b_adj[i]) ty.push_back({pins[pn.first].second, pn.second});
        if (ty.empty() || hi<lo-TOL) return false;
        double ny=min(max(wmedian(ty)-h/2, lo), hi);
        if (fabs(ny-p[i].y)>TOL && hwire(i,p[i].x,ny)<hwire(i,p[i].x,p[i].y)-TOL){ p[i].y=ny; return true; }
        return false;
    };
    // ── M24 jump: rip a single out and re-insert it at the best non-overlapping
    // slot inside the frozen bbox. The slides above only move a block within its
    // own void interval (they can never cross an obstacle) and the M16 swap only
    // fires on bit-identical partners, so a block whose weighted median lies
    // beyond a blocking rect keeps its full HPWL detour forever. Candidates: the
    // 8 corner-aligned abutment slots against every other rect + the 4 bbox
    // corners, clamped into the frozen bbox, deduped; strict per-block HPWL
    // decrease only (sorted by score, first feasible = best). Downside-free by
    // the slide argument: targets stay inside the frozen bbox (area non-
    // increasing; satisfied boundary blocks co-define their edges so bv cannot
    // rise), no cluster member moves (gf), dims unchanged (mib). Boundary
    // singles jump 1-D along their free axis only (pinned coordinate kept ->
    // edge-contact preserved).
    auto fits=[&](int i,double x,double y)->bool{
        double w=p[i].w, h=p[i].h;
        for (int j=0;j<N;j++){ if (j==i) continue;
            if (x<p[j].x+p[j].w-TOL && p[j].x<x+w-TOL &&
                y<p[j].y+p[j].h-TOL && p[j].y<y+h-TOL) return false; }
        return true;
    };
    auto try_jump=[&](int i,bool free_xy)->bool{
        if (b2b_adj[i].empty() && p2b_adj[i].empty()) return false;
        double w=p[i].w, h=p[i].h;
        vector<pair<double,double>> cands;
        auto add=[&](double x,double y){
            cands.push_back({min(max(x,xmin),xmax-w), min(max(y,ymin),ymax-h)}); };
        if (free_xy){
            cands.reserve(8*(size_t)N+4);
            add(xmin,ymin); add(xmax-w,ymin); add(xmin,ymax-h); add(xmax-w,ymax-h);
            for (int j=0;j<N;j++){ if (j==i) continue;
                double jx=p[j].x, jy=p[j].y, jw=p[j].w, jh=p[j].h;
                add(jx+jw,jy); add(jx+jw,jy+jh-h); add(jx-w,jy); add(jx-w,jy+jh-h);
                add(jx,jy+jh); add(jx+jw-w,jy+jh); add(jx,jy-h); add(jx+jw-w,jy-h);
            }
        } else if (blocks[i].boundary&(B_LEFT|B_RIGHT)){     // x pinned: jump along y
            add(p[i].x,ymin); add(p[i].x,ymax-h);
            for (int j=0;j<N;j++){ if (j==i) continue;
                add(p[i].x,p[j].y+p[j].h); add(p[i].x,p[j].y-h); }
        } else {                                             // y pinned: jump along x
            add(xmin,p[i].y); add(xmax-w,p[i].y);
            for (int j=0;j<N;j++){ if (j==i) continue;
                add(p[j].x+p[j].w,p[i].y); add(p[j].x-w,p[i].y); }
        }
        sort(cands.begin(),cands.end());
        cands.erase(unique(cands.begin(),cands.end()),cands.end());
        double hcur=hwire(i,p[i].x,p[i].y);
        vector<tuple<double,double,double>> scored; scored.reserve(cands.size());
        for (auto&c:cands) scored.push_back(make_tuple(hwire(i,c.first,c.second),c.first,c.second));
        sort(scored.begin(),scored.end());
        for (auto&s:scored){
            double hc=get<0>(s), x=get<1>(s), y=get<2>(s);
            if (hc>=hcur-TOL) break;                         // sorted: nothing better left
            if (fabs(x-p[i].x)<=TOL && fabs(y-p[i].y)<=TOL) continue;
            if (fits(i,x,y)){ p[i].x=x; p[i].y=y; return true; }
        }
        return false;
    };
    // M16 swap groups: exact (w,h,boundary) over non-preplaced non-cluster blocks.
    // dims never change inside the push loop, so build once.
    vector<vector<int>> swap_groups;
    if (PUSH_SWAP){
        map<tuple<double,double,int>,vector<int>> sg;
        for (int i=0;i<N;i++){
            if (blocks[i].is_preplaced || blocks[i].cluster!=0) continue;
            sg[make_tuple(p[i].w,p[i].h,blocks[i].boundary)].push_back(i);
        }
        for (auto&kv:sg) if (kv.second.size()>=2) swap_groups.push_back(kv.second);
    }
    auto swap_pass=[&]()->bool{
        bool any=false;
        for (auto&mem:swap_groups){
            for (size_t a=0;a<mem.size();a++) for (size_t b=a+1;b<mem.size();b++){
                int i=mem[a], j=mem[b];
                double h0=hwire(i,p[i].x,p[i].y)+hwire(j,p[j].x,p[j].y);
                swap(p[i].x,p[j].x); swap(p[i].y,p[j].y);
                // an i-j edge contributes identically before/after and cancels
                double h1=hwire(i,p[i].x,p[i].y)+hwire(j,p[j].x,p[j].y);
                if (h1<h0-TOL) any=true;
                else { swap(p[i].x,p[j].x); swap(p[i].y,p[j].y); }
            }
        }
        return any;
    };
    for (int pass=0; pass<PUSH_PASSES; pass++){
        bool moved=false;
        for (int i:bndb){                                  // boundary: free axis only
            if (blocks[i].boundary&(B_LEFT|B_RIGHT)) { if (slide_y(i)) moved=true; }
            else                                          { if (slide_x(i)) moved=true; }
        }
        for (int i:freeb){ if (slide_x(i)) moved=true; if (slide_y(i)) moved=true; }
        if (PUSH_JUMP){
            for (int i:freeb) if (try_jump(i,true))  moved=true;
            if (PUSH_BND) for (int i:bndb) if (try_jump(i,false)) moved=true;
        }
        if (PUSH_SWAP && swap_pass()) moved=true;
        if (!moved) break;
    }
}

// ─── fallback ─────────────────────────────────────────────────────────────────
static vector<XYWH> shelf_fallback(const vector<int>& order){
    vector<XYWH> p=pos; double x0=0;
    for (int i=0;i<N;i++) if (placed[i]) x0=max(x0,pos[i].x+pos[i].w);
    x0+=1.0;
    double tot=0; for(int i=0;i<N;i++) tot+=dims[i].first*dims[i].second;
    double row_w=sqrt(max(1.0,tot))*2.0, cx=x0, cy=0, rowh=0;
    for (int b:order){ double w=dims[b].first,h=dims[b].second;
        if (cx-x0+w>row_w&&cx>x0){cx=x0;cy+=rowh;rowh=0;}
        p[b]={cx,cy,w,h}; cx+=w; rowh=max(rowh,h); }
    return p;
}

// ─── solve ────────────────────────────────────────────────────────────────────
static void solve() {
    double t_solve=m46_now(), t_stage=t_solve;
    dims.assign(N,{1,1}); pos.assign(N,{0,0,0,0}); placed.assign(N,0);
    for (int i=0;i<N;i++){
        if (blocks[i].is_preplaced){ pos[i]={blocks[i].tx,blocks[i].ty,blocks[i].tw,blocks[i].th}; dims[i]={blocks[i].tw,blocks[i].th}; placed[i]=1; }
        else if (blocks[i].is_fixed){ double w=blocks[i].tw,h=blocks[i].th; if(w<=0||h<=0){double s=sqrt(blocks[i].area);w=h=s;} dims[i]={w,h}; }
        else dims[i]=default_soft_dim(blocks[i].area, blocks[i].boundary);
    }
    apply_safe_mib_dims();
    // M33 cluster-member uniform aspect: reshape pure-movable INTERIOR cluster members
    // before make_group_item reads dims. boundary!=0 keep their LR/TB aspect; mib!=0 keep
    // the unified shape. Default 1.0 -> skipped -> bit-identical.
    if (CLUSTER_ASPECT != 1.0)
        for (int i=0;i<N;i++)
            if (blocks[i].cluster>0 && blocks[i].mib==0 && blocks[i].boundary==0
                && !blocks[i].is_preplaced && !blocks[i].is_fixed && blocks[i].area>0){
                double A=blocks[i].area, w=sqrt(A*CLUSTER_ASPECT); dims[i]={w, A/w};
            }
    estimate_anchors();

    b2b_adj.assign(N,{}); p2b_adj.assign(N,{});
    for (auto& e:b2b_edges){ if(e.w<=0||e.i<0||e.j<0||e.i>=N||e.j>=N) continue;
        b2b_adj[e.i].push_back({e.j,e.w}); b2b_adj[e.j].push_back({e.i,e.w}); }
    for (auto& e:p2b_edges){ if(e.w<=0||e.j<0||e.j>=N||e.i<0||e.i>=(int)pins.size()) continue;
        p2b_adj[e.j].push_back({e.i,e.w}); }

    // build items: pure-movable clusters -> compound item; mixed clusters
    // (preplaced + movable) -> anchored (movable attach to preplaced walls in
    // pack_in_frame's first-pass, and also appear as singles as a fallback);
    // everything else -> singles.
    map<int,vector<int>> cluster_map;
    for (int i=0;i<N;i++) if (blocks[i].cluster>0)
        cluster_map[blocks[i].cluster].push_back(i);
    anchored_clusters.clear();
    vector<char> used(N,0);
    vector<Item> items;
    for (auto& kv:cluster_map){
        vector<int> mov, pre;
        for (int b:kv.second){ if (blocks[b].is_preplaced) pre.push_back(b); else mov.push_back(b); }
        if (!pre.empty() && !mov.empty()){
            anchored_clusters.push_back({pre, mov});   // handled in pack_in_frame
            continue;
        }
        if (mov.size()<2) continue;
        Item it=make_group_item(mov);
        set_item_anchor(it); items.push_back(it);
        for (int b:mov) used[b]=1;
    }
    for (int i=0;i<N;i++){
        if (blocks[i].is_preplaced||used[i]) continue;
        Item it; it.blocks={i}; it.offs={{0,0}}; finalize_item(it); set_item_anchor(it);
        items.push_back(it);
    }
    m46.build=m46_now()-t_stage; t_stage=m46_now();
    if (WIRE_ORDER) {
        // Sort by total wire weight first (hpwl-driven packing: most-connected items placed first)
        sort(items.begin(),items.end(),[](const Item&a,const Item&b){
            if (fabs(a.total_wire-b.total_wire)>1e-6) return a.total_wire>b.total_wire;
            if (a.bscore!=b.bscore) return a.bscore>b.bscore;
            if (a.blocks.size()!=b.blocks.size()) return a.blocks.size()>b.blocks.size();
            double aa=a.w*a.h,ab=b.w*b.h; if (fabs(aa-ab)>TOL) return aa>ab;
            return max(a.w,a.h)>max(b.w,b.h);
        });
    } else if (WIRE_TIEBREAK) {
        // boundary priority intact (WIRE_ORDER's failure was wire-first ordering,
        // vBd 390); inside each bscore class place most-connected items first so
        // the greedy wire term sees its heavy neighbours early.
        sort(items.begin(),items.end(),[](const Item&a,const Item&b){
            if (a.bscore!=b.bscore) return a.bscore>b.bscore;
            if (fabs(a.total_wire-b.total_wire)>1e-6) return a.total_wire>b.total_wire;
            if (a.blocks.size()!=b.blocks.size()) return a.blocks.size()>b.blocks.size();
            double aa=a.w*a.h,ab=b.w*b.h; if (fabs(aa-ab)>TOL) return aa>ab;
            return max(a.w,a.h)>max(b.w,b.h);
        });
    } else {
        sort(items.begin(),items.end(),[](const Item&a,const Item&b){
            if (a.bscore!=b.bscore) return a.bscore>b.bscore;
            if (a.blocks.size()!=b.blocks.size()) return a.blocks.size()>b.blocks.size();
            double aa=a.w*a.h,ab=b.w*b.h; if (fabs(aa-ab)>TOL) return aa>ab;
            return max(a.w,a.h)>max(b.w,b.h);
        });
    }
    if (WIRE_BFS) {
        // BFS-connectivity ordering layered over the base sort: bscore classes stay
        // intact (wire-first ordering across classes was the WIRE_ORDER failure,
        // vBd 390), but inside each class items are emitted greedily by largest
        // edge weight into the already-ordered set (any class) plus preplaced
        // blocks. The greedy wire term then sees the placed side of an item's
        // heavy edges instead of placing early items blind. Ties (attachment 0,
        // e.g. unconnected items) keep the base order, so cases without wire
        // signal reduce to the base/WT behaviour.
        int I=(int)items.size();
        vector<int> blk2item(N,-1);
        for (int t=0;t<I;t++) for (int b:items[t].blocks) blk2item[b]=t;
        vector<double> attach(I,0.0);
        for (int t=0;t<I;t++) for (int b:items[t].blocks){
            for (auto& nb:b2b_adj[b])
                if (blocks[nb.first].is_preplaced) attach[t]+=nb.second;
            if (BFS_PIN) for (auto& pw:p2b_adj[b]) attach[t]+=pw.second;
        }
        vector<Item> bfs_out; bfs_out.reserve(I);
        vector<char> emitted(I,0);
        for (int s=0;s<I;){
            int e=s; while (e<I && items[e].bscore==items[s].bscore) e++;
            for (int k=s;k<e;k++){
                int pick=-1; double pickv=0;
                for (int t=s;t<e;t++){
                    if (emitted[t]) continue;
                    double v=attach[t];
                    if (BFS_NORM) v/=sqrt(max(items[t].w*items[t].h,1e-12));
                    if (pick<0 || v>pickv+1e-9){ pick=t; pickv=v; }
                }
                emitted[pick]=1; bfs_out.push_back(items[pick]);
                for (int b:items[pick].blocks) for (auto& nb:b2b_adj[b]){
                    int ot=blk2item[nb.first];
                    if (ot>=0 && !emitted[ot]) attach[ot]+=nb.second;
                }
            }
            s=e;
        }
        items.swap(bfs_out);
    }
    if (CLUSTER_ORD==1 || CLUSTER_ORD==2) {
        // Compound cluster items' position inside their bscore class is an axis the
        // base sort never moves (its size key always puts them first; WT/BFS shuffle
        // them by wire). Force them to the class front (1) or back (2): back packs
        // singles tight first and lets the big compound land on the perimeter.
        int I=(int)items.size();
        for (int s=0;s<I;){
            int e=s; while (e<I && items[e].bscore==items[s].bscore) e++;
            stable_partition(items.begin()+s, items.begin()+e,
                [](const Item& it){ return (it.blocks.size()>1) == (CLUSTER_ORD==1); });
            s=e;
        }
    }

    // Oracle-perm probe (OFFLINE only, never shipped). If ICCAD_ORDER_FILE is set,
    // reorder items by an externally supplied per-block key (computed offline from
    // fp_sol). ICCAD_ORDER_GLOBAL ignores bscore (pure global sweep); otherwise the
    // key is a within-bscore-class tiebreak. Measures the ordering ceiling: decides
    // whether pack-order still has ore (>=0.5%) or is permanently closed (<0.2%).
    if (const char* of=getenv("ICCAD_ORDER_FILE")){
        vector<double> okey(N, 1e18);
        if (FILE* fp=fopen(of,"r")){
            int id; double k;
            while (fscanf(fp,"%d %lf",&id,&k)==2) if (id>=0&&id<N) okey[id]=k;
            fclose(fp);
        }
        auto ik=[&](const Item& it){ double m=1e18; for(int b:it.blocks) m=min(m,okey[b]); return m; };
        bool global = getenv("ICCAD_ORDER_GLOBAL")!=nullptr;
        stable_sort(items.begin(),items.end(),[&](const Item& a,const Item& b){
            if (!global && a.bscore!=b.bscore) return a.bscore>b.bscore;
            return ik(a)<ik(b);
        });
    }

    vector<int> order; for(int i=0;i<N;i++) if(!blocks[i].is_preplaced) order.push_back(i);

    m46.order=m46_now()-t_stage;
    auto frames=frame_candidates();
    // A few tight frames win: trying all overshoots layout_score's 150000*bv weight
    // (picks low-violation but area-bloated outlines). 4/5 measured best (deterministic).
    int max_trials=(N>=60)?4:5;
    auto run_frame=[&](double fw,double fh,bool prev,const vector<XYWH>& pv,vector<XYWH>& out)->bool{
        use_prev=prev; if (prev) prev_pos=pv;
        double t0=m46_now();
        bool ok=pack_in_frame(fw,fh,items,out);
        m46.pack+=m46_now()-t0; m46.packs++;
        use_prev=false;
        if (!ok){ m46.pack_fail++; return false; }
        t0=m46_now();
        final_boundary_nudge(out);
        final_group_boundary_nudge(out);
        final_single_edge_escape(out);
        m46.nudge+=m46_now()-t0;
        return true;
    };
    vector<Item> items_base=items;   // original sorted order, captured once (reframe reuses it)
    // M26 reframe: wrap the whole frames -> best(compacted+pushed) pipeline as a
    // reusable unit so it can be re-run on a measured-bbox-seeded frame set.
    auto run_pipeline=[&](const vector<pair<double,double>>& frms)->vector<XYWH>{
    double t_pl=m46_now();
    vector<XYWH> best; bool have_best=false; double best_score=1e300; int trials=0;
    int m49_best_trial=-1, m49_best_last_imp=-1;   // M49 trace: winner attribution
    for (auto& f:frms){
        items=items_base;
        vector<XYWH> c1, dummy;
        if (!run_frame(f.first,f.second,false,dummy,c1)) continue;
        trials++; m46.frames++;
        double sc=layout_score(c1);
        double m49_init_sc=sc; int m49_last_imp=-1;   // M49 trace (per frame)
        // Pack-order pair-swap hill-climb: greedy ordering misplaces an item the
        // force-directed refinement can only nudge, not relocate. Swapping two
        // order positions re-routes the whole downstream pack — a jump move.
        // Compared pack-once vs pack-once (fair), downside-protected, before
        // refinement so the better order seeds the guide passes.
        double t_sm=m46_now();
        if (ORDER_SWAP>0 && (int)items.size()>2){
            vector<int> top;
            for (int i=0;i<(int)items.size();i++) top.push_back(i);
            sort(top.begin(),top.end(),[&](int a,int b){
                return items[a].total_wire>items[b].total_wire; });
            top.resize(min((size_t)ORDER_SWAP, top.size()));
            for (size_t a=0;a+1<top.size();a++) for (size_t b=a+1;b<top.size();b++){
                swap(items[top[a]],items[top[b]]);
                vector<XYWH> c2;
                bool ok=run_frame(f.first,f.second,false,dummy,c2);
                double sc2=ok?layout_score(c2):1e300;
                if (sc2<sc-1e-9){ sc=sc2; c1.swap(c2); }
                else swap(items[top[a]],items[top[b]]);
            }
        }
        // Pack-order relocation hill-climb: pull one top-K wire item out and insert
        // it at another top-K item's slot (the segment between shifts by one) — a
        // structural jump ORDER_SWAP can't reach since swap fixes everyone else.
        // Same protocol: pack-once comparisons, strict improvement only. Items are
        // tracked by their first block id because accepted moves shift positions.
        if (ORDER_MOVE>0 && (int)items.size()>2){
            vector<int> mtop;
            for (int i=0;i<(int)items.size();i++) mtop.push_back(i);
            sort(mtop.begin(),mtop.end(),[&](int a,int b){
                return items[a].total_wire>items[b].total_wire; });
            mtop.resize(min((size_t)ORDER_MOVE, mtop.size()));
            vector<int> ids; for (int t:mtop) ids.push_back(items[t].blocks[0]);
            auto pos_of=[&](int id)->int{
                for (int i=0;i<(int)items.size();i++)
                    if (items[i].blocks[0]==id) return i;
                return -1; };
            for (size_t a=0;a<ids.size();a++) for (size_t b=0;b<ids.size();b++){
                if (a==b) continue;
                int pa=pos_of(ids[a]), pb=pos_of(ids[b]);
                if (pa<0||pb<0||pa==pb) continue;
                if (pa<pb) rotate(items.begin()+pa, items.begin()+pa+1, items.begin()+pb+1);
                else       rotate(items.begin()+pb, items.begin()+pa,   items.begin()+pa+1);
                vector<XYWH> c2;
                bool ok=run_frame(f.first,f.second,false,dummy,c2);
                double sc2=ok?layout_score(c2):1e300;
                if (sc2<sc-1e-9){ sc=sc2; c1.swap(c2); }
                else {
                    if (pa<pb) rotate(items.begin()+pa, items.begin()+pb,   items.begin()+pb+1);
                    else       rotate(items.begin()+pb, items.begin()+pb+1, items.begin()+pa+1);
                }
            }
        }
        m46.swapmove+=m46_now()-t_sm;
        // refinement passes: re-pack with full-neighbor wire pulling toward the
        // previous layout (coordinate descent). Keep the best by layout_score per
        // frame (downside-protected); advance the guide each iteration to converge.
        if (REFINE){
            double t_rf=m46_now();
            vector<XYWH> guide=c1;
            m59_dump_state(trials-1,-1,sc,1,f.first,f.second,c1);   // pre-refine c1
            for (int r=0;r<REFINE_ITERS;r++){
                vector<XYWH> c2;
                m46.refine_passes++;
                if (!run_frame(f.first,f.second,true,guide,c2)) break;
                double sc2=layout_score(c2);
                bool m49_imp = sc2<sc;
                if (m49_imp){ sc=sc2; c1=c2; m49_last_imp=r; }
                m59_dump_state(trials-1,r,sc2,m49_imp?1:0,f.first,f.second,c2);
                if (REFINE_TRACE)
                    fprintf(stderr,"RTRACE pass f=%d r=%d sc=%.17g imp=%d h=%016llx\n",
                            trials-1, r, sc2, m49_imp?1:0,
                            m49_fnv(c2.data(), c2.size()*sizeof(XYWH)));
                // M46 opt-D: fixed point -> every later pass reproduces this
                // exact c2/sc2 (deterministic in guide) and can never update
                // sc again -> breaking here is result-identical.
                if (c2.size()==guide.size() &&
                    !memcmp(c2.data(), guide.data(), c2.size()*sizeof(XYWH))){
                    m46.refine_fixed++;
                    if (REFINE_TRACE)
                        fprintf(stderr,"RTRACE fixpoint f=%d r=%d\n", trials-1, r);
                    break;
                }
                guide.swap(c2);
            }
            m46.refine+=m46_now()-t_rf;
        }
        if (REFINE_TRACE)
            fprintf(stderr,"RTRACE fdone f=%d init=%.17g final=%.17g last_imp=%d\n",
                    trials-1, m49_init_sc, sc, m49_last_imp);
        if (!have_best||sc<best_score){ best_score=sc; best=c1; have_best=true;
            m49_best_trial=trials-1; m49_best_last_imp=m49_last_imp; }
        if (trials>=max_trials) break;
    }
    if (REFINE_TRACE)
        fprintf(stderr,"RTRACE winner f=%d last_imp=%d trials=%d\n",
                m49_best_trial, m49_best_last_imp, trials);
    if (!have_best) best=shelf_fallback(order);
    // Compaction squeezes void out of the SELECTED layout (downside-protected:
    // returns the original if no directional pack beats it by layout_score).
    // Applied to the single chosen frame, not per-frame: feeding the imperfect
    // layout_score proxy every frame's compacted variant lets it overfit and pick
    // a low-proxy / high-true-cost outline (same failure mode as trying too many
    // frames). Compacting only the winner matched the validated prototype.
    // A csc-pool variant (compact+push every frame finalist, pick by csc — the
    // "M17" experiment) was also tried and is a DEAD END: csc's fixed hw weight
    // mis-ranks across different outlines (it trades cluster fragments for
    // boundary violations; single-base 1.5197->1.5293) and as a portfolio profile
    // its oracle-min gain is +0.008% (1 case). Cross-frame selection needs the
    // wrapper's shapely proxy, which already does exactly this across profiles.
    { double t0=m46_now(); if (COMPACT) best=compact_layout(best); m46.compact+=m46_now()-t0; }
    // Post-compaction HPWL push: slide free singles toward connectivity centroids
    // into remaining void. Downside-free (area/bv/gf/mib unchanged), attacks the
    // dominant hgap. Must run after compaction (its frame-face packs spread wire).
    { double t0=m46_now(); hpwl_push(best); m46.push+=m46_now()-t0; }
    m46.pipeline+=m46_now()-t_pl;
    return best;
    };  // run_pipeline

    vector<XYWH> best = run_pipeline(frames);
    if (REFRAME){
        // Measure the compacted bbox; seed a small frame set at that aspect (a shape
        // the blocks demonstrably fit, unlike the dead M13 preplaced-frame attempt)
        // and re-run the pipeline. Output the single result -- NO internal cross-
        // layout selection (that was the dead M17 per-frame-csc path); the wrapper
        // proxy arbitrates this profile against the others.
        double x0=1e18,y0=1e18,x1=-1e18,y1=-1e18;
        for (auto&q:best){x0=min(x0,q.x);y0=min(y0,q.y);x1=max(x1,q.x+q.w);y1=max(y1,q.y+q.h);}
        double wc=max(x1-x0,1.0), hc=max(y1-y0,1.0);
        double max_iw=1,max_ih=1,pre_w=0,pre_h=0;
        for (int i=0;i<N;i++){ max_iw=max(max_iw,dims[i].first); max_ih=max(max_ih,dims[i].second);
            if (placed[i]){ pre_w=max(pre_w,pos[i].x+pos[i].w); pre_h=max(pre_h,pos[i].y+pos[i].h);} }
        vector<pair<double,double>> seeds; set<pair<long long,long long>> seen;
        for (double s:{1.00,1.02,1.05}) for (double am:{1.0,1.12,0.89}){
            double w=wc*s*am, h=hc*s/am;
            w=max(w,max(pre_w+MARGIN,max_iw+MARGIN)); h=max(h,max(pre_h+MARGIN,max_ih+MARGIN));
            auto key=make_pair((long long)llround(w*1e6),(long long)llround(h*1e6));
            if (seen.insert(key).second) seeds.push_back({w,h});
        }
        sort(seeds.begin(),seeds.end(),[](const pair<double,double>&A,const pair<double,double>&B){
            double aa=A.first*A.second,ab=B.first*B.second;
            if (fabs(aa-ab)>TOL) return aa<ab; return max(A.first,A.second)<max(B.first,B.second);});
        best = run_pipeline(seeds);
    }

    if (getenv("CONSTRUCTIVE_DEBUG"))
        fprintf(stderr,"[dbg] N=%d frames=%d reframe=%d bv=%d gf=%d\n",
                N,(int)frames.size(),REFRAME?1:0,
                count_boundary_violations(best),count_group_fragments(best));
    // Baseline-free proxy metrics for the portfolio selector (read from stderr).
    {
        double xmin=1e18,ymin=1e18,xmax=-1e18,ymax=-1e18;
        for (auto&q:best){xmin=min(xmin,q.x);ymin=min(ymin,q.y);xmax=max(xmax,q.x+q.w);ymax=max(ymax,q.y+q.h);}
        double area=(xmax-xmin)*(ymax-ymin), hpwl=approx_hpwl(best);
        int vbd=count_boundary_violations(best), vcl=count_group_fragments(best);
        map<int,set<pair<long long,long long>>> shp; int nsoft=0;
        map<int,int> clc;
        for (int i=0;i<N;i++){
            if (blocks[i].boundary) nsoft++;
            if (blocks[i].mib>0) shp[blocks[i].mib].insert({llround(dims[i].first*1e4),llround(dims[i].second*1e4)});
            if (blocks[i].cluster>0) clc[blocks[i].cluster]++;
        }
        int vmb=0; for (auto&kv:shp){ nsoft+=max(0,(int)kv.second.size()-1); vmb+=max(0,(int)kv.second.size()-1); }
        for (auto&kv:clc) nsoft+=max(0,kv.second-1);
        fprintf(stderr,"METRICS %.6f %.6f %d %d %d %d\n",area,hpwl,vbd,vcl,vmb,nsoft);
    }
    m46.total=m46_now()-t_solve;
    if (STAGE_TIMING){
        fprintf(stderr,"TIMING total=%.1f build=%.1f order=%.1f pipeline=%.1f pack=%.1f "
                       "anch=%.1f free=%.1f generic=%.1f cand_gen=%.1f nudge=%.1f "
                       "refine=%.1f swapmove=%.1f compact=%.1f push=%.1f\n",
                m46.total,m46.build,m46.order,m46.pipeline,m46.pack,m46.anch,m46.freep,
                m46.generic,m46.cand,m46.nudge,m46.refine,m46.swapmove,m46.compact,m46.push);
        fprintf(stderr,"TIMING packs=%lld fail=%lld frames=%lld refine_passes=%lld items=%lld "
                       "cands=%lld feas=%lld free_items=%lld anch_members=%lld refine_fixed=%lld\n",
                m46.packs,m46.pack_fail,m46.frames,m46.refine_passes,m46.items,
                m46.cands,m46.feas,m46.free_items,m46.anch_members,m46.refine_fixed);
        fprintf(stderr,"TIMING sampled_x32 ov_ms=%.1f wire_ms=%.1f\n",
                m46.ov32*32.0,m46.wire32*32.0);
    }
    printf("%d\n",N);
    // %.17g (not %.10f): compaction creates EXACT abutments (block A right edge ==
    // block B left edge). %.10f rounding can shift a coordinate ~1e-10, opening a
    // sub-nm gap that shapely (exact) treats as a cluster fragment — a spurious
    // grouping violation worth ~9% cost on dense cases. %.17g round-trips doubles
    // bit-exactly, so abutments survive C++ -> Python -> harness scoring.
    for (int i=0;i<N;i++) printf("%.17g %.17g %.17g %.17g\n",best[i].x,best[i].y,best[i].w,best[i].h);
}

// ─── I/O ──────────────────────────────────────────────────────────────────────
int main() {
    if (scanf("%d",&N)!=1) return 0;
    blocks.resize(N); area_targets.resize(N,0.0);
    for (int i=0;i<N;i++){ scanf("%lf",&area_targets[i]); blocks[i].area=area_targets[i]>0?area_targets[i]:1.0; }
    int nb2b; scanf("%d",&nb2b); b2b_edges.resize(nb2b);
    for (int i=0;i<nb2b;i++) scanf("%d %d %lf",&b2b_edges[i].i,&b2b_edges[i].j,&b2b_edges[i].w);
    int np2b; scanf("%d",&np2b); p2b_edges.resize(np2b);
    for (int i=0;i<np2b;i++) scanf("%d %d %lf",&p2b_edges[i].i,&p2b_edges[i].j,&p2b_edges[i].w);
    int npins; scanf("%d",&npins); pins.resize(npins);
    for (int i=0;i<npins;i++) scanf("%lf %lf",&pins[i].first,&pins[i].second);
    for (int i=0;i<N;i++){ int fx,pp,mib,cl,bnd; scanf("%d %d %d %d %d",&fx,&pp,&mib,&cl,&bnd);
        blocks[i].is_fixed=fx!=0; blocks[i].is_preplaced=pp!=0; blocks[i].mib=mib; blocks[i].cluster=cl; blocks[i].boundary=bnd; }
    for (int i=0;i<N;i++) scanf("%lf %lf %lf %lf",&blocks[i].tx,&blocks[i].ty,&blocks[i].tw,&blocks[i].th);
    if (const char* e=getenv("ICCAD_BP_WEIGHT")) { double v=atof(e); if (v>0) BP_W=v; }
    if (const char* e=getenv("ICCAD_WIRE_MULT")) { double v=atof(e); if (v>0) WIRE_MULT=v; }
    if (const char* e=getenv("ICCAD_ANCHOR_W"))  { double v=atof(e); if (v>=0) ANCHOR_W=v; }
    if (const char* e=getenv("ICCAD_LR_ASPECT"))  { double v=atof(e); if (v>0) LR_ASPECT=v; }
    if (const char* e=getenv("ICCAD_TB_ASPECT"))  { double v=atof(e); if (v>0) TB_ASPECT=v; }
    if (const char* e=getenv("ICCAD_SOFT_ASPECT")){ double v=atof(e); if (v>0) SOFT_ASPECT=v; }
    if (const char* e=getenv("ICCAD_FREE_ASPECT")){ int v=atoi(e); if (v>0) FREE_ASPECT=v; }
    if (const char* e=getenv("ICCAD_CLUSTER_ASPECT")){ double v=atof(e); if (v>0) CLUSTER_ASPECT=v; }
    if (const char* e=getenv("ICCAD_MIB_ASPECT")){ double v=atof(e); if (v>0) MIB_ASPECT=v; } // M37 probe
    if (const char* e=getenv("ICCAD_FREE_CLUSTER")){ int v=atoi(e); if (v>0) FREE_CLUSTER=v; }
    if (const char* e=getenv("ICCAD_FREE_ANCHORED")){ int v=atoi(e); if (v>0) FREE_ANCHORED=v; } // M35 probe
    if (const char* e=getenv("ICCAD_FREE_ANCHORED_BND")){ int v=atoi(e); if (v>0) FREE_ANCHORED_BND=v; } // M36 probe
    if (getenv("ICCAD_NO_REFINE")) REFINE=false;
    if (getenv("ICCAD_NO_COMPACT")) COMPACT=false;
    if (getenv("ICCAD_NO_PUSH")) PUSH=false;
    if (getenv("ICCAD_NO_BND_PUSH")) PUSH_BND=false;
    if (getenv("ICCAD_NO_SWAP")) PUSH_SWAP=false;
    if (getenv("ICCAD_NO_JUMP")) PUSH_JUMP=false;
    if (const char* e=getenv("ICCAD_PUSH_PASSES")){ int v=atoi(e); if (v>=0) PUSH_PASSES=v; }
    if (const char* e=getenv("ICCAD_REFINE_ITERS")){ int v=atoi(e); if (v>=0) REFINE_ITERS=v; }
    if (const char* e=getenv("ICCAD_COMPACT_ITERS")){ int v=atoi(e); if (v>=0) COMPACT_ITERS=v; }
    if (getenv("ICCAD_WIRE_FOR_ALL")) WIRE_FOR_ALL=true;
    if (getenv("ICCAD_WIRE_ORDER")) WIRE_ORDER=true;
    if (getenv("ICCAD_WIRE_TIEBREAK")) WIRE_TIEBREAK=true;
    if (getenv("ICCAD_WIRE_BFS")) WIRE_BFS=true;
    if (getenv("ICCAD_BFS_PIN")) BFS_PIN=true;
    if (getenv("ICCAD_BFS_NORM")) BFS_NORM=true;
    if (const char* e=getenv("ICCAD_ORDER_SWAP")){ int v=atoi(e); if (v>0) ORDER_SWAP=v; }
    if (const char* e=getenv("ICCAD_ORDER_MOVE")){ int v=atoi(e); if (v>0) ORDER_MOVE=v; }
    if (const char* e=getenv("ICCAD_CLUSTER_ORD")){ int v=atoi(e); if (v==1||v==2) CLUSTER_ORD=v; }
    if (getenv("ICCAD_REFRAME")) REFRAME=true;
    if (getenv("ICCAD_GUIDE_MED")) GUIDE_MED=true;
    if (const char* e=getenv("ICCAD_STAGE_TIMING")){ int v=atoi(e); if (v>0) STAGE_TIMING=v; }
    if (const char* e=getenv("ICCAD_REFINE_TRACE")){ int v=atoi(e); if (v>0) REFINE_TRACE=v; }
    if (const char* e=getenv("ICCAD_REFINE_DUMP")){ if (*e) REFINE_DUMP_FP=fopen(e,"w"); }
    auto parse_list=[](const char* name, vector<double>& out){
        const char* e=getenv(name); if(!e||!*e) return;
        string s=e; size_t i=0;
        while (i<=s.size()){
            size_t j=s.find(',',i); if (j==string::npos) j=s.size();
            if (j>i){ try{ double v=stod(s.substr(i,j-i)); if (v>0) out.push_back(v); }catch(...){} }
            i=j+1;
        }
    };
    parse_list("ICCAD_FRAME_ASPECTS", FRAME_ASPECTS);
    parse_list("ICCAD_FRAME_SCALES",  FRAME_SCALES);
    if (getenv("ICCAD_FREE_CLUSTER_RATIOS")){            // M34: override per-member search set
        FREE_CLUSTER_RATIOS.clear();
        parse_list("ICCAD_FREE_CLUSTER_RATIOS", FREE_CLUSTER_RATIOS);
        if (FREE_CLUSTER_RATIOS.empty()) FREE_CLUSTER_RATIOS={1.0,1.5,0.6667,2.0,0.5};
    }
    if (getenv("ICCAD_FREE_ANCHORED_RATIOS")){          // M36: override anchored search set
        FREE_ANCHORED_RATIOS.clear();
        parse_list("ICCAD_FREE_ANCHORED_RATIOS", FREE_ANCHORED_RATIOS);
        if (FREE_ANCHORED_RATIOS.empty()) FREE_ANCHORED_RATIOS={1.0,1.5,0.6667,2.0,0.5};
    }
    solve();
    if (REFINE_DUMP_FP) fclose(REFINE_DUMP_FP);
    return 0;
}
