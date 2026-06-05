// Constructive fixed-outline floorplanner (C++ port of teammate's my_optimizer
// architecture). Replaces our SA+skyline-BL placer (oracle-perm ceiling ~3.27).
//
// M1: boundary-aspect dims + anchor-guided greedy frame packing (singles).
// M2 (this file): cluster blocks become compound "items" with an internal layout
//     and are placed as a unit, so grouping violations collapse. The M1 diagnosis
//     showed hpwl/area already match the teammate's v5; the whole gap was vrel.
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
using namespace std;

static const int B_LEFT = 1, B_RIGHT = 2, B_TOP = 4, B_BOTTOM = 8;
static const double MARGIN = 1e-4;
static const double TOL = 1e-6;

struct Edge { int i, j; double w; };
struct Block {
    double area; bool is_fixed, is_preplaced;
    int mib, cluster, boundary; double tx, ty, tw, th;
};
struct XYWH { double x, y, w, h; };

static int N;
static double BP_W = 30000.0;   // boundary-miss penalty in greedy item scoring
static double WIRE_MULT = 1.0;  // extra scale on the incremental-HPWL term (env)
static double ANCHOR_W = 0.10;  // anchor pull in greedy item scoring
static vector<double> area_targets;
static vector<Edge>   b2b_edges, p2b_edges;
static vector<pair<double,double>> pins;
static vector<Block>  blocks;

static vector<pair<double,double>> dims;   // (w,h) per block
static vector<XYWH> pos;                    // working positions
static vector<char> placed;

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
    double ax, ay, aw;   // anchor
};

// ─── basic helpers ────────────────────────────────────────────────────────────
static double soft_ratio(int code) {
    bool lr = (code & (B_LEFT | B_RIGHT)) != 0;
    bool tb = (code & (B_TOP | B_BOTTOM)) != 0;
    if (lr && !tb) return 2.50;
    if (tb && !lr) return 0.40;
    return 1.0;
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
        if (okall){ double side=sqrt(avg); for (int i:mov) dims[i]={side,side}; }
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
    int bs=0; for (int b: it.blocks) bs+=block_boundary_score(b);
    it.bscore=bs;
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
    vector<double> aspects={1.0,1.35,0.75,1.8,0.55};
    vector<double> scales=(N>=60&&N<80)?vector<double>{1.00,1.03,1.05,1.15,1.35,1.65,2.10}
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
static vector<pair<double,double>> item_candidates(
        const Item& it, double fw, double fh, const vector<XYWH>& rects) {
    double iw=it.w, ih=it.h, xmax=max(0.0,fw-iw), ymax=max(0.0,fh-ih);
    vector<pair<double,double>> cands={{0,0},{xmax,0},{0,ymax},{xmax,ymax}};
    if (it.aw>0) cands.push_back({it.ax-iw/2, it.ay-ih/2});
    set<long long> xs,ys; xs.insert(0);xs.insert(llround(xmax*1e6));ys.insert(0);ys.insert(llround(ymax*1e6));
    auto addx=[&](double v){ xs.insert(llround(min(max(0.0,v),xmax)*1e6)); };
    auto addy=[&](double v){ ys.insert(llround(min(max(0.0,v),ymax)*1e6)); };
    for (const XYWH& r:rects){
        double rx2=r.x+r.w+MARGIN, ry2=r.y+r.h+MARGIN;
        double lx=max(0.0,r.x-iw-MARGIN), by=max(0.0,r.y-ih-MARGIN);
        addx(rx2);addx(lx);addx(r.x); addy(ry2);addy(by);addy(r.y);
        cands.push_back({rx2,r.y}); cands.push_back({rx2,max(0.0,r.y+r.h-ih)});
        cands.push_back({r.x,ry2}); cands.push_back({max(0.0,r.x+r.w-iw),ry2});
        cands.push_back({lx,r.y});  cands.push_back({lx,max(0.0,r.y+r.h-ih)});
        cands.push_back({r.x,by});  cands.push_back({max(0.0,r.x+r.w-iw),by});
    }
    // boundary-exact: origin that makes a boundary member touch the frame edge
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
    if (any){
        if (xv.empty()){ for(long long v:xs) xv.push_back(v/1e6); }
        if (yv.empty()){ for(long long v:ys) yv.push_back(v/1e6); }
        for (double xx:xv) for (double yy:yv) cands.push_back({xx,yy});
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

// greedy pack of items into frame
static bool pack_in_frame(double fw,double fh,const vector<Item>& items,vector<XYWH>& out){
    out=pos; vector<XYWH> rects; bbox_reset();
    vector<char> done(N,0);
    // Wire weight calibrated high: bbox-area minimisation alone scatters connected
    // blocks; the baseline we reconstruct is wire-driven. Swept optimum ~2000-3000
    // (broad basin, area_gap stays flat) → bake the old 0.025-0.075 base ×2000.
    double ww=(N>=116)?150.0:(N>=100?70.0:50.0);
    for (int i=0;i<N;i++) if (placed[i]){
        if (pos[i].x<-TOL||pos[i].y<-TOL||pos[i].x+pos[i].w>fw+TOL||pos[i].y+pos[i].h>fh+TOL) return false;
        for (const XYWH&r:rects) if (rect_overlap(pos[i].x,pos[i].y,pos[i].w,pos[i].h,r.x,r.y,r.w,r.h)) return false;
        rects.push_back(pos[i]); bbox_add(pos[i].x,pos[i].y,pos[i].w,pos[i].h); done[i]=1;
    }
    // First-pass: attach anchored-cluster movable members to their preplaced walls
    // (teammate _pack_in_frame 637-689). Placed members are skipped by the singles
    // loop below; any that don't fit here fall back to that loop.
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
            double bw=dims[b].first, bh=dims[b].second;
            auto cands=adjacent_candidates_for_block(bw,bh,cluster_rects,fw,fh,blocks[b].boundary);
            double best=1e300, bx=0, by=0; bool found=false;
            for (auto&c:cands){
                double x=c.first, y=c.second;
                if (x<-TOL||y<-TOL||x+bw>fw+TOL||y+bh>fh+TOL) continue;
                bool ov=false;
                for (const XYWH&r:rects) if (rect_overlap(x,y,bw,bh,r.x,r.y,r.w,r.h)){ ov=true; break; }
                if (ov) continue;
                double cx=x+bw/2, cy=y+bh/2;
                double ad=anchors[b].w>0?fabs(cx-anchors[b].x)+fabs(cy-anchors[b].y):0.0;
                int bp=boundary_penalty_est(b,x,y,bw,bh,fw,fh);
                double area=bbox_area_with(x,y,bw,bh), wire=0.0;
                if (bp==0){
                    for (auto& nb:b2b_adj[b]) if (done[nb.first]){
                        double ncx=out[nb.first].x+out[nb.first].w/2, ncy=out[nb.first].y+out[nb.first].h/2;
                        wire+=nb.second*(fabs(cx-ncx)+fabs(cy-ncy));
                    }
                    for (auto& pn:p2b_adj[b])
                        wire+=pn.second*(fabs(cx-pins[pn.first].first)+fabs(cy-pins[pn.first].second));
                }
                double score=area+ANCHOR_W*ad+ww*WIRE_MULT*wire+BP_W*bp+1e-3*y+1e-4*x;
                if (!rect_touches_any(x,y,bw,bh,cluster_rects)) score+=7000.0; // keep group connected
                if (score<best){ best=score; bx=x; by=y; found=true; }
            }
            if (found){
                out[b]={bx,by,bw,bh}; rects.push_back({bx,by,bw,bh});
                bbox_add(bx,by,bw,bh); done[b]=1; cluster_rects.push_back({bx,by,bw,bh});
            }
        }
    }
    for (const Item& it:items){
        bool all_done=true; for (int b:it.blocks) if(!done[b]){ all_done=false; break; }
        if (all_done) continue;                       // already placed in first-pass
        bool any_done=false; for (int b:it.blocks) if(done[b]){ any_done=true; break; }
        if (any_done) continue;                       // partial: leave to other items/frames
        auto cands=item_candidates(it,fw,fh,rects);
        double best=1e300, bx=0, by=0; bool found=false;
        for (auto&c:cands){
            double x=c.first,y=c.second;
            if (x<-TOL||y<-TOL||x+it.w>fw+TOL||y+it.h>fh+TOL) continue;
            bool ov=false;
            for (size_t k=0;k<it.blocks.size()&&!ov;k++){
                int b=it.blocks[k]; double rx=x+it.offs[k].first, ry=y+it.offs[k].second;
                double bw=dims[b].first, bh=dims[b].second;
                for (const XYWH&r:rects) if (rect_overlap(rx,ry,bw,bh,r.x,r.y,r.w,r.h)){ ov=true; break; }
            }
            if (ov) continue;
            double cx=x+it.w/2, cy=y+it.h/2;
            double ad=it.aw>0?fabs(cx-it.ax)+fabs(cy-it.ay):0.0;
            double bp=item_boundary_penalty(it,x,y,fw,fh);
            double area=bbox_area_with(x,y,it.w,it.h);
            // incremental HPWL to already-placed connected neighbors + pins
            double wire=0.0;
            if (bp==0){
                for (size_t k=0;k<it.blocks.size();k++){
                    int b=it.blocks[k];
                    double mcx=x+it.offs[k].first+dims[b].first/2, mcy=y+it.offs[k].second+dims[b].second/2;
                    for (auto& nb:b2b_adj[b]) if (done[nb.first]){
                        double ncx=out[nb.first].x+out[nb.first].w/2, ncy=out[nb.first].y+out[nb.first].h/2;
                        wire+=nb.second*(fabs(mcx-ncx)+fabs(mcy-ncy));
                    }
                    for (auto& pn:p2b_adj[b])
                        wire+=pn.second*(fabs(mcx-pins[pn.first].first)+fabs(mcy-pins[pn.first].second));
                }
            }
            double score=area+ANCHOR_W*ad+ww*WIRE_MULT*wire+BP_W*bp+1e-3*y+1e-4*x;
            if (score<best){ best=score; bx=x; by=y; found=true; }
        }
        if (!found) return false;
        for (size_t k=0;k<it.blocks.size();k++){
            int b=it.blocks[k]; double rx=bx+it.offs[k].first, ry=by+it.offs[k].second;
            double bw=dims[b].first, bh=dims[b].second;
            out[b]={rx,ry,bw,bh}; rects.push_back({rx,ry,bw,bh}); bbox_add(rx,ry,bw,bh); done[b]=1;
        }
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
    dims.assign(N,{1,1}); pos.assign(N,{0,0,0,0}); placed.assign(N,0);
    for (int i=0;i<N;i++){
        if (blocks[i].is_preplaced){ pos[i]={blocks[i].tx,blocks[i].ty,blocks[i].tw,blocks[i].th}; dims[i]={blocks[i].tw,blocks[i].th}; placed[i]=1; }
        else if (blocks[i].is_fixed){ double w=blocks[i].tw,h=blocks[i].th; if(w<=0||h<=0){double s=sqrt(blocks[i].area);w=h=s;} dims[i]={w,h}; }
        else dims[i]=default_soft_dim(blocks[i].area, blocks[i].boundary);
    }
    apply_safe_mib_dims();
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
    sort(items.begin(),items.end(),[](const Item&a,const Item&b){
        if (a.bscore!=b.bscore) return a.bscore>b.bscore;
        if (a.blocks.size()!=b.blocks.size()) return a.blocks.size()>b.blocks.size();
        double aa=a.w*a.h,ab=b.w*b.h; if (fabs(aa-ab)>TOL) return aa>ab;
        return max(a.w,a.h)>max(b.w,b.h);
    });

    vector<int> order; for(int i=0;i<N;i++) if(!blocks[i].is_preplaced) order.push_back(i);

    auto frames=frame_candidates();
    // A few tight frames win: trying all overshoots layout_score's 150000*bv weight
    // (picks low-violation but area-bloated outlines). 4/5 measured best (deterministic).
    int max_trials=(N>=60)?4:5;
    vector<XYWH> best; bool have_best=false; double best_score=1e300; int trials=0;
    for (auto& f:frames){
        vector<XYWH> cand;
        if (!pack_in_frame(f.first,f.second,items,cand)) continue;
        trials++;
        final_boundary_nudge(cand);
        final_group_boundary_nudge(cand);
        final_single_edge_escape(cand);
        double sc=layout_score(cand);
        if (!have_best||sc<best_score){ best_score=sc; best=cand; have_best=true; }
        if (trials>=max_trials) break;
    }
    if (!have_best) best=shelf_fallback(order);

    if (getenv("CONSTRUCTIVE_DEBUG"))
        fprintf(stderr,"[dbg] N=%d frames=%d ok=%d shelf=%d bv=%d gf=%d\n",
                N,(int)frames.size(),trials,have_best?0:1,
                count_boundary_violations(best),count_group_fragments(best));
    printf("%d\n",N);
    for (int i=0;i<N;i++) printf("%.10f %.10f %.10f %.10f\n",best[i].x,best[i].y,best[i].w,best[i].h);
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
    solve();
    return 0;
}
