// ICCAD 2026 FloorSet Challenge - C++ SA Floorplanner
// Reads problem from stdin, writes positions to stdout.
// Compile: g++ -O3 -std=c++17 -o optimizer_claude optimizer_claude.cpp
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <map>
#include <set>
#include <unordered_map>
#include <cassert>
#include <numeric>
#include <queue>

using namespace std;
using Clock = chrono::steady_clock;

static double elapsed_sec(chrono::time_point<Clock> t0) {
    return chrono::duration<double>(Clock::now() - t0).count();
}

// ─── Constants ────────────────────────────────────────────────────────────────
static const double TIME_LIMIT  = 12.00;
static const double AREA_TOL    = 0.005;
static double W_VIOL            = 10.0;   // calibrated dynamically (∝ initial HPWL)
static double W_BOUNDARY        = 10.0;   // soft boundary-distance gradient (fixed scale)
static double W_AREA            = 0.05;   // calibrated dynamically

static const int COL_FIXED      = 0;
static const int COL_PREPLACED  = 1;
static const int COL_MIB        = 2;
static const int COL_CLUSTER    = 3;
static const int COL_BOUNDARY   = 4;

static const int B_LEFT=1, B_RIGHT=2, B_TOP=4, B_BOTTOM=8;
static const int B_TL=5, B_TR=6, B_BL=9, B_BR=10;

// ─── Problem data ─────────────────────────────────────────────────────────────
struct Edge { int i, j; double w; };
struct Block {
    double area;
    bool   is_fixed, is_preplaced;
    int    mib, cluster, boundary;
    double tx, ty, tw, th;
};
struct Pos { double x, y, w, h; };

static int N;
static vector<double> area_targets;
static vector<Edge>   b2b_edges, p2b_edges;
static vector<pair<double,double>> pins;
static vector<Block>  blocks;

// ─── Skyline packer ───────────────────────────────────────────────────────────
// pts[i] = (x_start, height); segment i covers [pts[i].x, pts[i+1].x)
struct Skyline {
    vector<pair<double,double>> pts;
    Skyline() { pts.push_back({0.0, 0.0}); }

    double height_at(double x) const {
        // binary search for last pt with pt.x <= x
        int lo = 0, hi = (int)pts.size() - 1, ans = 0;
        while (lo <= hi) {
            int mid = (lo + hi) / 2;
            if (pts[mid].first <= x) { ans = mid; lo = mid + 1; }
            else hi = mid - 1;
        }
        return pts[ans].second;
    }

    double max_height(double x0, double x1) const {
        double h = 0.0;
        for (int i = 0; i < (int)pts.size(); i++) {
            double sx = pts[i].first;
            double nx = (i + 1 < (int)pts.size()) ? pts[i+1].first : x1 + 1.0;
            if (sx >= x1) break;
            if (nx <= x0) continue;
            if (pts[i].second > h) h = pts[i].second;
        }
        return h;
    }

    void raise_region(double x0, double x1, double new_h) {
        // Collect unique x coords
        vector<double> xs;
        xs.reserve(pts.size() + 2);
        for (auto& p : pts) xs.push_back(p.first);
        xs.push_back(x0);
        xs.push_back(x1);
        sort(xs.begin(), xs.end());
        xs.erase(unique(xs.begin(), xs.end()), xs.end());

        vector<pair<double,double>> npts;
        npts.reserve(xs.size());
        double prev_h = -1e18;
        for (double x : xs) {
            double h = (x >= x0 && x < x1) ? new_h : height_at(x);
            if (h != prev_h) {
                npts.push_back({x, h});
                prev_h = h;
            }
        }
        if (npts.empty()) npts.push_back({0.0, 0.0});
        pts = move(npts);
    }

    // BL: leftmost then lowest, respecting max_width (0 = no limit)
    pair<double,double> find_bl(double w, double h, double max_width = 0.0) const {
        double bx = 0.0, by = 1e18;
        for (auto& [sx, _] : pts) {
            if (max_width > 0 && sx + w > max_width + 1e-9) continue;
            double y = max_height(sx, sx + w);
            if (y < by || (y == by && sx < bx)) { by = y; bx = sx; }
        }
        // Fallback: ignore width constraint if nothing fits
        if (by >= 1e17) {
            for (auto& [sx, _] : pts) {
                double y = max_height(sx, sx + w);
                if (y < by || (y == by && sx < bx)) { by = y; bx = sx; }
            }
        }
        return {bx, by};
    }

    // place: raises skyline to max(current, y+h) in [x, x+w).
    // Must never lower (e.g. preplaced blocks may have y+h < existing height).
    void place(double x, double y, double w, double h) {
        double new_h = y + h;
        // Collect breakpoints
        vector<double> xs;
        xs.reserve(pts.size() + 2);
        for (auto& p : pts) xs.push_back(p.first);
        xs.push_back(x);
        xs.push_back(x + w);
        sort(xs.begin(), xs.end());
        xs.erase(unique(xs.begin(), xs.end()), xs.end());

        vector<pair<double,double>> npts;
        npts.reserve(xs.size());
        double prev_h = -1e18;
        for (double xi : xs) {
            double cur = height_at(xi);
            double h_val = (xi >= x && xi < x + w) ? max(cur, new_h) : cur;
            if (h_val != prev_h) {
                npts.push_back({xi, h_val});
                prev_h = h_val;
            }
        }
        if (npts.empty()) npts.push_back({0.0, 0.0});
        pts = move(npts);
    }
};

// Cluster group ID per block (0 = no cluster). Set by run_sa.
static vector<int> s_cluster_gid_of;
static const int MAX_PACK_SIZE = 120;  // apply structured packing for up to this many members

// Soft-constraint group tables (populated in run_sa).
// Forward-declared here so skyline_decode can detect preplaced anchors.
static map<int,vector<int>> mib_groups, cluster_groups;
static vector<pair<int,int>> boundary_blocks;
static int n_soft = 0;

// Multi-row cluster packing: guarantees connectivity while keeping compact footprint.
// Splits members into rows of ≤ row_len blocks.
// KEY: The tallest block of each row is placed leftmost, so it always touches
//      the block above it in the next row (sharing the horizontal boundary).
static void pack_cluster_multirow(
    Skyline& sky,
    vector<int> members,       // by value (sorted in place)
    const vector<double>& widths,
    const vector<double>& heights,
    vector<Pos>& pos,
    vector<bool>& placed,
    double max_width,
    int row_len)               // blocks per row (except possibly the last)
{
    int nm = (int)members.size();
    if (nm == 0) return;

    // Sort tallest-first so each row starts with the tallest block.
    // This ensures the row boundary always has a touching point with the next row.
    sort(members.begin(), members.end(),
         [&](int a, int b){ return heights[a] > heights[b]; });

    // Compute per-row stats
    int n_rows = (nm + row_len - 1) / row_len;
    double max_row_w = 0.0, total_h = 0.0;
    for (int r = 0; r < n_rows; r++) {
        double rw = 0.0, rh = 0.0;
        for (int i = r * row_len; i < min((r+1)*row_len, nm); i++) {
            rw += widths[members[i]]; rh = max(rh, heights[members[i]]);
        }
        max_row_w = max(max_row_w, rw);
        total_h += rh;
    }

    // Find BL for the full cluster footprint
    auto [x0, _yt] = sky.find_bl(max_row_w, total_h, max_width);
    double y_base = sky.max_height(x0, x0 + max_row_w);

    // Place rows bottom-to-top
    double cur_y = y_base;
    for (int r = 0; r < n_rows; r++) {
        double row_h = 0.0;
        double cx = x0;
        for (int i = r * row_len; i < min((r+1)*row_len, nm); i++) {
            int m = members[i];
            if (!placed[m]) {
                pos[m] = {cx, cur_y, widths[m], heights[m]};
                sky.place(cx, cur_y, widths[m], heights[m]);
                placed[m] = true;
                cx += widths[m];
                row_h = max(row_h, heights[m]);
            }
        }
        cur_y += row_h;
    }
}

// Variant of pack_cluster_multirow that anchors the pack adjacent to a
// preplaced cluster member, so the free pack touches the preplaced footprint
// and the cluster stays connected. Returns false if the pack doesn't fit
// anywhere adjacent to the anchor (caller should fall back to BL packing).
static bool pack_cluster_anchored(
    Skyline& sky,
    vector<int> members,
    const vector<double>& widths,
    const vector<double>& heights,
    vector<Pos>& pos,
    vector<bool>& placed,
    double max_width,
    int row_len,
    Pos anchor)
{
    int nm = (int)members.size();
    if (nm == 0) return true;

    sort(members.begin(), members.end(),
         [&](int a, int b){ return heights[a] > heights[b]; });

    int n_rows = (nm + row_len - 1) / row_len;
    double max_row_w = 0.0, total_h = 0.0;
    for (int r = 0; r < n_rows; r++) {
        double rw = 0.0, rh = 0.0;
        for (int i = r * row_len; i < min((r+1)*row_len, nm); i++) {
            rw += widths[members[i]]; rh = max(rh, heights[members[i]]);
        }
        max_row_w = max(max_row_w, rw);
        total_h += rh;
    }

    // Candidate 1: right of anchor. The first member's left edge touches
    // anchor.x + anchor.w; first member sits at y = anchor.y if skyline allows.
    double x0_r = anchor.x + anchor.w;
    bool fit_r = (max_width <= 0 || x0_r + max_row_w <= max_width + 1e-9) && x0_r >= 0;
    double y0_r = fit_r ? max(anchor.y, sky.max_height(x0_r, x0_r + max_row_w)) : 1e18;
    bool touch_r = fit_r && y0_r <= anchor.y + anchor.h - 1e-9;

    // Candidate 2: top of anchor. First row sits on top of anchor.
    double x0_t = anchor.x;
    bool fit_t = (max_width <= 0 || x0_t + max_row_w <= max_width + 1e-9);
    double y0_t = fit_t ? max(anchor.y + anchor.h, sky.max_height(x0_t, x0_t + max_row_w)) : 1e18;
    bool touch_t = fit_t && fabs(y0_t - (anchor.y + anchor.h)) < 1e-6;

    double x0, y_base;
    if      (touch_r) { x0 = x0_r; y_base = y0_r; }
    else if (touch_t) { x0 = x0_t; y_base = y0_t; }
    else if (fit_r)   { x0 = x0_r; y_base = y0_r; }     // not strictly touching, but close
    else if (fit_t)   { x0 = x0_t; y_base = y0_t; }
    else              return false;

    double cur_y = y_base;
    for (int r = 0; r < n_rows; r++) {
        double row_h = 0.0;
        double cx = x0;
        for (int i = r * row_len; i < min((r+1)*row_len, nm); i++) {
            int m = members[i];
            if (!placed[m]) {
                pos[m] = {cx, cur_y, widths[m], heights[m]};
                sky.place(cx, cur_y, widths[m], heights[m]);
                placed[m] = true;
                cx += widths[m];
                row_h = max(row_h, heights[m]);
            }
        }
        cur_y += row_h;
    }
    return true;
}

static vector<Pos> skyline_decode(
    const vector<int>& perm,
    const vector<double>& widths,
    const vector<double>& heights,
    const map<int,Pos>& preplaced,
    double max_width = 0.0)
{
    Skyline sky;
    vector<Pos> pos(N);
    vector<bool> placed(N, false);

    for (auto& [bid, p] : preplaced) {
        pos[bid] = p;
        sky.place(p.x, p.y, p.w, p.h);
        placed[bid] = true;
    }

    // Collect each cluster group's free members (in perm order)
    map<int, vector<int>> cluster_perm_members;
    for (int bid : perm) {
        int cl = (!s_cluster_gid_of.empty() && bid < (int)s_cluster_gid_of.size())
                 ? s_cluster_gid_of[bid] : 0;
        if (cl > 0) cluster_perm_members[cl].push_back(bid);
    }

    set<int> cluster_done;
    int n_perm = (int)perm.size();

    for (int ki = 0; ki < n_perm; ki++) {
        int bid = perm[ki];
        if (placed[bid]) continue;

        int cl = (!s_cluster_gid_of.empty() && bid < (int)s_cluster_gid_of.size())
                 ? s_cluster_gid_of[bid] : 0;

        if (cl > 0 && !cluster_done.count(cl)) {
            auto& members = cluster_perm_members[cl];
            int nm = (int)members.size();
            // Filter out already-placed (preplaced)
            vector<int> free_m;
            for (int m : members) if (!placed[m]) free_m.push_back(m);

            if (!free_m.empty() && nm <= MAX_PACK_SIZE) {
                int row_len = max(2, (int)ceil(sqrt((double)nm)));
                pack_cluster_multirow(sky, free_m, widths, heights,
                                      pos, placed, max_width, row_len);
            } else {
                // Standard BL for large clusters or already-placed members
                for (int m : free_m) {
                    double w = widths[m], h = heights[m];
                    auto [x, y] = sky.find_bl(w, h, max_width);
                    pos[m] = {x, y, w, h};
                    sky.place(x, y, w, h);
                    placed[m] = true;
                }
            }
            cluster_done.insert(cl);
        } else {
            double w = widths[bid], h = heights[bid];
            auto [x, y] = sky.find_bl(w, h, max_width);
            pos[bid] = {x, y, w, h};
            sky.place(x, y, w, h);
            placed[bid] = true;
        }
    }
    return pos;
}

// ─── Cost ─────────────────────────────────────────────────────────────────────
static double calc_hpwl_b2b(const vector<Pos>& pos) {
    double t = 0.0;
    for (auto& e : b2b_edges) {
        if (e.i < 0 || e.i >= N || e.j < 0 || e.j >= N) continue;
        double dx = (pos[e.i].x + pos[e.i].w/2) - (pos[e.j].x + pos[e.j].w/2);
        double dy = (pos[e.i].y + pos[e.i].h/2) - (pos[e.j].y + pos[e.j].h/2);
        t += e.w * (fabs(dx) + fabs(dy));
    }
    return t;
}

static double calc_hpwl_p2b(const vector<Pos>& pos) {
    double t = 0.0;
    for (auto& e : p2b_edges) {
        int pi = e.i, bi = e.j;
        if (pi < 0 || pi >= (int)pins.size() || bi < 0 || bi >= N) continue;
        double dx = (pos[bi].x + pos[bi].w/2) - pins[pi].first;
        double dy = (pos[bi].y + pos[bi].h/2) - pins[pi].second;
        t += e.w * (fabs(dx) + fabs(dy));
    }
    return t;
}

static double calc_bbox_area(const vector<Pos>& pos) {
    double xmn=1e18, ymn=1e18, xmx=-1e18, ymx=-1e18;
    for (auto& p : pos) {
        xmn = min(xmn, p.x);       ymn = min(ymn, p.y);
        xmx = max(xmx, p.x + p.w); ymx = max(ymx, p.y + p.h);
    }
    return (xmx - xmn) * (ymx - ymn);
}

// ─── Soft violations ─────────────────────────────────────────────────────────
// (mib_groups / cluster_groups / boundary_blocks / n_soft declared above
//  to allow skyline_decode to inspect cluster membership.)

static double calc_violation(const vector<Pos>& pos) {
    if (n_soft == 0) return 0.0;
    double v = 0.0;

    // MIB: distinct (w,h) per group
    for (auto& [gid, members] : mib_groups) {
        set<pair<int,int>> shapes;
        for (int b : members)
            shapes.insert({(int)round(pos[b].w * 1000), (int)round(pos[b].h * 1000)});
        v += (int)shapes.size() - 1;
    }

    // Cluster: connected components
    for (auto& [gid, members] : cluster_groups) {
        int m = (int)members.size();
        if (m < 2) continue;
        vector<vector<int>> adj(m);
        for (int ii = 0; ii < m; ii++) {
            int bi = members[ii];
            for (int jj = ii+1; jj < m; jj++) {
                int bj = members[jj];
                double ox = min(pos[bi].x+pos[bi].w, pos[bj].x+pos[bj].w) - max(pos[bi].x, pos[bj].x);
                double oy = min(pos[bi].y+pos[bi].h, pos[bj].y+pos[bj].h) - max(pos[bi].y, pos[bj].y);
                bool touch = (oy > 1e-6 && (fabs(pos[bi].x+pos[bi].w - pos[bj].x) < 1e-4 ||
                                            fabs(pos[bj].x+pos[bj].w - pos[bi].x) < 1e-4)) ||
                             (ox > 1e-6 && (fabs(pos[bi].y+pos[bi].h - pos[bj].y) < 1e-4 ||
                                            fabs(pos[bj].y+pos[bj].h - pos[bi].y) < 1e-4));
                if (touch) { adj[ii].push_back(jj); adj[jj].push_back(ii); }
            }
        }
        vector<bool> vis(m, false); int comps = 0;
        for (int s = 0; s < m; s++) {
            if (vis[s]) continue;
            comps++;
            queue<int> q; q.push(s); vis[s] = true;
            while (!q.empty()) {
                int cur = q.front(); q.pop();
                for (int nb : adj[cur]) if (!vis[nb]) { vis[nb]=true; q.push(nb); }
            }
        }
        v += comps - 1;
    }

    // Boundary
    if (!boundary_blocks.empty()) {
        double xmn=1e18,ymn=1e18,xmx=-1e18,ymx=-1e18;
        for (auto& p : pos) {
            xmn=min(xmn,p.x); ymn=min(ymn,p.y);
            xmx=max(xmx,p.x+p.w); ymx=max(ymx,p.y+p.h);
        }
        for (auto& [b, flag] : boundary_blocks) {
            auto& p = pos[b];
            bool lt = fabs(p.x         - xmn) < 1e-4;
            bool rt = fabs(p.x + p.w   - xmx) < 1e-4;
            bool bt = fabs(p.y         - ymn) < 1e-4;
            bool tt = fabs(p.y + p.h   - ymx) < 1e-4;
            bool ok = false;
            if      (flag == B_LEFT  ) ok = lt;
            else if (flag == B_RIGHT ) ok = rt;
            else if (flag == B_TOP   ) ok = tt;
            else if (flag == B_BOTTOM) ok = bt;
            else if (flag == B_TL    ) ok = tt && lt;
            else if (flag == B_TR    ) ok = tt && rt;
            else if (flag == B_BL    ) ok = bt && lt;
            else if (flag == B_BR    ) ok = bt && rt;
            if (!ok) v += 1.0;
        }
    }
    return v / n_soft;
}

static double target_width_sq = 0.0;  // set in run_sa; 0 = disabled

// Continuous boundary distance penalty (gives SA a gradient signal).
// For each boundary block, measures normalized distance to required edge.
static double calc_boundary_dist(const vector<Pos>& pos) {
    if (boundary_blocks.empty()) return 0.0;
    double xmn=1e18,ymn=1e18,xmx=-1e18,ymx=-1e18;
    for (auto& p : pos) {
        xmn=min(xmn,p.x); ymn=min(ymn,p.y);
        xmx=max(xmx,p.x+p.w); ymx=max(ymx,p.y+p.h);
    }
    double W = max(xmx-xmn, 1e-6), H = max(ymx-ymn, 1e-6);
    double penalty = 0.0;
    for (auto& [b, flag] : boundary_blocks) {
        auto& p = pos[b];
        if (flag & B_LEFT  ) penalty += max(0.0, p.x          - xmn) / W;
        if (flag & B_RIGHT ) penalty += max(0.0, xmx - (p.x+p.w)  ) / W;
        if (flag & B_BOTTOM) penalty += max(0.0, p.y          - ymn) / H;
        if (flag & B_TOP   ) penalty += max(0.0, ymx - (p.y+p.h)  ) / H;
    }
    return penalty;
}

static double proxy_cost(const vector<Pos>& pos) {
    double c = calc_hpwl_b2b(pos) + calc_hpwl_p2b(pos)
             + calc_bbox_area(pos) * W_AREA
             + calc_violation(pos) * W_VIOL          // hard violations (MIB, cluster, boundary)
             + calc_boundary_dist(pos) * W_BOUNDARY; // soft gradient toward boundary
    // Soft aspect-ratio penalty: encourage square bounding box
    if (target_width_sq > 0) {
        double xmn=1e18,ymn=1e18,xmx=-1e18,ymx=-1e18;
        for (auto& p : pos) {
            xmn=min(xmn,p.x); ymn=min(ymn,p.y);
            xmx=max(xmx,p.x+p.w); ymx=max(ymx,p.y+p.h);
        }
        double W = xmx - xmn, H = ymx - ymn;
        if (W > 0 && H > 0) {
            double ratio = W / H;
            // penalize elongation: log(max(ratio, 1/ratio)) * W_AREA * bbox
            double ar_pen = fabs(log(ratio)) * W_AREA * W * H * 0.3;
            c += ar_pen;
        }
    }
    return c;
}

// ─── Connectivity ordering ────────────────────────────────────────────────────
// Greedy nearest-neighbour ordering on connectivity graph to put connected
// blocks adjacent in the permutation (reduces HPWL).
static vector<int> connectivity_order(const vector<int>& free_blocks,
                                      const vector<double>& widths,
                                      const vector<double>& heights)
{
    // Build edge weights per block pair
    unordered_map<int,double> block_weight; // block → total connectivity
    // Build adjacency weight
    vector<vector<pair<int,double>>> adj(N);
    for (auto& e : b2b_edges) {
        if (e.i < 0 || e.i >= N || e.j < 0 || e.j >= N) continue;
        adj[e.i].push_back({e.j, e.w});
        adj[e.j].push_back({e.i, e.w});
    }
    for (auto& e : p2b_edges) {
        if (e.j < 0 || e.j >= N) continue;
        block_weight[e.j] += e.w;
    }

    int n = (int)free_blocks.size();
    if (n == 0) return free_blocks;

    set<int> in_set(free_blocks.begin(), free_blocks.end());
    vector<bool> used(N, false);

    // Start with the free block having highest pin connectivity
    int start = free_blocks[0];
    double best_w = -1;
    for (int b : free_blocks) {
        double w = block_weight.count(b) ? block_weight[b] : 0.0;
        if (w > best_w) { best_w = w; start = b; }
    }

    vector<int> order;
    order.reserve(n);
    order.push_back(start);
    used[start] = true;

    while ((int)order.size() < n) {
        int last = order.back();
        double best_score = -1;
        int best_next = -1;
        for (auto& [nb, w] : adj[last]) {
            if (!in_set.count(nb) || used[nb]) continue;
            if (w > best_score) { best_score = w; best_next = nb; }
        }
        if (best_next == -1) {
            // No connected neighbor: pick unvisited with highest pin weight
            for (int b : free_blocks) {
                if (used[b]) continue;
                double w = block_weight.count(b) ? block_weight[b] : 0.0;
                if (w > best_score) { best_score = w; best_next = b; }
            }
        }
        if (best_next == -1) {
            for (int b : free_blocks) if (!used[b]) { best_next = b; break; }
        }
        order.push_back(best_next);
        used[best_next] = true;
    }
    return order;
}

// ─── Post-processing: Boundary snap ──────────────────────────────────────────
// For each free non-cluster boundary block, push it toward its required edge
// using available slack (no overlap). Repeats a few passes because moves can
// change the bbox. Guaranteed to never create overlaps; may reduce boundary
// violations directly. Cluster blocks are skipped to avoid breaking adjacency.
static vector<Pos> boundary_snap(vector<Pos> pos) {
    int n = (int)pos.size();
    if (boundary_blocks.empty() || n == 0) return pos;

    for (int pass = 0; pass < 6; pass++) {
        double xmn=1e18, ymn=1e18, xmx=-1e18, ymx=-1e18;
        for (auto& p : pos) {
            xmn = min(xmn, p.x); ymn = min(ymn, p.y);
            xmx = max(xmx, p.x+p.w); ymx = max(ymx, p.y+p.h);
        }

        int moved = 0;
        for (auto& [b, flag] : boundary_blocks) {
            if (blocks[b].is_fixed || blocks[b].is_preplaced) continue;
            if (blocks[b].cluster > 0) continue;          // skip cluster blocks

            double bw = pos[b].w, bh = pos[b].h;

            // Apply X axis then Y axis (independent slack)
            for (int axis = 0; axis < 2; axis++) {
                double bx = pos[b].x, by = pos[b].y;
                double bx1 = bx + bw, by1 = by + bh;
                double target_d = 0.0;
                if (axis == 0) {
                    if (flag & B_LEFT)  target_d = xmn - bx;       // ≤ 0
                    if (flag & B_RIGHT) target_d = xmx - bx1;      // ≥ 0
                } else {
                    if (flag & B_BOTTOM) target_d = ymn - by;      // ≤ 0
                    if (flag & B_TOP)    target_d = ymx - by1;     // ≥ 0
                }
                if (fabs(target_d) < 1e-9) continue;

                double need = fabs(target_d);
                double slack = need;
                for (int o = 0; o < n; o++) {
                    if (o == b) continue;
                    if (axis == 0) {
                        bool yov = pos[o].y < by1-1e-6 && pos[o].y+pos[o].h > by+1e-6;
                        if (!yov) continue;
                        if (target_d < 0 && pos[o].x+pos[o].w <= bx+1e-9)
                            slack = min(slack, bx - (pos[o].x+pos[o].w));
                        else if (target_d > 0 && pos[o].x >= bx1-1e-9)
                            slack = min(slack, pos[o].x - bx1);
                    } else {
                        bool xov = pos[o].x < bx1-1e-6 && pos[o].x+pos[o].w > bx+1e-6;
                        if (!xov) continue;
                        if (target_d < 0 && pos[o].y+pos[o].h <= by+1e-9)
                            slack = min(slack, by - (pos[o].y+pos[o].h));
                        else if (target_d > 0 && pos[o].y >= by1-1e-9)
                            slack = min(slack, pos[o].y - by1);
                    }
                }
                if (slack <= 1e-9) continue;

                double d = (target_d < 0) ? -slack : slack;
                if (axis == 0) pos[b].x += d; else pos[b].y += d;
                moved++;
            }
        }
        if (moved == 0) break;
    }
    return pos;
}

// ─── Post-processing: Cluster snap ───────────────────────────────────────────
// For each cluster with multiple connected components, find the largest
// component (anchor) and slide non-anchor free members toward the anchor's
// centroid using available slack. Reduces cluster connectivity violations
// without creating overlaps. Skips fixed/preplaced members.
static vector<Pos> cluster_snap(vector<Pos> pos) {
    int n = (int)pos.size();
    if (cluster_groups.empty()) return pos;

    for (int pass = 0; pass < 20; pass++) {
        int moved = 0;

        for (auto& [gid, members] : cluster_groups) {
            int m = (int)members.size();
            if (m < 2) continue;

            // Adjacency: cluster members touch if they share an edge segment
            vector<vector<int>> adj(m);
            for (int ii = 0; ii < m; ii++) {
                int bi = members[ii];
                for (int jj = ii+1; jj < m; jj++) {
                    int bj = members[jj];
                    double ox = min(pos[bi].x+pos[bi].w, pos[bj].x+pos[bj].w) - max(pos[bi].x, pos[bj].x);
                    double oy = min(pos[bi].y+pos[bi].h, pos[bj].y+pos[bj].h) - max(pos[bi].y, pos[bj].y);
                    bool touch =
                        (oy > 1e-6 && (fabs(pos[bi].x+pos[bi].w - pos[bj].x) < 1e-4 ||
                                       fabs(pos[bj].x+pos[bj].w - pos[bi].x) < 1e-4)) ||
                        (ox > 1e-6 && (fabs(pos[bi].y+pos[bi].h - pos[bj].y) < 1e-4 ||
                                       fabs(pos[bj].y+pos[bj].h - pos[bi].y) < 1e-4));
                    if (touch) { adj[ii].push_back(jj); adj[jj].push_back(ii); }
                }
            }

            // BFS to find components
            vector<int> comp(m, -1);
            vector<int> comp_size;
            int n_comps = 0;
            for (int s = 0; s < m; s++) {
                if (comp[s] != -1) continue;
                int cs = n_comps++;
                queue<int> q; q.push(s); comp[s] = cs;
                int sz = 0;
                while (!q.empty()) {
                    int c = q.front(); q.pop(); sz++;
                    for (int nb : adj[c]) if (comp[nb] == -1) { comp[nb] = cs; q.push(nb); }
                }
                comp_size.push_back(sz);
            }
            if (n_comps == 1) continue;     // already connected

            // Anchor = largest component
            int anchor_comp = 0;
            for (int c = 1; c < n_comps; c++)
                if (comp_size[c] > comp_size[anchor_comp]) anchor_comp = c;

            // Anchor centroid
            double acx = 0, acy = 0; int an = 0;
            for (int i = 0; i < m; i++) if (comp[i] == anchor_comp) {
                int b = members[i];
                acx += pos[b].x + pos[b].w/2;
                acy += pos[b].y + pos[b].h/2;
                an++;
            }
            if (an > 0) { acx /= an; acy /= an; }

            // Slide each non-anchor free member toward anchor centroid.
            // Try dominant axis first; if blocked, fall back to the other axis.
            for (int i = 0; i < m; i++) {
                if (comp[i] == anchor_comp) continue;
                int b = members[i];
                if (blocks[b].is_fixed || blocks[b].is_preplaced) continue;

                double bw = pos[b].w, bh = pos[b].h;
                double bcx = pos[b].x + bw/2;
                double bcy = pos[b].y + bh/2;
                double dx_t = acx - bcx, dy_t = acy - bcy;
                int bflag = blocks[b].boundary;

                bool primary_x = fabs(dx_t) >= fabs(dy_t);

                for (int axis_try = 0; axis_try < 2; axis_try++) {
                    bool move_x = (axis_try == 0) ? primary_x : !primary_x;
                    double dir = move_x ? dx_t : dy_t;
                    if (fabs(dir) < 1e-9) continue;

                    if (move_x) {
                        if ((bflag & B_LEFT)  && dir > 0) continue;
                        if ((bflag & B_RIGHT) && dir < 0) continue;
                    } else {
                        if ((bflag & B_BOTTOM) && dir > 0) continue;
                        if ((bflag & B_TOP)    && dir < 0) continue;
                    }

                    double bx = pos[b].x, by = pos[b].y;
                    double bx1 = bx + bw, by1 = by + bh;
                    double slack = fabs(dir);

                    for (int o = 0; o < n; o++) {
                        if (o == b) continue;
                        if (move_x) {
                            bool yov = pos[o].y < by1-1e-6 && pos[o].y+pos[o].h > by+1e-6;
                            if (!yov) continue;
                            if (dir < 0 && pos[o].x+pos[o].w <= bx+1e-9)
                                slack = min(slack, bx - (pos[o].x+pos[o].w));
                            else if (dir > 0 && pos[o].x >= bx1-1e-9)
                                slack = min(slack, pos[o].x - bx1);
                        } else {
                            bool xov = pos[o].x < bx1-1e-6 && pos[o].x+pos[o].w > bx+1e-6;
                            if (!xov) continue;
                            if (dir < 0 && pos[o].y+pos[o].h <= by+1e-9)
                                slack = min(slack, by - (pos[o].y+pos[o].h));
                            else if (dir > 0 && pos[o].y >= by1-1e-9)
                                slack = min(slack, pos[o].y - by1);
                        }
                    }
                    if (slack <= 1e-9) continue;

                    double d = (dir > 0 ? slack : -slack);
                    double v_before = calc_violation(pos);
                    double old_x = pos[b].x, old_y = pos[b].y;
                    if (move_x) pos[b].x += d; else pos[b].y += d;
                    double v_after = calc_violation(pos);
                    if (v_after > v_before + 1e-9) {
                        pos[b].x = old_x; pos[b].y = old_y;
                    } else {
                        moved++;
                        break;     // accepted in this axis; don't try the other
                    }
                }
            }
        }
        if (moved == 0) break;
    }
    return pos;
}

// ─── Post-processing: HPWL hill-climber ──────────────────────────────────────
// For each non-cluster free block, compute connectivity force, slide it toward
// its neighbours using the available gap (slack), accept only if local HPWL
// improves.  Guaranteed to never increase wire length or create overlaps.
static vector<Pos> slack_hpwl_opt(vector<Pos> pos,
                                   chrono::time_point<Clock> t0,
                                   double budget_sec = 0.15)
{
    int n = (int)pos.size();
    if (n == 0) return pos;

    // Pre-build per-block adjacency
    vector<vector<pair<int,double>>> badj(n), padj(n);
    for (auto& e : b2b_edges)
        if (e.i>=0&&e.i<n&&e.j>=0&&e.j<n) {
            badj[e.i].push_back({e.j,e.w});
            badj[e.j].push_back({e.i,e.w});
        }
    for (auto& e : p2b_edges)
        if (e.j>=0&&e.j<n) padj[e.j].push_back({e.i,e.w});

    // Iterate: for each non-cluster, non-fixed block, try sliding it toward
    // its connectivity neighbours. Accept only if local HPWL strictly improves
    // and no overlap is introduced.
    for (int pass = 0; pass < 40; pass++) {
        if (elapsed_sec(t0) > TIME_LIMIT + budget_sec) break;
        int moved = 0;

        for (int b = 0; b < n; b++) {
            if (blocks[b].is_fixed || blocks[b].is_preplaced) continue;
            if (blocks[b].cluster > 0) continue;  // skip cluster blocks (adjacency must be preserved)

            double bcx = pos[b].x + pos[b].w/2;
            double bcy = pos[b].y + pos[b].h/2;
            double fx=0, fy=0, hpwl0=0;

            for (auto& [nb,w] : badj[b]) {
                double ox=pos[nb].x+pos[nb].w/2, oy=pos[nb].y+pos[nb].h/2;
                fx += w*(ox-bcx); fy += w*(oy-bcy);
                hpwl0 += w*(fabs(ox-bcx)+fabs(oy-bcy));
            }
            for (auto& [pi,w] : padj[b]) {
                fx += w*(pins[pi].first-bcx); fy += w*(pins[pi].second-bcy);
                hpwl0 += w*(fabs(pins[pi].first-bcx)+fabs(pins[pi].second-bcy));
            }
            if (fabs(fx)<1e-9 && fabs(fy)<1e-9) continue;

            // Respect boundary constraints
            int bflag = blocks[b].boundary;
            if ((bflag&B_LEFT)   && fx>0) fx=0;
            if ((bflag&B_RIGHT)  && fx<0) fx=0;
            if ((bflag&B_BOTTOM) && fy>0) fy=0;
            if ((bflag&B_TOP)    && fy<0) fy=0;
            if (fabs(fx)<1e-9 && fabs(fy)<1e-9) continue;

            // Move in the dominant axis only (simpler; less chance of corner issues)
            bool move_x = fabs(fx) >= fabs(fy);
            double bx0=pos[b].x, by0=pos[b].y, bx1=pos[b].x+pos[b].w, by1=pos[b].y+pos[b].h;

            // Compute available slack in the chosen direction
            double slack = 1e18;
            for (int o=0; o<n; o++) {
                if (o==b) continue;
                if (move_x) {
                    bool yov = pos[o].y<by1-1e-6 && pos[o].y+pos[o].h>by0+1e-6;
                    if (!yov) continue;
                    if (fx>0 && pos[o].x>=bx1-1e-9)
                        slack=min(slack, pos[o].x-bx1);
                    else if (fx<0 && pos[o].x+pos[o].w<=bx0+1e-9)
                        slack=min(slack, bx0-(pos[o].x+pos[o].w));
                } else {
                    bool xov = pos[o].x<bx1-1e-6 && pos[o].x+pos[o].w>bx0+1e-6;
                    if (!xov) continue;
                    if (fy>0 && pos[o].y>=by1-1e-9)
                        slack=min(slack, pos[o].y-by1);
                    else if (fy<0 && pos[o].y+pos[o].h<=by0+1e-9)
                        slack=min(slack, by0-(pos[o].y+pos[o].h));
                }
            }
            if (slack <= 1e-9) continue;

            // Proposed move: force direction, clamped by 90% of slack
            double dx=0, dy=0;
            if (move_x)
                dx = (fx>0) ? min(fx, slack*0.9) : -min(-fx, slack*0.9);
            else
                dy = (fy>0) ? min(fy, slack*0.9) : -min(-fy, slack*0.9);
            if (fabs(dx)<1e-9 && fabs(dy)<1e-9) continue;

            // Accept only if local HPWL strictly improves
            double new_bcx=bcx+dx, new_bcy=bcy+dy, hpwl1=0;
            for (auto& [nb,w] : badj[b]) {
                double ox=pos[nb].x+pos[nb].w/2, oy=pos[nb].y+pos[nb].h/2;
                hpwl1 += w*(fabs(ox-new_bcx)+fabs(oy-new_bcy));
            }
            for (auto& [pi,w] : padj[b]) {
                hpwl1 += w*(fabs(pins[pi].first-new_bcx)+fabs(pins[pi].second-new_bcy));
            }
            if (hpwl1 < hpwl0 - 1e-9) {
                pos[b].x += dx; pos[b].y += dy;
                moved++;
            }
        }
        if (moved == 0) break;
    }
    return pos;
}

// ─── SA ───────────────────────────────────────────────────────────────────────
static bool   use_max_width = true;   // set false by --no-width flag
static mt19937 rng(42);

static void run_sa(chrono::time_point<Clock> t0) {
    // Parse blocks into local structures
    map<int,Pos> preplaced_map;
    vector<bool> is_fixed(N, false), is_preplaced(N, false);

    mib_groups.clear(); cluster_groups.clear(); boundary_blocks.clear(); n_soft = 0;

    for (int i = 0; i < N; i++) {
        auto& b = blocks[i];
        if (b.is_preplaced) {
            is_preplaced[i] = true;
            preplaced_map[i] = {b.tx, b.ty, b.tw, b.th};
        } else if (b.is_fixed) {
            is_fixed[i] = true;
        }
        if (b.mib     > 0) mib_groups    [b.mib    ].push_back(i);
        if (b.cluster > 0) cluster_groups[b.cluster].push_back(i);
        if (b.boundary> 0) boundary_blocks.push_back({i, b.boundary});
    }
    n_soft = (int)boundary_blocks.size();
    for (auto& [g, m] : cluster_groups) n_soft += max(0, (int)m.size()-1);
    for (auto& [g, m] : mib_groups    ) n_soft += max(0, (int)m.size()-1);

    // Initial widths/heights
    vector<double> widths(N), heights(N);
    for (int i = 0; i < N; i++) {
        if ((is_fixed[i] || is_preplaced[i]) && blocks[i].tw > 0 && blocks[i].th > 0) {
            widths[i] = blocks[i].tw; heights[i] = blocks[i].th;
        } else {
            double a = area_targets[i] > 0 ? area_targets[i] : 1.0;
            widths[i] = heights[i] = sqrt(a);
        }
    }

    // Pre-unify MIB shapes (soft blocks only)
    for (auto& [gid, members] : mib_groups) {
        vector<int> soft;
        for (int b : members) if (!is_preplaced[b] && !is_fixed[b]) soft.push_back(b);
        if (!soft.empty()) {
            double rw = widths[soft[0]], rh = heights[soft[0]];
            for (int b : soft) { widths[b] = rw; heights[b] = rh; }
        }
    }

    // Free / resizable blocks
    vector<int> free_blocks, resizable;
    for (int i = 0; i < N; i++) if (!is_preplaced[i]) free_blocks.push_back(i);
    for (int b : free_blocks) if (!is_fixed[b]) resizable.push_back(b);

    // MIB free groups (>= 2 soft members)
    map<int,vector<int>> mib_free;
    for (auto& [gid, members] : mib_groups) {
        vector<int> soft;
        for (int b : members) if (!is_preplaced[b] && !is_fixed[b]) soft.push_back(b);
        if (soft.size() >= 2) mib_free[gid] = soft;
    }
    vector<int> mib_keys;
    for (auto& [g, _] : mib_free) mib_keys.push_back(g);

    // Cluster free groups (>= 2 free members) for cluster-adjacency moves
    map<int,vector<int>> cluster_free;
    for (auto& [gid, members] : cluster_groups) {
        vector<int> free_m;
        for (int b : members) if (!is_preplaced[b]) free_m.push_back(b);
        if (free_m.size() >= 2) cluster_free[gid] = free_m;
    }
    vector<int> cluster_keys;
    for (auto& [g, _] : cluster_free) cluster_keys.push_back(g);

    // MIB group ID lookup per block (for sync in resize/rotate)
    vector<int> mib_gid_of(N, -1);
    for (auto& [gid, members] : mib_groups)
        for (int b : members) mib_gid_of[b] = gid;

    // Cluster group ID lookup per block (for adjacency in perm and row-packing)
    vector<int> cluster_gid_of(N, 0);  // 0 = no cluster
    for (auto& [gid, members] : cluster_groups)
        for (int b : members) cluster_gid_of[b] = gid;
    s_cluster_gid_of = cluster_gid_of;  // expose to skyline_decode

    // Sync MIB group members after block b is resized to (nw[b], nh[b])
    auto sync_mib = [&](int b, vector<double>& nw, vector<double>& nh) {
        int gid = mib_gid_of[b];
        if (gid < 0) return;
        auto& members = mib_groups[gid];
        double lw = nw[b], lh = nh[b], la = lw * lh;
        for (int m : members) {
            if (m == b || is_preplaced[m] || is_fixed[m]) continue;
            double ab = area_targets[m];
            if (ab > 0) {
                if (fabs(la - ab) / (ab + 1e-9) <= AREA_TOL) {
                    nw[m] = lw; nh[m] = lh;
                } else {
                    double rat = (lh > 1e-9) ? lw / lh : 1.0;
                    nw[m] = sqrt(ab * rat); nh[m] = ab / nw[m];
                }
            } else { nw[m] = lw; nh[m] = lh; }
        }
    };

    // --- Build b2b adjacency list (for connectivity moves) ---
    vector<vector<pair<int,double>>> b2b_adj(N);
    for (auto& e : b2b_edges) {
        if (e.i < 0 || e.i >= N || e.j < 0 || e.j >= N) continue;
        b2b_adj[e.i].push_back({e.j, e.w});
        b2b_adj[e.j].push_back({e.i, e.w});
    }

    // Connectivity-driven initial perm
    vector<int> perm = connectivity_order(free_blocks, widths, heights);

    // Re-order perm to improve soft constraints:
    // 1) Group cluster members consecutively (encourages abutment).
    // 2) LEFT/BOTTOM boundary blocks first, RIGHT/TOP boundary blocks last.
    {
        // Build initial position map for stable secondary sort
        vector<int> init_pos(N, 0);
        for (int k = 0; k < (int)perm.size(); k++) init_pos[perm[k]] = k;

        // Priority ordering:
        //   0: Cluster blocks (grouped by gid) — placed first so row-packing works cleanly
        //   1: LEFT/BOTTOM boundary blocks — placed early for boundary satisfaction
        //   2: Unconstrained blocks (by connectivity order)
        //   3: RIGHT/TOP boundary blocks — placed last (push to outer edge)
        // Within clusters, keep original connectivity order (init_pos).
        auto priority = [&](int b) -> pair<int,int> {
            int flag = blocks[b].boundary;
            int cl = cluster_gid_of[b];
            bool is_lb = (flag & (B_LEFT | B_BOTTOM)) != 0;
            bool is_rt = (flag & (B_RIGHT | B_TOP)) != 0 && !is_lb;
            int bp;
            if      (cl > 0) bp = 0;        // cluster blocks first
            else if (is_lb)  bp = 1;        // LEFT/BOTTOM boundary next
            else if (is_rt)  bp = 3;        // RIGHT/TOP boundary last
            else             bp = 2;        // others in middle
            // Secondary key: cluster members group by gid (so they're consecutive)
            int sec = (cl > 0) ? (cl * 10000 + init_pos[b]) : (N + init_pos[b]);
            return {bp, sec};
        };

        stable_sort(perm.begin(), perm.end(), [&](int a, int b) {
            return priority(a) < priority(b);
        });
    }

    // Index of each block in perm (for O(1) lookup)
    vector<int> perm_idx(N, -1);
    for (int k = 0; k < (int)perm.size(); k++) perm_idx[perm[k]] = k;

    // Compute target width for rectangular packings (disabled by --no-width)
    double mw = 0.0;
    if (use_max_width) {
        double total_area = 0.0;
        for (int b : free_blocks) total_area += widths[b] * heights[b];
        for (auto& [bid, p] : preplaced_map) total_area += p.w * p.h;
        mw = sqrt(total_area * 1.4);
    }

    // Initial decode
    auto cur_pos  = skyline_decode(perm, widths, heights, preplaced_map, mw);

    // Calibrate W_AREA: balance area vs HPWL sensitivity in proxy cost.
    // Contest weights HPWL_gap and Area_gap equally, so target:
    //   W_AREA * d(area) == d(hpwl)  ⟹  W_AREA ≈ HPWL / area
    {
        double h0 = calc_hpwl_b2b(cur_pos) + calc_hpwl_p2b(cur_pos);
        double a0 = calc_bbox_area(cur_pos);
        if (a0 > 0 && h0 > 0) {
            W_AREA = h0 / a0;
            W_AREA = max(0.01, min(W_AREA, 2.0));
        }
        // Calibrate W_VIOL so each violation is ≈6× more costly than 1 unit
        // of HPWL gap in the proxy (matches actual scoring formula gradient ratio
        // ∂cost/∂V_rel / ∂cost/∂HPWL_gap ≈ 6 at typical operating points).
        if (h0 > 0 && n_soft > 0) {
            W_VIOL = max(50.0, h0 * 9.0);      // 1.5× best from sweep
            // W_BOUNDARY stays at 10 — it's just a guidance gradient
        }
    }

    // Enable aspect-ratio penalty if max_width is active
    target_width_sq = use_max_width ? 1.0 : 0.0;

    double cur_cost = proxy_cost(cur_pos);

    vector<int>    best_perm  = perm;
    vector<double> best_w     = widths, best_h = heights;
    vector<Pos>    best_pos   = cur_pos;
    double         best_cost  = cur_cost;

    int n_free = (int)free_blocks.size();
    int n_res  = (int)resizable.size();

    // Distributions
    uniform_real_distribution<double> unif(0.0, 1.0);
    uniform_real_distribution<double> ar_dist(0.2, 5.0);

    // SA schedule: time-based cooling so we always use the full time budget.
    // temperature = T_start * (T_end/T_start)^(elapsed/TIME_LIMIT)
    const double T_START = 200.0;
    const double T_END   = 0.05;
    const double LOG_RATIO = log(T_END / T_START);  // negative

    // Perm position map (which index in perm each block lives at)
    // Rebuild lazily after relocate/swap moves
    auto rebuild_idx = [&]() {
        for (int k = 0; k < (int)perm.size(); k++) perm_idx[perm[k]] = k;
    };

    double temp = T_START;
    while (elapsed_sec(t0) < TIME_LIMIT) {
        double progress = elapsed_sec(t0) / TIME_LIMIT;
        temp = T_START * exp(LOG_RATIO * progress);
        double r = unif(rng);

        vector<int>    np = perm;
        vector<double> nw = widths, nh = heights;

        if (r < 0.30 && n_free >= 2) {
            // Swap two positions
            int i = rng() % n_free, j = rng() % n_free;
            swap(np[i], np[j]);
        }
        else if (r < 0.44 && n_free >= 2) {
            // Relocate
            int i = rng() % n_free;
            int j = rng() % n_free;
            int blk = np[i];
            np.erase(np.begin() + i);
            np.insert(np.begin() + (j >= (int)np.size() ? np.size() : j), blk);
        }
        else if (r < 0.52 && n_free >= 2) {
            // Connectivity move: pick a b2b edge, move j next to i in perm
            if (!b2b_edges.empty()) {
                const auto& e = b2b_edges[rng() % b2b_edges.size()];
                int bi = e.i, bj = e.j;
                if (bi >= 0 && bi < N && bj >= 0 && bj < N &&
                    perm_idx[bi] >= 0 && perm_idx[bj] >= 0) {
                    int pi = perm_idx[bi], pj = perm_idx[bj];
                    int dst = pi + 1 < n_free ? pi + 1 : pi;
                    int blk = np[pj];
                    np.erase(np.begin() + pj);
                    dst = min(dst, (int)np.size());
                    np.insert(np.begin() + dst, blk);
                }
            }
        }
        else if (r < 0.62 && n_res > 0) {
            // Resize aspect ratio — sync MIB group members to same shape
            int b = resizable[rng() % n_res];
            double a = area_targets[b];
            if (a > 0) {
                double rat = ar_dist(rng);
                nw[b] = sqrt(a * rat);
                nh[b] = a / nw[b];
                sync_mib(b, nw, nh);
            }
        }
        else if (r < 0.70 && n_res > 0) {
            // Rotate — sync MIB group members to same shape
            int b = resizable[rng() % n_res];
            swap(nw[b], nh[b]);
            sync_mib(b, nw, nh);
        }
        else if (r < 0.82 && !mib_keys.empty()) {
            // MIB unify: pick a random MIB group and unify all members' shapes
            int gid = mib_keys[rng() % mib_keys.size()];
            auto& members = mib_free[gid];
            int leader = members[rng() % members.size()];
            double lw = nw[leader], lh = nh[leader], la = lw * lh;
            for (int b : members) {
                if (b == leader) continue;
                double ab = area_targets[b];
                if (ab > 0) {
                    if (fabs(la - ab) / (ab + 1e-9) <= AREA_TOL) {
                        nw[b] = lw; nh[b] = lh;
                    } else {
                        double rat = (lh > 1e-9) ? lw / lh : 1.0;
                        nw[b] = sqrt(ab * rat);
                        nh[b] = ab / nw[b];
                    }
                } else {
                    nw[b] = lw; nh[b] = lh;
                }
            }
        }
        else if (r < 0.90 && !cluster_keys.empty()) {
            // Cluster adjacency move: move a cluster member next to another.
            // This encourages abutment by making cluster members consecutive in perm.
            int gid = cluster_keys[rng() % cluster_keys.size()];
            auto& members = cluster_free[gid];
            int sz = (int)members.size();
            int ia = rng() % sz, ib = rng() % sz;
            if (ia != ib) {
                int ba = members[ia], bb = members[ib];
                int pa = perm_idx[ba], pb = perm_idx[bb];
                if (pa >= 0 && pb >= 0) {
                    int blk = np[pb];
                    np.erase(np.begin() + pb);
                    int dst = (pa < pb) ? pa + 1 : pa;
                    dst = min(dst, (int)np.size());
                    np.insert(np.begin() + dst, blk);
                }
            }
        }
        else if (n_free >= 2) {
            // Double swap (fallback)
            int i = rng() % n_free, j = rng() % n_free;
            swap(np[i], np[j]);
        }
        else { continue; }

        auto np_pos   = skyline_decode(np, nw, nh, preplaced_map, mw);
        double np_cost = proxy_cost(np_pos);
        double delta   = np_cost - cur_cost;

        if (delta < 0.0 || unif(rng) < exp(-delta / temp)) {
            perm   = np;
            widths = nw; heights = nh;
            cur_cost = np_cost; cur_pos = np_pos;
            rebuild_idx();

            if (cur_cost < best_cost) {
                best_cost = cur_cost;
                best_perm = perm;
                best_w    = widths; best_h = heights;
                best_pos  = cur_pos;
            }
        }
    }

    // Post-process: cluster_snap → boundary_snap → HPWL slack → boundary_snap
    // 1) cluster_snap: pull disconnected free cluster members toward anchors
    // 2) first boundary_snap: push boundary blocks to required edges
    // 3) hill-climber: optimise wire length (boundary direction respected)
    // 4) final boundary_snap: re-align any block displaced by hill-climber
    best_pos = cluster_snap(best_pos);
    best_pos = boundary_snap(best_pos);
    best_pos = slack_hpwl_opt(best_pos, t0, 0.15);
    best_pos = boundary_snap(best_pos);

    // Output
    printf("%d\n", N);
    for (int i = 0; i < N; i++)
        printf("%.10f %.10f %.10f %.10f\n",
               best_pos[i].x, best_pos[i].y,
               best_pos[i].w, best_pos[i].h);
}

// ─── I/O ──────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--no-width") use_max_width = false;
    }
    auto t0 = Clock::now();

    scanf("%d", &N);
    blocks.resize(N);
    area_targets.resize(N, 0.0);

    for (int i = 0; i < N; i++) scanf("%lf", &area_targets[i]);

    int nb2b; scanf("%d", &nb2b);
    b2b_edges.resize(nb2b);
    for (int i = 0; i < nb2b; i++)
        scanf("%d %d %lf", &b2b_edges[i].i, &b2b_edges[i].j, &b2b_edges[i].w);

    int np2b; scanf("%d", &np2b);
    p2b_edges.resize(np2b);
    for (int i = 0; i < np2b; i++)
        scanf("%d %d %lf", &p2b_edges[i].i, &p2b_edges[i].j, &p2b_edges[i].w);

    int npins; scanf("%d", &npins);
    pins.resize(npins);
    for (int i = 0; i < npins; i++)
        scanf("%lf %lf", &pins[i].first, &pins[i].second);

    for (int i = 0; i < N; i++) {
        int fx, pp, mib, cl, bnd;
        scanf("%d %d %d %d %d", &fx, &pp, &mib, &cl, &bnd);
        blocks[i].is_fixed     = fx != 0;
        blocks[i].is_preplaced = pp != 0;
        blocks[i].mib          = mib;
        blocks[i].cluster      = cl;
        blocks[i].boundary     = bnd;
    }
    for (int i = 0; i < N; i++)
        scanf("%lf %lf %lf %lf",
              &blocks[i].tx, &blocks[i].ty, &blocks[i].tw, &blocks[i].th);

    run_sa(t0);
    return 0;
}
