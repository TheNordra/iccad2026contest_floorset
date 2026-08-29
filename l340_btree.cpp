// L340: B*-tree simulated annealing on the LABEL'S OWN SPACE, minimising
// area + wirelength -- the representation the generator used, the objective we
// are actually scored on.
//
// WHY THIS SHAPE (all measured, see L320-L336b):
//   * the labels ARE B*-tree placements: tree_sol in the 1M training shards obeys
//     left-child x = x_p + w_p (4419/4419), right-child x = x_p (3869/3869) and
//     y = contour (8400/8400).
//   * every label coordinate is an integer and every soft block's area equals its
//     target EXACTLY -- zero use of the contest's 1% slack -- so the shape space is
//     the integer divisor pairs of A with aspect <= 3 (7050/7050 blocks).
//   * a pure-Python version of this reached 0.9455 utilisation at n=40 and was
//     still climbing at 160k iterations, above our shipped packer's 0.877 and above
//     the 85.4% "density ceiling" L284 measured for OUR pool. That ceiling is a
//     property of our packer's reachable set, not of the instance.
//   * BUT the generator's own objective is AREA ONLY, and replaying it gives
//     hpwl_gap 1.13-1.60 against our 0.240, because the netlist was sampled FROM
//     the generator's finished layout (paper Alg. 4) and does not fit a different
//     area optimum. So: keep the representation, drop the objective.
//
// The wirelength here is the contest's, which is NOT half-perimeter: it is weighted
// centre-to-centre Manhattan over 2-pin edges (iccad2026_evaluate.calculate_hpwl_b2b),
// plus the same against fixed pins. That makes it O(E) to evaluate and, importantly,
// INCREMENTAL is not worth it -- a B*-tree move relocates many blocks at once.
//
// Offline research tool. Never shipped, reads no label, writes no package.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

struct Edge { int a, b; double w; };
struct PEdge { int p, b; double w; };

static int N;                            // blocks
static std::vector<std::vector<std::pair<int,int>>> SH;   // per block: (w,h) options
static std::vector<Edge> B2B;
static std::vector<PEdge> P2B;
static std::vector<std::pair<double,double>> PIN;
static double HW = 0.0;                  // wirelength weight, in area units

// ---- B*-tree ---------------------------------------------------------------
struct Tree {
    std::vector<int> L, R, par, si;
    int root = 0;
    void init(int n, std::mt19937 &rng) {
        L.assign(n, -1); R.assign(n, -1); par.assign(n, -1); si.assign(n, 0);
        std::vector<int> ord(n);
        for (int i = 0; i < n; i++) ord[i] = i;
        std::shuffle(ord.begin(), ord.end(), rng);
        root = ord[0];
        for (int k = 1; k < n; k++) {
            int c = ord[k];
            int p = ord[rng() % k];
            for (;;) {
                if (rng() & 1) { if (L[p] < 0) { L[p] = c; par[c] = p; break; } p = L[p]; }
                else           { if (R[p] < 0) { R[p] = c; par[c] = p; break; } p = R[p]; }
            }
        }
    }
};

// contour: xs[] sorted breakpoints, hs[i] = height on [xs[i], xs[i+1])
static std::vector<int> cx, ch, stk_n, stk_x;
static std::vector<int> PX, PY, PW, PH;

// decode; returns bbox W,H and fills PX/PY/PW/PH
static void pack(const Tree &t, int &W, int &H) {
    cx.assign(1, 0); ch.assign(1, 0);
    cx.push_back(1 << 30); ch.push_back(0);
    W = H = 0;
    stk_n.clear(); stk_x.clear();
    stk_n.push_back(t.root); stk_x.push_back(0);
    while (!stk_n.empty()) {
        int k = stk_n.back(); stk_n.pop_back();
        int x = stk_x.back(); stk_x.pop_back();
        int w = SH[k][t.si[k]].first, h = SH[k][t.si[k]].second;
        int xe = x + w;
        int i = int(std::upper_bound(cx.begin(), cx.end(), x) - cx.begin()) - 1;
        int j = int(std::lower_bound(cx.begin(), cx.end(), xe) - cx.begin());
        int y = 0;
        for (int q = i; q < j; q++) if (ch[q] > y) y = ch[q];
        int top = y + h;
        int tail = ch[j - 1];
        // replace [i, j) with (x, top) and, if the block ends before the next
        // breakpoint, (xe, tail)
        std::vector<int> nx, nh;
        if (cx[i] < x) { nx.push_back(cx[i]); nh.push_back(ch[i]); }
        nx.push_back(x); nh.push_back(top);
        if (xe < cx[j]) { nx.push_back(xe); nh.push_back(tail); }
        cx.erase(cx.begin() + i, cx.begin() + j);
        ch.erase(ch.begin() + i, ch.begin() + j);
        cx.insert(cx.begin() + i, nx.begin(), nx.end());
        ch.insert(ch.begin() + i, nh.begin(), nh.end());
        PX[k] = x; PY[k] = y; PW[k] = w; PH[k] = h;
        if (xe > W) W = xe;
        if (top > H) H = top;
        if (t.R[k] >= 0) { stk_n.push_back(t.R[k]); stk_x.push_back(x); }
        if (t.L[k] >= 0) { stk_n.push_back(t.L[k]); stk_x.push_back(xe); }
    }
}

static double wirelength() {
    double s = 0.0;
    for (const Edge &e : B2B) {
        double x1 = PX[e.a] + PW[e.a] * 0.5, y1 = PY[e.a] + PH[e.a] * 0.5;
        double x2 = PX[e.b] + PW[e.b] * 0.5, y2 = PY[e.b] + PH[e.b] * 0.5;
        s += e.w * (std::fabs(x2 - x1) + std::fabs(y2 - y1));
    }
    for (const PEdge &e : P2B) {
        double x1 = PX[e.b] + PW[e.b] * 0.5, y1 = PY[e.b] + PH[e.b] * 0.5;
        s += e.w * (std::fabs(PIN[e.p].first - x1) + std::fabs(PIN[e.p].second - y1));
    }
    return s;
}

static double cost(const Tree &t, int &W, int &H, double &area, double &wl) {
    pack(t, W, H);
    area = double(W) * double(H);
    wl = HW > 0.0 ? wirelength() : 0.0;
    return area + HW * wl;
}

static bool detach(Tree &t, int k) {
    if (t.L[k] >= 0 && t.R[k] >= 0) return false;
    int c = t.L[k] >= 0 ? t.L[k] : t.R[k];
    int p = t.par[k];
    if (p < 0) { if (c < 0) return false; t.root = c; t.par[c] = -1; }
    else {
        if (t.L[p] == k) t.L[p] = c; else t.R[p] = c;
        if (c >= 0) t.par[c] = p;
    }
    t.L[k] = t.R[k] = -1; t.par[k] = -1;
    return true;
}

static void attach(Tree &t, int k, int p, int side, std::mt19937 &rng) {
    int c = side ? t.R[p] : t.L[p];
    (side ? t.R[p] : t.L[p]) = k;
    t.par[k] = p;
    if (c >= 0) { (rng() & 1 ? t.L[k] : t.R[k]) = c; t.par[c] = k; }
}

int main(int argc, char **argv) {
    // stdin:  n  hw  iters  seed
    //         n lines: k  n_opts  w1 h1 w2 h2 ...
    //         nE   then nE lines: a b w
    //         nP   then nP lines: px py
    //         nPE  then nPE lines: p b w
    long long iters = 200000;
    unsigned seed = 1;
    if (scanf("%d %lf %lld %u", &N, &HW, &iters, &seed) != 4) return 1;
    SH.assign(N, {});
    for (int i = 0; i < N; i++) {
        int k, m; scanf("%d %d", &k, &m);
        for (int j = 0; j < m; j++) { int w, h; scanf("%d %d", &w, &h); SH[k].push_back({w, h}); }
    }
    int nE; scanf("%d", &nE);
    B2B.resize(nE);
    for (int i = 0; i < nE; i++) scanf("%d %d %lf", &B2B[i].a, &B2B[i].b, &B2B[i].w);
    int nP; scanf("%d", &nP);
    PIN.resize(nP);
    for (int i = 0; i < nP; i++) scanf("%lf %lf", &PIN[i].first, &PIN[i].second);
    int nPE; scanf("%d", &nPE);
    P2B.resize(nPE);
    for (int i = 0; i < nPE; i++) scanf("%d %d %lf", &P2B[i].p, &P2B[i].b, &P2B[i].w);

    PX.assign(N, 0); PY.assign(N, 0); PW.assign(N, 0); PH.assign(N, 0);
    std::mt19937 rng(seed);
    Tree t; t.init(N, rng);
    int W, H; double area, wl;
    double cur = cost(t, W, H, area, wl), best = cur;
    Tree bt = t;
    double T0 = cur * 0.05;
    std::uniform_real_distribution<double> U(0.0, 1.0);
    for (long long it = 0; it < iters; it++) {
        double frac = 1.0 - double(it) / double(iters);
        double T = T0 * frac * frac + 1e-9;
        Tree save = t;
        double m = U(rng);
        if (m < 0.40) {
            int k = rng() % N;
            if (SH[k].size() > 1) t.si[k] = int(rng() % SH[k].size());
        } else if (m < 0.70) {
            int a = rng() % N, b = rng() % N;
            if (a != b) {
                for (int i = 0; i < N; i++) {
                    if (t.L[i] == a) t.L[i] = b; else if (t.L[i] == b) t.L[i] = a;
                    if (t.R[i] == a) t.R[i] = b; else if (t.R[i] == b) t.R[i] = a;
                }
                std::swap(t.par[a], t.par[b]);
                std::swap(t.L[a], t.L[b]);
                std::swap(t.R[a], t.R[b]);
                if (t.L[a] >= 0) t.par[t.L[a]] = a;
                if (t.R[a] >= 0) t.par[t.R[a]] = a;
                if (t.L[b] >= 0) t.par[t.L[b]] = b;
                if (t.R[b] >= 0) t.par[t.R[b]] = b;
                if (t.root == a) t.root = b; else if (t.root == b) t.root = a;
            }
        } else {
            int k = rng() % N;
            if (!detach(t, k)) { t = save; continue; }
            int p = rng() % N;
            while (p == k) p = rng() % N;
            attach(t, k, p, int(rng() & 1), rng);
        }
        double nw = cost(t, W, H, area, wl);
        if (nw <= cur || U(rng) < std::exp((cur - nw) / T)) {
            cur = nw;
            if (nw < best) { best = nw; bt = t; }
        } else t = save;
    }
    t = bt;
    cost(t, W, H, area, wl);
    printf("%d %d %.10f %.10f\n", W, H, area, wl);
    for (int i = 0; i < N; i++) printf("%d %d %d %d\n", PX[i], PY[i], PW[i], PH[i]);
    return 0;
}
