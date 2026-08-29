"""L144v2 -- generate constructive_l144v2.cpp from constructive.cpp.

NEW FILE ONLY. constructive.cpp is read, never written.
Mechanism: ICCAD_BND_DEMAND_ORDER (0 = off = bit-identical, 1 = on).
"""
import sys
from pathlib import Path

D = Path(__file__).parent
src = (D / "constructive.cpp").read_text(encoding="utf-8", errors="surrogateescape")
orig = src

# ---------------------------------------------------------------- 1. the flag
A0 = 'static int CLUSTER_ORD = 0;        // ICCAD_CLUSTER_ORD: 1 = compound (multi-block cluster)'
A1 = '''// \xe2\x94\x80\xe2\x94\x80\xe2\x94\x80 L144v2: ICCAD_BND_DEMAND_ORDER \xe2\x94\x80\xe2\x94\x80\xe2\x94\x80
// Boundary items already sort first (bscore is the primary key) but INSIDE a
// bscore class the order is member-count -> area -> max dimension: nothing about
// how much of the EDGE the item consumes. The L144 trace showed the greedy takes
// a compliant slot 584001/584001 times it has one and has none 21.7% of the
// time, i.e. the miss is edge AVAILABILITY. A large boundary item placed late
// cannot fit any surviving run; the same item placed first always fits. So order
// each bscore class by along-edge extent (edge demand) descending.
// 0 = off => the whole block is skipped => bit-identical.
static int BND_DEMAND_ORDER = 0;
'''
assert src.count(A0) == 1
src = src.replace(A0, A1 + A0)

# ------------------------------------------------------- 2. the demand metric
B0 = '''    it.bscore=bs; it.total_wire=tw;
}
'''
B1 = '''    it.bscore=bs; it.total_wire=tw;
}
// L144v2: how much of its required EDGE this item consumes. The union of the
// members' boundary codes decides the side; the along-edge extent is the item
// dimension PARALLEL to that edge (LEFT/RIGHT are vertical edges -> height;
// TOP/BOTTOM are horizontal edges -> width).
// AMBIGUOUS CASE (documented choice): if the union code names both a vertical
// and a horizontal side -- a true corner block, or a cluster holding e.g. one
// LEFT member and one BOTTOM member -- the item consumes run on BOTH edges, so
// we take max(w,h). That is the larger of the two demands, which is the
// conservative key for a "hardest to fit first" order.
// Items with no boundary member return 0, so the bscore==0 class (which is
// exactly those items) keeps its base order under a stable_sort.
static double item_edge_demand(const Item& it){
    int code=0;
    for (int b: it.blocks) code |= blocks[b].boundary;
    if (code==0) return 0.0;
    bool lr = (code & (B_LEFT|B_RIGHT)) != 0;
    bool tb = (code & (B_TOP|B_BOTTOM)) != 0;
    if (lr && !tb) return it.h;
    if (tb && !lr) return it.w;
    return max(it.w, it.h);
}
'''
assert src.count(B0) == 1, src.count(B0)
src = src.replace(B0, B1, 1)

# ------------------------------------------------------- 3. the ordering pass
C0 = '''    // Oracle-perm probe (OFFLINE only, never shipped). If ICCAD_ORDER_FILE is set,'''
C1 = '''    if (BND_DEMAND_ORDER) {
        // L144v2 "largest edge demand first". Same class-preserving shape as the
        // CLUSTER_ORD block above: walk the maximal runs of equal bscore and
        // stable_sort INSIDE each run only, so bscore classes never mix and the
        // boundary-first property is untouched. Ties (and the whole bscore==0
        // class, whose demand is 0) keep whatever order the base/WT/BFS/ORD
        // layers produced.
        int I=(int)items.size();
        for (int s=0;s<I;){
            int e=s; while (e<I && items[e].bscore==items[s].bscore) e++;
            stable_sort(items.begin()+s, items.begin()+e,
                [](const Item& a, const Item& b){
                    double da=item_edge_demand(a), db=item_edge_demand(b);
                    if (fabs(da-db)>TOL) return da>db;
                    return false;
                });
            s=e;
        }
    }

''' + C0
assert src.count(C0) == 1
src = src.replace(C0, C1, 1)

# ------------------------------------------------------------ 4. env parsing
D0 = '''    if (const char* e=getenv("ICCAD_BP_WEIGHT")) { double v=atof(e); if (v>0) BP_W=v; }'''
D1 = D0 + '''
    if (const char* e=getenv("ICCAD_BND_DEMAND_ORDER")) { int v=atoi(e); if (v==1) BND_DEMAND_ORDER=1; }   // L144v2'''
assert src.count(D0) == 1
src = src.replace(D0, D1, 1)

out = D / "constructive_l144v2.cpp"
assert not out.exists() or "--force" in sys.argv, "refusing to overwrite"
out.write_text(src, encoding="utf-8", errors="surrogateescape")
print("wrote", out, len(orig), "->", len(src))
