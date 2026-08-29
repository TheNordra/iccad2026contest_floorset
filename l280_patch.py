"""L280 -- commit connected blocks TOGETHER. The one axis with no verdict anywhere.

WHY THIS AND NOT ANOTHER WIRE HEURISTIC. L276 measured that ~99 % of `hpwl_gap`
survives an exact minimisation of the official objective inside our own topology,
so wire is an ADJACENCY problem. And two attempts to fix it by giving the greedy
better wire *information* both made hpwl WORSE:

    L272   the L137 hint feeding the wire term      hpwl 0.2924 -> 0.2999
    GUIDE_MED   a wire-optimal candidate origin     hpwl 0.2484 -> 0.2538

Both are M78's "adding candidates is harmful by default". The diagnosis those two
failures point at is that the greedy is **commitment-limited, not
information-limited**: by the time an item is placed, the blocks it should sit next
to are already fixed somewhere else. No scoring rule can undo that. Only committing
them together can.

WHAT IS ACTUALLY UNTRIED. `constructive.cpp:1772-1789` builds compound items from
`cluster_map[blocks[i].cluster]` and nothing else -- i.e. only from a grouping the
PROBLEM gives us. Connectivity has never formed an item. It has only ever
re-ORDERED items (`WIRE_ORDER`, `WIRE_TIEBREAK`, `WIRE_BFS`, `BFS_PIN`, all
<= 0.063 % and all measured in-set) or re-scored candidate positions (M71, M78).
Grouping and ordering are different levers: ordering decides WHEN a block is
committed, grouping decides that two blocks are committed as ONE object and
therefore land adjacent by construction.

THE RULE, AND IT HAS NO FITTED CONSTANT. Two movable, non-cluster blocks are made
one compound item iff they are each other's HEAVIEST b2b neighbour -- mutual
top-1. That is a matching (every block is in at most one pair), it needs no
threshold, no K, and no weight, so L266's over-fitting mode cannot apply and
L278's "does this corpus contain the antecedent" question has a countable answer:
the number of mutual top-1 pairs.

    ICCAD_L280=1        form mutual-top-1 compound items
    ICCAD_L280_REPORT=1 print the pair count to stderr (antecedent size)

Default off => bit-identical to the shipped binary.

⚠️ Synthetic pairs carry `blocks[i].cluster == 0`, so they cannot create grouping
violations -- `count_group_fragments` keys on the real cluster id. The risk is not
new violations; it is that forcing two blocks adjacent costs more in area/wire
than the adjacency buys, which is what the measurement is for.
"""
import hashlib
import sys
from pathlib import Path

DIR = Path(__file__).parent
SRC = DIR / "constructive.cpp"
DST = DIR / "constructive_l280.cpp"
EXPECT_SRC_MD5 = "e2c7b2f418ef2b70b6bff99f7adfbd37"

PAIRING = r'''
    // ── L280: commit connected blocks together ───────────────────────────────
    // Mutual top-1 b2b pairing over movable, non-cluster, not-yet-used blocks.
    // Parameter-free: i and j pair iff each is the other's heaviest neighbour.
    // That is a matching, so every block joins at most one synthetic item, and
    // the pair count is the antecedent size (reported so a null can be read).
    if (L280){
        vector<int> tbest(N,-1); vector<double> tw(N,0.0);
        for (int i=0;i<N;i++){
            if (blocks[i].is_preplaced||blocks[i].cluster>0||used[i]) continue;
            for (size_t k=0;k<b2b_adj[i].size();k++){
                int j=b2b_adj[i][k].first; double w=b2b_adj[i][k].second;
                if (j<0||j>=N) continue;
                if (blocks[j].is_preplaced||blocks[j].cluster>0||used[j]) continue;
                if (w>tw[i]+1e-12){ tw[i]=w; tbest[i]=j; }
            }
        }
        int npair=0;
        for (int i=0;i<N;i++){
            int j=tbest[i];
            if (j<0||j<=i) continue;              // each unordered pair once
            if (tbest[j]!=i) continue;            // MUTUAL top-1 only
            if (used[i]||used[j]) continue;
            vector<int> mv; mv.push_back(i); mv.push_back(j);
            Item it=make_group_item(mv);
            set_item_anchor(it); items.push_back(it);
            used[i]=1; used[j]=1; npair++;
        }
        if (L280_REPORT) fprintf(stderr,"L280PAIRS %d %d\n",npair,N);
    }

'''

PATCHES = [
    ('static bool FRAME_REPORT = false;',
     'static int  L280 = 0;            // ICCAD_L280=1: mutual-top-1 compound items\n'
     'static bool L280_REPORT = false; // ICCAD_L280_REPORT=1: stderr pair count\n'
     'static bool FRAME_REPORT = false;'),
    ('    if (getenv("ICCAD_GUIDE_MED")) GUIDE_MED=true;',
     '    if (const char* e=getenv("ICCAD_L280")){ int v=atoi(e); if (v>=0&&v<=1) L280=v; }\n'
     '    if (getenv("ICCAD_L280_REPORT")) L280_REPORT=true;\n'
     '    if (getenv("ICCAD_GUIDE_MED")) GUIDE_MED=true;'),
    # insert between the cluster loop (which sets used[]) and the singles loop
    # (which skips used[]) -- so a paired block is emitted once, as part of its
    # compound item, and never again as a single.
    ('''    for (int i=0;i<N;i++){
        if (blocks[i].is_preplaced||used[i]) continue;
        Item it; it.blocks={i}; it.offs={{0,0}}; finalize_item(it); set_item_anchor(it);
        items.push_back(it);
    }''',
     PAIRING +
     '''    for (int i=0;i<N;i++){
        if (blocks[i].is_preplaced||used[i]) continue;
        Item it; it.blocks={i}; it.offs={{0,0}}; finalize_item(it); set_item_anchor(it);
        items.push_back(it);
    }'''),
]


def main():
    got = hashlib.md5(SRC.read_bytes()).hexdigest()
    if got != EXPECT_SRC_MD5:
        print("!! constructive.cpp is {} not {}".format(got, EXPECT_SRC_MD5))
        return 1
    out = SRC.read_bytes().decode("utf-8")
    for i, (old, new) in enumerate(PATCHES, 1):
        if out.count(old) != 1:
            print("!! patch {} matches {} times, expected 1".format(i, out.count(old)))
            return 1
        out = out.replace(old, new)
    DST.write_bytes(out.encode("utf-8"))
    print("wrote {}  ({} patches)".format(DST.name, len(PATCHES)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
