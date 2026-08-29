"""L281: measure the handoff's LITERAL move, so deviating from it is evidenced.

HANDOFF S1.4 step 2 says: for unit u and target ordinal p, force "u before v for
all v now after it, and after v for all v now before it".  Read literally on a
1-D ordering that is one axis, so u ends up horizontally separated from EVERY
other unit -- a full-height column.  This runs the same certificate on that move
and on the relocation move, on identical geometry, so the choice in S2 of the
report rests on a number instead of an argument.

No cache writes.
"""
import json
import sys
from pathlib import Path

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR))
import l281_reloc_probe as L                                       # noqa: E402

cases = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1
                          else "85,88,91").split(",")]
per = int(sys.argv[2]) if len(sys.argv) > 2 else 8

aj = json.loads(open(_DIR / "results_L274_base_48c.json", "rb").read())
ANCH = {t["test_id"]: t for t in aj["test_results"]}
L.ANCH = ANCH

tot = dict(coherent=0, cyclic=0, oversized=0, total=0)
for ci in cases:
    P0 = [tuple(p) for p in ANCH[ci]["positions"]]
    ranked, G = L.rank_units(ci, P0)
    box, keys, bb = G["box"], G["keys"], G["bb"]
    unit_of, ukey = G["unit_of"], G["ukey"]
    # the 1-D ordering the literal reading needs: units by current x
    order = sorted([k for k in keys], key=lambda k: (box[k][0], k))
    sub = dict(coherent=0, cyclic=0, oversized=0, total=0)
    done = 0
    for rec in ranked:
        if done >= per or rec["pinned"]:
            continue
        ku = rec["ku"]
        cur = order.index(ku)
        EHb, EVb = L.base_graph(ci, P0, unit_of, ukey, ku)
        for p in (0, len(order) // 4, len(order) // 2,
                  3 * len(order) // 4, len(order) - 1):
            if p == cur:
                continue
            seq = [k for k in order if k != ku]
            seq.insert(min(p, len(seq)), ku)
            pos_of = {k: i for i, k in enumerate(seq)}
            fr = {}
            for kv in keys:
                if kv == ku:
                    continue
                k = 0 if pos_of[ku] < pos_of[kv] else 1   # u left of v / v left
                fr[(ku, kv) if ku <= kv else (kv, ku)] = (
                    k if ku <= kv else L.MIRROR[k])
            EHu, EVu = L.unit_edges(ci, P0, unit_of, ukey, ku, fr)
            cert = L.certificate(ci, P0, EHb, EVb, EHu, EVu, bb)
            sub["total"] += 1
            sub["coherent" if cert["ok"] else
                ("cyclic" if cert["why"] == "cyclic" else "oversized")] += 1
        done += 1
    for k in tot:
        tot[k] += sub[k]
    print(f"case {ci:3d}: literal-ordinal moves {sub['total']:4d}  "
          f"coherent {sub['coherent']:3d} "
          f"({100.0 * sub['coherent'] / max(sub['total'], 1):.1f} %)  "
          f"cyclic {sub['cyclic']:3d}  oversized {sub['oversized']:3d}",
          flush=True)

print(f"\n== literal 1-D ordinal reading, {tot['total']} moves ==")
print(f"   coherent {tot['coherent']} "
      f"({100.0 * tot['coherent'] / max(tot['total'], 1):.1f} %)   "
      f"cyclic {tot['cyclic']}   oversized {tot['oversized']}")
print("   (relocation, same geometry: 16.6 % coherent, 0.1 % cyclic)")
