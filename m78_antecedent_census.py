"""M78 Step 0 — antecedent census for the candidate-set axis.  NEVER SHIPPED.

Question this answers, BEFORE any C++ is written:

    M71 enriched the candidate set + sort key of ONE of the two cluster paths
    (pure-movable -> make_group_item) and bought -1.589% in-set / -4.04% OOS.
    The OTHER path (mixed = preplaced + movable -> the anchored first-pass in
    pack_in_frame:792-856 + adjacent_candidates_for_block) still carries the
    pre-M71 candidate set.  Is there enough of it to be worth a probe?

The ledger's own lesson ([[m57-codex-gap-plan]], [[m60-anchored-wall-red]]):
a line-level fact being true is not the same as it being worth points.  M60
died on an EMPTY antecedent (2 violators).  So: count the antecedent first,
weighted by exp(n/12), and calibrate it against the M71 path -- the one axis
we know the exchange rate for.

Reads the dataset only; runs no placer.  Usage:

    python m78_antecedent_census.py [inset|s1|s2|all]
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest"))
sys.path.insert(0, str(_DIR))

# constraint columns, in _serialize_input()'s order (optimizer_claude.py:561-568)
C_FIXED, C_PREPLACED, C_MIB, C_CLUSTER, C_BOUNDARY = 0, 1, 2, 3, 4


# --------------------------------------------------------------------------- #
# per-case census                                                              #
# --------------------------------------------------------------------------- #
def census_case(n, cons):
    """Mirror solve():1610-1632's item construction and bucket every block.

    Returns a dict of counts.  The three buckets are exhaustive and disjoint
    over the n blocks, which is the check that keeps this honest.
    """
    pre = [int(cons[i][C_PREPLACED]) != 0 for i in range(n)]
    cl = [int(cons[i][C_CLUSTER]) for i in range(n)]
    bnd = [int(cons[i][C_BOUNDARY]) for i in range(n)]
    mib = [int(cons[i][C_MIB]) for i in range(n)]

    groups = defaultdict(list)
    for i in range(n):
        if cl[i] > 0:
            groups[cl[i]].append(i)

    r = dict(n=n,
             n_pre=sum(pre),
             n_cluster_blocks=sum(len(v) for v in groups.values()),
             n_groups=len(groups),
             # M71 path: pure-movable cluster with >=2 movable -> compound item
             m71_groups=0, m71_mov=0, m71_mov_bnd=0,
             # M78 A1 path: mixed cluster -> anchored first-pass
             anch_groups=0, anch_mov=0, anch_mov_bnd=0, anch_pre=0,
             anch_mov_mib=0,
             # clusters that fall through to singles (all-preplaced, or 1 movable)
             fall_groups=0, fall_mov=0)

    used = [False] * n
    for _g, members in groups.items():
        mov = [b for b in members if not pre[b]]
        pp = [b for b in members if pre[b]]
        if pp and mov:
            r["anch_groups"] += 1
            r["anch_mov"] += len(mov)
            r["anch_pre"] += len(pp)
            r["anch_mov_bnd"] += sum(1 for b in mov if bnd[b] != 0)
            r["anch_mov_mib"] += sum(1 for b in mov if mib[b] != 0)
            # NOTE: anchored members are NOT marked used -- pack_in_frame's
            # first-pass places them, but they also appear as singles as a
            # fallback (solve():1607-1609).  They belong to the A1 antecedent.
            continue
        if len(mov) < 2:
            r["fall_groups"] += 1
            r["fall_mov"] += len(mov)
            continue
        r["m71_groups"] += 1
        r["m71_mov"] += len(mov)
        r["m71_mov_bnd"] += sum(1 for b in mov if bnd[b] != 0)
        for b in mov:
            used[b] = True

    # generic item_candidates() path: every movable block not inside a compound
    # item.  Anchored members are counted here too (they fall back to singles),
    # which is why the buckets below overlap by exactly anch_mov.
    r["singles"] = sum(1 for i in range(n) if not pre[i] and not used[i])
    return r


# --------------------------------------------------------------------------- #
# corpora                                                                      #
# --------------------------------------------------------------------------- #
def iter_inset():
    from iccad2026_evaluate import ContestEvaluator
    ev = ContestEvaluator(data_path=str(_DIR), verbose=False)
    ev._load_dataset()
    for idx in range(len(ev.dataset)):
        at, _b2b, _p2b, _pins, cons = ev.dataset[idx]["input"]
        n = int((at != -1).sum().item())
        yield f"case{idx}", n, cons[:n]


def iter_manifest(sample):
    path = _DIR / f"m77_oos_manifest_{sample}.json"
    doc = json.load(open(path))
    byfile = defaultdict(list)
    for c in doc["cases"]:
        byfile[c["file"]].append((c["key"], int(c["layout"]), int(c["n"])))
    for f in sorted(byfile):
        d = torch.load(_DIR / f)
        for key, L, n_expect in sorted(byfile[f]):
            at_all = d[0][L][:, 0]
            n = int((at_all != -1).sum().item())
            assert n == n_expect, f"{key}: n {n} != manifest {n_expect}"
            yield key, n, d[0][L][:n, 1:]


# --------------------------------------------------------------------------- #
# aggregation                                                                  #
# --------------------------------------------------------------------------- #
def _pct(num, den):
    return 100.0 * num / den if den else 0.0


def run(corpus, rows):
    W = sum(w for _k, _n, w, _r in rows)
    print(f"\n{'=' * 78}\n{corpus}: {len(rows)} cases   Sigma exp(n/12) = {W:.1f}")

    # weighted share of BLOCKS on each path (the thing a per-block mechanism scales with)
    def wshare(field):
        return _pct(sum(w * r[field] for _k, _n, w, r in rows),
                    sum(w * r["n"] for _k, _n, w, r in rows))

    # weighted share of CASES whose antecedent is non-empty (can the mechanism fire at all)
    def wcases(field):
        return _pct(sum(w for _k, _n, w, r in rows if r[field] > 0), W)

    print(f"\n  {'path':<34} {'wt%% blocks':>11} {'wt%% cases':>11} {'raw cases':>10}")
    for label, field in (("M71  pure-movable cluster members", "m71_mov"),
                         ("M78-A1  anchored (mixed) movable", "anch_mov"),
                         ("   ...of those, boundary!=0", "anch_mov_bnd"),
                         ("   ...of those, mib!=0", "anch_mov_mib"),
                         ("M78-A2  generic singles", "singles"),
                         ("preplaced (immovable)", "n_pre")):
        raw = sum(1 for _k, _n, _w, r in rows if r[field] > 0)
        print(f"  {label:<34} {wshare(field):10.2f}% {wcases(field):10.2f}% "
              f"{raw:>6}/{len(rows)}")

    # Disjointness from M71: a case with an anchored antecedent but NO pure-movable
    # cluster got nothing at all from M71, so any A1 gain there is additive rather
    # than competing with a mechanism that already fired.
    w_a1_only = sum(w for _k, _n, w, r in rows
                    if r["anch_mov"] > 0 and r["m71_mov"] == 0)
    w_both = sum(w for _k, _n, w, r in rows
                 if r["anch_mov"] > 0 and r["m71_mov"] > 0)
    print(f"\n  disjointness from M71 (by weight):")
    print(f"    A1 antecedent, M71 antecedent EMPTY : {_pct(w_a1_only, W):6.2f}%")
    print(f"    A1 antecedent, M71 also present     : {_pct(w_both, W):6.2f}%")

    # where the weight actually sits: heavy band only (n>100 is ~53% of weight)
    heavy = [t for t in rows if t[1] > 100]
    if heavy:
        Wh = sum(w for _k, _n, w, _r in heavy)
        print(f"\n  n>100 band ({len(heavy)} cases, {_pct(Wh, W):.1f}% of weight):")
        for label, field in (("M71 pure-movable", "m71_mov"),
                             ("M78-A1 anchored movable", "anch_mov")):
            sh = _pct(sum(w * r[field] for _k, _n, w, r in heavy),
                      sum(w * r["n"] for _k, _n, w, r in heavy))
            nz = sum(1 for _k, _n, _w, r in heavy if r[field] > 0)
            print(f"    {label:<26} {sh:6.2f}% of blocks   "
                  f"{nz}/{len(heavy)} cases non-empty")

    # top antecedent cases -- these are the ones a probe would move
    print(f"\n  top-12 by weighted anchored-movable mass:")
    print(f"    {'case':<26} {'n':>4} {'anchG':>6} {'anchMov':>8} {'bnd':>4} "
          f"{'m71Mov':>7} {'wt%':>6}")
    for k, n, w, r in sorted(rows, key=lambda t: -t[2] * t[3]["anch_mov"])[:12]:
        print(f"    {k:<26} {n:>4} {r['anch_groups']:>6} {r['anch_mov']:>8} "
              f"{r['anch_mov_bnd']:>4} {r['m71_mov']:>7} {100 * w / W:5.2f}%")
    return dict(W=W, n_cases=len(rows),
                anch_blocks=wshare("anch_mov"), anch_cases=wcases("anch_mov"),
                m71_blocks=wshare("m71_mov"), m71_cases=wcases("m71_mov"))


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    corpora = {"inset": iter_inset}
    for s in ("s1", "s2"):
        corpora[s] = (lambda s=s: iter_manifest(s))
    todo = list(corpora) if which == "all" else [which]

    summary = {}
    for name in todo:
        rows = []
        for key, n, cons in corpora[name]():
            r = census_case(n, cons)
            rows.append((key, n, math.exp(n / 12.0), r))
        summary[name] = run(name, rows)

    print(f"\n{'=' * 78}\nVERDICT (M78 Step 0)")
    print("  bar: the anchored antecedent must be a non-trivial fraction of the\n"
          "       M71 antecedent, otherwise A1 cannot repay M71-scale numbers.\n")
    print(f"  {'corpus':<8} {'A1 blocks':>10} {'A1 cases':>9} "
          f"{'M71 blocks':>11} {'M71 cases':>10} {'A1/M71':>8}")
    for name, s in summary.items():
        ratio = s["anch_blocks"] / s["m71_blocks"] if s["m71_blocks"] else float("inf")
        print(f"  {name:<8} {s['anch_blocks']:9.2f}% {s['anch_cases']:8.2f}% "
              f"{s['m71_blocks']:10.2f}% {s['m71_cases']:9.2f}% {ratio:7.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
