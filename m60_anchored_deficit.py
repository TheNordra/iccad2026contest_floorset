"""M60 diagnostic probe (OFFLINE, never shipped): anchored first-pass wall-capacity.

M57_PLAN.md section 5 (P3). Hypothesis: the anchored greedy (constructive.cpp
pack_in_frame first-pass) eats the wall segment a LATER boundary member needs --
boundary_penalty_est only sees the current block, and the member order is
direction-insensitive (boundary-bit count + area). The ledger's "ordering closed"
entry (M26 oracle-perm) does NOT cover this internal ordering (plan section 0).

Pipeline:
  Phase A (cache-only, zero exe calls): for all 100 cases, winner host ksel via
    the M59 winner_host mirror (top-32 pre-LP proxy + post-LP proxy re-selection,
    asserted against the anchor json); runP = m53_l3_cache ("run", ci, ksel);
    official-bbox-semantics boundary touch test (EPS 1e-6) restricted to movable
    members of MIXED clusters (>=1 preplaced AND >=1 movable, the C++ anchored
    split at constructive.cpp:1360-1366). Cases with such violators = targets.
  Phase B (per target): byte-gate constructive_m60.exe (trace off) == cache run;
    trace run ICCAD_ANCHOR_TRACE=<file> with stdout still bit-identical; parse
    PACK/PRE/MEM/CAND/WIN records; keep the WIN pack (last WIN line).
  Phase C (per violator, WIN pack): replay commits in order (state = PRE rects +
    prior found=1 commits, sequential across clusters). Wall-strip math per
    pack-time-missing boundary bit: strip depth = violator dim perpendicular to
    the wall, needed = dim along the wall; occupied = wall-axis projections of
    state rects intersecting the strip; deficit = needed - max_free_gap.
    Classifications:
      self-fork    -- a bp=0 candidate existed at V's own commit but lost on
                      score (BP_W=30000 vs raw bbox-area term => possible);
      generator-miss -- wall had a big-enough free gap at V's commit but no
                      candidate was proposed inside it (candidate-grid blind
                      spot, NOT a greedy fork -- beam would not fix it);
      preplaced-blocked -- deficit > 0 already with PRE rects alone (structural,
                      no first-pass fork can help);
      upstream STRONG fork -- an earlier commit's alternative candidate keeps
                      deficit <= 0 while the selected one pushes it > 0;
      upstream WEAK fork -- alternative leaves a strictly larger max free gap.
  Kill gate (plan section 5, 10-minute version): zero forks (self-forks +
    upstream STRONG) across all violators -> immediate RED, no beam. Any ->
    list only (m60_forks.json); beam B=4 lexicographic is stage 2, NOT here.

  SECONDARY scan (informational, added after Phase A found only 2 primary
  violators, both feather-weight): for hard cases' NON-anchored movable
  violators (placed later in the items loop, commit not traced), ask whether
  the anchored first-pass ALONE already destroyed their wall: deficit with
  PRE rects only vs deficit after all first-pass commits. Implicated ->
  same local fork scan over first-pass commits. 'wall-open-after-fp' means
  the items loop ate the wall, i.e. NOT this axis. Does not move the spec
  kill gate; gives A6 the full picture of where wall capacity actually dies.

Honest scope: fork check swaps ONE commit locally (state frozen at t', no
downstream cascade re-simulation); V keeps its selected dims (no joint reshape);
corner (two-bit) feasibility judged per-bit independently. This is a pure
quality/structure diagnostic -- no runtime measurement, no new layouts scored.

Skeleton (dataset load / winner_host / run_exe / cache) copied from
m59_refine_seed_probe.py per the global no-import-and-patch rule.
Caches: m60_cache.pkl (own); m53_l3_cache.pkl READ-ONLY.
"""
import argparse
import hashlib
import json
import math
import os
import pickle
import sys
import subprocess
import time
from pathlib import Path

import numpy as np

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest"))
sys.path.insert(0, str(_DIR))

os.environ["ICCAD_L1_POOL"] = "1"        # BEFORE oc import (84-profile L1 pool)
os.environ["ICCAD_ADAPTIVE_POOL"] = "0"

from iccad2026_evaluate import ContestEvaluator  # noqa: E402
import optimizer_constructive as oc  # noqa: E402
from optimizer_claude import _serialize_input, _parse_output  # noqa: E402
from proxy_analysis import build_opt_target_pos  # noqa: E402

EPS_BND = 1e-6     # official boundary tolerance (mirrors m59/touch_ok)
CPP_TOL = 1e-6     # constructive.cpp TOL (boundary_penalty_est / overlap)
GTOL = 1e-9        # strict-improvement tolerance for gap comparisons
RH = 1.4
HARD = (89, 85, 62, 65, 88, 97, 82, 52, 61, 79, 66, 91)
ANCHOR_JSON = _DIR / "results_L3_port_top32_area.json"
L3_CACHE = _DIR / "m53_l3_cache.pkl"          # READ-ONLY
M60_CACHE = _DIR / "m60_cache.pkl"
EXE60 = str(_DIR / "constructive_m60.exe")
EXE_SHIPPED = str(_DIR / "constructive.exe")
SCRATCH = Path(os.environ.get(
    "M60_SCRATCH",
    r"C:\Users\Nordra\AppData\Local\Temp\claude"
    r"\C--Users-Nordra-Downloads-ICCAD2026-FloorSet-FloorSet"
    r"\22e65eb1-b7fb-48f3-a922-4cb045baf288\scratchpad"))

PROFILES = list(oc._PROFILES)
assert len(PROFILES) == 84, f"L1 pool expected 84 profiles, got {len(PROFILES)}"

BIT_NAMES = {1: "L", 2: "R", 4: "T", 8: "B"}

# ── dataset (copied from m59_refine_seed_probe.py) ───────────────────────────
print("[load] dataset ...", flush=True)
_ev = ContestEvaluator(data_path=str(_DIR), verbose=False)
_ev._load_dataset()
CASES, W = {}, {}
for _idx in range(100):
    _s = _ev.dataset[_idx]
    _inp, _lab = _s["input"], _s["label"]
    _at, _b2b, _p2b, _pins, _cons = _inp
    _n = int((_at != -1).sum().item())
    W[_idx] = math.exp(_n / 12.0)
    _base, _tp = _ev._extract_baseline(_idx, _lab, _b2b, _p2b, _pins, _n)
    CASES[_idx] = dict(
        idx=_idx, n=_n, base=_base, tp=_tp, at=_at, b2b=_b2b, p2b=_p2b,
        pins=_pins, cons=_cons,
        cn=[[int(v) for v in _cons[i].tolist()] for i in range(_n)],
    )
TOTW = sum(W.values())
print("[load] 100 cases ready", flush=True)


def winner_host(ci, db, anch_cost):
    """Mirror mode_portfull's selection (copied from m59_refine_seed_probe.py):
    top-32 by pre-LP proxy, post-LP proxy re-selection -> ksel."""
    c = CASES[ci]
    A_hat = 1.035 * max(sum(max(0.0, float(c["at"][i]))
                            for i in range(c["n"])), 1e-9)
    pm = {k: db[("pm", ci, k)] for k in range(84)}
    hmin = min(v[1] for v in pm.values()) or 1.0
    prox = {k: (pm[k][0] / A_hat + RH * pm[k][1] / hmin)
            * math.exp(2 * pm[k][2]) for k in pm}
    top = sorted(pm, key=lambda k: prox[k])[:32]
    res = {k: db[("lp", ci, k, 8, True)] for k in top}
    h2 = {k: res[k][1] for k in top}
    hmin2 = min(h2.values()) or 1.0
    ab = float(c["base"].get("area_baseline", 1.0))
    prox2 = {k: ((1 + res[k][2]) * ab / A_hat + RH * h2[k] / hmin2)
             * math.exp(2 * res[k][3]) for k in top}
    ksel = min(top, key=lambda k: prox2[k])
    assert res[ksel][0] == anch_cost, (
        f"case {ci}: cache ksel={ksel} lp cost {res[ksel][0]!r} != anchor "
        f"{anch_cost!r} (data drift)")
    return ksel


def run_exe(ci, k, trace_path=None):
    c = CASES[ci]
    otp = build_opt_target_pos(c["tp"], c["cons"], c["n"])
    txt = _serialize_input(c["n"], c["at"], c["b2b"], c["p2b"], c["pins"],
                           c["cons"], otp, gnn_hint=None)
    env = dict(os.environ)
    env.update(PROFILES[k])
    if trace_path is not None:
        env["ICCAD_ANCHOR_TRACE"] = str(trace_path)
    r = subprocess.run([EXE60], input=txt, capture_output=True, text=True,
                       env=env, timeout=600)
    return _parse_output(r.stdout, c["n"])


# ── Phase A: violator prescan (cache only) ───────────────────────────────────
def mixed_clusters(ci):
    """gid -> (preplaced members, movable members) for clusters with both
    (the C++ anchored split: is_preplaced == cons col 1)."""
    c = CASES[ci]
    out = {}
    for g in sorted({c["cn"][i][3] for i in range(c["n"]) if c["cn"][i][3] > 0}):
        mem = [i for i in range(c["n"]) if c["cn"][i][3] == g]
        pre = [i for i in mem if c["cn"][i][1] != 0]
        mov = [i for i in mem if c["cn"][i][1] == 0]
        if pre and mov:
            out[g] = (pre, mov)
    return out


def miss_bits(code, x, y, w, h, x0, x1, y0, y1, eps):
    """Boundary bits of `code` NOT satisfied vs box [x0,x1]x[y0,y1]."""
    m = 0
    if code & 1 and abs(x - x0) >= eps:
        m |= 1
    if code & 2 and abs(x + w - x1) >= eps:
        m |= 2
    if code & 4 and abs(y + h - y1) >= eps:
        m |= 4
    if code & 8 and abs(y - y0) >= eps:
        m |= 8
    return m


def movable_violators(ci, P):
    """ALL movable blocks failing the official (bbox-recomputed) boundary touch
    test on layout P -> [(block, code, missing bits, class)] where class is
    'anchored' (mixed-cluster member: the spec's primary scope), 'cluster'
    (pure-movable-cluster member) or 'single'."""
    c = CASES[ci]
    n = c["n"]
    x0 = min(P[i][0] for i in range(n))
    x1 = max(P[i][0] + P[i][2] for i in range(n))
    y0 = min(P[i][1] for i in range(n))
    y1 = max(P[i][1] + P[i][3] for i in range(n))
    mixed = mixed_clusters(ci)
    out = []
    for i in range(n):
        code = c["cn"][i][4]
        if code == 0 or c["cn"][i][1] != 0:
            continue
        mb = miss_bits(code, *P[i], x0, x1, y0, y1, EPS_BND)
        if not mb:
            continue
        g = c["cn"][i][3]
        vc = ("anchored" if g in mixed else
              "cluster" if g > 0 else "single")
        out.append((i, code, mb, vc))
    return out


# ── trace parsing ─────────────────────────────────────────────────────────────
def parse_trace(path):
    """-> (packs: seq -> dict(fw, fh, pre=[(b,rect)], mems=[...]), win_seq)."""
    packs, win, cur = {}, None, None
    with open(path) as fh:
        for line in fh:
            tok = line.split()
            if tok[0] == "PACK":
                kv = dict(t.split("=") for t in tok[1:])
                cur = dict(fw=float(kv["fw"]), fh=float(kv["fh"]),
                           pre=[], mems=[])
                packs[int(kv["seq"])] = cur
            elif tok[0] == "PRE":
                b = int(tok[1].split("=")[1])
                cur["pre"].append((b, tuple(float(v) for v in tok[2:6])))
            elif tok[0] == "MEM":
                kv = dict(t.split("=") for t in tok[1:])
                cur["mems"].append(dict(
                    cl=int(kv["cl"]), b=int(kv["b"]), code=int(kv["code"]),
                    found=int(kv["found"]), cands=[], sel=-1))
            elif tok[0] == "CAND":
                m = cur["mems"][-1]
                if int(tok[7].split("=")[1]):
                    m["sel"] = len(m["cands"])
                m["cands"].append((float(tok[1]), float(tok[2]),
                                   float(tok[3]), float(tok[4]),
                                   float(tok[5]), int(tok[6])))
            elif tok[0] == "WIN":
                win = int(tok[1].split("=")[1])
    return packs, win


# ── wall-strip interval math ─────────────────────────────────────────────────
def wall_blocked(bit, rects, fw, fh, depth):
    """Wall-axis intervals blocked by rects intersecting the wall strip."""
    iv = []
    for (x, y, w, h) in rects:
        if bit == 1:                              # LEFT: strip x in [0, depth)
            if x < depth - GTOL and x + w > GTOL:
                iv.append((y, y + h))
        elif bit == 2:                            # RIGHT: strip x in (fw-depth, fw]
            if x + w > fw - depth + GTOL and x < fw - GTOL:
                iv.append((y, y + h))
        elif bit == 4:                            # TOP: strip y in (fh-depth, fh]
            if y + h > fh - depth + GTOL and y < fh - GTOL:
                iv.append((x, x + w))
        else:                                     # 8 BOTTOM: strip y in [0, depth)
            if y < depth - GTOL and y + h > GTOL:
                iv.append((x, x + w))
    return iv


def gap_stats(blocked, L):
    """(max free gap, total free length) on [0, L] minus merged intervals."""
    iv = sorted((max(0.0, a), min(L, b)) for a, b in blocked if b > 0 and a < L)
    mx, tot, cur = 0.0, 0.0, 0.0
    for a, b in iv:
        if a > cur:
            mx = max(mx, a - cur)
            tot += a - cur
        cur = max(cur, b)
    if L > cur:
        mx = max(mx, L - cur)
        tot += L - cur
    return mx, tot


def need_depth(bit, w, h):
    """(needed wall length, strip depth) for a w x h block on wall `bit`."""
    return (h, w) if bit in (1, 2) else (w, h)


# ── Phase C: per-violator deficit replay + fork scan ─────────────────────────
def analyze_violator(ci, pack, v, final_missbits):
    """-> dict(class, per-bit stats, forks) for a PRIMARY (anchored-member)
    violator v=(b, code, mb, vc): full replay at its own first-pass commit."""
    b, code, _, _ = v
    fw, fh = pack["fw"], pack["fh"]
    mems = pack["mems"]
    tV = next((t for t, m in enumerate(mems) if m["b"] == b), None)
    if tV is None:
        return dict(b=b, code=code, cls="not-first-pass")
    mV = mems[tV]
    if not mV["found"]:
        return dict(b=b, code=code, cls="no-slot-fallback",
                    nc=len(mV["cands"]))
    sx, sy, sw, sh, ssc, sbp = mV["cands"][mV["sel"]]
    if sbp == 0:
        return dict(b=b, code=code, cls="pack-time-satisfied",
                    final_missbits=final_missbits)
    pt_miss = miss_bits(code, sx, sy, sw, sh, 0.0, fw, 0.0, fh, CPP_TOL)
    has_bp0 = any(c[5] == 0 for c in mV["cands"])
    self_fork = None
    if has_bp0:
        alt = min((c for c in mV["cands"] if c[5] == 0), key=lambda c: c[4])
        self_fork = dict(alt_xywh=alt[:4], alt_score=alt[4], sel_score=ssc,
                         score_gap=alt[4] - ssc)

    # state replay: PRE + prior found=1 selected rects, in commit order
    state = [r for _, r in pack["pre"]]
    commits = []                           # (mem idx, rect) actually committed
    for t in range(tV):
        m = mems[t]
        if m["found"]:
            r = m["cands"][m["sel"]][:4]
            commits.append((t, r))
            state.append(r)

    bits = []
    forks = []
    for bit in (1, 2, 4, 8):
        if not pt_miss & bit:
            continue
        needed, depth = need_depth(bit, sw, sh)
        L = fh if bit in (1, 2) else fw
        pre_rects = [r for _, r in pack["pre"]]
        g_pre, _ = gap_stats(wall_blocked(bit, pre_rects, fw, fh, depth), L)
        g_now, tot_now = gap_stats(wall_blocked(bit, state, fw, fh, depth), L)
        bits.append(dict(bit=BIT_NAMES[bit], needed=needed, depth=depth,
                         gap_pre=g_pre, gap_at_commit=g_now,
                         free_total_at_commit=tot_now,
                         deficit_pre=needed - g_pre,
                         deficit_at_commit=needed - g_now))
        # upstream fork scan (local single-commit swap, state frozen at t')
        run = [r for _, r in pack["pre"]]
        for t, r_sel in commits:
            g_before, _ = gap_stats(wall_blocked(bit, run, fw, fh, depth), L)
            g_after, _ = gap_stats(
                wall_blocked(bit, run + [r_sel], fw, fh, depth), L)
            if g_after < g_before - GTOL:      # this commit ate the wall
                m = mems[t]
                best = None
                for q, cand in enumerate(m["cands"]):
                    if q == m["sel"]:
                        continue
                    g_alt, _ = gap_stats(
                        wall_blocked(bit, run + [cand[:4]], fw, fh, depth), L)
                    if best is None or g_alt > best[0]:
                        best = (g_alt, q, cand)
                if best is not None and best[0] > g_after + GTOL:
                    g_alt, q, cand = best
                    forks.append(dict(
                        bit=BIT_NAMES[bit], t=t, member=m["b"],
                        strong=bool(g_alt >= needed - GTOL
                                    and g_after < needed - GTOL),
                        gap_sel=g_after, gap_alt=g_alt, needed=needed,
                        sel_xywh=r_sel, alt_xywh=cand[:4],
                        sel_score=m["cands"][m["sel"]][4], alt_score=cand[4]))
            run.append(r_sel)

    if bits and all(d["deficit_pre"] > GTOL for d in bits) and not has_bp0:
        cls = "preplaced-blocked"
    elif has_bp0:
        cls = "self-fork"
    elif bits and any(d["deficit_at_commit"] <= GTOL for d in bits):
        cls = "generator-miss"
    else:
        cls = "wall-eaten"
    return dict(b=b, code=code, cls=cls, sel_xywh=(sx, sy, sw, sh),
                sel_bp=sbp, pt_missbits=pt_miss, final_missbits=final_missbits,
                has_bp0=has_bp0, self_fork=self_fork, bits=bits, forks=forks)


def analyze_secondary(ci, pack, v, P):
    """SECONDARY (informational, non-anchored movable violator v placed later
    in the items loop, so its own commit is not in the trace): did the anchored
    FIRST-PASS alone already destroy its wall?  Assessed with the violator's
    final dims against the frame walls: deficit with PRE rects only vs deficit
    after all first-pass commits.  Implicated (open -> eaten) => same local
    single-swap fork scan over the first-pass commits."""
    b, code, mb, vc = v
    fw, fh = pack["fw"], pack["fh"]
    _, _, w, h = P[b]
    pre_rects = [r for _, r in pack["pre"]]
    fp_rects = [m["cands"][m["sel"]][:4] for m in pack["mems"] if m["found"]]
    bits, forks, implicated = [], [], False
    for bit in (1, 2, 4, 8):
        if not mb & bit:
            continue
        needed, depth = need_depth(bit, w, h)
        L = fh if bit in (1, 2) else fw
        g_pre, _ = gap_stats(wall_blocked(bit, pre_rects, fw, fh, depth), L)
        g_fp, tot_fp = gap_stats(
            wall_blocked(bit, pre_rects + fp_rects, fw, fh, depth), L)
        imp = g_pre >= needed - GTOL and g_fp < needed - GTOL
        implicated |= imp
        bits.append(dict(bit=BIT_NAMES[bit], needed=needed, depth=depth,
                         gap_pre=g_pre, gap_post_fp=g_fp,
                         free_total_post_fp=tot_fp,
                         deficit_pre=needed - g_pre,
                         deficit_post_fp=needed - g_fp, implicated=imp))
        if not imp:
            continue
        run = list(pre_rects)
        for t, m in enumerate(pack["mems"]):
            if not m["found"]:
                continue
            r_sel = m["cands"][m["sel"]][:4]
            g_before, _ = gap_stats(wall_blocked(bit, run, fw, fh, depth), L)
            g_after, _ = gap_stats(
                wall_blocked(bit, run + [r_sel], fw, fh, depth), L)
            if g_after < g_before - GTOL:
                best = None
                for q, cand in enumerate(m["cands"]):
                    if q == m["sel"]:
                        continue
                    g_alt, _ = gap_stats(
                        wall_blocked(bit, run + [cand[:4]], fw, fh, depth), L)
                    if best is None or g_alt > best[0]:
                        best = (g_alt, q, cand)
                if best is not None and best[0] > g_after + GTOL:
                    g_alt, q, cand = best
                    forks.append(dict(
                        bit=BIT_NAMES[bit], t=t, member=m["b"],
                        strong=bool(g_alt >= needed - GTOL
                                    and g_after < needed - GTOL),
                        gap_sel=g_after, gap_alt=g_alt, needed=needed,
                        sel_xywh=r_sel, alt_xywh=cand[:4],
                        sel_score=m["cands"][m["sel"]][4],
                        alt_score=cand[4]))
            run.append(r_sel)
    if not bits:
        cls2 = "no-bits"
    elif implicated:
        cls2 = "fp-implicated"
    elif all(d["deficit_pre"] > GTOL for d in bits):
        cls2 = "preplaced-blocked"
    elif all(d["deficit_post_fp"] <= GTOL for d in bits):
        cls2 = "wall-open-after-fp"       # items loop ate it -> not M60's axis
    else:
        cls2 = "mixed"
    return dict(b=b, code=code, vclass=vc, cls=cls2, bits=bits, forks=forks)


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=None,
                    help="comma list override (default: all prescan hits)")
    ap.add_argument("--max", type=int, default=25,
                    help="max cases to trace (hard list first, then weight)")
    ap.add_argument("--prescan-only", action="store_true")
    args = ap.parse_args()

    aj = json.load(open(ANCHOR_JSON))
    ANCH = {t["test_id"]: t for t in aj["test_results"]}
    print(f"[anchor] {ANCHOR_JSON.name} total={aj['total_score']:.10f}",
          flush=True)

    l3 = pickle.load(open(L3_CACHE, "rb"))          # READ-ONLY
    sig_expect = repr((repr(PROFILES),
                       hashlib.md5(open(EXE_SHIPPED, "rb").read()).hexdigest()))
    assert l3.get("sig") == sig_expect, \
        "m53_l3_cache signature != current pool/exe -> data drift, abort"
    db = l3["db"]

    msig = repr((repr(PROFILES),
                 hashlib.md5(open(EXE60, "rb").read()).hexdigest()))
    m60 = {}
    if M60_CACHE.exists():
        try:
            _c = pickle.load(open(M60_CACHE, "rb"))
            if _c.get("sig") == msig:
                m60 = _c["db"]
            else:
                print("[cache] m60 signature mismatch -> reset", flush=True)
        except Exception:
            print("[cache] m60 unreadable -> reset", flush=True)

    def save():
        tmp = M60_CACHE.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump({"sig": msig, "db": m60}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, M60_CACHE)

    # Phase A ── prescan all 100 cases from cache
    print("\n== Phase A: movable-violator prescan (cache only) ==", flush=True)
    pres = {}
    for ci in range(100):
        ksel = winner_host(ci, db, ANCH[ci]["cost"])
        runP = [tuple(p) for p in db[("run", ci, ksel)]]
        mv = movable_violators(ci, runP)
        av = [v for v in mv if v[3] == "anchored"]
        pres[ci] = dict(ksel=ksel, runP=runP, mv=mv, av=av,
                        nmix=len(mixed_clusters(ci)))
        if av:
            print(f"  case {ci:3d} n={CASES[ci]['n']:3d} k={ksel:2d} "
                  f"mixed={pres[ci]['nmix']} PRIMARY anchored-violators="
                  f"{[(b, BIT_NAMES.get(mb, mb)) for b, _, mb, _ in av]}",
                  flush=True)
    hits = [ci for ci in range(100) if pres[ci]["av"]]
    nv = sum(len(pres[ci]["av"]) for ci in hits)
    nmv = sum(len(pres[ci]["mv"]) for ci in range(100))
    print(f"[prescan] PRIMARY: {len(hits)} cases / {nv} anchored-cluster "
          f"violators; all movable violators: {nmv} over "
          f"{sum(1 for ci in range(100) if pres[ci]['mv'])} cases", flush=True)
    if args.prescan_only:
        return

    if args.cases:
        targets = [int(x) for x in args.cases.split(",")]
    else:
        # primary-scope cases first, then hard cases with ANY movable violator
        # (secondary informational scan: is the anchored first-pass the wall
        # eater for their violators?), capped by --max
        sec = [ci for ci in HARD if pres[ci]["mv"] and ci not in hits]
        targets = (sorted(hits, key=lambda ci: -W[ci])
                   + sorted(sec, key=lambda ci: -W[ci]))
        targets = targets[:args.max]
    print(f"[targets] {targets}", flush=True)

    SCRATCH.mkdir(parents=True, exist_ok=True)
    report_cases = []
    t0 = time.perf_counter()
    for ci in targets:
        ksel, runP, av = pres[ci]["ksel"], pres[ci]["runP"], pres[ci]["av"]
        # byte-gate: trace-off run reproduces the cached shipped positions
        gk = ("gate", ci, ksel)
        if gk not in m60:
            p_off = [tuple(p) for p in run_exe(ci, ksel)]
            assert p_off == runP, f"case {ci}: m60 exe (trace off) != cache run"
            m60[gk] = True
            save()
        trace_path = SCRATCH / f"m60_trace_{ci}.txt"
        tk = ("traceok", ci, ksel)
        if tk not in m60 or not trace_path.exists():
            p_on = [tuple(p) for p in run_exe(ci, ksel, trace_path=trace_path)]
            assert p_on == runP, f"case {ci}: trace run perturbed stdout"
            m60[tk] = True
            save()
        packs, win = parse_trace(trace_path)
        assert win is not None and win in packs, \
            f"case {ci}: WIN seq {win} missing from trace"
        pack = packs[win]
        # replay consistency self-check: committed rects must not overlap
        chk = [r for _, r in pack["pre"]]
        for m in pack["mems"]:
            if not m["found"]:
                continue
            x, y, w, h = m["cands"][m["sel"]][:4]
            for (X, Y, Ww, Hh) in chk:
                assert (x + w <= X + CPP_TOL or X + Ww <= x + CPP_TOL
                        or y + h <= Y + CPP_TOL or Y + Hh <= y + CPP_TOL), \
                    f"case {ci}: replay overlap at member {m['b']}"
            chk.append((x, y, w, h))

        av = pres[ci]["av"]
        vios = [analyze_violator(ci, pack, v, v[2]) for v in av]
        sec = [analyze_secondary(ci, pack, v, runP)
               for v in pres[ci]["mv"] if v[3] != "anchored"]
        nstrong = sum(sum(f["strong"] for f in v.get("forks", []))
                      for v in vios)
        nweak = sum(sum(not f["strong"] for f in v.get("forks", []))
                    for v in vios)
        nself = sum(1 for v in vios if v.get("self_fork"))
        s_strong = sum(sum(f["strong"] for f in v["forks"]) for v in sec)
        s_imp = sum(1 for v in sec if v["cls"] == "fp-implicated")
        print(f"case {ci:3d} n={CASES[ci]['n']:3d} k={ksel:2d} win_seq={win} "
              f"packs={len(packs)} fp_mems={len(pack['mems'])} | "
              f"PRIMARY viol={len(vios)} cls={[v['cls'] for v in vios]} "
              f"self={nself} strong={nstrong} weak={nweak} | "
              f"SECONDARY viol={len(sec)} "
              f"cls={sorted(set(v['cls'] for v in sec))} "
              f"fp-implicated={s_imp} strong={s_strong} "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)
        report_cases.append(dict(ci=ci, n=CASES[ci]["n"], ksel=ksel,
                                 win_seq=win, npacks=len(packs),
                                 fp_mems=len(pack["mems"]),
                                 primary=vios, secondary=sec))

    tot_self = sum(1 for rc in report_cases for v in rc["primary"]
                   if v.get("self_fork"))
    tot_strong = sum(f["strong"] for rc in report_cases
                     for v in rc["primary"] for f in v.get("forks", []))
    tot_weak = sum((not f["strong"]) for rc in report_cases
                   for v in rc["primary"] for f in v.get("forks", []))
    sec_strong = sum(f["strong"] for rc in report_cases
                     for v in rc["secondary"] for f in v["forks"])
    sec_imp = sum(1 for rc in report_cases
                  for v in rc["secondary"] if v["cls"] == "fp-implicated")
    verdict = "RED" if (tot_self + tot_strong) == 0 else "FORKS-FOUND"
    print(f"\n== M60 PRIMARY verdict (spec kill gate): {verdict}  "
          f"(self-forks={tot_self}, upstream STRONG={tot_strong}, "
          f"WEAK={tot_weak}) ==", flush=True)
    print(f"== SECONDARY (informational): fp-implicated violators={sec_imp}, "
          f"STRONG forks={sec_strong}, {len(report_cases)} cases traced ==",
          flush=True)

    out = dict(
        verdict=verdict, primary_self_forks=tot_self,
        primary_strong_forks=tot_strong, primary_weak_forks=tot_weak,
        secondary_fp_implicated=sec_imp, secondary_strong_forks=sec_strong,
        prescan=dict(cases_with_anchored_violators=hits,
                     total_anchored_violators=nv,
                     total_movable_violators=nmv,
                     per_case={str(ci): dict(
                         ksel=pres[ci]["ksel"], nmix=pres[ci]["nmix"],
                         violators=[dict(b=b, code=c, missbits=mb, cls=vc)
                                    for b, c, mb, vc in pres[ci]["mv"]])
                         for ci in range(100) if pres[ci]["mv"]}),
        traced=report_cases,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(_DIR / "m60_forks.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"[dump] m60_forks.json", flush=True)


if __name__ == "__main__":
    main()
