"""OFFLINE ONLY — M52 Phase 0 generative-ML feasibility probe (never shipped).

Measures, with ZERO training, whether the M52 generative track (sample a
B*-tree -> decode -> legalize -> portfolio proxy selection) is viable:

  pipeline (Phase 0-i)   true tree_sol + oracle-Y + true dims -> render_bstar
                         -> compact_down -> legalize_v0 must reproduce fp_sol
                         EXACTLY under the official evaluate_solution with
                         STRICT hard-constraint checking (target_positions).
  slack    (Phase 0-ii)  perturb the three things a model must emit — tree
                         tokens / Y-order / movable dims — and measure the
                         official-cost degradation AFTER legalization
                         (= the model-error tolerance band).
  legalize (Phase 0-iii) legalizer v0 repair rate: pre/post hard feasibility,
                         self-inflation on unperturbed input (must be ~0),
                         and model-vs-legalizer attribution.
  vocab    (extra)       training-set dims statistics -> Phase 1 dims-head
                         vocabulary design.
  all                    all four + the GO/NO-GO verdict box.

Facts this probe is built on (explored 2026-07-10):
  * The validation set has NO tree_sol -> everything here runs on TRAINING
    .th files; the training fp_sol floor is NOT the validation 1.1079.
  * tree_decode_probe._cost_of passes target_positions=None, and the official
    check_dimension_hard_constraints returns 0 in that case -> it never
    checks preplaced/fixed hard constraints. This probe uses _cost_strict.
  * Preplaced positions are HARD constraints (tol 1e-4, violation -> 10.0),
    so slack curves are only meaningful AFTER legalize_v0.
  * Training fp_sol itself carries vrel>0 (boundary+MIB label-inherent) ->
    all tables report R = cost / cost(fp_self) per case (floor-relative).

Run:  C:/Users/Nordra/.conda/envs/iccadv/python.exe m52_phase0_probe.py all quick
      C:/Users/Nordra/.conda/envs/iccadv/python.exe m52_phase0_probe.py all
      modes: pipeline | slack | legalize | vocab | all   (+ optional "quick")
"""
import glob
import math
import os
import pickle
import random
import sys
import time
from collections import Counter
from pathlib import Path

import torch

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest"))
sys.path.insert(0, str(_DIR))

from iccad2026_evaluate import evaluate_solution                    # noqa: E402
from tree_decode_probe import load_layout, render_bstar, compact_down  # noqa: E402
from recon_slice_probe import _baseline                             # noqa: E402

SEED = 52
PROBE_VERSION = 1
CACHE_PATH = _DIR / "m52_phase0_cache.pkl"

BANDS = (("S", 20, 60), ("M", 60, 100), ("B", 100, 130))
QUOTA_PIPE = {"S": 60, "M": 80, "B": 120}
QUOTA_SLACK = {"S": 24, "M": 32, "B": 40}
PER_FILE_CAP = 4

LEVELS = (0.02, 0.05, 0.10, 0.20, 0.40)
REPS = 3
QUANT_KS = (8, 16, 32)
COMBOS = {"mild": (0.02, 0.02, 0.05), "moderate": (0.05, 0.05, 0.10)}
OP_W = (("MOVE_SUB", 0.40), ("MOVE_LEAF", 0.30), ("SWAP", 0.20), ("FLIP", 0.10))
ASPECT_LO, ASPECT_HI = 1.0 / 3.0, 3.0

HEADROOM_ABS = 1.3265 - 1.1079   # local validation headroom, scale reference only
BAR_TREE_Y = 1.10                # wR bar for tree/Y axes at s=0.05
BAR_DIMS = 1.03                  # wR bar for dims_quant K=16
SIG = repr(("m52p0", PROBE_VERSION, SEED, OP_W))

# --------------------------------------------------------------------------- #
# cache (atomic, sig-guarded; index survives sig change)                       #
# --------------------------------------------------------------------------- #
_C = {"sig": SIG, "index": {}, "metrics": {}}
_EVAL_MS = []          # (n, ms) for cache-miss evaluate_solution calls
_LAYS_MEMO = {}        # quota-tuple -> loaded cases (per process)


def _cload():
    global _C
    if CACHE_PATH.exists():
        try:
            c = pickle.load(open(CACHE_PATH, "rb"))
            if c.get("sig") == SIG:
                _C = c
            else:
                _C = {"sig": SIG, "index": c.get("index", {}), "metrics": {}}
                print("[cache] sig changed -> metrics reset (file index kept)")
        except Exception:
            pass


def _csave():
    tmp = CACHE_PATH.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(_C, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, CACHE_PATH)


def _eval_variant(lay, vkey, fn):
    key = (lay["key"], vkey)
    t = _C["metrics"].get(key)
    if t is None:
        t = fn()
        _C["metrics"][key] = t
    return t


# --------------------------------------------------------------------------- #
# strict official scoring                                                      #
# --------------------------------------------------------------------------- #
def _cost_strict(positions, lay):
    """Official per-case metrics with STRICT hard-constraint checking:
    target_positions = fp_sol tuples, so preplaced position / fixed dims
    deviations make the solution infeasible (exactly like the harness)."""
    t0 = time.perf_counter()
    m = evaluate_solution({"positions": positions, "runtime": 1.0}, lay["base"],
                          lay["cons"], lay["b2b"], lay["p2b"], lay["pins"],
                          lay["at"], target_positions=lay["tp"],
                          median_runtime=1.0)
    _EVAL_MS.append((lay["n"], (time.perf_counter() - t0) * 1e3))
    return m


def _mt(m, extra=0.0):
    """Metrics tuple stored in the cache.
    [0]=cost [1]=feasible [2]=hgap [3]=agap [4]=vrel [5]=vb [6]=vg [7]=vm
    [8]=extra (maxerr / tau / preDisp) [9]=overlaps [10]=dimviol [11]=areaviol"""
    return (float(m.cost), 1 if m.is_feasible else 0, float(m.hpwl_gap),
            float(m.area_gap), float(m.violations_relative),
            int(m.boundary_violations), int(m.grouping_violations),
            int(m.mib_violations), float(extra), int(m.overlap_violations),
            int(m.dimension_violations), int(m.area_violations))


# --------------------------------------------------------------------------- #
# data: index -> stratified sample -> load + augment                           #
# --------------------------------------------------------------------------- #
def _index_files(workers=(0, 1, 2)):
    idx = _C.get("index", {})
    if idx.get("workers") == workers and idx.get("files"):
        return idx["files"]
    todo = []
    for w in workers:
        todo += sorted(glob.glob(str(_DIR / "floorset_lite" / f"worker_{w}"
                                      / "layouts_*.th")))
    print(f"[index] scanning {len(todo)} files for n (one-time, cached) ...")
    t0 = time.time()
    files = {}
    for i, f in enumerate(todo):
        d = torch.load(f)
        files[str(f)] = (int((d[0][0][:, 0] != -1).sum().item()),
                         int(d[0].shape[0]))
        if (i + 1) % 60 == 0:
            print(f"[index] {i + 1}/{len(todo)}  ({time.time() - t0:.0f}s)")
    _C["index"] = {"workers": workers, "files": files}
    _csave()
    return files


def _shuffled(seq, seedstr):
    lst = list(seq)
    random.Random(seedstr).shuffle(lst)
    return lst


def _band_of(n):
    for b, lo, hi in BANDS:
        if lo < n <= hi:
            return b
    return "?"


def _sample_cases(files, quota):
    """Deterministic per-band case order with the PREFIX property: a smaller
    quota yields a prefix of a larger one (quick ⊂ full, slack ⊂ pipeline ->
    cache reuse). Round-robin over files (≤PER_FILE_CAP each for the first
    rounds); in band B the first 8 files are n>=116 heavyweights."""
    specs = []
    for band, lo, hi in BANDS:
        fl = sorted((f, n, nl) for f, (n, nl) in files.items() if lo < n <= hi)
        big = _shuffled([x for x in fl if x[1] >= 116], f"{SEED}:s:{band}:big")
        rest = _shuffled([x for x in fl if x[1] < 116], f"{SEED}:s:{band}:rest")
        if band == "B":
            ordered = big[:8] + _shuffled(big[8:] + rest, f"{SEED}:s:{band}:mix")
        else:
            ordered = _shuffled(rest + big, f"{SEED}:s:{band}:mix")
        picks = {f: _shuffled(range(nl), f"{SEED}:pick:{f}")
                 for f, n, nl in ordered}
        take, rounds = [], 0
        while len(take) < quota[band] and rounds < 400:
            prog = False
            for f, n, nl in ordered:
                if len(take) >= quota[band]:
                    break
                used = sum(1 for t in take if t[0] == f)
                cap = PER_FILE_CAP if rounds < PER_FILE_CAP else nl
                if used >= min(cap, nl):
                    continue
                take.append((f, picks[f][used], n))
                prog = True
            rounds += 1
            if not prog:
                break
        specs.extend(take)
    return specs


def _augment(lay, key):
    n = lay["n"]
    lay["key"] = key
    base, pos_fp = _baseline(lay)
    lay["base"] = base
    lay["pos_fp"] = pos_fp
    lay["tp"] = pos_fp                       # strict targets = fp_sol itself
    cons = lay["cons"]
    lay["fixed"] = [b for b in range(n) if int(cons[b][0]) != 0]
    lay["preplaced"] = [b for b in range(n) if int(cons[b][1]) != 0]
    lay["mib_id"] = [int(cons[b][2]) for b in range(n)]
    lay["oracleY"] = sorted(range(n), key=lambda b: lay["Y"][b])
    lay["fp"] = _eval_variant(lay, ("fp",),
                              lambda: _mt(_cost_strict(pos_fp, lay)))
    return lay


def _get_cases(quota):
    qk = tuple(sorted(quota.items()))
    if qk in _LAYS_MEMO:
        return _LAYS_MEMO[qk]
    specs = _sample_cases(_index_files(), quota)
    byf = {}
    for f, L, n in specs:
        byf.setdefault(f, []).append(L)
    lays = []
    t0 = time.time()
    for i, (f, Ls) in enumerate(sorted(byf.items())):
        d = torch.load(f)
        stem = f"{Path(f).parent.name}/{Path(f).stem}"
        for L in sorted(Ls):
            try:
                lay = load_layout(d, L)
            except AssertionError:
                continue
            lays.append(_augment(lay, f"{stem}/L{L}"))
        if (i + 1) % 20 == 0:
            print(f"[load] {i + 1}/{len(byf)} files ({time.time() - t0:.0f}s)")
    _csave()
    _LAYS_MEMO[qk] = lays
    return lays


# --------------------------------------------------------------------------- #
# tree edit operators (validity-preserving)                                    #
# --------------------------------------------------------------------------- #
def _lr_from_ins(ins, n):
    left, right = [None] * n, [None] * n
    for a, nb, dr in ins:
        if dr == 0:
            left[a] = nb
        else:
            right[a] = nb
    return left, right


def _parents(left, right, n):
    parent, pdir = [None] * n, [None] * n
    for u in range(n):
        if left[u] is not None:
            parent[left[u]], pdir[left[u]] = u, 0
        if right[u] is not None:
            parent[right[u]], pdir[right[u]] = u, 1
    return parent, pdir


def _subtree_nodes(v, left, right):
    out, stack = set(), [v]
    while stack:
        u = stack.pop()
        out.add(u)
        if left[u] is not None:
            stack.append(left[u])
        if right[u] is not None:
            stack.append(right[u])
    return out


def _tree_to_ins(left, right, seed, n):
    ins, seen, stack = [], {seed}, [seed]
    while stack:
        u = stack.pop()
        for dr, c in ((0, left[u]), (1, right[u])):
            if c is not None:
                assert c not in seen, "cycle/dup in perturbed tree"
                ins.append((u, c, dr))
                seen.add(c)
                stack.append(c)
    assert len(ins) == n - 1 and len(seen) == n, "perturbed tree not spanning"
    return ins


def perturb_tree(lay, s, rng, movable_only=False, op_mix=None):
    """Apply k=ceil(s*(n-1)) validity-preserving edits to the B*-tree; each
    edit ~ one wrong (anchor, dir) token from an autoregressive decoder.
    movable_only refuses edits that directly relocate a preplaced block
    (moved subtree containing one / identity swap involving one)."""
    n, seed = lay["n"], lay["seed"]
    left, right = _lr_from_ins(lay["ins"], n)
    pre = set(lay["preplaced"])
    k = max(1, math.ceil(s * (n - 1)))
    mix = op_mix if op_mix is not None else OP_W
    ops = [o for o, _ in mix]
    wts = [w for _, w in mix]
    applied = guard = 0
    while applied < k and guard < 300 * k:
        guard += 1
        op = rng.choices(ops, weights=wts)[0]
        if op == "SWAP":                      # swap two block identities
            a, b = rng.sample(range(n), 2)
            if movable_only and (a in pre or b in pre):
                continue
            perm = list(range(n))
            perm[a], perm[b] = b, a
            L2, R2 = [None] * n, [None] * n
            for u in range(n):
                if left[u] is not None:
                    L2[perm[u]] = perm[left[u]]
                if right[u] is not None:
                    R2[perm[u]] = perm[right[u]]
            left, right, seed = L2, R2, perm[seed]
            applied += 1
            continue
        parent, pdir = _parents(left, right, n)
        cand = [v for v in range(n) if v != seed]
        if op == "MOVE_LEAF":
            cand = [v for v in cand if left[v] is None and right[v] is None]
        if not cand:
            continue
        v = rng.choice(cand)
        sub = _subtree_nodes(v, left, right)
        if movable_only and (sub & pre):
            continue
        p, d = parent[v], pdir[v]
        if op == "FLIP":                      # move v to parent's other slot
            if (right[p] if d == 0 else left[p]) is not None:
                continue
            if d == 0:
                left[p], right[p] = None, v
            else:
                left[p], right[p] = v, None
            applied += 1
            continue
        slots = []                            # MOVE_SUB / MOVE_LEAF
        for u in range(n):
            if u in sub:
                continue
            if left[u] is None and (u, 0) != (p, d):
                slots.append((u, 0))
            if right[u] is None and (u, 1) != (p, d):
                slots.append((u, 1))
        if not slots:
            continue
        u2, d2 = rng.choice(slots)
        if d == 0:
            left[p] = None
        else:
            right[p] = None
        if d2 == 0:
            left[u2] = v
        else:
            right[u2] = v
        applied += 1
    return _tree_to_ins(left, right, seed, n), seed


def near_miss_one(lay, rng):
    """The most benign REALISTIC single decoder error: detach one movable
    leaf and re-attach it to an empty slot whose anchor is among the 3
    geometrically nearest (fp_sol L1) to the ORIGINAL anchor. Lower-bounds
    the cost of a near-miss token, vs perturb_tree's uniform-random slot."""
    n, seed = lay["n"], lay["seed"]
    left, right = _lr_from_ins(lay["ins"], n)
    parent, pdir = _parents(left, right, n)
    pre = set(lay["preplaced"])
    leaves = [v for v in range(n) if v != seed and v not in pre
              and left[v] is None and right[v] is None]
    if not leaves:
        return lay["ins"], seed
    v = rng.choice(leaves)
    p, d = parent[v], pdir[v]
    X, Y = lay["X"], lay["Y"]
    slots = []
    for u in range(n):
        if u == v:
            continue
        for dd, c in ((0, left[u]), (1, right[u])):
            if c is None:
                slots.append((abs(X[u] - X[p]) + abs(Y[u] - Y[p]), u, dd))
    if d == 0:
        left[p] = None
    else:
        right[p] = None
    slots.sort()
    _, u2, d2 = slots[rng.randrange(min(3, len(slots)))]
    if d2 == 0:
        left[u2] = v
    else:
        right[u2] = v
    return _tree_to_ins(left, right, seed, n), seed


def perturb_order(order, s, rng):
    o = list(order)
    for _ in range(max(1, math.ceil(s * len(o)))):
        i, j = rng.sample(range(len(o)), 2)
        o[i], o[j] = o[j], o[i]
    return o


def _kendall_tau(o, ref):
    pos = {b: i for i, b in enumerate(ref)}
    r = [pos[b] for b in o]
    n = len(r)
    inv = 0
    for i in range(n):
        ri = r[i]
        for j in range(i + 1, n):
            if ri > r[j]:
                inv += 1
    return inv / (n * (n - 1) / 2.0)


def perturb_dims(lay, mode, level, rng):
    """Reshape movable blocks area-EXACTLY (W'=sqrt(a*r'), H'=a/W' keeps
    W'*H'==a to ~1 ulp << the 1% tolerance). fixed/preplaced exempt. MIB
    groups share one jitter epsilon (relation-preserving: equal dims stay
    equal); quant maps per member (equal stays equal, near-equal may merge
    -> vmib can only drop, reported as a delta)."""
    n = lay["n"]
    W2, H2 = list(lay["W"]), list(lay["H"])
    skip = set(lay["fixed"]) | set(lay["preplaced"])
    eps_g = {}
    lo, hi = math.log(ASPECT_LO), math.log(ASPECT_HI)
    for b in range(n):
        if b in skip:
            continue
        a = float(lay["at"][b])
        if a <= 0:
            continue
        r = lay["W"][b] / lay["H"][b]
        if mode == "jitter":
            g = lay["mib_id"][b]
            if g > 0:
                if g not in eps_g:
                    eps_g[g] = rng.uniform(-level, level)
                eps = eps_g[g]
            else:
                eps = rng.uniform(-level, level)
            r2 = r * math.exp(eps)
        else:                                  # "quant": level = K bins
            K = int(level)
            t = (math.log(max(min(r, ASPECT_HI), ASPECT_LO)) - lo) / (hi - lo)
            idx = min(max(int(round(t * (K - 1))), 0), K - 1)
            r2 = math.exp(lo + idx * (hi - lo) / (K - 1))
        r2 = max(min(r2, ASPECT_HI), ASPECT_LO)
        W2[b] = math.sqrt(a * r2)
        H2[b] = a / W2[b]
    return W2, H2


# --------------------------------------------------------------------------- #
# legalizer v0                                                                 #
# --------------------------------------------------------------------------- #
def legalize_v0(lay, px, W, H, order):
    """Obstacle-aware ordered gravity. X stays fixed (movables keep decoded
    px; preplaced snapped to target x); fixed/preplaced dims restored to
    target; blocks drop in `order`; a preplaced block is FORCED onto its
    target (x, y) when its turn comes and reserves that rectangle before-
    hand, so movables bump above the reservation (y only grows ->
    terminates). Constructive guarantees: zero overlap + every dimension/
    position hard constraint -> the output is always hard-feasible. On
    unperturbed input every step coincides with compact_down (fp_sol is
    overlap-free so no bump ever triggers) -> reproduces fp_sol exactly."""
    n = lay["n"]
    tp = lay["tp"]
    pre = set(lay["preplaced"])
    X2, W2, H2 = list(px), list(W), list(H)
    for b in set(lay["fixed"]) | pre:
        W2[b], H2[b] = tp[b][2], tp[b][3]
    for b in pre:
        X2[b] = tp[b][0]
    Y2 = [0.0] * n
    resv = {b: tp[b] for b in pre}
    done = []
    for b in order:
        if b in pre:
            Y2[b] = tp[b][1]
            del resv[b]
            done.append(b)
            continue
        x0, x1 = X2[b], X2[b] + W2[b]
        y = 0.0
        for c in done:
            if X2[c] < x1 - 1e-9 and X2[c] + W2[c] > x0 + 1e-9:
                t = Y2[c] + H2[c]
                if t > y:
                    y = t
        moved = True
        while moved:
            moved = False
            for rx, ry, rw, rh in resv.values():
                if (rx < x1 - 1e-9 and rx + rw > x0 + 1e-9 and
                        ry < y + H2[b] - 1e-9 and ry + rh > y + 1e-9):
                    y = ry + rh
                    moved = True
        Y2[b] = y
        done.append(b)
    return [(X2[b], Y2[b], W2[b], H2[b]) for b in range(n)]


def decode_eval(lay, ins=None, seed=None, order=None, W=None, H=None,
                legalize=True):
    """One candidate through the full pipeline: render X -> raw gravity (for
    preDisp) -> legalize_v0 (or raw) -> strict official metrics. Returns
    (SolutionMetrics, positions, preDisp) with preDisp = mean L1 displacement
    of preplaced blocks in the RAW decode (legalizer-intervention gauge)."""
    lay2 = dict(lay)
    if ins is not None:
        lay2["ins"] = ins
    if seed is not None:
        lay2["seed"] = seed
    if W is not None:
        lay2["W"] = W
    if H is not None:
        lay2["H"] = H
    Wd, Hd = lay2["W"], lay2["H"]
    if order is None:
        order = lay["oracleY"]
    px, _ = render_bstar(lay2)
    _, py = compact_down(px, Wd, Hd, order)
    pre = lay["preplaced"]
    pd = (sum(abs(px[b] - lay["tp"][b][0]) + abs(py[b] - lay["tp"][b][1])
              for b in pre) / len(pre)) if pre else 0.0
    if legalize:
        pos = legalize_v0(lay, px, Wd, Hd, order)
    else:
        pos = [(px[b], py[b], Wd[b], Hd[b]) for b in range(lay["n"])]
    return _cost_strict(pos, lay), pos, pd


# --------------------------------------------------------------------------- #
# variant dispatcher (cache-keyed)                                             #
# --------------------------------------------------------------------------- #
def _perturbed_tree_eval(lay, axis, s, r, legalize):
    rng = random.Random(f"{SEED}:{lay['key']}:{axis}:{s}:{r}")
    ins2, seed2 = perturb_tree(lay, s, rng, movable_only=(axis == "tree_mov"))
    m, _, pd = decode_eval(lay, ins=ins2, seed=seed2, legalize=legalize)
    return _mt(m, pd)


def _run_variant(lay, vkey):
    kind = vkey[0]
    if kind == "clean":
        m, _, pd = decode_eval(lay)
        return _mt(m, pd)
    if kind == "contourY":                    # no-Y-label anchor (single point)
        px, pyc = render_bstar(dict(lay))
        order = sorted(range(lay["n"]), key=lambda b: pyc[b])
        m, _, pd = decode_eval(lay, order=order)
        return _mt(m, pd)
    if kind == "pre":                         # RAW decode (no legalizer)
        _, axis, s, r = vkey
        if axis == "clean":
            m, _, pd = decode_eval(lay, legalize=False)
            return _mt(m, pd)
        return _perturbed_tree_eval(lay, axis, s, r, legalize=False)
    if kind in ("tree_any", "tree_mov"):
        _, s, r = vkey
        return _perturbed_tree_eval(lay, kind, s, r, legalize=True)
    if kind == "tree_leaf1":                  # exactly ONE most-benign token
        r = vkey[1]                           # error: relocate one movable leaf
        rng = random.Random(f"{SEED}:{lay['key']}:tree_leaf1:{r}")
        ins2, seed2 = perturb_tree(lay, 0.0, rng, movable_only=True,
                                   op_mix=(("MOVE_LEAF", 1.0),))
        m, _, pd = decode_eval(lay, ins=ins2, seed=seed2)
        return _mt(m, pd)
    if kind == "tree_near1":                  # ONE near-miss token (nearest-
        r = vkey[1]                           # anchor re-attach, softest error)
        rng = random.Random(f"{SEED}:{lay['key']}:tree_near1:{r}")
        ins2, seed2 = near_miss_one(lay, rng)
        m, _, pd = decode_eval(lay, ins=ins2, seed=seed2)
        return _mt(m, pd)
    if kind == "Y":
        _, s, r = vkey
        rng = random.Random(f"{SEED}:{lay['key']}:Y:{s}:{r}")
        o2 = perturb_order(lay["oracleY"], s, rng)
        tau = _kendall_tau(o2, lay["oracleY"])
        m, _, _ = decode_eval(lay, order=o2)
        return _mt(m, tau)
    if kind == "dims_jit":
        _, s, r = vkey
        rng = random.Random(f"{SEED}:{lay['key']}:dims_jit:{s}:{r}")
        W2, H2 = perturb_dims(lay, "jitter", s, rng)
        m, _, _ = decode_eval(lay, W=W2, H=H2)
        return _mt(m)
    if kind == "dims_q":
        W2, H2 = perturb_dims(lay, "quant", vkey[1], None)
        m, _, _ = decode_eval(lay, W=W2, H=H2)
        return _mt(m)
    if kind == "comb":
        _, name, r = vkey
        st, sy, sd = COMBOS[name]
        rng = random.Random(f"{SEED}:{lay['key']}:comb:{name}:{r}")
        ins2, seed2 = perturb_tree(lay, st, rng, False)
        o2 = perturb_order(lay["oracleY"], sy, rng)
        W2, H2 = perturb_dims(lay, "jitter", sd, rng)
        m, _, pd = decode_eval(lay, ins=ins2, seed=seed2, order=o2, W=W2, H=H2)
        return _mt(m, pd)
    raise ValueError(vkey)


def _pairs(lays, vkeys):
    out = []
    for lay in lays:
        for vk in vkeys:
            out.append((lay, _eval_variant(
                lay, vk, lambda l=lay, v=vk: _run_variant(l, v))))
    return out


# --------------------------------------------------------------------------- #
# aggregation helpers                                                          #
# --------------------------------------------------------------------------- #
def _w(lay):
    return math.exp(lay["n"] / 12.0)


def _wr(pairs):
    sw = sr = 0.0
    for lay, t in pairs:
        w = _w(lay)
        sw += w
        sr += w * (t[0] / lay["fp"][0])
    return sr / sw if sw else float("nan")


def _q(t):
    """Quality factor 1+0.5(hgap+agap), gaps clamped at 0 (official form)."""
    return 1.0 + 0.5 * (max(0.0, t[2]) + max(0.0, t[3]))


def _wrq(pairs):
    """Quality-only wR: what wR would be if a (hypothetical ideal) soft-
    constraint repair restored vrel to the fp floor without moving gaps —
    the upper bound a constraint-enforcing legalizer v1 could reach."""
    sw = sr = 0.0
    for lay, t in pairs:
        w = _w(lay)
        sw += w
        sr += w * (_q(t) / _q(lay["fp"]))
    return sr / sw if sw else float("nan")


def _wcost(pairs):
    sw = sc = 0.0
    for lay, t in pairs:
        w = _w(lay)
        sw += w
        sc += w * t[0]
    return sc / sw if sw else float("nan")


def _hu(pairs):
    fp = [(lay, lay["fp"]) for lay, _ in pairs]
    return (_wcost(pairs) - _wcost(fp)) / HEADROOM_ABS * 100.0


def _feas(pairs):
    return 100.0 * sum(t[1] for _, t in pairs) / len(pairs) if pairs else float("nan")


def _dv(pairs, i):
    return (sum(t[i] - lay["fp"][i] for lay, t in pairs) / len(pairs)
            if pairs else float("nan"))


def _mean(pairs, i):
    return sum(t[i] for _, t in pairs) / len(pairs) if pairs else float("nan")


def _bsel(pairs, band):
    return [(lay, t) for lay, t in pairs if _band_of(lay["n"]) == band]


def _pctl(vals, p):
    if not vals:
        return float("nan")
    s = sorted(vals)
    return s[min(len(s) - 1, int(p / 100.0 * len(s)))]


# --------------------------------------------------------------------------- #
# pipeline (Phase 0-i)                                                         #
# --------------------------------------------------------------------------- #
def cmd_pipeline(quick):
    print("=" * 78)
    print("PHASE 0-i  PIPELINE: true tree + oracle-Y + true dims --> fp_sol ?")
    print("(training-set floor; NOT comparable to the validation 1.1079)")
    print("=" * 78)
    quota = {b: (q // 2 if quick else q) for b, q in QUOTA_PIPE.items()}
    lays = _get_cases(quota)
    per = {b: dict(cases=0, exact=0, costeq=0, fpfeas=0, ns=[], fpc=[], fpv=[],
                   dec=[], leg=[]) for b, _, _ in BANDS}
    fails = []
    for lay in lays:
        n, W, H = lay["n"], lay["W"], lay["H"]
        t0 = time.perf_counter()
        px, _ = render_bstar(lay)
        _, py = compact_down(px, W, H, lay["oracleY"])
        t1 = time.perf_counter()
        pos_dec = [(px[k], py[k], W[k], H[k]) for k in range(n)]
        pos_leg = legalize_v0(lay, px, W, H, lay["oracleY"])
        t2 = time.perf_counter()
        me_dec = max(max(abs(pos_dec[k][0] - lay["tp"][k][0]),
                         abs(pos_dec[k][1] - lay["tp"][k][1]))
                     for k in range(n))
        me_leg = max(max(abs(pos_leg[k][0] - lay["tp"][k][0]),
                         abs(pos_leg[k][1] - lay["tp"][k][1]))
                     for k in range(n))
        t = _eval_variant(lay, ("dec",), lambda l=lay, p=pos_dec, e=me_dec:
                          _mt(_cost_strict(p, l), e))
        R = per[_band_of(n)]
        R["cases"] += 1
        R["ns"].append(n)
        R["dec"].append((t1 - t0) * 1e3)
        R["leg"].append((t2 - t1) * 1e3)
        R["fpc"].append(lay["fp"][0])
        R["fpv"].append(lay["fp"][4])
        ce = False
        if me_leg < 1e-6:
            R["exact"] += 1
            ce = abs(t[0] - lay["fp"][0]) < 1e-9
            if ce:
                R["costeq"] += 1
        if lay["fp"][1]:
            R["fpfeas"] += 1
        if me_leg >= 1e-6 or not ce or not lay["fp"][1]:
            fails.append((lay["key"], n, me_dec, me_leg, t[0], lay["fp"][0],
                          lay["fp"][1]))
    _csave()

    def _wmean(ns, vals):
        ws = [math.exp(x / 12.0) for x in ns]
        return sum(a * b for a, b in zip(ws, vals)) / sum(ws)

    print(f"{'band':<5}{'cases':>6}{'exact%':>8}{'costEq%':>9}{'fpFeas%':>9}"
          f"{'fp_wCost':>10}{'fp_wVrel':>10}{'dec_ms p50/p95':>16}"
          f"{'leg_ms p50/p95':>16}")
    tot = dict(cases=0, exact=0, costeq=0, fpfeas=0)
    for b, _, _ in BANDS:
        R = per[b]
        if not R["cases"]:
            continue
        for k in tot:
            tot[k] += R[k]
        print(f"{b:<5}{R['cases']:>6}"
              f"{100.0 * R['exact'] / R['cases']:>8.1f}"
              f"{100.0 * R['costeq'] / R['cases']:>9.1f}"
              f"{100.0 * R['fpfeas'] / R['cases']:>9.1f}"
              f"{_wmean(R['ns'], R['fpc']):>10.4f}"
              f"{_wmean(R['ns'], R['fpv']):>10.4f}"
              f"{_pctl(R['dec'], 50):>8.1f}/{_pctl(R['dec'], 95):<7.1f}"
              f"{_pctl(R['leg'], 50):>8.1f}/{_pctl(R['leg'], 95):<7.1f}")
    ems = {b: [ms for n, ms in _EVAL_MS if _band_of(n) == b] for b, _, _ in BANDS}
    if any(ems.values()):
        print("eval_ms (cache-miss only): " + "  ".join(
            f"{b} p50={_pctl(v, 50):.0f} p95={_pctl(v, 95):.0f}"
            for b, v in ems.items() if v))
    g1a = 100.0 * tot["exact"] / max(tot["cases"], 1)
    g1b = 100.0 * tot["costeq"] / max(tot["exact"], 1)
    g1c = 100.0 * tot["fpfeas"] / max(tot["cases"], 1)
    ok = g1a >= 99.5 and g1b >= 100.0 - 1e-9 and g1c >= 100.0 - 1e-9
    print(f"\n[G1] exact {g1a:.2f}% (bar 99.5) | costEq {g1b:.2f}% of exact "
          f"(bar 100) | fpFeas {g1c:.2f}% (bar 100)  -> "
          f"{'PASS' if ok else 'FAIL'}")
    if fails:
        print(f"     failures ({len(fails)}):")
        for key, n, ed, el, c, cf, ff in fails[:12]:
            print(f"       {key} n={n} errDec={ed:.3g} errLeg={el:.3g} "
                  f"cost={c:.6f} fp={cf:.6f} fpFeas={ff}")
        if len(fails) > 12:
            print(f"       ... +{len(fails) - 12} more")
    return dict(ok=ok, g1a=g1a, g1b=g1b, g1c=g1c, cases=tot["cases"])


# --------------------------------------------------------------------------- #
# slack (Phase 0-ii)                                                           #
# --------------------------------------------------------------------------- #
def _slack_params(quick):
    quota = {b: (q // 2 if quick else q) for b, q in QUOTA_SLACK.items()}
    levels = (0.05, 0.20) if quick else LEVELS
    reps = 2 if quick else REPS
    return quota, levels, reps


def _slack_row(label, pairs):
    cells = [f"{_wr(pairs):7.4f}", f"{_wrq(pairs):7.4f}"]
    for b, _, _ in BANDS:
        bp = _bsel(pairs, b)
        cells.append(f"{_wr(bp):7.4f}" if bp else "      -")
    print(f"{label:<15}| " + " ".join(cells) +
          f" |{_hu(pairs):7.1f} |{_mean(pairs, 8):9.3f} |"
          f"{_dv(pairs, 5):6.2f}{_dv(pairs, 6):7.2f}{_dv(pairs, 7):6.2f} |"
          f"{_feas(pairs):6.1f}")


def cmd_slack(quick):
    print("=" * 78)
    print("PHASE 0-ii  SLACK: cost degradation vs model-error rate "
          "(post-legalize)")
    print("R = cost/fp_floor per case; wR = exp(n/12)-weighted mean; "
          "HU% = share of the 0.2186 local headroom")
    print("=" * 78)
    quota, levels, reps = _slack_params(quick)
    qks = (16,) if quick else QUANT_KS
    combos = ("mild",) if quick else tuple(COMBOS)
    lays = _get_cases(quota)
    print(f"cases={len(lays)}  levels={levels}  reps={reps}")
    print("wRq = quality-only wR (gaps ratio; = ideal-soft-repair upper bound)")
    print(f"{'variant':<15}| {'wR_all':>7} {'wRq':>7} {'wR_S':>7} {'wR_M':>7}"
          f" {'wR_B':>7} |{'HU%':>7} |{'x(extra)':>9} |{'d_vb':>6}{'d_vg':>7}"
          f"{'d_vm':>6} |{'feas%':>6}")
    _slack_row("ref_clean", _pairs(lays, [("clean",)]))
    res = {}
    p1 = _pairs(lays, [("tree_leaf1", r) for r in range(REPS)])
    res[("tree_leaf1",)] = p1
    _slack_row("tree_leaf1", p1)
    pn = _pairs(lays, [("tree_near1", r) for r in range(REPS)])
    res[("tree_near1",)] = pn
    _slack_row("tree_near1", pn)
    print("-" * 78)
    for axis in ("tree_any", "tree_mov", "Y", "dims_jit"):
        for s in levels:
            p = _pairs(lays, [(axis, s, r) for r in range(reps)])
            res[(axis, s)] = p
            _slack_row(f"{axis} {s:.2f}", p)
        _csave()
        print("-" * 78)
    for K in qks:
        p = _pairs(lays, [("dims_q", K)])
        res[("dims_q", K)] = p
        _slack_row(f"dims_q K={K}", p)
    for name in combos:
        _slack_row(f"comb {name}", _pairs(lays, [("comb", name, r)
                                                 for r in range(reps)]))
    _slack_row("ref_contourY", _pairs(lays, [("contourY",)]))
    _csave()

    s_ref = 0.05
    smin = min(levels)
    out = dict(cases=len(lays))
    for axis in ("tree_any", "tree_mov", "Y"):
        wr5 = _wr(res[(axis, s_ref)])
        steep = _wr(res[(axis, smin)]) > BAR_TREE_Y
        out[axis] = (wr5, "STEEP" if steep else
                     ("OK" if wr5 <= BAR_TREE_Y else "OVER"))
    wq = _wr(res[("dims_q", 16 if 16 in qks else qks[0])])
    out["dims_q"] = (wq, "OK" if wq <= BAR_DIMS else "OVER")
    out["leaf1"] = (_wr(res[("tree_leaf1",)]), _wrq(res[("tree_leaf1",)]))
    out["near1"] = (_wr(res[("tree_near1",)]), _wrq(res[("tree_near1",)]))
    print(f"\n[S2] bars: tree/Y wR@{s_ref:.2f} <= {BAR_TREE_Y}, "
          f"dims_q K=16 <= {BAR_DIMS}; STEEP = wR@{smin:.2f} already > "
          f"{BAR_TREE_Y}")
    for k in ("tree_any", "tree_mov", "Y", "dims_q"):
        v, tag = out[k]
        print(f"     {k:<9} wR={v:.4f}  {tag}")
    print(f"     tree_leaf1 (single random-slot token error): "
          f"wR={out['leaf1'][0]:.4f}  wRq={out['leaf1'][1]:.4f}")
    print(f"     tree_near1 (single NEAR-MISS token error):   "
          f"wR={out['near1'][0]:.4f}  wRq={out['near1'][1]:.4f} "
          f"(zero-tolerance test)")
    return out


# --------------------------------------------------------------------------- #
# legalize (Phase 0-iii)                                                       #
# --------------------------------------------------------------------------- #
def cmd_legalize(quick):
    print("=" * 78)
    print("PHASE 0-iii  LEGALIZER v0: repair rate + attribution")
    print("=" * 78)
    quota, _, reps = _slack_params(quick)
    lays = _get_cases(quota)
    scen = [("clean", None)] + [(ax, s) for ax in ("tree_any", "tree_mov")
                                for s in (0.05, 0.20)]
    print(f"{'scenario':<15}| {'preFeas%':>8} {'preOvl':>7} {'preDim':>7} |"
          f" {'postFeas%':>9} {'wR_post':>8} |{'d_vb':>6}{'d_vg':>7}{'d_vm':>6}"
          f" |{'preDisp':>9}")
    post_all = []
    rows = {}
    for ax, s in scen:
        if ax == "clean":
            pre_p = _pairs(lays, [("pre", "clean", 0, 0)])
            post_p = _pairs(lays, [("clean",)])
            label = "unperturbed"
        else:
            pre_p = _pairs(lays, [("pre", ax, s, r) for r in range(reps)])
            post_p = _pairs(lays, [(ax, s, r) for r in range(reps)])
            label = f"{ax} {s:.2f}"
        rows[(ax, s)] = post_p
        post_all += post_p
        print(f"{label:<15}| {_feas(pre_p):>8.1f} {_mean(pre_p, 9):>7.2f} "
              f"{_mean(pre_p, 10):>7.2f} | {_feas(post_p):>9.1f} "
              f"{_wr(post_p):>8.4f} |{_dv(post_p, 5):6.2f}"
              f"{_dv(post_p, 6):7.2f}{_dv(post_p, 7):6.2f} |"
              f"{_mean(pre_p, 8):>9.2f}")
        _csave()
    self_infl = _wr(rows[("clean", None)]) - 1.0
    model = _wr(rows[("tree_mov", 0.05)]) - 1.0
    inter = _wr(rows[("tree_any", 0.05)]) - _wr(rows[("tree_mov", 0.05)])
    post_feas = _feas(post_all)
    g3a = post_feas >= 100.0 - 1e-9
    g3b = self_infl <= 0.01
    g3c = _wr(rows[("tree_any", 0.05)]) <= BAR_TREE_Y and inter < max(model, 1e-9)
    ok = g3a and g3b and g3c
    print(f"\n[attribution @0.05] model(tree_mov)={model * 100:+.2f}%  "
          f"legalizer-interact={inter * 100:+.2f}%  "
          f"self-inflation={self_infl * 100:+.2f}%")
    print(f"[G3] postFeas {post_feas:.1f}% (bar 100) | self-infl "
          f"{self_infl * 100:+.2f}% (bar <=1%) | interact<model "
          f"{'yes' if inter < max(model, 1e-9) else 'NO'}  -> "
          f"{'PASS' if ok else 'FAIL'}")
    return dict(ok=ok, post_feas=post_feas, self_infl=self_infl,
                model=model, inter=inter)


# --------------------------------------------------------------------------- #
# vocab (Phase 1 dims-head design input)                                       #
# --------------------------------------------------------------------------- #
def cmd_vocab(quick):
    print("=" * 78)
    print("VOCAB: training-set dims statistics (Phase 1 dims-head design)")
    print("=" * 78)
    files = []
    for w in ((0,) if quick else (0, 1)):
        files += sorted(glob.glob(str(_DIR / "floorset_lite" / f"worker_{w}"
                                      / "layouts_*.th")))
    if quick:
        files = files[::3]
    intWH = [0, 0]
    intXY = [0, 0]
    areaEx = [0, 0]
    asp = Counter()
    areas = []
    seen_mov = 0
    mib_uni = [0, 0]
    consn = Counter()
    cases = 0
    maxcoord = 0.0
    rngres = random.Random(f"{SEED}:vocab")
    t0 = time.time()
    for i, f in enumerate(files):
        d = torch.load(f)
        for L in range(d[0].shape[0]):
            try:
                lay = load_layout(d, L)
            except AssertionError:
                continue
            n = lay["n"]
            cases += 1
            cons = lay["cons"]
            pre = [b for b in range(n) if int(cons[b][1]) != 0]
            fx = [b for b in range(n) if int(cons[b][0]) != 0]
            skip = set(pre) | set(fx)
            consn["preplaced"] += len(pre)
            consn["fixed"] += len(fx)
            consn["boundary"] += sum(1 for b in range(n)
                                     if int(cons[b][4]) != 0)
            mib = {}
            clus = set()
            for b in range(n):
                g = int(cons[b][2])
                if g > 0:
                    mib.setdefault(g, []).append(b)
                if int(cons[b][3]) > 0:
                    clus.add(int(cons[b][3]))
            consn["mib_groups"] += len(mib)
            consn["clusters"] += len(clus)
            for g, mem in mib.items():
                mib_uni[1] += 1
                dims = {(round(lay["W"][b], 4), round(lay["H"][b], 4))
                        for b in mem}
                if len(dims) == 1:
                    mib_uni[0] += 1
            for b in range(n):
                w_, h_ = lay["W"][b], lay["H"][b]
                x_, y_ = lay["X"][b], lay["Y"][b]
                intWH[1] += 2
                intWH[0] += ((abs(w_ - round(w_)) < 1e-9)
                             + (abs(h_ - round(h_)) < 1e-9))
                intXY[1] += 2
                intXY[0] += ((abs(x_ - round(x_)) < 1e-9)
                             + (abs(y_ - round(y_)) < 1e-9))
                if x_ + w_ > maxcoord:
                    maxcoord = x_ + w_
                if y_ + h_ > maxcoord:
                    maxcoord = y_ + h_
                a = float(lay["at"][b])
                areaEx[1] += 1
                areaEx[0] += (abs(w_ * h_ - a) <= 1e-9 * max(1.0, a))
                if b in skip:
                    continue
                seen_mov += 1
                asp[round(w_ / h_, 4)] += 1
                if len(areas) < 3000:
                    areas.append(a)
                elif rngres.random() < 3000.0 / seen_mov:
                    areas[rngres.randrange(3000)] = a
        if (i + 1) % 30 == 0:
            print(f"[vocab] {i + 1}/{len(files)} files "
                  f"({time.time() - t0:.0f}s)")
    tot = sum(asp.values()) or 1
    freq = sorted(asp.values(), reverse=True)
    cov = {K: 100.0 * sum(freq[:K]) / tot for K in (8, 16, 32, 64)}
    divs = []
    for a in areas:
        ai = int(round(a))
        if abs(a - ai) > 1e-6 or ai <= 0:
            divs.append(0)
            continue
        wlo = max(1, int(math.ceil(math.sqrt(ai / 3.0) - 1e-9)))
        whi = int(math.floor(math.sqrt(ai * 3.0) + 1e-9))
        divs.append(sum(1 for w_ in range(wlo, whi + 1) if ai % w_ == 0))
    divs.sort()
    print(f"cases={cases}  movable blocks={seen_mov}")
    print(f"W/H integer: {100.0 * intWH[0] / max(intWH[1], 1):.2f}%   "
          f"X/Y integer: {100.0 * intXY[0] / max(intXY[1], 1):.2f}%   "
          f"area==W*H exact: {100.0 * areaEx[0] / max(areaEx[1], 1):.2f}%   "
          f"max coord: {maxcoord:.0f}")
    print(f"movable aspect: distinct(4dp)={len(asp)}  "
          f"top-K coverage: " + "  ".join(f"K={K}:{cov[K]:.1f}%"
                                          for K in (8, 16, 32, 64)))
    if divs:
        print(f"integer divisor-pairs per movable area (aspect in [1/3,3]): "
              f"p25={_pctl(divs, 25):.0f} p50={_pctl(divs, 50):.0f} "
              f"p75={_pctl(divs, 75):.0f} p95={_pctl(divs, 95):.0f} "
              f"(n={len(divs)} sampled)")
    print(f"per-case means: preplaced={consn['preplaced'] / max(cases, 1):.2f} "
          f"fixed={consn['fixed'] / max(cases, 1):.2f} "
          f"boundary={consn['boundary'] / max(cases, 1):.2f} "
          f"mib_groups={consn['mib_groups'] / max(cases, 1):.2f} "
          f"clusters={consn['clusters'] / max(cases, 1):.2f}")
    print(f"MIB groups with uniform dims: {mib_uni[0]}/{mib_uni[1]} "
          f"({100.0 * mib_uni[0] / max(mib_uni[1], 1):.1f}%)")
    if cov[16] >= 90.0:
        rec = "16-bin aspect head covers >=90% of movables -> aspect-K-bin head"
    elif divs and _pctl(divs, 50) <= 8:
        rec = ("aspect bins cover poorly but divisor-pairs are few -> "
               "integer-W pointer head")
    else:
        rec = ("use aspect-K-bin head + exact-area decode; quant loss is "
               "bounded by slack dims_q row")
    print(f"[vocab] recommendation: {rec}")
    return dict(cov=cov, rec=rec)


# --------------------------------------------------------------------------- #
# all + verdict                                                                #
# --------------------------------------------------------------------------- #
def cmd_all(quick):
    t0 = time.time()
    g1 = cmd_pipeline(quick)
    print()
    g2 = cmd_slack(quick)
    print()
    g3 = cmd_legalize(quick)
    print()
    gv = cmd_vocab(quick)
    print()
    print("=" * 62)
    print("M52 PHASE 0 VERDICT" + ("  (QUICK)" if quick else ""))
    print("=" * 62)
    print(f"(i)   pipeline reproduce : {'PASS' if g1['ok'] else 'FAIL'}   "
          f"exact {g1['g1a']:.2f}% costEq {g1['g1b']:.2f}% "
          f"fpFeas {g1['g1c']:.2f}% ({g1['cases']} cases)")
    print(f"(ii)  slack steepness    : tree_any {g2['tree_any'][0]:.4f}"
          f"({g2['tree_any'][1]})  tree_mov {g2['tree_mov'][0]:.4f}"
          f"({g2['tree_mov'][1]})  Y {g2['Y'][0]:.4f}({g2['Y'][1]})  "
          f"dims_q16 {g2['dims_q'][0]:.4f}({g2['dims_q'][1]})")
    print(f"      single-token limit : tree_leaf1 wR={g2['leaf1'][0]:.4f} "
          f"wRq={g2['leaf1'][1]:.4f} | tree_near1 wR={g2['near1'][0]:.4f} "
          f"wRq={g2['near1'][1]:.4f}")
    print(f"(iii) legalizer repair   : {'PASS' if g3['ok'] else 'FAIL'}   "
          f"postFeas {g3['post_feas']:.1f}%  selfInfl "
          f"{g3['self_infl'] * 100:+.2f}%  model {g3['model'] * 100:+.2f}% "
          f"vs interact {g3['inter'] * 100:+.2f}%")
    print("-" * 62)
    steep_ty = (g2["tree_any"][1] == "STEEP" and g2["Y"][1] == "STEEP")
    if not g1["ok"]:
        verdict = "DEAD (wiring) -> fix decoder/label reading before anything"
    elif steep_ty and not g3["ok"]:
        verdict = "NO-GO tree-space -> switch to direct-layout model B"
    else:
        verdict = "GO Phase 1 (imitation training script on user CUDA)"
    print(f"RULE: (i) FAIL -> DEAD | (ii) tree&Y STEEP and (iii) FAIL -> "
          f"model B | else GO")
    print(f"==> {verdict}")
    print(f"[total wall {time.time() - t0:.0f}s; cache "
          f"{len(_C['metrics'])} entries]")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    quick = "quick" in sys.argv[2:]
    _cload()
    t0 = time.time()
    if mode == "pipeline":
        cmd_pipeline(quick)
    elif mode == "slack":
        cmd_slack(quick)
    elif mode == "legalize":
        cmd_legalize(quick)
    elif mode == "vocab":
        cmd_vocab(quick)
    elif mode == "all":
        cmd_all(quick)
    else:
        print(__doc__)
        return
    _csave()
    if mode != "all":
        print(f"[wall {time.time() - t0:.0f}s; cache {len(_C['metrics'])} "
              f"entries]")


if __name__ == "__main__":
    main()
