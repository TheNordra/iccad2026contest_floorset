"""L351 Gate 0 -- does the FORM of the greedy step score reach beyond M80's coefficient space?

THE QUESTION (pre-registered in L351_TIER3_EVAL.md sec.5). Tier-3 item 11 is "evolve the
packer's priority/tie-break function". It is the only Tier-3 item left alive, because it is
the only one that changes the packer's REACHABLE SET rather than its inputs (CLAUDE.md
conclusion 7: perfect order +0.005 %, perfect seed +0.001 %, perfect shape +0.099 %).

Its whole risk is one thing: **M80 already swept the COEFFICIENT space** -- 512 random joint
vectors, per-case oracle +3.081 %, held-out saturating and getting WORSE from R 256->512.
Item 11 bets that new functional FORMS reach further. This gate tests that bet before any
evolution machinery is built.

DECISION RULE, fixed in advance:

    oracle increment over the 512-vector cloud
      >= +0.5 %      form reaches beyond coefficients -> build the evolution loop
      +0.1 - 0.5 %   ambiguous -> measure held-out transfer before spending more
      <= +0.1 %      coefficient-saturated -> item 11 collapses into M80 -> CLOSE it

THE FORMS (constructive_l351.cpp, all default 0 => off-path bit-identical):
  WIREUNPL  the shipped wire term counts only ALREADY-PLACED neighbours; estimate the
            unplaced ones at the frame centre.  <- attacks the greedy short-sightedness
            L276/M78 diagnose, on hpwl, the one axis L349 found still has room
  WIRENORM  normalise wire by the item's own incident weight
  FILLNORM  make the area term's weight state-dependent (grows as the frame fills)
  ASPECT    penalise the resulting bbox drifting from the frame's aspect (a NEW term)

🔑 THE OFF-PATH GATE IS ALSO THE CACHE'S VALIDITY PROOF. `m79_knob_cloud_probe._sig()`
pins the exe md5, so a different binary would normally invalidate the 512-vector cloud.
Instead of regenerating 51 200 runs, this gate re-runs a SAMPLE of cached (case, vector)
jobs through `constructive_l351.exe` with no L351 flag set and requires the positions to
come back BIT-IDENTICAL. That simultaneously proves the flags are inert and that reusing
the cache under the new binary is legitimate.

  <python> l351_gate0.py gate     off-path bit-identity vs the cached cloud (fast)
  <python> l351_gate0.py base     score all 512 cloud vectors -> l351_gate0_cache.pkl
  <python> l351_gate0.py forms    run + score the form profiles
  <python> l351_gate0.py report   the oracle increment and the verdict
"""
import concurrent.futures as cf
import math
import pickle
import subprocess
import sys
import time
from pathlib import Path

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR))
import m79_knob_cloud_probe as kc  # noqa: E402  (loads the dataset and CASES)

EXE_L351 = _DIR / "constructive_l351.exe"
CACHE = _DIR / "l351_gate0_cache.pkl"
R = 512

# One flag per FORM, swept over magnitudes. Hosted on the shipped base profile so the
# comparison isolates FORM from COEFFICIENT: the cloud already contains every coefficient
# combination this host could have.
FORMS = []
for _mag in ("0.25", "1.0", "4.0"):
    FORMS.append(("wireunpl_" + _mag, {"ICCAD_L351_WIREUNPL": _mag}))
    FORMS.append(("wirenorm_" + _mag, {"ICCAD_L351_WIRENORM": _mag}))
for _mag in ("0.05", "0.2", "0.8"):
    FORMS.append(("fillnorm_" + _mag, {"ICCAD_L351_FILLNORM": _mag}))
    FORMS.append(("aspect_" + _mag, {"ICCAD_L351_ASPECT": _mag}))

# L353: ICCAD_WIRE_FOR_ALL is a SHIPPED flag (constructive.cpp:1122) that defaults OFF and
# gates whether wirelength is scored at all for candidates carrying a boundary miss. It is
# NOT in M80's 512-vector cloud, and the only place it appears in the tree is
# l271_quality.py, always COUPLED to ICCAD_L268=4. A flag only ever tested in combination
# has not been tested -- M80's "單獨死不代表聯合死", read the other way round.
FORMS.append(("wireforall", {"ICCAD_WIRE_FOR_ALL": "1"}))
for _m in ("0.5", "2.0"):
    FORMS.append(("wfa_wm_" + _m, {"ICCAD_WIRE_FOR_ALL": "1", "ICCAD_WIRE_MULT": _m}))


def _load():
    return pickle.load(open(CACHE, "rb")) if CACHE.exists() else {}


def _save(d):
    pickle.dump(d, open(CACHE, "wb"))


def _run_one(exe, case, env_extra):
    env = dict(kc._CHILD_BASE)
    env.update(env_extra)
    env.update(kc._overlay(case["n"]))
    out = subprocess.run([str(exe)], input=case["txt"], capture_output=True,
                         text=True, env=env).stdout
    return kc._parse_output(out, case["n"])


def mode_gate(nsample=24):
    """constructive_l351.exe with no L351 flag must reproduce the cached cloud positions
    bit-for-bit. Proves the flags are inert AND that the cache may be reused."""
    if not EXE_L351.exists():
        sys.exit("constructive_l351.exe missing -- build it first (PowerShell)")
    cloud = kc.build_cloud(R)
    data = pickle.load(open(kc.CACHE, "rb"))["data"]
    # the cache carries a few non-(case, profile) bookkeeping keys; keep only real jobs
    keys = sorted(k for k in data
                  if isinstance(k, tuple) and len(k) == 2 and isinstance(k[0], int))
    step = max(1, len(keys) // nsample)
    sel = keys[::step][:nsample]
    print("== L351 Gate 0 -- off-path bit-identity (%d sampled cloud jobs) ==" % len(sel))
    # Compare against the LIVE shipped binary, not against the cached positions. The
    # cache pins an exe md5 in its own _sig(); if it is stale, comparing to it tests the
    # cache's age rather than the flags' inertness. (Learned the hard way: the first
    # version of this gate reported 6/10 "mismatches" that the shipped exe reproduced
    # exactly -- the flags were inert all along and the cache was pre-L124.)
    csig = pickle.load(open(kc.CACHE, "rb")).get("sig")
    print("   cloud cache sig %s vs live %s  -> %s"
          % (csig, kc._sig(), "current" if csig == kc._sig() else "STALE"))
    pk = {kc._pkey(p): p for p in cloud}
    bad = 0
    checked = 0
    for ci, ph in sel:
        prof = pk.get(ph)
        if prof is None:
            continue
        got = _run_one(EXE_L351, kc.CASES[ci], prof)
        want = _run_one(kc.EXE, kc.CASES[ci], prof)
        checked += 1
        if got != want:
            bad += 1
            print("   MISMATCH case %d profile %s" % (ci, ph))
    print("   checked %d   mismatches %d   %s"
          % (checked, bad, "PASS" if (checked and not bad) else "*** FAIL ***"))
    if not checked:
        print("   nothing checked -- cloud/profile keys did not line up; stopping.")
        return 1
    return 0 if not bad else 1


def _score_jobs(jobs, exe, label):
    """jobs = [(key, case_idx, env)] -> fills the cost cache."""
    C = _load()
    todo = [j for j in jobs if j[0] not in C]
    print("   %s: %d jobs, %d already cached" % (label, len(jobs), len(jobs) - len(todo)))
    if not todo:
        return C
    t0 = time.time()
    done = 0
    with cf.ThreadPoolExecutor(max_workers=kc.WORKERS) as ex:
        futs = {ex.submit(_run_one, exe, kc.CASES[ci], env): (k, ci)
                for k, ci, env in todo}
        for f in cf.as_completed(futs):
            k, ci = futs[f]
            try:
                pos = f.result()
                cost, feas = kc._true(kc.CASES[ci], pos)
            except Exception:
                cost, feas = float("inf"), False
            C[k] = (cost if feas else float("inf"))
            done += 1
            if done % 500 == 0:
                print("      %d/%d  %.0fs" % (done, len(todo), time.time() - t0),
                      flush=True)
                _save(C)
    _save(C)
    print("   %s done in %.0fs" % (label, time.time() - t0))
    return C


def mode_base():
    """Score every cached cloud run. Positions are already on disk; only the official
    scoring has to be paid."""
    cloud = kc.build_cloud(R)
    data = pickle.load(open(kc.CACHE, "rb"))["data"]
    C = _load()
    todo = []
    for ki, p in enumerate(cloud):
        ph = kc._pkey(p)
        for ci in range(len(kc.CASES)):
            k = ("cloud", ci, ph)
            if k in C or (ci, ph) not in data:
                continue
            todo.append((k, ci, ph))
    print("== L351 Gate 0 -- scoring the 512-vector cloud (%d to do) ==" % len(todo))
    t0 = time.time()
    for i, (k, ci, ph) in enumerate(todo):
        pos = data[(ci, ph)][0]
        try:
            cost, feas = kc._true(kc.CASES[ci], pos)
        except Exception:
            cost, feas = float("inf"), False
        C[k] = cost if feas else float("inf")
        if (i + 1) % 2000 == 0:
            print("   %d/%d  %.0fs" % (i + 1, len(todo), time.time() - t0), flush=True)
            _save(C)
    _save(C)
    print("   done in %.0fs" % (time.time() - t0))
    return 0


def mode_forms():
    jobs = [(("form", ci, name), ci, env)
            for name, env in FORMS for ci in range(len(kc.CASES))]
    print("== L351 Gate 0 -- running %d form profiles x %d cases =="
          % (len(FORMS), len(kc.CASES)))
    _score_jobs(jobs, EXE_L351, "forms")
    return 0


def mode_report():
    """The 512-vector cloud's PER-CASE oracle is already written by
    `m79_knob_cloud_probe.py run` into m79_knob_cloud_oraclepick.json (cost per case =
    the min over the 512). So the whole `base` step is unnecessary -- only the forms
    have to be run and scored."""
    import json
    op = _DIR / "m79_knob_cloud_oraclepick.json"
    if not op.exists():
        sys.exit("m79_knob_cloud_oraclepick.json missing -- run "
                 "`m79_knob_cloud_probe.py run 512` first")
    J = json.load(open(op))
    cloud_cost = {r["test_id"]: (r["cost"] if r.get("is_feasible", True)
                                 else float("inf")) for r in J["test_results"]}
    C = _load()
    W = kc.TOTW
    tot_cloud = tot_all = 0.0
    wins, missing = [], 0
    for ci, c in enumerate(kc.CASES):
        mc = cloud_cost.get(ci)
        if mc is None:
            missing += 1
            continue
        ff = [(nm, C[("form", ci, nm)]) for nm, _ in FORMS
              if ("form", ci, nm) in C and C[("form", ci, nm)] < float("inf")]
        ma = mc
        if ff:
            bn, bv = min(ff, key=lambda z: z[1])
            if bv < mc:
                ma = bv
                wins.append((c["n"], bn, 100 * (1 - bv / mc)))
        tot_cloud += c["w"] * mc
        tot_all += c["w"] * ma
    Tc, Ta = tot_cloud / W, tot_all / W
    inc = 100 * (1 - Ta / Tc)
    print("== L351 Gate 0 -- VERDICT ==")
    print("   cloud baseline : m79_knob_cloud_oraclepick.json (512 vectors, CURRENT exe)")
    print("   its own total  : %.9f   (report says %.9f)"
          % (Tc, J.get("total_score", float("nan"))))
    print("   cases missing a cloud oracle: %d" % missing)
    print("   form profiles run: %d x %d cases" % (len(FORMS), len(kc.CASES)))
    print()
    print("   per-case oracle, 512-vector cloud                %.9f" % Tc)
    print("   per-case oracle, cloud + %-2d form profiles        %.9f" % (len(FORMS), Ta))
    print("   *** ORACLE INCREMENT FROM THE FORMS: %+.4f %% ***" % inc)
    print()
    print("   cases where a form beats the ENTIRE cloud: %d/%d"
          % (len(wins), len(kc.CASES)))
    for n, nm, d in sorted(wins, key=lambda z: -z[2])[:12]:
        print("      n=%-4d %-16s %+.3f %%" % (n, nm, d))
    # which forms ever win, and by how much in aggregate
    from collections import Counter
    cnt = Counter(nm for _, nm, _ in wins)
    if cnt:
        print()
        print("   wins by form: %s" % dict(cnt.most_common()))
    print()
    if inc >= 0.5:
        v = "BUILD THE EVOLUTION LOOP -- form reaches beyond coefficients"
    elif inc > 0.1:
        v = "AMBIGUOUS -- measure held-out transfer before spending more"
    else:
        v = "CLOSE ITEM 11 -- coefficient-saturated, it collapses into M80"
    print("   pre-registered rule (>=0.5 / 0.1-0.5 / <=0.1)  =>  **%s**" % v)
    return 0


if __name__ == "__main__":
    m = sys.argv[1] if len(sys.argv) > 1 else "gate"
    sys.exit({"gate": mode_gate, "base": mode_base,
              "forms": mode_forms, "report": mode_report}[m]())
