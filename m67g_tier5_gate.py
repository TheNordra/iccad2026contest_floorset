"""OFFLINE ONLY — M67-F tier-5 pool-identity gate (never shipped).

Tier-5 restores the M42 `_BIG_REDUNDANT_IDX` cut when >=_M67F_CORES_MIN (40)
cores are detected, because past the restored pool's c* = sum(dt)/max(dt)
(min 15.5 / median 19.3 / MAX 22.5 over the 20 n>100 cases) the wall is the
max-setter and the cost saturates at +6.0%, while the cut still pays
theta_pool = 0.7636 of the +2.825% out-of-sample quality tax (M67-F Phase 1).
Below c* the wall grows like sum/cores and gets expensive fast (measured: on a
16-logical box whose EFFECTIVE parallelism is only ~10, restoring doubled the
n>100 wall), which is why the threshold carries ~1.8x headroom over c*max.
It is the FIRST tier whose gate fires on a HIGH core count, which makes two
things newly fallible:

  1. fail direction — `_effective_cores()` maps unknown to 9999 so tier-4
     (cores <= 8) stays off; reusing it for a `>= 32` test would make tier-5
     fire wherever detection FAILS. Tier-5 therefore uses `_effective_cores_hi()`
     (unknown -> 0). This gate asserts both directions.
  2. blast radius — tier-5 must touch ONLY the M42 layer on n > _free_n. The mid
     band (tier-3) is ship-RED and must stay cut; the M49/M50 REFINE overlay must
     not move with it.

Checks (all vs the committed HEAD copy of optimizer_constructive.py, n=1..130):
  V1  pool+overlay identical to HEAD at auto/4/8/12/16/24/31 cores and with the
      ICCAD_M67F_TIER5=0 kill switch at 48c
  V2  at 32/48/96 cores: n>100 pool == the offline ICCAD_M67F_RESTORE=1 pool,
      n<=100 pool == shipped
  V3  restored set == _BIG_REDUNDANT_IDX exactly; REFINE overlay unchanged
  V4  fail-closed: _effective_cores_hi() returns 0 (not 9999) when detection
      raises, so tier-5 stays off

Re-run after ANY change to _PROFILES, _BIG_REDUNDANT_IDX, the cores helpers, or
the tier constants. Exit 0 = PASS.

Usage:  <python> m67g_tier5_gate.py
"""
import os
import subprocess
import sys
import tempfile
import importlib.util
from pathlib import Path

_DIR = Path(__file__).parent
NS = range(1, 131)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sweep(mod, cores=None, extra=None):
    """(pool, refine-overlay) for every n under a forced core count."""
    if cores is None:
        os.environ.pop("ICCAD_ADAPTIVE_CORES", None)
    else:
        os.environ["ICCAD_ADAPTIVE_CORES"] = str(cores)
    for k, v in (extra or {}).items():
        os.environ[k] = v
    out = {n: (tuple(mod._pool_indices(n)),
               tuple(sorted(mod._band_env(n).items()))) for n in NS}
    for k in (extra or {}):
        os.environ.pop(k, None)
    os.environ.pop("ICCAD_ADAPTIVE_CORES", None)
    return out


def main() -> int:
    for k in [k for k in os.environ if k.startswith("ICCAD_")]:
        del os.environ[k]
    sys.path.insert(0, str(_DIR))

    head_src = subprocess.run(["git", "show", "HEAD:optimizer_constructive.py"],
                              cwd=_DIR, capture_output=True, text=True,
                              encoding="utf-8").stdout
    if not head_src:
        print("FAIL: could not read HEAD:optimizer_constructive.py")
        return 1
    hf = Path(tempfile.mkdtemp()) / "oc_head.py"
    hf.write_text(head_src, encoding="utf-8")

    live = _load(_DIR / "optimizer_constructive.py", "oc_live_t5")
    head = _load(hf, "oc_head_t5")
    if len(live._PROFILES) != len(head._PROFILES):
        print(f"FAIL: pool size drift live={len(live._PROFILES)} "
              f"head={len(head._PROFILES)}")
        return 1
    head_has_t5 = hasattr(head, "_effective_cores_hi")
    print(f"pool={len(live._PROFILES)} profiles; HEAD already has tier-5: {head_has_t5}")

    fails = 0

    # ---- V1: inert everywhere it must be inert ------------------------------
    # Reference = tier-5 switched OFF. Once tier-5 is committed, HEAD contains it
    # too, so comparing against HEAD is vacuous; the kill switch is the invariant
    # that keeps testing "tier-5 does nothing below _M67F_CORES_MIN" forever.
    # HEAD is still used as a second reference while it predates tier-5.
    for cores in (None, 4, 8, 12, 16, 24, 32, 39):
        a = _sweep(live, cores)
        b = _sweep(live, cores, {"ICCAD_M67F_TIER5": "0"})
        bad = [n for n in NS if a[n] != b[n]]
        tag = "auto" if cores is None else f"{cores}c"
        extra = ""
        if not head_has_t5:
            h = _sweep(head, cores)
            hbad = [n for n in NS if a[n] != h[n]]
            fails += bool(hbad)
            extra = f"; ==HEAD: {'PASS' if not hbad else 'FAIL ' + str(hbad[:6])}"
        print(f"V1 tier-5 inert @ {tag:>5}: "
              f"{'PASS' if not bad else 'FAIL ' + str(bad[:6])}{extra}")
        fails += bool(bad)
    # and the switch must also neutralise it ABOVE the threshold.
    # M74: this used to compare 48c vs 12c and demand equality on every n. That
    # premise died when tier-3 became cores-gated (_M45_MID_CORES_MAX): with
    # tier-5 off, 48c and 12c are now LEGITIMATELY different on the mid band,
    # because tier-3 fires at 12c and not at 48c. Split the claim in two so it
    # still isolates tier-5:
    #   (a) two core counts BOTH above the tier-3 threshold must agree on every
    #       n -- nothing else varies up there, so any diff is tier-5 leaking;
    #   (b) n>100, where tier-3 never applies, must still match the 12c pool.
    mid_max = live._M45_MID_CORES_MAX
    a = _sweep(live, 48, {"ICCAD_M67F_TIER5": "0"})
    b = _sweep(live, mid_max + 1, {"ICCAD_M67F_TIER5": "0"})
    c12 = _sweep(live, 12, {"ICCAD_M67F_TIER5": "0"})
    bad = [n for n in NS if a[n] != b[n]]
    bad_big = [n for n in NS if n > 100 and a[n] != c12[n]]
    print(f"V1 kill-switch @48c   : "
          f"vs {mid_max + 1}c (tier-3 off both sides) "
          f"{'PASS' if not bad else 'FAIL ' + str(bad[:6])}; "
          f"n>100 vs 12c {'PASS' if not bad_big else 'FAIL ' + str(bad_big[:6])}")
    fails += bool(bad) + bool(bad_big)

    # ---- V2: at >=32c == restore knob on n>100, shipped on n<=100 -----------
    for cores in (40, 48, 96):
        t5 = _sweep(live, cores)
        knob = _sweep(live, cores, {"ICCAD_M67F_TIER5": "0",
                                    "ICCAD_M67F_RESTORE": "1"})
        base = _sweep(live, cores, {"ICCAD_M67F_TIER5": "0"})
        big_bad = [n for n in NS if n > 100 and t5[n][0] != knob[n][0]]
        mid_bad = [n for n in NS if n <= 100 and t5[n] != base[n]]
        print(f"V2 @{cores:>3}c  n>100 == restore-knob: "
              f"{'PASS' if not big_bad else 'FAIL ' + str(big_bad[:6])}   "
              f"n<=100 == shipped: "
              f"{'PASS' if not mid_bad else 'FAIL ' + str(mid_bad[:6])}")
        fails += bool(big_bad) + bool(mid_bad)

    # ---- V3: exact blast radius --------------------------------------------
    t5, base = _sweep(live, 48), _sweep(live, 12)
    want = sorted(live._BIG_REDUNDANT_IDX)
    for n in (101, 105, 120, 130):
        got = sorted(set(t5[n][0]) - set(base[n][0]))
        if got != want or t5[n][1] != base[n][1]:
            print(f"V3 n={n}: FAIL restored={got} refine {base[n][1]}->{t5[n][1]}")
            fails += 1
        else:
            print(f"V3 n={n}: PASS pool {len(base[n][0])}->{len(t5[n][0])}, "
                  f"restored == _BIG_REDUNDANT_IDX ({len(want)}), REFINE unchanged")

    # ---- V4: fail-closed ----------------------------------------------------
    real_hi, real_lo = live._effective_cores_hi(), live._effective_cores()
    saved = os.environ.pop("ICCAD_ADAPTIVE_CORES", None)
    import builtins
    orig_cpu = os.cpu_count
    orig_aff = getattr(os, "sched_getaffinity", None)
    try:
        os.cpu_count = lambda: (_ for _ in ()).throw(OSError("boom"))
        if orig_aff is not None:
            os.sched_getaffinity = lambda p: (_ for _ in ()).throw(OSError("boom"))
        hi_unknown, lo_unknown = live._effective_cores_hi(), live._effective_cores()
    finally:
        os.cpu_count = orig_cpu
        if orig_aff is not None:
            os.sched_getaffinity = orig_aff
        if saved is not None:
            os.environ["ICCAD_ADAPTIVE_CORES"] = saved
    ok4 = (hi_unknown < live._M67F_CORES_MIN) and (lo_unknown > live._M45_CORES_MAX)
    print(f"V4 fail-closed: detection-failure hi={hi_unknown} (tier5 off: "
          f"{hi_unknown < live._M67F_CORES_MIN}), lo={lo_unknown} (tier4 off: "
          f"{lo_unknown > live._M45_CORES_MAX}) -> {'PASS' if ok4 else 'FAIL'}")
    fails += (not ok4)
    print(f"   this box: _effective_cores_hi()={real_hi} "
          f"(tier5 fires: {real_hi >= live._M67F_CORES_MIN}), "
          f"_effective_cores()={real_lo}")

    print("\nM67G-TIER5 GATE:", "ALL PASS" if fails == 0 else f"{fails} FAILURES")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
