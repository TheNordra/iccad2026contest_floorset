"""OFFLINE ONLY — M80 knob-cloud tier identity gate (never shipped).

The M80 tier appends K profiles past the shipped prefix and switches them on only
at >= _M80_CORES_MIN detected cores, because the wall is
max(max_i dt_i, sum_i dt_i / cores): at 48 cores 100/100 cases are max-setter
bound so K extra profiles cost ~nothing, while at 12 cores the sum term dominates
and M79 measured dRF +10.614% at K=8 with every case getting a higher wall.

It is the SECOND tier gated on a high core count (after M67-F tier-5), which is
what makes each check below fallible in its own way:

  V1  inert below the threshold — a tier that leaked at 12-16 cores would be a
      large net LOSS, not a small one, so this is the check that protects the
      local box, WSL and any grader with modest parallelism.
  V2  exact blast radius at 40/48/96c: the pool gains exactly _M80_IDX and the
      M49/M50 REFINE overlay does not move with it.
  V3  the shipped prefix is untouched — _PROFILES[:_M55_BASE_LEN] identical to
      HEAD, and the pool restricted to that prefix identical with the tier on and
      off. This is the invariant that keeps audit_cache*.pkl / m67_oos_cache*.pkl
      / m77_oos_audit.pkl valid (all four anchor their signature on the prefix),
      and it is why the vectors are appended rather than merged into the 41.
  V4  fail-CLOSED: _effective_cores_hi() must map detection failure to 0, not to
      _effective_cores()'s 9999 sentinel, or the tier fires exactly where the
      machine could not be measured.
  V5  vector identity: _M80_EXTRA must equal m80_vectors.json verbatim. The
      vectors come out of a seeded cloud whose contents depend on the shipped
      prefix, so without a pinned file "#100" is a moving target.
  V6  the tier is reachable: no M80 profile may carry ORDER_SWAP/ORDER_MOVE,
      which M41 filters by CONTENT, and every M80 index must receive the M71
      overlay it was measured under.

Re-run after ANY change to _M80_EXTRA, the cores helpers or the tier constants.
Exit 0 = PASS.

Usage:  <python> m80_tier_gate.py
"""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_DIR = Path(__file__).parent
NS = range(1, 131)
VECFILE = _DIR / "m80_vectors.json"

# Symmetric to m67g_tier5_gate.py's _ISOLATE: pin the OTHER high-core tier so a
# difference set here can only be M80's.
_ISOLATE = {"ICCAD_M67F_TIER5": "0"}


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
    env = dict(_ISOLATE)
    env.update(extra or {})
    for k, v in env.items():
        os.environ[k] = v
    out = {n: (tuple(mod._pool_indices(n)),
               tuple(sorted(mod._band_env(n).items()))) for n in NS}
    for k in env:
        os.environ.pop(k, None)
    os.environ.pop("ICCAD_ADAPTIVE_CORES", None)
    return out


def main() -> int:
    for k in [k for k in os.environ if k.startswith("ICCAD_")]:
        del os.environ[k]
    sys.path.insert(0, str(_DIR))

    live = _load(_DIR / "optimizer_constructive.py", "oc_live_m80")
    if not hasattr(live, "_M80_IDX"):
        print("FAIL: optimizer_constructive.py has no M80 tier")
        return 1
    on, off = {"ICCAD_M80_TIER": "1"}, {"ICCAD_M80_TIER": "0"}
    want = sorted(live._M80_IDX)
    cmin = live._M80_CORES_MIN
    print(f"pool={len(live._PROFILES)} profiles; M80 tier = {len(want)} indices "
          f"{want[:3]}..{want[-1:]} at >={cmin} cores")
    fails = 0

    # ---- V1: inert everywhere below the threshold ---------------------------
    for cores in (None, 4, 8, 12, 16, 24, 32, cmin - 1):
        a = _sweep(live, cores, on)
        b = _sweep(live, cores, off)
        bad = [n for n in NS if a[n] != b[n]]
        tag = "auto" if cores is None else f"{cores}c"
        print(f"V1 M80 inert @ {tag:>5}: "
              f"{'PASS' if not bad else 'FAIL ' + str(bad[:6])}")
        fails += bool(bad)

    # ---- V2: exact blast radius above the threshold -------------------------
    for cores in (cmin, 48, 96):
        a, b = _sweep(live, cores, on), _sweep(live, cores, off)
        bad = [n for n in NS
               if sorted(set(a[n][0]) - set(b[n][0])) != want
               or set(b[n][0]) - set(a[n][0])
               or a[n][1] != b[n][1]]
        print(f"V2 @{cores:>3}c  adds exactly _M80_IDX, REFINE unchanged: "
              f"{'PASS' if not bad else 'FAIL ' + str(bad[:6])}"
              + ("" if bad else f"   pool {len(b[120][0])}->{len(a[120][0])}"))
        fails += bool(bad)
    # band gate: ICCAD_M80_MIN_N=100 must leave n<=100 exactly as the tier-off pool
    a = _sweep(live, 48, dict(on, ICCAD_M80_MIN_N="100"))
    b = _sweep(live, 48, off)
    bad = [n for n in NS if (a[n] != b[n]) != (n > 100)]
    print(f"V2 band gate MIN_N=100 splits at n=100: "
          f"{'PASS' if not bad else 'FAIL ' + str(bad[:6])}")
    fails += bool(bad)

    # ---- V3: the shipped prefix is untouched --------------------------------
    head_src = subprocess.run(["git", "show", "HEAD:optimizer_constructive.py"],
                              cwd=_DIR, capture_output=True, text=True,
                              encoding="utf-8").stdout
    if not head_src:
        print("V3 FAIL: could not read HEAD:optimizer_constructive.py")
        return 1
    hf = Path(tempfile.mkdtemp()) / "oc_head.py"
    hf.write_text(head_src, encoding="utf-8")
    head = _load(hf, "oc_head_m80")
    same_len = live._M55_BASE_LEN == head._M55_BASE_LEN
    same_pre = (repr(live._PROFILES[:live._M55_BASE_LEN])
                == repr(head._PROFILES[:head._M55_BASE_LEN]))
    print(f"V3 shipped prefix == HEAD ({live._M55_BASE_LEN} profiles): "
          f"{'PASS' if same_len and same_pre else 'FAIL'}"
          f"   <- this is what keeps the four offline caches valid")
    fails += (not (same_len and same_pre))
    a, b = _sweep(live, 48, on), _sweep(live, 48, off)
    bad = [n for n in NS
           if [i for i in a[n][0] if i < live._M55_BASE_LEN]
           != [i for i in b[n][0] if i < live._M55_BASE_LEN]]
    print(f"V3 prefix pool identical with the tier on/off @48c: "
          f"{'PASS' if not bad else 'FAIL ' + str(bad[:6])}")
    fails += bool(bad)
    # ADAPTIVE_POOL=0 must track the gate (M72's port leaked here; ours reads the
    # gate before the early return).
    f_on = _sweep(live, 48, dict(on, ICCAD_ADAPTIVE_POOL="0"))
    f_off = _sweep(live, 48, dict(off, ICCAD_ADAPTIVE_POOL="0"))
    ok = (len(f_off[120][0]) == live._M55_BASE_LEN
          and sorted(set(f_on[120][0]) - set(f_off[120][0])) == want)
    print(f"V3 ADAPTIVE_POOL=0 tracks the gate (no leak when off): "
          f"{'PASS' if ok else 'FAIL'}   {len(f_off[120][0])} -> "
          f"{len(f_on[120][0])}")
    fails += (not ok)

    # ---- V4: fail-closed ----------------------------------------------------
    real_hi = live._effective_cores_hi()
    saved = os.environ.pop("ICCAD_ADAPTIVE_CORES", None)
    orig_cpu = os.cpu_count
    orig_aff = getattr(os, "sched_getaffinity", None)
    try:
        os.cpu_count = lambda: (_ for _ in ()).throw(OSError("boom"))
        if orig_aff is not None:
            os.sched_getaffinity = lambda p: (_ for _ in ()).throw(OSError("boom"))
        hi_unknown = live._effective_cores_hi()
        os.environ["ICCAD_M80_TIER"] = "1"
        pool_unknown = len(live._pool_indices(120))
        os.environ.pop("ICCAD_M80_TIER", None)
    finally:
        os.cpu_count = orig_cpu
        if orig_aff is not None:
            os.sched_getaffinity = orig_aff
        if saved is not None:
            os.environ["ICCAD_ADAPTIVE_CORES"] = saved
    ok4 = hi_unknown < cmin and pool_unknown <= live._M55_BASE_LEN
    print(f"V4 fail-closed: detection-failure hi={hi_unknown} -> tier off "
          f"({hi_unknown < cmin}), n=120 pool {pool_unknown} <= "
          f"{live._M55_BASE_LEN} -> {'PASS' if ok4 else 'FAIL'}")
    fails += (not ok4)
    print(f"   this box: _effective_cores_hi()={real_hi} "
          f"(M80 fires: {real_hi >= cmin})")

    # ---- V5: vector identity ------------------------------------------------
    if not VECFILE.exists():
        print(f"V5 FAIL: {VECFILE.name} missing (run m79_knob_cloud_probe.py "
              f"greedy/dump)")
        fails += 1
    else:
        j = json.loads(VECFILE.read_text(encoding="utf-8"))
        vecs = j["vectors"][:len(live._M80_EXTRA)]
        same = (len(vecs) == len(live._M80_EXTRA)
                and all(a == b for a, b in zip(vecs, live._M80_EXTRA)))
        print(f"V5 _M80_EXTRA == {VECFILE.name} verbatim "
              f"(source={j.get('source')} R={j.get('R')} order={j.get('order')}): "
              f"{'PASS' if same else 'FAIL'}")
        fails += (not same)

    # ---- V6: the tier is actually reachable and carries the M71 overlay -----
    swap = [i for i in want if "ICCAD_ORDER_SWAP" in live._PROFILES[i]
            or "ICCAD_ORDER_MOVE" in live._PROFILES[i]]
    print(f"V6 no M80 profile carries ORDER_SWAP/MOVE (M41 filters by content): "
          f"{'PASS' if not swap else 'FAIL ' + str(swap)}")
    fails += bool(swap)
    knobs = set(live._M71_ENV)
    badm71 = [i for i in want if not knobs <= set(live._profile_env(i, 120))]
    print(f"V6 every M80 index gets the M71 overlay it was measured under: "
          f"{'PASS' if not badm71 else 'FAIL ' + str(badm71)}")
    fails += bool(badm71)

    print("\nM80-TIER GATE:", "ALL PASS" if fails == 0 else f"{fails} FAILURES")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
