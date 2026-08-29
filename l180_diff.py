"""L180 - find the 1.89x that is NOT M80, not the L124 twins, not L137, not M71.

L179: with all four off, the current wrapper runs the SAME 35 profiles as M73,
with the SAME constructive.exe, and takes 208.82s against M73's 110.73s on the
same box. That 1.89x is worth rank 5 -> rank 2, so it is the most valuable
unknown left.

Everything the wrapper hands the placer is (profile dict) + (band overlay) +
(M71 overlay) + (L137 overlay). Compare them side by side, per selected index.
If the env dicts differ, the placer is being asked to do more work and the diff
says exactly which knob.
"""
import os
import sys
from pathlib import Path

DIR = Path(__file__).resolve().parent
os.environ["ICCAD_ADAPTIVE_CORES"] = "48"
os.environ["ICCAD_M80_TIER"] = "0"
os.environ["ICCAD_M124_TWIN"] = "0"
os.environ["ICCAD_HINT_MODE"] = "0"

sys.path.insert(0, str(DIR / "iccad2026contest"))


def load(path, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


cur = load(DIR / "optimizer_constructive.py", "cur_opt")
m73 = load(DIR / "_m73win" / "optimizer_constructive.py", "m73_opt")

for n in (120, 100, 60):
    ic = list(cur._pool_indices(n))
    im = list(m73._pool_indices(n))
    print("=" * 74)
    print("n={}   current {} profiles   M73 {} profiles   same index set: {}"
          .format(n, len(ic), len(im), sorted(ic) == sorted(im)))
    if sorted(ic) != sorted(im):
        print("   only in current: {}".format(sorted(set(ic) - set(im))))
        print("   only in M73    : {}".format(sorted(set(im) - set(ic))))

    def eff(mod, i, nn):
        """M73 builds the overlay inline (band + M71); the current tree factored
        it into _profile_env. Reproduce M73's exactly rather than guess."""
        p = dict(mod._PROFILES[i])
        if hasattr(mod, "_profile_env"):
            p.update(mod._profile_env(i, nn))
        else:
            ov = dict(mod._band_env(nn))
            ov.update(mod._m71_env())
            p.update(ov)
        return p

    shared = sorted(set(ic) & set(im))
    diffs = {}
    for i in shared:
        a, b = eff(cur, i, n), eff(m73, i, n)
        for k in set(a) | set(b):
            if a.get(k) != b.get(k):
                diffs.setdefault(k, []).append((i, b.get(k), a.get(k)))
    if not diffs:
        print("   effective env IDENTICAL on all {} shared indices".format(len(shared)))
    else:
        print("   knobs that differ, on how many of the {} shared indices:"
              .format(len(shared)))
        for k, v in sorted(diffs.items(), key=lambda x: -len(x[1])):
            ex = v[0]
            print("     {:<28} {:>3} indices   e.g. idx {}: M73 {!r} -> now {!r}"
                  .format(k, len(v), ex[0], ex[1], ex[2]))
print("=" * 74)
print("base pool length: current {}  M73 {}"
      .format(cur._M55_BASE_LEN, m73._M55_BASE_LEN))
same = sum(1 for i in range(min(cur._M55_BASE_LEN, m73._M55_BASE_LEN))
           if cur._PROFILES[i] == m73._PROFILES[i])
print("_PROFILES[:41] identical entries: {} of {}".format(same, m73._M55_BASE_LEN))
