"""L274 -- turn the measured `l269p1` arm into a SHIPPABLE code default.

WHY A CODE DEFAULT AND NOT AN ENV VAR. L158: a mechanism that can only be turned
on by an environment variable is INERT inside the package -- the grader sets no
ICCAD_* at all. So shipping means the C++ default itself has to change.

WHY THIS SOURCE AND NOT A FRESH MINIMAL PATCH. The arm that was measured is
`constructive_l270.exe` run with ICCAD_L269=1 / ICCAD_L269_PROBES=1. If the ship
binary were re-derived from a new hand-written patch it would be a DIFFERENT
artefact from the one that carries the numbers, and the whole L267-L269 evidence
chain would no longer attach to it. So this takes the exact probe source and
flips only the two defaults -- which makes the equivalence checkable:

    l274_gate.py:  constructive_ship.exe with NO env
                == constructive_l270.exe with ICCAD_L269=1 ICCAD_L269_PROBES=1
       byte-for-byte, on every (case, profile) pair.

That gate is the reason to do it this way. It proves the shipped default IS the
measured configuration rather than something that resembles it.

The L267 / L268 branches stay in the source but are unreachable: they are read
only from getenv, the package sets no environment, and their statics default to
off. They are carried rather than stripped so that the file remains the measured
one; `make_submission._binary_matches_source()` requires every ICCAD_* the source
reads to appear literally in the ELF, and they all do.

    input   constructive_l267.cpp   (the other session's probe source)
    output  constructive_ship.cpp   (L269 = 1, L269_PROBES = 1 by default)
"""
import hashlib
import shutil
import sys
from pathlib import Path

DIR = Path(__file__).parent
SRC = DIR / "constructive_l267.cpp"
SNAP = DIR / "constructive_ship_src.cpp"      # frozen copy: the probe source can move
DST = DIR / "constructive_ship.cpp"

FLIPS = [
    ("static int  L269 = 0;            // ICCAD_L269: 1 = in-loop bisection, 2 = start loosest",
     "static int  L269 = 1;            // L274 SHIPPED DEFAULT (was 0, env-only). In-loop\n"
     "                                 // bisection: the trial loop proposes its own next\n"
     "                                 // frame between the loosest scale known to fail and\n"
     "                                 // the tightest known to pack, anchored on the aspect\n"
     "                                 // that just succeeded. A proposal that packs becomes\n"
     "                                 // a real trial. ICCAD_L269=0 is the kill switch."),
    # 🚨 Found by l274_gate.py G2, and it would have shipped silently. The probe's
    # parser is `if (v==1||v==2) L269=v;` -- a value of 0 is IGNORED. That is
    # harmless while the default is 0 (env-only), and becomes a missing KILL
    # SWITCH the moment the default is flipped to 1: ICCAD_L269=0 would leave the
    # mechanism on. Every shipped mechanism in this project has an off switch
    # (ICCAD_M71=0, ICCAD_M80_TIER=0, ICCAD_HINT_MODE=0, ICCAD_SHAPE_LP_DEPTH2=0),
    # and without one there is no way to produce the "off" control that every
    # future bit-equality gate needs.
    ('if (const char* e=getenv("ICCAD_L269")){ int v=atoi(e); if (v==1||v==2) L269=v; }',
     'if (const char* e=getenv("ICCAD_L269")){ int v=atoi(e); if (v>=0&&v<=2) L269=v; }'
     '   // L274: accept 0 = kill switch'),
    ("static int  L269_PROBES = 5;     // bisection proposals per pipeline",
     "static int  L269_PROBES = 1;     // L274 SHIPPED DEFAULT (was 5). Chosen on the\n"
     "                                 // ISOLATED cost column and the measured wall, not on\n"
     "                                 // the portfolio headline -- the headline ranks p3>p2>p1\n"
     "                                 // and NET ranks p1>p2>p3. Every proposal that packs\n"
     "                                 // consumes one of max_trials(=4), so a larger budget\n"
     "                                 // crowds the aspect search out: isolated cost is\n"
     "                                 // monotone in the budget and changes sign at p3."),
]


def main():
    if not SRC.exists():
        print("!! {} missing".format(SRC.name))
        return 1
    shutil.copyfile(SRC, SNAP)
    out = SNAP.read_bytes().decode("utf-8")
    for i, (old, new) in enumerate(FLIPS, 1):
        if out.count(old) != 1:
            print("!! flip {} matches {} times, expected 1".format(i, out.count(old)))
            return 1
        out = out.replace(old, new)
    DST.write_bytes(out.encode("utf-8"))
    print("snapshot {}  md5 {}".format(SNAP.name, hashlib.md5(SNAP.read_bytes()).hexdigest()))
    print("wrote    {}  md5 {}".format(DST.name, hashlib.md5(DST.read_bytes()).hexdigest()))
    print("  L269 default      0 -> 1")
    print("  L269_PROBES       5 -> 1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
