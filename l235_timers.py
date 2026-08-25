"""L235 - build optimizer_l235t.py: optimizer_l235lp.py + phase timers.

cProfile charges per-call overhead to the caller, which on a function called
55k times per run is most of what it reports. These timers bracket whole PHASES
instead, so the numbers are what the LP actually spends.

Diagnostic only: optimizer_l235t.py is never A/B'd for identity and never ships.

  <python> l235_timers.py
  <python> l235_lpbench.py timers --mod optimizer_l235t
"""
import pathlib
import sys

SRC = pathlib.Path("optimizer_l235lp.py")
DST = pathlib.Path("optimizer_l235t.py")

OPEN_AT = "    units, unit_of, group_units, group_comp0 = decompose(ci, P)"

MARKS = [
    ("    b2l_items, p2l_items = _aggregate_pairwise_edges(c, unit_of)",
     "prologue"),
    ("    _uoa = np.fromiter(((-1 if _u is None else _u) for _u in unit_of),",
     "hpwl"),
    ("    keep_mask = (_sep_reduction_mask(_sep_ax, _sep_bi, _sep_bj, _sep_rhs,",
     "sep_build"),
    ("    xmin0 = min(P[i][0] for i in range(n))", "sep_reduce+emit"),
    ("    t_build0 = time.perf_counter()", "bnd+env+tangent"),
]


def main():
    s = SRC.read_text(encoding="utf-8")
    hdr = ("import collections as _l235c\n"
           "_L235T = _l235c.defaultdict(float)\n"
           "_L235N = _l235c.defaultdict(int)\n\n\n")
    if s.count(OPEN_AT) != 1:
        print("!! open anchor matched {} times".format(s.count(OPEN_AT)))
        return 1
    s = s.replace(OPEN_AT, "    _t0 = time.perf_counter()\n" + OPEN_AT)
    for anchor, name in MARKS:
        if s.count(anchor) != 1:
            print("!! anchor {!r} matched {} times".format(name, s.count(anchor)))
            return 1
        stamp = ('    _t1 = time.perf_counter()\n'
                 '    _L235T["{}"] += _t1 - _t0\n'
                 '    _L235N["{}"] += 1\n'
                 '    _t0 = _t1\n').format(name, name)
        s = s.replace(anchor, stamp + anchor)
    close_at = "    t_solve = time.perf_counter() - t_solve0"
    if s.count(close_at) != 1:
        print("!! close anchor matched {} times".format(s.count(close_at)))
        return 1
    s = s.replace(close_at,
                  close_at + '\n    _L235T["sparse"] += t_build'
                             '\n    _L235T["solve"] += t_solve')
    DST.write_text(hdr + s, encoding="utf-8", newline="\n")
    print("wrote {} with {} phase marks".format(DST, len(MARKS) + 2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
