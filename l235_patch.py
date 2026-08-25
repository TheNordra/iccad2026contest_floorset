"""L235 - build optimizer_l235lp.py: the shape LP with its Python half rewritten.

EVERY edit here is required to be OUTPUT-IDENTICAL, not merely equivalent-ish.
The LP is massively degenerate (L119: Windows and Linux land on different optima
of the same program), so "the objective matched" is not a gate -- the gate is
that the triplet arrays handed to HiGHS are built in the SAME ORDER with the
SAME values, which makes the whole thing bit-identical by construction. Three
rules every patch below obeys:

  1. float accumulation order is never changed (`obj[col] += ...` runs in the
     same sequence, so the rounding is the same),
  2. row emission order is never changed (a row's index is len(bub) at the
     moment it is added),
  3. tie-breaks are never changed -- `max(cands, key=lambda t: t[0])` returns
     the FIRST maximal element, and np.argmax returns the first maximum too.

MEASURED PHASE MAP (21 cases n>=100, min-of-3, real timers not cProfile):

    solve            67.7%      <- untouchable without changing the answer
    hpwl             11.7%
    sep_build        11.0%
    sep_reduce+emit   3.9%
    sparse            2.7%
    prologue          1.9%
    bnd+env+tangent   1.1%
                            ceiling if all Python went to zero: 1.48x

The one structural fact the separation rewrite turns on: **only ~11% of the
candidate pairs survive the transitive reduction** (25,568 rows emitted from
~233k pairs over 37 builds), yet the original built a dict and a terms list for
every single one before the mask was even computed.

  <python> l235_patch.py
"""
import pathlib
import sys

SRC = pathlib.Path("optimizer_constructive.py")
DST = pathlib.Path("optimizer_l235lp.py")

PATCHES = []


def P(old, new, why):
    PATCHES.append((old, new, why))


# --------------------------------------------------------------------------
# 1. _sep_reduction_mask takes parallel arrays instead of a list of dicts, so
#    the caller never has to materialise the dicts.
# --------------------------------------------------------------------------
P('''def _sep_reduction_mask(rows, n, P, unit_of, sv, resh, rho):''',
  '''def _sep_reduction_mask(ax_l, bi_l, bj_l, rhs_l, n, P, unit_of, sv, resh,
                        rho):''',
  "_sep_reduction_mask: array signature")

P('''    keep = [True] * len(rows)
    for axis in (0, 1):
        idxs = [k for k, r in enumerate(rows) if r["axis"] == axis]''',
  '''    keep = [True] * len(ax_l)
    for axis in (0, 1):
        idxs = [k for k, a in enumerate(ax_l) if a == axis]''',
  "_sep_reduction_mask: axis selection on the array")

P('''        adj = [0] * n
        for k in idxs:
            if rows[k]["rhs"] >= 0.0:
                adj[rows[k]["bi"]] |= 1 << rows[k]["bj"]''',
  '''        adj = [0] * n
        for k in idxs:
            if rhs_l[k] >= 0.0:
                adj[bi_l[k]] |= 1 << bj_l[k]''',
  "_sep_reduction_mask: adjacency on the array")

P('''        for k in idxs:
            if via[rows[k]["bi"]] >> rows[k]["bj"] & 1:
                keep[k] = False

    for k, r in enumerate(rows):
        if not r["terms"] and r["rhs"] >= 0.0:
            keep[k] = False        # `0 <= nonneg`, true without the row
    return keep''',
  '''        for k in idxs:
            if via[bi_l[k]] >> bj_l[k] & 1:
                keep[k] = False

    # The original ended with `if not r["terms"] and r["rhs"] >= 0.0`. That
    # branch is unreachable and always was: a row exists only for a pair whose
    # two units DIFFER, so at most one of (ul, ur) is None and `terms` is never
    # empty. Dropping it removes a full pass over every candidate pair.
    return keep''',
  "_sep_reduction_mask: drop the unreachable empty-terms pass")

# --------------------------------------------------------------------------
# 2. bound-method locals for the triplet arrays, and per-axis size lookups
#    that replace dsize() on the hot paths.
# --------------------------------------------------------------------------
P('''    def add_ub(terms, rhs, origin):
        r = len(bub)
        bub.append(rhs)
        for col, coef in terms:
            rub.append(r), cub.append(col), vub.append(coef)
        rows_by_origin[origin] += 1''',
  '''    # L235: bound methods hoisted, and the throwaway 3-tuple that
    # `a.append(x), b.append(y), c.append(z)` allocates on EVERY triplet is
    # gone. Same appends, same order.
    _rub_a = rub.append
    _cub_a = cub.append
    _vub_a = vub.append
    _bub_a = bub.append

    def add_ub(terms, rhs, origin):
        r = len(bub)
        _bub_a(rhs)
        for col, coef in terms:
            _rub_a(r)
            _cub_a(col)
            _vub_a(coef)
        rows_by_origin[origin] += 1''',
  "add_ub: hoisted appends, no tuple allocation")

P('''    def dsize(u, axis):
        if u is None or u not in sv:
            return None
        return sv[u][axis]''',
  '''    def dsize(u, axis):
        if u is None or u not in sv:
            return None
        return sv[u][axis]

    # L235: dsize() was called 141k times per census run purely to index a
    # 2-tuple. Split it once per axis; `.get()` returns None exactly where
    # dsize() did, because u is never None on the paths that use these.
    _svax = ({_u: _v[0] for _u, _v in sv.items()},
             {_u: _v[1] for _u, _v in sv.items()})
    # 0.5 * rho * P[resh[u]][2 + axis], which add_hpwl_rows recomputed for
    # every reshapeable unit of every hpwl term.
    _hslack = ({_u: 0.5 * rho * P[resh[_u]][2] for _u in sv},
               {_u: 0.5 * rho * P[resh[_u]][3] for _u in sv})''',
  "dsize: per-axis dicts + precomputed hpwl slack")

# --------------------------------------------------------------------------
# 3. add_hpwl_rows -- loop unrolled, dsize inlined, rows emitted straight into
#    the triplet arrays.
# --------------------------------------------------------------------------
P('''    def add_hpwl_rows(wsc, ui, uj, off, dC, axis):
        nonlocal prune_const
        tid = term_id[0]
        term_id[0] += 1
        lin = []
        slack = 0.0
        for u, s in ((ui, 1.0), (uj, -1.0)):
            if u is None:
                continue
            lin.append((off + u, s))
            slack += prune_B or 0.0
            k = dsize(u, axis)
            if k is not None:
                lin.append((k, 0.5 * s))
                slack += 0.5 * rho * P[resh[u]][2 + axis]''',
  '''    _pb0 = prune_B or 0.0

    def add_hpwl_rows(wsc, ui, uj, off, dC, axis):
        nonlocal prune_const, nv
        tid = term_id[0]
        term_id[0] += 1
        lin = []
        _ap = lin.append
        slack = 0.0
        _svk = _svax[axis]
        _hsk = _hslack[axis]
        # L235: the two-element loop unrolled in its original order (ui with
        # s=+1 first, then uj with s=-1), so `lin` is built identically and the
        # float accumulation below sees the same sequence.
        if ui is not None:
            _ap((off + ui, 1.0))
            slack += _pb0
            k = _svk.get(ui)
            if k is not None:
                _ap((k, 0.5))
                slack += _hsk[ui]
        if uj is not None:
            _ap((off + uj, -1.0))
            slack += _pb0
            k = _svk.get(uj)
            if k is not None:
                _ap((k, -0.5))
                slack += _hsk[uj]''',
  "add_hpwl_rows: unrolled, dsize/slack precomputed")

P('''        prune_stat[1] += 1
        t = new_aux(wsc)
        t1 = [(t, -1.0)] + lin
        t2 = [(t, -1.0)] + [(col, -coef) for col, coef in lin]
        add_ub(t1, -dC, "hpwl")
        add_ub(t2, dC, "hpwl")''',
  '''        prune_stat[1] += 1
        # L235: new_aux + two add_ub calls inlined. Identical emission order:
        # row -dC first with (t,-1) then lin, row +dC second with (t,-1) then
        # the negated lin -- exactly what add_ub(t1)/add_ub(t2) produced.
        obj.append(wsc)
        nv += 1
        t = nv - 1
        r1 = len(bub)
        _bub_a(-dC)
        _rub_a(r1)
        _cub_a(t)
        _vub_a(-1.0)
        for col, coef in lin:
            _rub_a(r1)
            _cub_a(col)
            _vub_a(coef)
        r2 = len(bub)
        _bub_a(dC)
        _rub_a(r2)
        _cub_a(t)
        _vub_a(-1.0)
        for col, coef in lin:
            _rub_a(r2)
            _cub_a(col)
            _vub_a(-coef)
        rows_by_origin["hpwl"] += 2''',
  "add_hpwl_rows: emit triplets directly")

# --------------------------------------------------------------------------
# 4. the separation rows. The pair enumeration, the four gaps and the argmax
#    go to numpy; the terms are built ONLY for the ~11% that survive.
# --------------------------------------------------------------------------
P('''    sep_rows = []
    for i in range(n):
        xi, yi, wi, hi = P[i]
        for j in range(i + 1, n):
            ui, uj = unit_of[i], unit_of[j]
            if ui == uj:
                continue
            xj, yj, wj, hj = P[j]
            cands = (
                (xj - (xi + wi), ui, uj, 0, 0, i, j),
                (xi - (xj + wj), uj, ui, 0, 0, j, i),
                (yj - (yi + hi), ui, uj, U, 1, i, j),
                (yi - (yj + hj), uj, ui, U, 1, j, i),
            )
            # key is t[0] only, so the extra block ids cannot change the pick
            gap, ul, ur, off, axis, bl, br = max(cands, key=lambda t: t[0])
            terms = []
            if ul is not None:
                terms.append((off + ul, 1.0))
                k = dsize(ul, axis)
                if k is not None:
                    terms.append((k, 1.0))
            if ur is not None:
                terms.append((off + ur, -1.0))
            sep_rows.append({"axis": axis, "bi": bl, "bj": br,
                             "terms": terms, "rhs": gap})

    keep_mask = (_sep_reduction_mask(sep_rows, n, P, unit_of, sv, resh, rho)
                 if sep_rows else [])
    sep_kept = sum(1 for x in keep_mask if x)
    for row, kf in zip(sep_rows, keep_mask if sep_trim else [True] * len(keep_mask)):
        if kf:
            add_ub(row["terms"], row["rhs"], "separation")''',
  '''    # L235: the pair enumeration, the four gaps and the argmax move to numpy;
    # only the rows the reduction KEEPS are ever turned into coefficients.
    # Measured: 25,568 rows survive out of ~233k candidate pairs (11%), and the
    # original built a dict and a terms list for all of them.
    #
    # Exactness, three points:
    #   * np.triu_indices(n, 1) enumerates (0,1),(0,2)...(1,2)... i.e. exactly
    #     the order the double loop produced, which the mask indexes into;
    #   * np.argmax returns the FIRST maximum, which is what
    #     `max(cands, key=lambda t: t[0])` did on ties;
    #   * every arithmetic op is the same IEEE double subtraction on the same
    #     operands -- `_Rx = _Px + _Pw` is `xi + wi`, not a re-association.
    # unit ids are non-negative, so mapping None -> -1 keeps None == None (the
    # pair the original skipped) and cannot collide with a real unit.
    # pick 0,2 keep (i, j); pick 1,3 swap to (j, i); axis is pick >> 1.
    _uoa = np.fromiter(((-1 if _u is None else _u) for _u in unit_of),
                       dtype=np.int64, count=n)
    _Px = np.fromiter((P[_k][0] for _k in range(n)), dtype=np.float64, count=n)
    _Py = np.fromiter((P[_k][1] for _k in range(n)), dtype=np.float64, count=n)
    _Pw = np.fromiter((P[_k][2] for _k in range(n)), dtype=np.float64, count=n)
    _Ph = np.fromiter((P[_k][3] for _k in range(n)), dtype=np.float64, count=n)
    _Rx = _Px + _Pw
    _Ty = _Py + _Ph
    _I, _J = np.triu_indices(n, 1)
    _kp = _uoa[_I] != _uoa[_J]
    _I = _I[_kp]
    _J = _J[_kp]
    if _I.size:
        _g = np.empty((4, _I.size), dtype=np.float64)
        _g[0] = _Px[_J] - _Rx[_I]
        _g[1] = _Px[_I] - _Rx[_J]
        _g[2] = _Py[_J] - _Ty[_I]
        _g[3] = _Py[_I] - _Ty[_J]
        _pick = _g.argmax(axis=0)
        _sep_rhs = _g[_pick, np.arange(_I.size)].tolist()
        _even = (_pick & 1) == 0
        _sep_ax = (_pick >> 1).tolist()
        _sep_bi = np.where(_even, _I, _J).tolist()
        _sep_bj = np.where(_even, _J, _I).tolist()
    else:
        _sep_rhs, _sep_ax, _sep_bi, _sep_bj = [], [], [], []

    keep_mask = (_sep_reduction_mask(_sep_ax, _sep_bi, _sep_bj, _sep_rhs,
                                     n, P, unit_of, sv, resh, rho)
                 if _sep_rhs else [])
    sep_kept = sum(1 for x in keep_mask if x)
    _emit = ([_k for _k in range(len(keep_mask)) if keep_mask[_k]] if sep_trim
             else range(len(keep_mask)))
    _nsep = 0
    for _k in _emit:
        axis = _sep_ax[_k]
        off = 0 if axis == 0 else U
        ul = unit_of[_sep_bi[_k]]
        ur = unit_of[_sep_bj[_k]]
        r = len(bub)
        _bub_a(_sep_rhs[_k])
        if ul is not None:
            _rub_a(r)
            _cub_a(off + ul)
            _vub_a(1.0)
            _kk = _svax[axis].get(ul)
            if _kk is not None:
                _rub_a(r)
                _cub_a(_kk)
                _vub_a(1.0)
        if ur is not None:
            _rub_a(r)
            _cub_a(off + ur)
            _vub_a(-1.0)
        _nsep += 1
    if _nsep:
        rows_by_origin["separation"] += _nsep''',
  "separation: numpy enumeration, terms built only for survivors")

P('''        sep_rows_total=len(sep_rows),''',
  '''        sep_rows_total=len(_sep_rhs),''',
  "telemetry: sep_rows_total off the array")


# --------------------------------------------------------------------------
# 5. add_hpwl_rows again -- `lin` as a ONE-ALLOCATION tuple instead of a list
#    plus 2-4 appends plus the tuple() copy `dropped` needed, and the two
#    loop-invariant multiplications hoisted. ~682k calls per census run, so an
#    allocation removed here is removed 682k times.
#
#    Float exactness: the original accumulated
#        slack = 0.0; slack += pb0; [slack += hs_i]; slack += pb0; [slack += hs_j]
#    and `0.0 + pb0` is exactly pb0, so the flat expressions below associate
#    left-to-right in the same order. `wsc * sgn * coef` is `(wsc*sgn)*coef`,
#    which is exactly what hoisting wss computes.
# --------------------------------------------------------------------------
P('''    _pb0 = prune_B or 0.0

    def add_hpwl_rows(wsc, ui, uj, off, dC, axis):
        nonlocal prune_const, nv
        tid = term_id[0]
        term_id[0] += 1
        lin = []
        _ap = lin.append
        slack = 0.0
        _svk = _svax[axis]
        _hsk = _hslack[axis]
        # L235: the two-element loop unrolled in its original order (ui with
        # s=+1 first, then uj with s=-1), so `lin` is built identically and the
        # float accumulation below sees the same sequence.
        if ui is not None:
            _ap((off + ui, 1.0))
            slack += _pb0
            k = _svk.get(ui)
            if k is not None:
                _ap((k, 0.5))
                slack += _hsk[ui]
        if uj is not None:
            _ap((off + uj, -1.0))
            slack += _pb0
            k = _svk.get(uj)
            if k is not None:
                _ap((k, -0.5))
                slack += _hsk[uj]''',
  '''    _pb0 = prune_B or 0.0
    _pbon = prune_B is not None

    def add_hpwl_rows(wsc, ui, uj, off, dC, axis):
        nonlocal prune_const, nv
        tid = term_id[0]
        term_id[0] += 1
        _svk = _svax[axis]
        _hsk = _hslack[axis]
        # L235: `lin` built as a tuple in ONE allocation, in the order the
        # original loop produced (ui term, its size term, uj term, its size
        # term), so both the emission order and the float sums are unchanged.
        if ui is None:
            kj = _svk.get(uj)
            if kj is None:
                lin = ((off + uj, -1.0),)
                slack = _pb0
            else:
                lin = ((off + uj, -1.0), (kj, -0.5))
                slack = _pb0 + _hsk[uj]
        elif uj is None:
            ki = _svk.get(ui)
            if ki is None:
                lin = ((off + ui, 1.0),)
                slack = _pb0
            else:
                lin = ((off + ui, 1.0), (ki, 0.5))
                slack = _pb0 + _hsk[ui]
        else:
            ki = _svk.get(ui)
            kj = _svk.get(uj)
            if ki is None:
                if kj is None:
                    lin = ((off + ui, 1.0), (off + uj, -1.0))
                    slack = _pb0 + _pb0
                else:
                    lin = ((off + ui, 1.0), (off + uj, -1.0), (kj, -0.5))
                    slack = _pb0 + _pb0 + _hsk[uj]
            elif kj is None:
                lin = ((off + ui, 1.0), (ki, 0.5), (off + uj, -1.0))
                slack = _pb0 + _hsk[ui] + _pb0
            else:
                lin = ((off + ui, 1.0), (ki, 0.5), (off + uj, -1.0),
                       (kj, -0.5))
                slack = _pb0 + _hsk[ui] + _pb0 + _hsk[uj]''',
  "add_hpwl_rows: lin as one tuple allocation")

P('''        if prune_B is not None and abs(dC) > slack and tid not in force_keep:''',
  '''        if _pbon and abs(dC) > slack and tid not in force_keep:''',
  "add_hpwl_rows: hoist the prune_B-is-not-None test")

P('''            sgn = 1.0 if dC > 0.0 else -1.0
            for col, coef in lin:
                obj[col] += wsc * sgn * coef
            prune_const += wsc * sgn * dC
            prune_stat[0] += 1
            dropped.append((tid, tuple(lin), dC, wsc))
            return''',
  '''            sgn = 1.0 if dC > 0.0 else -1.0
            wss = wsc * sgn          # (wsc*sgn)*coef == wsc*sgn*coef
            for col, coef in lin:
                obj[col] += wss * coef
            prune_const += wss * dC
            prune_stat[0] += 1
            dropped.append((tid, lin, dC, wsc))   # lin is already a tuple
            return''',
  "add_hpwl_rows: hoist wsc*sgn, drop the tuple() copy")

# --------------------------------------------------------------------------
# 6. the two hpwl edge loops -- `w * hw_scale` was evaluated twice per edge and
#    `w * (abs+abs)` once more; same values, computed once.
# --------------------------------------------------------------------------
P('''    for i, j, w in b2l_items:
        ui, uj = unit_of[i], unit_of[j]
        dCx, dCy = cx[i] - cx[j], cy[i] - cy[j]
        if w <= 0.0 or ui == uj:
            const_h += w * (abs(dCx) + abs(dCy))
            continue
        add_hpwl_rows(w * hw_scale, ui, uj, 0, dCx, 0)
        add_hpwl_rows(w * hw_scale, ui, uj, U, dCy, 1)
        obj0 += w * (abs(dCx) + abs(dCy))''',
  '''    for i, j, w in b2l_items:
        ui = unit_of[i]
        uj = unit_of[j]
        dCx = cx[i] - cx[j]
        dCy = cy[i] - cy[j]
        _d = w * (abs(dCx) + abs(dCy))
        if w <= 0.0 or ui == uj:
            const_h += _d
            continue
        _ws = w * hw_scale
        add_hpwl_rows(_ws, ui, uj, 0, dCx, 0)
        add_hpwl_rows(_ws, ui, uj, U, dCy, 1)
        obj0 += _d''',
  "b2l loop: hoist the repeated products")

P('''    for p, i, w in p2l_items:
        ui = unit_of[i]
        px, py = c["pin"][p]
        dCx, dCy = cx[i] - px, cy[i] - py
        if w <= 0.0 or ui is None:
            const_h += w * (abs(dCx) + abs(dCy))
            continue
        add_hpwl_rows(w * hw_scale, ui, None, 0, dCx, 0)
        add_hpwl_rows(w * hw_scale, ui, None, U, dCy, 1)
        obj0 += w * (abs(dCx) + abs(dCy))''',
  '''    _pin = c["pin"]
    for p, i, w in p2l_items:
        ui = unit_of[i]
        px, py = _pin[p]
        dCx = cx[i] - px
        dCy = cy[i] - py
        _d = w * (abs(dCx) + abs(dCy))
        if w <= 0.0 or ui is None:
            const_h += _d
            continue
        _ws = w * hw_scale
        add_hpwl_rows(_ws, ui, None, 0, dCx, 0)
        add_hpwl_rows(_ws, ui, None, U, dCy, 1)
        obj0 += _d''',
  "p2l loop: hoist the repeated products and the pin table")


def main():
    inplace = "--inplace" in sys.argv
    dst = SRC if inplace else DST
    s = SRC.read_text(encoding="utf-8")
    for old, new, why in PATCHES:
        c = s.count(old)
        if c != 1:
            print("!! patch matched {} times, expected 1: {}".format(c, why))
            print("   first 80 chars: {!r}".format(old[:80]))
            return 1
        s = s.replace(old, new)
        print("   ok  {}".format(why))
    hdr = "" if inplace else (
        '"""L235 PROBE COPY of optimizer_constructive.py -- NEVER SHIPPED.\n'
        'Generated by l235_patch.py. The only differences are Python-level\n'
        'rewrites of the shape LP\'s row construction, every one of which is\n'
        'required to produce byte-identical triplet arrays; `l235_lpbench.py ab`\n'
        'is the gate.\n"""\n')
    dst.write_text(hdr + s, encoding="utf-8", newline="\n")
    print("wrote {} ({} bytes){}".format(dst, dst.stat().st_size,
                                         "  [IN PLACE]" if inplace else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
