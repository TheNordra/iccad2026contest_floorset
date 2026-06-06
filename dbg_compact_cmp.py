"""Compare Python-prototype compaction vs C++ in-binary compaction on the SAME
starting layout, to isolate where they diverge. Run: python dbg_compact_cmp.py 99 96
"""
import os, sys, math, subprocess
from pathlib import Path
import torch
_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR / "iccad2026contest")); sys.path.insert(0, str(_DIR))
from iccad2026_evaluate import ContestEvaluator, evaluate_solution
from optimizer_claude import _serialize_input, _parse_output
from proxy_analysis import build_opt_target_pos
import dbg_compact as dc

EXE = str(_DIR / "constructive.exe")


def metrics(ps, codes, clus, b2b, p2b, pins, n):
    xs = [p[0] for p in ps]; ys = [p[1] for p in ps]
    ws = [p[2] for p in ps]; hs = [p[3] for p in ps]
    xmin, ymin, xmax, ymax = dc._bbox(xs, ys, ws, hs)
    area = (xmax - xmin) * (ymax - ymin)
    bv = dc.count_bv(xs, ys, ws, hs, codes)
    gf = dc.count_gf(xs, ys, ws, hs, clus)
    sc = dc.layout_score(xs, ys, ws, hs, codes, clus, b2b, p2b, pins, n)
    return area, bv, gf, sc


def main():
    ev = ContestEvaluator(data_path=str(_DIR), verbose=False); ev._load_dataset()
    ids = [int(a) for a in sys.argv[1:]] or [99]
    for idx in ids:
        s = ev.dataset[idx]; inp, lab = s["input"], s["label"]
        at, b2b, p2b, pins, cons = inp
        n = int((at != -1).sum().item())
        base, tp = ev._extract_baseline(idx, lab, b2b, p2b, pins, n)
        otp = build_opt_target_pos(tp, cons, n)
        txt = _serialize_input(n, at, b2b, p2b, pins, cons, otp, gnn_hint=None)
        codes = [int(cons[i, 4]) for i in range(n)]
        clus = [int(cons[i, 3]) for i in range(n)]
        pre = {i for i in range(n) if int(cons[i, 1]) != 0}

        env0 = dict(os.environ); env0["ICCAD_NO_COMPACT"] = "1"
        L0 = _parse_output(subprocess.run([EXE], input=txt, capture_output=True,
                                          text=True, env=env0).stdout, n)
        Lcpp = _parse_output(subprocess.run([EXE], input=txt, capture_output=True,
                                            text=True).stdout, n)
        Lpy = dc.compact(L0, codes, pre, clus, b2b, p2b, pins, n)

        def cost(ps):
            return evaluate_solution({'positions': ps, 'runtime': 1.0}, base,
                                     cons[:n], b2b, p2b, pins, at[:n],
                                     target_positions=tp[:n], median_runtime=1.0).cost

        print(f"\n=== case {idx} (n={n}, nclus={sum(1 for c in clus if c>0)}, "
              f"nbnd={sum(1 for c in codes if c>0)}, npre={len(pre)}) ===")
        for tag, L in (("L0(nocompact)", L0), ("Lpy(proto)", Lpy), ("Lcpp(binary)", Lcpp)):
            a, bv, gf, sc = metrics(L, codes, clus, b2b, p2b, pins, n)
            print(f"  {tag:16s} area={a:11.1f} bv={bv:>2} gf={gf:>2} "
                  f"lscore={sc:13.1f} cost={cost(L):.4f}")


if __name__ == "__main__":
    main()
