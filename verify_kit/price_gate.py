#!/usr/bin/env python3
"""把 LP 閘的「品質 vs RF」帳自己算一遍，不依賴組員的任何中間值。

輸入全部是我方自己量的：
  --lpall-json  LP 開在 100 案（ICCAD_LP_GATE=0）那輪的 results json
  --lpall-log   同一輪的 stderr（ICCAD_LP_TIMING=1 的 [lptime] 行）
  --medians     主辦公布的 C_median_runtimes_beta_hidden.csv
  --arm NAME=<op_wrapper.py>=<results.json>   可給多次

模型就是官方 evaluator 自己的式子:
    RF_n   = max(0.7, (t_n / M_n) ** 0.3)
    graded = sum_n w_n * quality_n * RF_n / sum_n w_n ,   w_n = exp(n/12)

唯一未知數 = 「我這台的秒 -> 評分機的秒」的比例 F（t_grader = t_local / F）。
所以輸出是**對 F 的曲線**，不是單一數字。

🚨 模型缺陷，必須跟數字一起讀：這台是 16 邏輯核卻被強制跑 48 核池形狀
   ⇒ t_pool 被非均勻放大（案子越大放大越多）。把機速當單一純量，正是組員
   的 t_beta*w_ours/w_m73 方法刻意要避開的假設。⇒ break-even F 的**位置**
   只能當指標；不需要 F 的結論在 §2（自洽性）與 §3（單調性）。
"""
import argparse
import ast
import csv
import json
import math
import re
import statistics as st
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GAMMA, RF_FLOOR = 0.3, 0.7
FLOOR_COEF = RF_FLOOR ** (1.0 / GAMMA)      # 0.304551 : t <= coef*M  =>  RF == 0.7
W = lambda n: math.exp(n / 12.0)


def load_medians(path):
    M = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            tid = int(row[list(row)[0]])
            M[tid + 21] = float(row["median_runtime_s"])   # test_id 0 == 21 blocks
    return M


LPT = re.compile(r"\[lptime\]\s*n=(\d+)\s+cpu=([0-9.]+)\s+wall=([0-9.]+)")


def load_lptime(path):
    cpu, wall = {}, {}
    for line in Path(path).read_text(errors="replace").splitlines():
        m = LPT.search(line)
        if m:
            n = int(m.group(1))
            cpu[n] = cpu.get(n, 0.0) + float(m.group(2))
            wall[n] = wall.get(n, 0.0) + float(m.group(3))
    return cpu, wall


def load_gate(op_wrapper):
    if op_wrapper == "ALL":          # LP 開在全部 100 案（ICCAD_LP_GATE=0）
        return {n: 1 for n in range(21, 121)}
    src = Path(op_wrapper).read_text(encoding="utf-8")
    m = re.search(r"_L196_LPGATE\s*=\s*(\{.*?\n\})", src, re.S)
    assert m, "no _L196_LPGATE in " + str(op_wrapper)
    return ast.literal_eval(m.group(1))


def graded(cost, t, M, F):
    num = den = 0.0
    for n in cost:
        rf = max(RF_FLOOR, (t[n] / F / M[n]) ** GAMMA)
        num += W(n) * cost[n] * rf
        den += W(n)
    return num / den


def cwrf(cost, t, M, F):
    """cost-weighted RF = graded / raw."""
    raw = sum(W(n) * cost[n] for n in cost) / sum(W(n) for n in cost)
    return graded(cost, t, M, F) / raw, raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lpall-json", required=True)
    ap.add_argument("--lpall-log", required=True)
    ap.add_argument("--medians", required=True)
    ap.add_argument("--arm", action="append", default=[],
                    metavar="NAME=op_wrapper.py=results.json")
    a = ap.parse_args()

    M = load_medians(a.medians)
    cpu, lpw = load_lptime(a.lpall_log)
    rows = json.load(open(a.lpall_json))["test_results"]
    t_full = {r["block_count"]: r["runtime_seconds"] for r in rows}

    print("=" * 78)
    print("§0  量到的東西")
    print("=" * 78)
    print("  medians          : %d 案" % len(M))
    print("  [lptime] 行       : %d 個 block count" % len(lpw))
    miss = sorted(set(t_full) - set(lpw))
    if miss:
        print("  ⚠️ 缺 LP 時間     : %s" % miss[:12])
    # LP 是序列後處理，它的 wall 直接加在該案的 runtime 上 ⇒ 預算算術要用 wall。
    t_pool = {n: max(0.0, t_full[n] - lpw.get(n, 0.0)) for n in t_full}
    print("  t_pool (本機)     : p50 %.2fs  max %.2fs"
          % (st.median(t_pool.values()), max(t_pool.values())))
    print("  dt_lp  wall (本機): p50 %.2fs  max %.2fs"
          % (st.median(lpw.values()), max(lpw.values())))
    print("  dt_lp  cpu  (本機): p50 %.2fs  max %.2fs"
          % (st.median(cpu.values()), max(cpu.values())))
    print()

    arms = {}
    for spec in a.arm:
        name, wrap, res = spec.split("=", 2)
        g = load_gate(wrap)
        c = {r["block_count"]: r["cost"]
             for r in json.load(open(res))["test_results"]}
        t = {n: t_pool[n] + (lpw.get(n, 0.0) if g.get(n) else 0.0) for n in t_pool}
        arms[name] = dict(gate=g, cost=c, t=t, on=sum(g.values()))

    # ---------------------------------------------------------------- §1 品質
    print("=" * 78)
    print("§1  品質（本機 harness 強制 RF=1.0 ⇒ 這一欄沒付 runtime 的錢）")
    print("=" * 78)
    base = None
    for name, A in arms.items():
        raw = sum(W(n) * A["cost"][n] for n in A["cost"]) / sum(W(n) for n in A["cost"])
        if base is None:
            base = raw
        print("  %-8s LP on %3d   raw = %.12f   (%+.4f%% vs 第一個臂)"
              % (name, A["on"], raw, (raw / base - 1) * 100))
    print()

    # ------------------------------------------------------- §2 閘的自洽性
    print("=" * 78)
    print("§2  自洽性：閘表能不能用一個門檻切開？（不需要 F，只需要排序）")
    print("=" * 78)
    print("  rho(n) = t(n, LP 開) / (0.3046 * M(n))   ← 1.0 = 剛好用完免費預算")
    for name, A in arms.items():
        rho = {n: (t_pool[n] + lpw.get(n, 0.0)) / (FLOOR_COEF * M[n]) for n in t_pool}
        on = sorted(n for n in A["gate"] if A["gate"][n])
        off = sorted(n for n in A["gate"] if not A["gate"][n])
        if not off:
            print("  %-8s 全開，無可分割性可言" % name)
            continue
        max_on = max(rho[n] for n in on)
        min_off = min(rho[n] for n in off)
        sep = max_on <= min_off
        print("  %-8s on 的 rho 上界 %.2f   off 的 rho 下界 %.2f   可分割: %s"
              % (name, max_on, min_off, "是" if sep else "否（表不是單純按 rho 排序）"))
        bad = sorted(n for n in off if rho[n] < max_on)
        if bad:
            print("           off 卻比某些 on 更便宜的 n: %s" % bad[:14])
    print()

    # ------------------------------------------------- §3 對 F 的 graded 曲線
    print("=" * 78)
    print("§3  graded 對 F 的曲線（F = 本機秒 / 評分機秒；越小代表評分機越像我這台）")
    print("=" * 78)
    names = list(arms)
    hdr = "  %-7s" % "F" + "".join("%14s" % n for n in names) + "   贏家"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    Fs = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 12.0, 20.0]
    for F in Fs:
        vals = [graded(arms[n]["cost"], arms[n]["t"], M, F) for n in names]
        best = names[vals.index(min(vals))]
        print("  %-7.1f" % F + "".join("%14.5f" % v for v in vals) + "   " + best)
    print()

    if len(names) >= 2:
        lo, hi = 0.5, 60.0
        f = lambda F: (graded(arms[names[1]]["cost"], arms[names[1]]["t"], M, F)
                       - graded(arms[names[0]]["cost"], arms[names[0]]["t"], M, F))
        if f(lo) * f(hi) < 0:
            for _ in range(80):
                mid = (lo + hi) / 2
                if f(lo) * f(mid) <= 0:
                    hi = mid
                else:
                    lo = mid
            print("  break-even F = %.2f" % ((lo + hi) / 2))
            print("     F 比它大 ⇒ %s 較好；比它小 ⇒ %s 較好"
                  % (names[1], names[0]))
        else:
            print("  在 F ∈ [0.5, 60] 內沒有交叉：%s 全程%s"
                  % (names[1], "較好" if f(2.0) < 0 else "較差"))
    print()
    print("  ⚠️ 讀法：這裡的 graded 用的是 **in-set 100 案的品質**，不是 hidden set，")
    print("     所以絕對值不可與排行榜比。**只有兩個臂的相對關係有意義。**")
    print()

    # ------------------------------------------------- §4 修掉單一純量的偏誤
    print("=" * 78)
    print("§4  修正：LP 在評分機上比在我這台貴得多（單一純量 F 是錯的）")
    print("=" * 78)
    sum_pool = sum(t_pool.values())
    sum_lp = sum(lpw.values())
    X = sum_lp / sum_pool
    print("  我這台（16 核跑 48 核池形狀）: sum dt_lp / sum t_pool = %.2fs / %.2fs = %.3f"
          % (sum_lp, sum_pool, X))
    print("  評分機（組員量到的）          : LP 全開 20.8 grader-s；beta pool 52.07 grader-s")
    print("                                 ⇒ 比值 %.3f" % (20.8 / 52.07))
    lam_true = (20.8 / 52.07) / X
    print("  ⇒ LP 在評分機上相對貴 lambda ~= %.1f 倍。單一純量 F 的模型把 LP 低估了這麼多。"
          % lam_true)
    print()
    print("  下面把 dt_lp 乘上 lambda，並且每個 lambda 都重新校準 F，")
    print("  讓 D71 這個臂剛好複現組員量到的 cwRF = 0.70423（她的錨）。")
    print()

    TARGET_CWRF = 0.70423

    def solve_F(cost, t):
        lo, hi = 0.01, 1e4
        for _ in range(200):
            mid = math.sqrt(lo * hi)
            if cwrf(cost, t, M, mid)[0] > TARGET_CWRF:
                lo = mid           # F 太小 => RF 太大 => 要加大 F
            else:
                hi = mid
        return math.sqrt(lo * hi)

    def t_of(name, lam):
        g = arms[name]["gate"]
        return {n: t_pool[n] + lam * lpw.get(n, 0.0) * (1 if g.get(n) else 0)
                for n in t_pool}

    hdr = "  %-8s %-8s" % ("lambda", "校準F") + "".join("%13s" % n for n in names)
    print(hdr + "   贏家")
    print("  " + "-" * (len(hdr) + 6))
    rowsout = []
    for lam in (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24):
        F = solve_F(arms[names[0]]["cost"], t_of(names[0], lam))
        gs = [graded(arms[n]["cost"], t_of(n, lam), M, F) for n in names]
        best = names[gs.index(min(gs))]
        rowsout.append((lam, F, gs))
        print("  %-8d %-8.2f" % (lam, F) + "".join("%13.5f" % g for g in gs)
              + "   " + best)
    print()
    print("  ⇒ 實測的 lambda ~= %.1f。" % lam_true)
    print()
    print("  🔎 模型證偽測試：組員量到 100 開 (0.89268) 比 71 開 (0.87819) **更差**，")
    print("     差 +1.449pp。我的模型必須算出同樣的方向，否則模型不可信。")
    if len(names) >= 3:
        for lam, F, gs in rowsout:
            d = (gs[2] - gs[0]) * 100
            tag = "方向相符" if d > 0 else "🚨 方向相反 ⇒ 模型錯"
            print("     lambda=%-3d  100開 − 71開 = %+.3f pp   %s" % (lam, d, tag))


if __name__ == "__main__":
    main()
