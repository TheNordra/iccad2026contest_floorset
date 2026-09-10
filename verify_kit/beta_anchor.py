#!/usr/bin/env python3
"""用 beta 的評分機實測逐案 runtime 重建 RF，驗證組員 L342 的關鍵論證。

L342 主張：§5(a) 所依據的 L230 模型，其「LP 關閉」的牆鐘估計 (54.90s) 超過
beta 實測的 LP-free 牆鐘 (52.07s) ⇒ 模型偏悲觀 ⇒ 每個 slack 都被壓小 ⇒
它把 RF-SAFE 的 12 個加項全判超支。判別依據是「加 LP 前就越過 RF floor 的案數」：
L230 模型 16/100、L312 模型 2/100，而 beta 實測 cwRF 幾乎貼著 0.70 floor。

本腳本不採信任何一方的中間值，只用：
  beta_2026-08-16/beta_evaluation_results.json   （評分機實測逐案 runtime + RF-free cost）
  C_median_runtimes_beta_hidden*.csv             （主辦公布的逐案 median，新舊兩份）
  C_beta_leaderboard*.csv                        （官方 total / raw，當已知答案自檢）

先自檢：重建的 raw 與 graded 必須對上官方排行榜，對不上就什麼都別信。
"""
import argparse
import csv
import json
import math
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GAMMA, RF_FLOOR = 0.3, 0.7
FLOOR_COEF = RF_FLOOR ** (1.0 / GAMMA)      # t <= coef*M  =>  RF == 0.7 (floor)
W = lambda n: math.exp(n / 12.0)


def load_med(path):
    M = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            M[int(row[list(row)[0]]) + 21] = float(row["median_runtime_s"])
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", required=True)
    ap.add_argument("--med", action="append", required=True,
                    metavar="LABEL=path.csv")
    ap.add_argument("--raw-official", type=float)
    ap.add_argument("--total-official", action="append", default=[],
                    metavar="LABEL=value")
    a = ap.parse_args()

    rows = json.load(open(a.beta))["test_results"]
    t = {r["block_count"]: r["runtime_seconds"] for r in rows}
    c = {r["block_count"]: r["cost"] for r in rows}
    SW = sum(W(n) for n in c)
    raw = sum(W(n) * c[n] for n in c) / SW

    print("=" * 76)
    print("§0  自檢：beta json 能不能重建官方公布的 raw_score")
    print("=" * 76)
    print("  案數            : %d" % len(rows))
    print("  sum runtime     : %.8f s   (排行榜 total_runtime 欄)" % sum(t.values()))
    print("  重建 raw        : %.16f" % raw)
    if a.raw_official:
        d = abs(raw - a.raw_official)
        print("  官方 raw        : %.16f" % a.raw_official)
        print("  |d|             : %.3e   %s" % (d, "PASS" if d < 1e-9 else "🚨 FAIL"))
        if d >= 1e-9:
            print("  ⇒ 重建不出 raw ⇒ 下面全部不算證據。停。")
            return 1
    print("  ⇒ json 的 cost 欄確實是 RF-free 的（否則對不上 raw）")
    print()

    official = {}
    for spec in a.total_official:
        k, v = spec.split("=", 1)
        official[k] = float(v)

    for spec in a.med:
        label, path = spec.split("=", 1)
        M = load_med(path)
        rf = {n: max(RF_FLOOR, (t[n] / M[n]) ** GAMMA) for n in c}
        graded = sum(W(n) * c[n] * rf[n] for n in c) / SW
        cwrf = graded / raw

        above = sorted(n for n in c if rf[n] > RF_FLOOR + 1e-12)
        w_above = sum(W(n) for n in above) / SW * 100

        print("=" * 76)
        print("§  median 版本 = %s" % label)
        print("=" * 76)
        print("  重建 graded total : %.16f" % graded)
        if label in official:
            d = abs(graded - official[label])
            print("  官方 total        : %.16f" % official[label])
            print("  |d|               : %.3e   %s"
                  % (d, "PASS 逐位級" if d < 1e-9 else
                        ("PASS 捨入級" if d < 1e-5 else "🚨 FAIL")))
        print("  cost-weighted RF  : %.16f   (floor = 0.70)" % cwrf)
        print()
        print("  🔑 加 LP 前就越過 RF floor 的案數 : **%d / 100**" % len(above))
        print("     它們佔的權重                   : %.2f%%" % w_above)
        if above:
            print("     n = %s" % above)
            worst = sorted(above, key=lambda n: -rf[n])[:6]
            print("     最超支的幾個 (n, t/med, RF):")
            for n in worst:
                print("        n=%-4d t=%6.2fs med=%6.3fs  t/med=%5.2f  RF=%.4f"
                      % (n, t[n], M[n], t[n] / M[n], rf[n]))
        slack = {n: FLOOR_COEF * M[n] - t[n] for n in c}
        neg = sorted(n for n in slack if slack[n] < 0)
        print("     免費預算 slack = 0.3046*med - t : 負的有 %d 個" % len(neg))
        print()

    print("=" * 76)
    print("§  這對 L230 vs L312 的意義")
    print("=" * 76)
    print("  上面那個「加 LP 前越過 floor 的案數」是 **beta 的實測值**，")
    print("  而出貨包的 pool 比 beta 更快（L219/L223/L231 把 REFINE 砍到 2/2）。")
    print("  ⇒ 出貨包在加 LP 之前，越過 floor 的案數應該 <= beta 的實測值。")
    print("  L230 模型說 16、L312 模型說 2。拿上面的實測值去卡。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
