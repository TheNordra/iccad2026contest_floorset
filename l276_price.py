"""L276 -- exact RF pricing against the CURRENT medians, from a measured dt vector.

    RF_i = max(0.7, (t_i / M_i) ** 0.3)          iccad2026_evaluate.py:552
    total = sum_i w_i * q_i * RF_i / sum_i w_i,   w_i = exp(n_i / 12)

🚨 WHY THIS EXISTS RATHER THAN `l146_rf_price.py`. That tool hardcodes
`C:/Users/.01/Downloads/C_median_runtimes_beta_hidden.csv` -- the medians as
published on 2026-08-19. They were REPUBLISHED on 2026-08-23 and every one of the
100 came down: ratio min 0.4837, p50 **0.7418**, max 0.9428. Lower medians mean a
higher t/M, so RF leaves its 0.7 floor sooner and every slowdown costs MORE.
Pricing a slow mechanism against the stale file understates its bill by roughly
the same factor it understates the medians. This module reads
`beta_2026-08-23/C_median_runtimes_beta_hidden_update.csv`.

🚨 AND IT PRICES SECONDS, NOT A MULTIPLIER. l146's own docstring is right about
this and it is worth restating: an added-time distribution is not a ratio. The
expensive cases are the big-n ones and they have the LEAST slack, so a mechanism
costed as a flat multiplier can read 3x cheaper than the same mechanism costed
with its measured per-case vector.

The population is the beta hidden set (real runtimes, real medians, real per-case
quality). The dt vector is measured locally and mapped onto it by block_count --
legitimate because the two corpora have the same n-range (21..120) and the same
100-case composition, and because dt is a property of the mechanism at a given n.
"""
import csv
import json
import math
import statistics
import sys
from pathlib import Path

DIR = Path(__file__).parent
CSV_NEW = DIR / "beta_2026-08-23" / "C_median_runtimes_beta_hidden_update.csv"
CSV_OLD = __import__("l146_rf_price").CSV      # repo copy / $ICCAD_MEDIAN_CSV /
#                                                Downloads -- see l146._median_csv
BETA = DIR / "beta_2026-08-16" / "beta_evaluation_results.json"


def _medians(p):
    M = {}
    with open(p) as f:
        for row in csv.DictReader(f):
            k = list(row)
            M[int(row[k[0]])] = float(row["median_runtime_s"])
    return M


def load(csv_path=None):
    B = {r["test_id"]: r for r in json.load(open(BETA))["test_results"]}
    M = _medians(csv_path or CSV_NEW)
    return [dict(i=i, n=B[i]["block_count"], t=B[i]["runtime_seconds"],
                 med=M[i], q=B[i]["cost"], w=math.exp(B[i]["block_count"] / 12.0))
            for i in sorted(B) if i in M]


def total(rows, dt_of=lambda r: 0.0):
    num = den = 0.0
    for r in rows:
        t = r["t"] + max(0.0, dt_of(r))
        num += r["w"] * r["q"] * max(0.7, (t / r["med"]) ** 0.3)
        den += r["w"]
    return num / den


def dt_by_n(base_json, arm_json):
    """Measured added seconds per case, keyed by block_count."""
    b = {r["test_id"]: r for r in json.load(open(base_json))["test_results"]}
    a = {r["test_id"]: r for r in json.load(open(arm_json))["test_results"]}
    d = {}
    for i in b:
        if i not in a:
            continue
        d.setdefault(b[i]["block_count"], []).append(
            a[i].get("runtime_seconds", 0.0) - b[i].get("runtime_seconds", 0.0))
    return {n: statistics.mean(v) for n, v in d.items()}


def quality_pct(base_json, arm_json):
    """Weighted quality delta, positive = better (RF-free: the harness forces 1.0)."""
    b = {r["test_id"]: r for r in json.load(open(base_json))["test_results"]}
    a = {r["test_id"]: r for r in json.load(open(arm_json))["test_results"]}
    W = lambda n: math.exp(n / 12.0)
    sw = sum(W(b[i]["block_count"]) for i in b)
    tb = sum(W(b[i]["block_count"]) * b[i]["cost"] for i in b) / sw
    ta = sum(W(a[i]["block_count"]) * a[i]["cost"] for i in a) / sw
    return 100.0 * (tb - ta) / tb


def price(base_json, arm_json, label):
    rows = load()
    dt = dt_by_n(base_json, arm_json)
    fallback = statistics.mean(dt.values()) if dt else 0.0
    q = quality_pct(base_json, arm_json)
    base = total(rows)
    slow = total(rows, lambda r: dt.get(r["n"], fallback))
    rf = 100.0 * (base - slow) / base
    # same, priced against the STALE medians, to show what the old tool would say
    rows_old = load(CSV_OLD)
    b0 = total(rows_old)
    rf_old = 100.0 * (b0 - total(rows_old, lambda r: dt.get(r["n"], fallback))) / b0
    dts = sorted(dt.values())
    print("  {:14s} quality {:+7.4f}%   RF {:+7.4f}%   NET {:+7.4f}%"
          "   | dt p50 {:+.3f}s max {:+.3f}s   | stale-medians RF would say {:+.4f}%"
          .format(label, q, rf, q + rf, statistics.median(dts), max(dts), rf_old))
    return q, rf, q + rf


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        rows = load()
        print("population: {} beta cases, medians = {}".format(len(rows), CSV_NEW.name))
        print("baseline weighted graded total {:.10f}".format(total(rows)))
        sys.exit(0)
    print("RF priced against {}  (published 2026-08-23)".format(CSV_NEW.name))
    for i in range(1, len(a), 2):
        price(a[0], a[i], a[i + 1] if i + 1 < len(a) else a[i])
