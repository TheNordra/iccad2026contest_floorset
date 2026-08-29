"""L312 -- build the RF-SAFE `_L196_LPGATE` from min-of-N dt.

L298 picked its RF-SAFE subset from a SINGLE ship/gate0 pair. The subset is a
function of measured dt, and dt is a measurement: the two gate0 repeats on this
box differ by 11 % of total wall (143.78 s vs 160.17 s), and two sessions were
running concurrently today. L296's rule is that min-of-N belongs on the WORK
UNIT, not the arm -- so this takes the per-CASE min over every repeat of each
side before differencing.

    dt[n] = max(0, min_over_repeats(gate0_t[n]) - min_over_repeats(ship_t[n]))

Selection is unchanged from L298 and still uses NO quality information:

    ungate n  iff  dt[n] / F  <=  slack[n]        slack[n] = 0.3046*med[n] - t_ship[n]

Run:  <python> l312_build_rfsafe_gate.py [F]        (default: both 3.17 and 2.38)
"""
import json
import sys
from pathlib import Path

DIR = Path(__file__).parent
sys.path.insert(0, str(DIR))
import l276_price as P                                            # noqa: E402
import l296_project as J                                          # noqa: E402

SHIP_S = J.SHIP_S
TH = 0.7 ** (1 / 0.3)

SHIP = ["l294_ship.json", "l294_ship_r2.json", "l301b_ship.json",
        "l301b_ship_r2.json", "l301b_ship_r3.json", "l302_ship_1.json",
        "l302_ship_2.json", "results_L237_post.json", "l285_ship_r2.json",
        "l285_lp_on.json"]
GATE0 = ["l294_gate0.json", "l294_gate0_r2.json",
         "l301b_gate0.json", "l301b_gate0_r2.json"]


def load(names, expect_cost):
    runs = []
    for n in names:
        d = {r["block_count"]: r for r in
             json.load(open(DIR / n))["test_results"]}
        runs.append(d)
    base = runs[0]
    for k, r in enumerate(runs[1:], 1):
        bad = [n for n in base if abs(r[n]["cost"] - base[n]["cost"]) > 0]
        assert not bad, f"{names[k]} is not the same arm as {names[0]}: {bad[:5]}"
    return {n: min(r[n]["runtime_seconds"] for r in runs) for n in base}


def gate_map():
    txt = (DIR / "optimizer_constructive.py").read_text(errors="replace")
    i = txt.index("_L196_LPGATE = {")
    return eval(txt[i + len("_L196_LPGATE = "):txt.index("}", i) + 1])


def main():
    ship = load(SHIP, None)
    gate = load(GATE0, None)
    print(f"min-of-N over {len(SHIP)} ship runs and {len(GATE0)} gate0 runs "
          f"(cost bit-identical within each group -- free determinism gate: PASS)")
    print(f"  ship  total of per-case minima: {sum(ship.values()):7.2f} s")
    print(f"  gate0 total of per-case minima: {sum(gate.values()):7.2f} s")

    dt = {n: max(0.0, gate[n] - ship[n]) for n in ship}
    meds = {x["n"]: x["med"] for x in P.load()}
    tship = {x["n"]: x["t"] * SHIP_S for x in P.load()}
    slack = {n: TH * meds[n] - tship[n] for n in meds}

    G = gate_map()
    OFF = sorted(n for n, v in G.items() if not v)
    print(f"  currently gated OFF: {len(OFF)} of {len(G)} block counts\n")

    # what L298 got from its single pair, for comparison
    import l298_selective_ungate as L298                           # noqa: E402
    single = {i["n"] for i in L298.items if i["fits"]}

    for F in ([float(sys.argv[1])] if len(sys.argv) > 1 else [3.17, 2.38]):
        sel = sorted(n for n in OFF if dt.get(n, 0.0) / F <= slack.get(n, 0.0))
        print(f"F = {F}:  RF-SAFE selects {len(sel)} block counts")
        print(f"  {sel}")
        gained = sorted(set(sel) - single)
        lost = sorted(single - set(sel))
        print(f"  vs L298's single-pair subset ({len(single)}): "
              f"+{len(gained)} {gained}  -{len(lost)} {lost}")
        newmap = {n: (1 if (G[n] or n in sel) else 0) for n in sorted(G)}
        out = DIR / f"l312_gate_rfsafe_F{str(F).replace('.', 'p')}.json"
        out.write_text(json.dumps({"F": F, "selected": sel,
                                   "gate": {str(k): v for k, v in newmap.items()}},
                                  indent=1))
        print(f"  on={sum(newmap.values())}/100  ->  {out.name}\n")


if __name__ == "__main__":
    main()
