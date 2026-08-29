"""Does ABUT's solo-profile boundary win land where the SHIPPED PORTFOLIO still
has boundary violations?

Solo profile 0 has 523 boundary violations over the 240 OOS s1 cases; the shipped
48-core portfolio (l140_oos_s1_c48.json) has only 254 on the same cases -- the
pool already removes 51% of them by profile diversity.  So a solo-profile
boundary win is only worth something if it happens on a case the portfolio has
NOT already fixed.  This joins my per-case solo A/B dump against the shipped
per-case v_bnd.  Read-only.
"""
import json
import math
import re
import sys

SOLO = sys.argv[1] if len(sys.argv) > 1 else "conc240.txt"
SHIPJ = r"C:\ICCAD_ml\ship_final\l140_oos_s1_c48.json"

ship = {r["key"]: r for r in json.load(open(SHIPJ))["test_results"]}
pat = re.compile(r"^\s+(\S+)\s+(\d+)\s+([\d.]+)%\s+([+-][\d.]+)\s+"
                 r"([+-][\d.]+)\s+(\d+)\s*->\s*(\d+)\s*$")
rows = []
for line in open(SOLO, encoding="utf-8", errors="replace"):
    m = pat.match(line.rstrip("\n"))
    if m:
        rows.append((m.group(1), int(m.group(2)), float(m.group(4)),
                     int(m.group(6)), int(m.group(7))))

print(f"parsed {len(rows)} moved cases from {SOLO}")
hit = [r for r in rows if r[3] != r[4]]
print(f"of those, {len(hit)} changed the solo boundary count")

W = sum(math.exp(r["n"] / 12.0) for r in ship.values())
tot_ship_bnd = sum(r["v_bnd"] for r in ship.values())
print(f"shipped portfolio total v_bnd over 240 = {tot_ship_bnd}")

print("\ncases where ABUT changed the SOLO boundary count, vs what the "
      "SHIPPED PORTFOLIO already achieves there:")
print(f"  {'case':>26} {'n':>4} {'solo bnd':>10} {'ship bnd':>9} "
      f"{'wshare':>7} {'dcost':>9}")
gain_where_ship_clean = 0
gain_where_ship_dirty = 0
for ck, n, dc, b0, bw in sorted(hit, key=lambda r: -math.exp(r[1] / 12.0)):
    s = ship.get(ck)
    sb = s["v_bnd"] if s else -1
    ws = 100 * math.exp(n / 12.0) / W
    print(f"  {ck:>26} {n:>4} {b0:>4} ->{bw:>3}   {sb:>9} {ws:>6.2f}% "
          f"{dc:>+9.5f}")
    if sb == 0:
        gain_where_ship_clean += 1
    else:
        gain_where_ship_dirty += 1
print(f"\nsolo boundary changes on cases the portfolio ALREADY has at 0 "
      f"violations: {gain_where_ship_clean}")
print(f"solo boundary changes on cases the portfolio still has >0: "
      f"{gain_where_ship_dirty}")
