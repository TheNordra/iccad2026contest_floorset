#!/usr/bin/env python3
"""驗收判官 — 獨立於組員的腳本，只讀 tar 與官方 results json。

子命令
  identity <tar>                       身分/結構/合規（不跑 solver）
  score    <results.json>              加權總分、feasible、帶別分佈
  diff     <a.json> <b.json>           逐案 cost/positions 比對（determinism / 迴歸）
  logcheck <logfile>                   SA fallback、scipy 來源、例外

退出碼 0 = 全過，1 = 有 FAIL。
"""
import argparse
import ast
import hashlib
import json
import math
import re
import sys
import tarfile
from pathlib import Path

# ---- 期望值（換包時只動這裡） -------------------------------------------
EXPECT = {
    "members": [
        "cadc1075/", "cadc1075/bin/", "cadc1075/bin/constructive_linux",
        "cadc1075/constructive.cpp", "cadc1075/op_src.py",
        "cadc1075/op_wrapper.py", "cadc1075/README.md",
        "cadc1075/requirements.txt",
    ],
    # D（2026-08-26 已上傳那顆）。新包來了用 --expect-md5 / --expect-elf 覆寫。
    "op_wrapper_md5": "1c326784de7cd9246cd1f380e2842668",
    "elf_md5": "bc9912072cd97b45b47a03adec7170ce",
    "linux_total": 1.2264069637381392,
    "lp_gate_on": 71,
}

STDLIB = set(getattr(sys, "stdlib_module_names", ()))
PROVIDED = {"iccad2026_evaluate"}

# Windows 主控台預設 cp1252，中文標籤會炸；WSL 上是 no-op。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_rows = []


def chk(ok, what, detail=""):
    _rows.append((ok, what, detail))


def _report(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    bad = 0
    for ok, what, detail in _rows:
        tag = "PASS" if ok is True else ("FAIL" if ok is False else "INFO")
        if ok is False:
            bad += 1
        print("  [{:4}] {:<44} {}".format(tag, what, detail))
    print("-" * 78)
    print("  {} checks, {} FAIL".format(len(_rows), bad))
    return 1 if bad else 0


# ---------------------------------------------------------------- identity
# 磁碟機代號前面只能排除「英數字」——不能排除引號，路徑正是寫在字串裡的。
# 排除 word char 就足以擋掉 http:// 之類（'p' 是 word char）。
ABS_PATH_RE = re.compile(r"(?<![\w])[A-Za-z]:[\\/]|(?<![\w])/home/|(?<![\w])/Users/")


def cmd_identity(a):
    tar_path = Path(a.tar)
    raw = tar_path.read_bytes()
    print("artefact : {}".format(tar_path))
    print("size     : {:,} bytes".format(len(raw)))
    print("tar md5  : {}  (不可重現，僅供記錄)".format(hashlib.md5(raw).hexdigest()))

    with tarfile.open(tar_path) as t:
        names = [m.name.rstrip("/") + ("/" if m.isdir() else "")
                 for m in t.getmembers()]
        blobs = {m.name: t.extractfile(m).read()
                 for m in t.getmembers() if m.isfile()}

    def md5(b):
        return hashlib.md5(b).hexdigest()

    # --- 結構 ---
    ok_members = sorted(names) == sorted(EXPECT["members"])
    chk(ok_members, "tar 成員恰為 8 個既定項",
        "" if ok_members else "got {}: {}".format(len(names), ",".join(sorted(names))))
    chk(not [n for n in names if n.startswith("cadc1075/vendor")],
        "無 vendor/ 項目")
    chk("cadc1075/op_wrapper.py" in blobs, "entry 名為 op_wrapper.py")
    extra = [n for n in names if n.endswith(".py")
             and n not in ("cadc1075/op_wrapper.py", "cadc1075/op_src.py")]
    chk(not extra, "無多餘的 optimizer .py", ",".join(extra))

    # --- 身分 ---
    ow = md5(blobs.get("cadc1075/op_wrapper.py", b""))
    osrc = md5(blobs.get("cadc1075/op_src.py", b""))
    elf = md5(blobs.get("cadc1075/bin/constructive_linux", b""))
    chk(ow == (a.expect_md5 or EXPECT["op_wrapper_md5"]), "op_wrapper.py md5", ow)
    chk(ow == osrc, "op_src.py == op_wrapper.py", osrc)
    chk(elf == (a.expect_elf or EXPECT["elf_md5"]), "bin/constructive_linux md5", elf)
    print("  cpp md5  : {}".format(md5(blobs.get("cadc1075/constructive.cpp", b""))))

    # --- CRLF（.gitattributes 沒蓋到新路徑時的無聲殺手）---
    chk(b"\r\n" not in blobs.get("cadc1075/op_wrapper.py", b""),
        "op_wrapper.py 無 CRLF")

    # --- 絕對路徑（官方 §4 checklist）---
    src = blobs.get("cadc1075/op_wrapper.py", b"").decode("utf-8", "replace")
    hits = []
    for i, line in enumerate(src.splitlines(), 1):
        if line.lstrip().startswith("#"):
            continue
        if ABS_PATH_RE.search(line):
            hits.append("{}:{}".format(i, line.strip()[:56]))
    chk(not hits, "code 內無絕對路徑", " | ".join(hits[:3]))

    # --- requirements 完整性（雙向）---
    req_raw = blobs.get("cadc1075/requirements.txt", b"").decode("utf-8", "replace")
    req = [l.split("#")[0].strip() for l in req_raw.splitlines()]
    req = [l for l in req if l]
    declared = {re.split(r"[<>=!\[ ]", l)[0].strip().lower() for l in req}
    chk(bool(req), "requirements.txt 非空",
        "{} 行: {}".format(len(req), ",".join(sorted(declared))))
    chk(all("==" not in l for l in req), "requirements 無 == 釘死（Case B 相容）")

    imported, guarded = set(), set()
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        chk(False, "op_wrapper.py 可 parse", str(e))
        return _report("IDENTITY / COMPLIANCE  —  {}".format(tar_path.name))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(n.name.split(".")[0] for n in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Import):
                    guarded.update(n.name.split(".")[0] for n in sub.names)
                elif isinstance(sub, ast.ImportFrom) and sub.module:
                    guarded.add(sub.module.split(".")[0])
    nonstd = {m for m in imported if m not in STDLIB and m not in PROVIDED}
    missing = sorted(m for m in nonstd if m.lower() not in declared)
    chk(not missing, "每個非 stdlib import 都在 requirements 裡", ",".join(missing))
    unused = sorted(d for d in declared if d not in {m.lower() for m in nonstd})
    chk(None, "requirements 宣告了但 op_wrapper 沒 import",
        ",".join(unused) + "  ← loader/evaluator 可能需要，非硬錯" if unused else "(無)")
    chk(None, "未包在 try 裡的非 stdlib import（pip 失敗即死）",
        ",".join(sorted(m for m in nonstd if m not in guarded)) or "(無)")

    return _report("IDENTITY / COMPLIANCE  —  {}".format(tar_path.name))


# ------------------------------------------------------------------- score
def _load(p):
    return json.load(open(p))["test_results"]


def _total(rows):
    def W(n):
        return math.exp(n / 12.0)
    sw = sum(W(r["block_count"]) for r in rows)
    return sum(W(r["block_count"]) * r["cost"] for r in rows) / sw


def cmd_score(a):
    rows = _load(a.results)
    tot = _total(rows)
    feas = sum(1 for r in rows if r["is_feasible"])
    print("file      : {}".format(a.results))
    print("total     : {!r}".format(tot))
    print("feasible  : {}/{}".format(feas, len(rows)))
    for lo, hi, lbl in ((0, 60, "  n<=60 "), (60, 100, " 60<n<=100"),
                        (100, 10 ** 9, "100<n   ")):
        seg = [r for r in rows if lo < r["block_count"] <= hi]
        if seg:
            print("  {} cases={:<3} total={:.9f}".format(lbl, len(seg), _total(seg)))
    chk(feas == len(rows), "feasible 100/100", "{}/{}".format(feas, len(rows)))
    chk(len(rows) == 100, "案數 = 100", str(len(rows)))
    if a.expect is not None:
        chk(abs(tot - a.expect) <= a.tol, "total == 期望值",
            "d={:.3e} (tol {:g})".format(tot - a.expect, a.tol))
    bad = [r["test_id"] for r in rows if r["cost"] > 9.99]
    chk(not bad, "無 SA fallback 級成本 (>9.99)", ",".join(map(str, bad[:5])))
    return _report("SCORE  —  {}".format(Path(a.results).name))


# -------------------------------------------------------------------- diff
def cmd_diff(a):
    A, B = _load(a.a), _load(a.b)
    ka = {r["test_id"]: r for r in A}
    kb = {r["test_id"]: r for r in B}
    common = sorted(set(ka) & set(kb))
    chk(len(common) == len(ka) == len(kb), "兩份案子集合相同",
        "{} vs {}".format(len(ka), len(kb)))

    cost_same = pos_same = 0
    worse, better = [], []
    for k in common:
        ra, rb = ka[k], kb[k]
        if ra["cost"] == rb["cost"]:
            cost_same += 1
        else:
            (worse if rb["cost"] > ra["cost"] else better).append(
                (k, ra["block_count"], ra["cost"], rb["cost"]))
        if json.dumps(ra.get("positions"), sort_keys=True) == \
           json.dumps(rb.get("positions"), sort_keys=True):
            pos_same += 1

    ta, tb = _total(A), _total(B)
    print("A = {}   total {!r}".format(a.a, ta))
    print("B = {}   total {!r}".format(a.b, tb))
    print("delta(B vs A) = {:+.4f}%".format((tb / ta - 1) * 100))
    print("cost identical  : {}/{}".format(cost_same, len(common)))
    print("positions ident : {}/{}".format(pos_same, len(common)))
    if worse:
        print("\n  B 比 A 差的案（迴歸）:")
        for k, n, ca, cb in sorted(worse, key=lambda x: -(x[3] - x[2]))[:12]:
            print("    test {:>3}  n={:<4} {:.6f} -> {:.6f}  ({:+.3f}%)".format(
                k, n, ca, cb, (cb / ca - 1) * 100))
    if better:
        print("\n  B 比 A 好的案: {} 個，最大 {:+.3f}%".format(
            len(better), min((cb / ca - 1) * 100 for _, _, ca, cb in better)))

    if a.mode == "determinism":
        chk(cost_same == len(common), "determinism cost 逐位相同",
            "{}/{}".format(cost_same, len(common)))
        chk(pos_same == len(common), "determinism positions 逐位相同",
            "{}/{}".format(pos_same, len(common)))
    elif a.mode == "regression":
        chk(not worse, "0 regressions vs 參考包", "{} 案變差".format(len(worse)))
        chk(tb <= ta * (1 + a.tol), "總分未變差",
            "{:+.4f}%".format((tb / ta - 1) * 100))
    else:
        chk(None, "movers", "{} 案".format(len(worse) + len(better)))
    return _report("DIFF [{}]".format(a.mode))


# ---------------------------------------------------------------- logcheck
def cmd_logcheck(a):
    txt = Path(a.log).read_text(errors="replace")
    sa = len(re.findall(r"SA fallback|falling back to SA|optimizer_claude", txt))
    chk(sa == 0, "log 內無 SA fallback", "{} 次".format(sa))
    m = re.findall(r"\[scipy\][^\n]*", txt)
    chk(None, "log 的 scipy 來源", m[0] if m else "(沒印，未必是問題)")
    tb = len(re.findall(r"Traceback \(most recent call last\)", txt))
    chk(tb == 0, "無 Python traceback", "{} 個".format(tb))
    for pat, name in ((r"FileNotFoundError", "FileNotFoundError"),
                      (r"Permission denied", "Permission denied"),
                      (r"GLIBC", "GLIBC 版本問題")):
        n = len(re.findall(pat, txt))
        chk(n == 0, "log 無 {}".format(name), "{} 次".format(n))
    leaked = sorted(set(re.findall(r"\bICCAD_[A-Z0-9_]+", txt)))
    chk(None, "log 內出現過的 ICCAD_* 名稱", ",".join(leaked[:8]) or "(無)")
    return _report("LOGCHECK  —  {}".format(Path(a.log).name))


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("identity")
    s.add_argument("tar")
    s.add_argument("--expect-md5")
    s.add_argument("--expect-elf")
    s.set_defaults(fn=cmd_identity)

    s = sub.add_parser("score")
    s.add_argument("results")
    s.add_argument("--expect", type=float)
    s.add_argument("--tol", type=float, default=1e-9)
    s.set_defaults(fn=cmd_score)

    s = sub.add_parser("diff")
    s.add_argument("a")
    s.add_argument("b")
    s.add_argument("--mode", choices=("determinism", "regression", "info"),
                   default="info")
    s.add_argument("--tol", type=float, default=0.0)
    s.set_defaults(fn=cmd_diff)

    s = sub.add_parser("logcheck")
    s.add_argument("log")
    s.set_defaults(fn=cmd_logcheck)

    a = p.parse_args()
    sys.exit(a.fn(a))


if __name__ == "__main__":
    main()
