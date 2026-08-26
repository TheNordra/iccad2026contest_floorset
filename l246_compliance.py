"""L246 - check the UPLOADED artefact against every requirement in both
official documents, clause by clause.

Checked against the file downloaded back FROM the Drive, not the local stage,
so this verifies what the organizers actually have.

  A = beta_submission_guidelines_problemC.txt   (md5 c7f7da58ba30f22ecbc35c09b8ae4963)
  B = C_beta_evaluation_report_hidden_final.txt (md5 721aa112a551db7919779e4a643adbc6)

Every row cites the document and line it comes from, so a FAIL can be argued
against the text rather than against my paraphrase of it.

  <python> l246_compliance.py "C:/Users/.01/Downloads/cadc1075 (2).tar.gz"
"""
import ast
import hashlib
import re
import sys
import tarfile
from pathlib import Path

DIR = Path(__file__).parent
STDLIB = {
    "os", "sys", "math", "time", "json", "re", "subprocess", "threading",
    "pathlib", "typing", "collections", "concurrent", "functools", "itertools",
    "random", "shutil", "tempfile", "warnings", "copy", "hashlib", "struct",
    "array", "dataclasses", "abc", "enum", "contextlib", "traceback",
    "platform", "getpass", "importlib", "io", "csv", "statistics", "bisect",
    "heapq", "operator", "textwrap", "argparse", "glob", "gc", "atexit",
    "signal", "errno", "zipfile", "tarfile", "base64", "pickle", "socket",
}
# the evaluator's own module, provided by the contest, not a dependency
PROVIDED = {"iccad2026_evaluate"}

ROWS = []


def chk(ok, doc, where, what, detail=""):
    ROWS.append((ok, doc, where, what, detail))


def main():
    tar_path = Path(sys.argv[1] if len(sys.argv) > 1
                    else r"C:/Users/.01/Downloads/cadc1075 (2).tar.gz")
    if not tar_path.exists():
        print("no such file:", tar_path)
        return 2
    raw = tar_path.read_bytes()
    print("artefact : {}".format(tar_path))
    print("size     : {:,} bytes".format(len(raw)))
    print("tar md5  : {}".format(hashlib.md5(raw).hexdigest()))

    with tarfile.open(tar_path) as t:
        members = t.getmembers()
        names = [m.name for m in members]
        files = {m.name: m for m in members if m.isfile()}
        src = {}
        for n in ("cadc1075/op_wrapper.py", "cadc1075/op_src.py",
                  "cadc1075/requirements.txt", "cadc1075/README.md"):
            if n in files:
                src[n] = t.extractfile(n).read()
        elf = t.extractfile("cadc1075/bin/constructive_linux").read() \
            if "cadc1075/bin/constructive_linux" in files else b""

    wrap = src.get("cadc1075/op_wrapper.py", b"").decode("utf-8", "replace")
    req = src.get("cadc1075/requirements.txt", b"").decode("utf-8", "replace")
    print("identity : op_wrapper md5 {}"
          .format(hashlib.md5(src.get("cadc1075/op_wrapper.py", b"")).hexdigest()))
    print()

    # ---- A section 1: structure (lines 39-59) ------------------------------
    tops = {n.split("/")[0] for n in names}
    chk(tops == {"cadc1075"}, "A", "39-40",
        "unpacks to a single flat directory cadc1075/", str(sorted(tops)))
    for f, line in (("op_wrapper.py", "43"), ("requirements.txt", "48")):
        chk("cadc1075/" + f in files, "A", line,
            "{} present and DIRECTLY inside cadc1075/".format(f))
    chk("cadc1075/op_src.py" in files, "A", "46",
        "op_src.py present (optional, strongly recommended)")
    depth = [n for n in files if n.count("/") > 1]
    chk(all(not n.split("/")[-1] in ("op_wrapper.py", "op_src.py",
                                     "requirements.txt") for n in depth),
        "A", "55-57", "no nesting: the three required files are not in subdirs",
        "subdir files: {}".format(sorted(depth)))

    # ---- A section 1: cleanliness (lines 61-69) ----------------------------
    bad_ext = sorted(n for n in files
                     if Path(n).suffix.lower() in {".pkl", ".json", ".log",
                                                   ".exe", ".pyc", ".ipynb"})
    chk(not bad_ext, "A", "62-64",
        "no unrelated files (result JSONs, logs, notebooks, checkpoints)",
        str(bad_ext))
    pys = sorted(n for n in files if n.endswith(".py"))
    chk(pys == ["cadc1075/op_src.py", "cadc1075/op_wrapper.py"], "A", "65-66",
        "exactly one op_wrapper.py, at most one op_src.py, NO other .py",
        str(pys))
    big = sorted((n, files[n].size) for n in files if files[n].size > 1_000_000)
    chk(all(n == "cadc1075/bin/constructive_linux" for n, _ in big), "A",
        "67-68", "large binaries only if ACTIVELY USED; no unused assets",
        "files >1MB: {}".format([(n, s) for n, s in big]))
    chk(not any("vendor/" in n for n in names), "A", "67-68",
        "no vendored package tree (would be unused under Case B)",
        "vendor entries: {}".format(sum(1 for n in names if "vendor/" in n)))

    # ---- A section 2: requirements.txt (lines 76-91) -----------------------
    reqs = [l.strip() for l in req.splitlines()
            if l.strip() and not l.strip().startswith("#")]
    chk(len(reqs) > 0, "A", "81-84",
        "Case B: requirements.txt is non-empty, so a fresh venv is built from it",
        "{} entries".format(len(reqs)))
    tree = ast.parse(wrap)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = (node.module if isinstance(node, ast.ImportFrom)
                   else node.names[0].name) or ""
            top = mod.split(".")[0]
            if top and top not in STDLIB and top not in PROVIDED:
                imported.add(top)
    listed = {re.split(r"[<>=!~\[]", r)[0].strip().lower() for r in reqs}
    missing = sorted(i for i in imported if i.lower() not in listed)
    chk(not missing, "A", "82-83 / B 62-66",
        "requirements.txt lists EVERY package the code imports",
        "imports {} | listed {} | MISSING {}".format(sorted(imported),
                                                     sorted(listed), missing))
    pinned = [r for r in reqs if "==" in r]
    chk(not pinned, "B", "75-76",
        "no pinned versions (>= only), so Python 3.13 can resolve", str(pinned))
    tor = [r for r in reqs if r.lower().startswith("torch")]
    ok_t = bool(tor) and all(">=2.5" in r.replace(" ", "") or
                             ">=2.6" in r.replace(" ", "") or
                             ">=3" in r.replace(" ", "") for r in tor)
    chk(ok_t, "B", "75-76", "torch >= 2.5.0 for Python 3.13", str(tor))

    # ---- A section 3: naming (lines 98-102) --------------------------------
    chk("cadc1075/op_wrapper.py" in files and "cadc1075/op_src.py" in files,
        "A", "98-99", "exact names op_wrapper.py / op_src.py")
    chk(tar_path.name.startswith("cadc1075") and
        "".join(tar_path.suffixes[-2:]) == ".tar.gz",
        "A", "141", "archive named cadc<team_id>.tar.gz",
        tar_path.name + "  (Chrome appends ' (n)' on re-download)")

    # ---- A section 4b / B: absolute paths (112-113) -------------------------
    abs_hits = []
    for m in re.finditer(r"[\"']([A-Za-z]:[\\/]|/home/|/mnt/|/Users/)[^\"'\n]{2,}",
                         wrap):
        line = wrap[:m.start()].count("\n") + 1
        abs_hits.append((line, m.group(0)[:70]))
    chk(not abs_hits, "A", "52-53 / 112-113", "no absolute paths in code",
        str(abs_hits[:4]))

    # ---- A section 5: entry point (129-134) --------------------------------
    subclass = re.search(r"class\s+\w+\s*\(\s*FloorplanOptimizer\s*\)", wrap)
    chk(bool(subclass), "A", "133-134",
        "defines a class subclassing FloorplanOptimizer",
        subclass.group(0) if subclass else "NOT FOUND")

    # ---- B: shipped binaries and the torch ABI (72-77) ---------------------
    chk(elf[:4] == b"\x7fELF", "B", "72-73",
        "the shipped binary is a Linux ELF", "magic {!r}".format(elf[:4]))
    tor_link = b"libtorch" in elf or b"torch" in elf[:200000]
    chk(not tor_link, "B", "72-77",
        "shipped binary is NOT a torch C++ extension, so no torch-ABI risk",
        "libtorch referenced: {}".format(tor_link))
    sos = sorted(n for n in files if n.endswith((".so", ".pyd")))
    chk(not sos, "B", "72-73",
        "no .so/.pyd Python extensions shipped (ABI clause does not apply)",
        str(sos))

    # ---- report ------------------------------------------------------------
    w = max(len(r[3]) for r in ROWS)
    npass = sum(1 for r in ROWS if r[0])
    print("{:<4}{:<4}{:<12}{:<{w}}".format("", "doc", "line", "requirement", w=w))
    print("-" * (20 + w))
    for ok, doc, where, what, detail in ROWS:
        print("{:<4}{:<4}{:<12}{:<{w}}".format("OK" if ok else "FAIL", doc,
                                               where, what, w=w))
        if detail and not ok:
            print("      -> {}".format(detail))
        elif detail:
            print("      .  {}".format(detail))
    print("-" * (20 + w))
    print("{} / {} checks pass".format(npass, len(ROWS)))
    print()
    print("A = beta_submission_guidelines_problemC.txt")
    print("B = C_beta_evaluation_report_hidden_final.txt")
    return 0 if npass == len(ROWS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
