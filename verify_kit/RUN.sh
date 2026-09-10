#!/usr/bin/env bash
# ===========================================================================
# GPU 機 WSL2 — RF-SAFE 出貨包的 Linux 驗收（計分平台）
#
#   bash RUN.sh --data /path/to/dir_containing_LiteTensorDataTest
#
# 選項:
#   --data DIR    必填。該目錄底下要有 LiteTensorDataTest/（100 個 config_*）
#   --skip-d      不跑 D 對照臂（少一半時間，但就沒有 0-regression 比對）
#   --runs2       RF-SAFE 跑兩輪驗 determinism（多 ~1/3 時間）
#   --python P    指定 python（預設 python3）
#
# 這支不需要網路、不需要 git、不需要組員的任何腳本。
# ===========================================================================
set -u
KIT="$(cd "$(dirname "$0")" && pwd)"
DATA=""; SKIP_D=0; RUNS2=0; PY="python3"

while [ $# -gt 0 ]; do
  case "$1" in
    --data) DATA="$2"; shift 2;;
    --skip-d) SKIP_D=1; shift;;
    --runs2) RUNS2=1; shift;;
    --python) PY="$2"; shift 2;;
    -h|--help) sed -n '2,18p' "$0"; exit 0;;
    *) echo "unknown flag: $1"; exit 2;;
  esac
done

RC=0
bad() { RC=1; echo "   >>> FAIL <<<"; }
sec() { echo; echo "=============================================================="; \
        echo "$*"; echo "=============================================================="; }

# ------------------------------------------------------------- P0 前置
sec "P0  前置檢查"
[ -n "$DATA" ] || { echo "!! 要給 --data <含 LiteTensorDataTest 的目錄>"; exit 2; }
[ -d "$DATA/LiteTensorDataTest" ] || { echo "!! $DATA 底下沒有 LiteTensorDataTest/"; exit 2; }
DATA="$(cd "$DATA" && pwd)"
echo "  dataset : $(ls "$DATA/LiteTensorDataTest" | grep -c '^config_') 個 config_*  ($DATA)"

command -v "$PY" >/dev/null || { echo "!! 找不到 $PY"; exit 2; }
echo "  python  : $($PY -V 2>&1)"

# bin/constructive_linux 的實測需求 = GLIBC_2.34（解析 .gnu.version_r 得到，
# 不是抄來的；另外它沒有任何 GLIBCXX/CXXABI 依賴，libstdc++ 是靜態連進去的）。
# Ubuntu 22.04 = 2.35 OK；20.04 = 2.31 不行。
GLIBC="$(ldd --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+$')"
echo "  glibc   : ${GLIBC:-unknown}   (ELF 需要 >= 2.34)"
if [ -n "$GLIBC" ] && \
   [ "$(printf '%s\n2.34\n' "$GLIBC" | sort -V | head -1)" != "2.34" ]; then
  echo "  ⚠️ glibc < 2.34 ⇒ bundled ELF 過不了 1-block smoke，會落到編譯鏈"
  echo "     （g++ / clang++ / c++），編得起來就照常跑，只是驗到的不是出貨的那顆 ELF。"
  for cc in g++ clang++ c++; do
    command -v $cc >/dev/null && echo "     找到編譯器: $cc" && break
  done
  command -v g++ >/dev/null || command -v clang++ >/dev/null || \
    command -v c++ >/dev/null || {
      echo "     🚨 而且這台沒有任何 C++ 編譯器 ⇒ 會無聲退回 Python SA"
      echo "        （症狀：Total 10.0000 配 Feasible 100/100）"; bad; }
fi

$PY - <<'EOF'
import importlib, sys
for m in ("numpy", "scipy", "shapely", "torch"):
    try:
        mod = importlib.import_module(m)
        print("  %-9s %s" % (m, getattr(mod, "__version__", "?")))
    except Exception as e:
        note = "裸 import，缺了整包直接死" if m == "torch" else \
               ("缺了 LP 整條靜默關閉，分數少 ~5%" if m == "scipy" else "缺了會死")
        print("  %-9s MISSING (%s)  <- %s" % (m, type(e).__name__, note))
EOF
echo "  ⚠️ 組員的 1.2178289924684162 是在她的 scipy 上量的。shape LP 高度退化，"
echo "     不同 scipy 版本會落在同一個 LP 的不同最佳解 ⇒ 用不變式判，不要用逐位相等。"

# ------------------------------------------------------------- P1 佈置
sec "P1  佈置評分機形狀的工作目錄"
WORK="$KIT/work"; rm -rf "$WORK"
ARMS="rfsafe"; [ "$SKIP_D" = "0" ] && ARMS="d $ARMS"
for arm in $ARMS; do
  mkdir -p "$WORK/$arm"
  cp "$KIT"/loaders/*.py "$WORK/$arm/"
  case "$arm" in
    d)      tar xzf "$KIT/cadc1075_D.tar.gz" -C "$WORK/$arm";;
    rfsafe) tar xzf "$KIT/cadc1075.tar.gz"   -C "$WORK/$arm";;
  esac
  cp "$KIT/iccad2026_evaluate.py" "$WORK/$arm/cadc1075/"
  chmod +x "$WORK/$arm/cadc1075/bin/constructive_linux"
  echo "  $arm  op_wrapper md5 = $(md5sum "$WORK/$arm/cadc1075/op_wrapper.py" | cut -d' ' -f1)"
done
echo "  期望: d = 1c326784de7cd9246cd1f380e2842668"
echo "        rfsafe = 62db6ee4569b31ddc8c546ccf3e7cd0b"

# ------------------------------------------------------------- P2 跑
UNSET=""
for v in $(env | sed -n 's/^\(ICCAD_[A-Za-z0-9_]*\)=.*/\1/p'); do UNSET="$UNSET -u $v"; done
[ -n "$UNSET" ] && echo "  已剝除殘留的 ICCAD_*"

run() {   # run <arm> <tag>
  local arm="$1" tag="$2" d="$WORK/$1/cadc1075"
  sec "R  $tag  （官方指令，ICCAD_* 全剝除，強制 48 核池形狀）"
  echo "  開始 $(date +%H:%M:%S)"
  ( cd "$d" && env $UNSET ICCAD_ADAPTIVE_CORES=48 \
      ICCAD_SHAPE_LP_STATS="$d/lp_stats_$tag.txt" \
      "$PY" -u iccad2026_evaluate.py --evaluate op_wrapper.py \
        -d "$DATA" -o "results_$tag.json" ) > "$WORK/$tag.out" 2> "$WORK/$tag.err"
  local rc=$?
  local n=0; [ -f "$d/lp_stats_$tag.txt" ] && n=$(wc -l < "$d/lp_stats_$tag.txt")
  echo "  結束 $(date +%H:%M:%S)   exit=$rc   lp_stats=$n 行"
  [ $rc -eq 0 ] || bad
  grep -qi "SA fallback" "$WORK/$tag.err" && { echo "  🚨 log 裡有 SA fallback"; bad; }
  # bundled-first 的判準：編譯鏈會把產物寫成 <pkg>/constructive.exe。
  # 它出現 = 出貨的 ELF 沒被用到（smoke 沒過），量到的就不是評分機會跑的那顆。
  if [ -e "$d/constructive.exe" ]; then
    echo "  🚨 出現 $d/constructive.exe ⇒ bundled ELF 沒被使用，這輪量的是現編的 binary"
    bad
  else
    echo "  bundled-first OK（無 constructive.exe 產物）"
  fi
}

[ "$SKIP_D" = "0" ] && run d d1
run rfsafe rf1
[ "$RUNS2" = "1" ] && run rfsafe rf2

# ------------------------------------------------------------- P3 判定
sec "G  判定"
RF="$WORK/rfsafe/cadc1075/results_rf1.json"
echo "--- RF-SAFE 分數（期望 1.2178289924684162，容差放寬給 scipy 版本差） ---"
$PY "$KIT/judge.py" score "$RF" --expect 1.2178289924684162 --tol 2e-3 || bad
echo
echo "--- log ---"
$PY "$KIT/judge.py" logcheck "$WORK/rf1.err" || bad

if [ "$RUNS2" = "1" ]; then
  echo; echo "--- determinism（同機兩輪必須逐位相同） ---"
  $PY "$KIT/judge.py" diff "$RF" "$WORK/rfsafe/cadc1075/results_rf2.json" \
      --mode determinism || bad
fi

if [ "$SKIP_D" = "0" ]; then
  echo; echo "--- 對 D 的 0-regression（組員宣稱 Linux 12 movers / 0 worse） ---"
  $PY "$KIT/judge.py" score "$WORK/d/cadc1075/results_d1.json" \
      --expect 1.2264069637381392 --tol 2e-3
  echo
  $PY "$KIT/judge.py" diff "$WORK/d/cadc1075/results_d1.json" "$RF" \
      --mode regression || bad
fi

sec "SUMMARY"
if [ $RC -eq 0 ]; then
  echo "  ✅ 全過。把上面整段輸出貼回給我。"
else
  echo "  ❌ 有 FAIL，把上面整段輸出貼回給我，不要自己判斷要不要上傳。"
fi
echo "  結果檔在 $WORK"
exit $RC
