#!/usr/bin/env bash
# ============================================================================
# ICCAD 2026 Problem C — 出貨包獨立驗收（Linux / WSL2）
#
# 用法（在 GPU 機的 WSL2 裡）:
#   ./verify_final.sh --tar /path/to/cadc1075.tar.gz --repo /path/to/FloorSet
#
# 常用旗標:
#   --tar   PATH     要驗的 tar（必填）
#   --repo  DIR      FloorSet checkout，供 dataset / loader / evaluator（必填）
#   --work  DIR      工作目錄，預設 ./vk_work（會被清空重建）
#   --ref   JSON     參考包的 results json ⇒ 做 0-regression 比對
#   --expect FLOAT   期望的加權總分（D = 1.2264069637381392）
#   --md5   HEX      期望的 op_wrapper.py md5（預設 = D 的）
#   --runs  N        跑幾輪，2 = 順便驗 determinism（預設 2）
#   --venv  MODE     system（用當前 python，預設）| fresh（照 Case B 建 venv）
#   --scipy-pin VER  fresh 模式下釘 scipy 版本（例如 1.18.0）
#   --skip-run       只做檔案級檢查，不跑 solver
#
# 這支腳本刻意「自己算分」，不採信包裡或組員腳本印出來的任何數字。
# ============================================================================
set -u

TAR=""; REPO=""; WORK="$(pwd)/vk_work"; REF=""; EXPECT=""; MD5=""
RUNS=2; VENVMODE="system"; SCIPY_PIN=""; SKIP_RUN=0
KIT="$(cd "$(dirname "$0")" && pwd)"

while [ $# -gt 0 ]; do
  case "$1" in
    --tar) TAR="$2"; shift 2;;
    --repo) REPO="$2"; shift 2;;
    --work) WORK="$2"; shift 2;;
    --ref) REF="$2"; shift 2;;
    --expect) EXPECT="$2"; shift 2;;
    --md5) MD5="$2"; shift 2;;
    --runs) RUNS="$2"; shift 2;;
    --venv) VENVMODE="$2"; shift 2;;
    --scipy-pin) SCIPY_PIN="$2"; shift 2;;
    --skip-run) SKIP_RUN=1; shift;;
    -h|--help) sed -n '2,25p' "$0"; exit 0;;
    *) echo "unknown flag: $1"; exit 2;;
  esac
done

fail() { echo; echo "!! $*"; exit 2; }
say()  { echo; echo "### $*"; }

[ -n "$TAR" ]  || fail "--tar 必填"
[ -f "$TAR" ]  || fail "找不到 tar: $TAR"
[ -n "$REPO" ] || fail "--repo 必填"
[ -d "$REPO" ] || fail "找不到 repo: $REPO"
TAR="$(cd "$(dirname "$TAR")" && pwd)/$(basename "$TAR")"
REPO="$(cd "$REPO" && pwd)"

RC=0
note_fail() { RC=1; echo "  >>> 這一關 FAIL <<<"; }

# ---------------------------------------------------------------- 前置檢查
say "P0  前置檢查"
PY="${PY:-python3}"
command -v "$PY" >/dev/null || fail "找不到 python3"
echo "  python   : $($PY -V 2>&1)  ($(command -v $PY))"
echo "  nproc    : $(nproc)"
echo "  glibc    : $(ldd --version 2>/dev/null | head -1)"
for f in litetestLoader.py lite_dataset_test.py liteLoader.py lite_dataset.py \
         prime_dataset.py cost.py utils.py visualize.py; do
  [ -f "$REPO/$f" ] || fail "repo 缺 loader: $f"
done
[ -f "$REPO/iccad2026contest/iccad2026_evaluate.py" ] || fail "repo 缺 iccad2026_evaluate.py"
[ -d "$REPO/LiteTensorDataTest" ] || fail "repo 缺 LiteTensorDataTest/"
echo "  dataset  : $(ls "$REPO/LiteTensorDataTest" | grep -c '^config_') 個 config_*"

# ------------------------------------------------------- G1 身分 / 合規
say "G1  身分 / 結構 / 合規（不跑 solver）"
IDARGS=""
[ -n "$MD5" ] && IDARGS="--expect-md5 $MD5"
$PY "$KIT/judge.py" identity "$TAR" $IDARGS || note_fail

say "G2  逐條規則檢查（組員的 l246，20 條，附文件行號）"
$PY "$KIT/l246_compliance.py" "$TAR" || true
echo "  ⚠️ 已知的唯一 FAIL = op_wrapper.py 的 msys64 絕對路徑（在 os.name=='nt' guard 內）。"
echo "     若新包已移除，這裡應該變成 20/20。"

if [ "$SKIP_RUN" = "1" ]; then
  say "SUMMARY"; echo "  --skip-run：只做了檔案級檢查。RC=$RC"; exit $RC
fi

# ------------------------------------------------------------ 佈置工作目錄
say "P1  佈置評分機形狀的工作目錄  $WORK"
rm -rf "$WORK"; mkdir -p "$WORK"; cd "$WORK"
tar xzf "$TAR" || fail "解 tar 失敗"
[ -d cadc1075 ] || fail "tar 裡沒有 cadc1075/"
chmod +x cadc1075/bin/constructive_linux
cp "$REPO/iccad2026contest/iccad2026_evaluate.py" cadc1075/
for f in litetestLoader.py lite_dataset_test.py liteLoader.py lite_dataset.py \
         prime_dataset.py cost.py utils.py visualize.py; do cp "$REPO/$f" .; done
ln -sfn "$REPO/LiteTensorDataTest" LiteTensorDataTest
echo "  ELF      : $(file cadc1075/bin/constructive_linux 2>/dev/null | cut -c1-90)"

# ------------------------------------------------------------------ venv
RUNPY="$PY"
if [ "$VENVMODE" = "fresh" ]; then
  say "P2  照官方 Section 2 Case B 建乾淨 venv（這是評分機真正走的路）"
  ( cd cadc1075 && "$PY" -m venv .venv_eval ) || fail "venv 建立失敗"
  ( cd cadc1075 && .venv_eval/bin/pip install --upgrade pip -q ) || true
  if ( cd cadc1075 && .venv_eval/bin/pip install -r requirements.txt ); then
    echo "  pip install: OK"
  else
    echo "  pip install: FAILED — 這正是官方 §4(a) 記錄的 Alpha 失敗模式"; note_fail
  fi
  if [ -n "$SCIPY_PIN" ]; then
    ( cd cadc1075 && .venv_eval/bin/pip install -q "scipy==$SCIPY_PIN" ) \
      && echo "  scipy 釘到 $SCIPY_PIN"
  fi
  RUNPY="$WORK/cadc1075/.venv_eval/bin/python"
fi
echo "  解析到的版本:"
"$RUNPY" - <<'EOF' 2>&1 | sed 's/^/    /'
for m in ("numpy", "scipy", "shapely", "torch"):
    try:
        mod = __import__(m)
        print("{:<10} {}".format(m, getattr(mod, "__version__", "?")))
    except Exception as e:
        print("{:<10} MISSING  ({})".format(m, type(e).__name__))
EOF

# ------------------------------------------------------------------- 跑
UNSET=""
for v in $(env | sed -n 's/^\(ICCAD_[A-Za-z0-9_]*\)=.*/\1/p'); do UNSET="$UNSET -u $v"; done
[ -n "$UNSET" ] && echo "  已剝除環境裡的: $(echo $UNSET | tr -d '-u ' )"

run_once() {
  local tag="$1"
  say "R$tag  官方指令，ICCAD_* 全剝除，強制 48 核池形狀"
  ( cd cadc1075 && rm -f "lp_stats_$tag.txt" &&
    env $UNSET ICCAD_ADAPTIVE_CORES=48 \
        ICCAD_SHAPE_LP_STATS="$WORK/cadc1075/lp_stats_$tag.txt" \
      "$RUNPY" -u iccad2026_evaluate.py --evaluate op_wrapper.py \
        -o "results_$tag.json" ) > "run_$tag.log" 2>&1
  local rc=$?
  echo "  exit code: $rc   log: $WORK/run_$tag.log"
  [ $rc -eq 0 ] || note_fail
  local n=0
  [ -f "cadc1075/lp_stats_$tag.txt" ] && n=$(wc -l < "cadc1075/lp_stats_$tag.txt")
  echo "  lp_stats : $n 行  (D 期望 71；0 = scipy 沒到位、LP 靜默關閉；100 = 閘沒生效)"
  tail -6 "run_$tag.log" | sed 's/^/    | /'
}

run_once 1
say "G3  分數 / feasibility"
EXPARG=""
[ -n "$EXPECT" ] && EXPARG="--expect $EXPECT --tol 1e-9"
$PY "$KIT/judge.py" score "cadc1075/results_1.json" $EXPARG || note_fail

say "G4  log 健康度"
$PY "$KIT/judge.py" logcheck "run_1.log" || note_fail

if [ "$RUNS" -ge 2 ]; then
  run_once 2
  say "G5  determinism（同機同旗標兩輪必須逐位相同）"
  $PY "$KIT/judge.py" diff "cadc1075/results_1.json" "cadc1075/results_2.json" \
      --mode determinism || note_fail
fi

if [ -n "$REF" ]; then
  say "G6  對參考包 0-regression（新包不得有任何一案比它差）"
  $PY "$KIT/judge.py" diff "$REF" "cadc1075/results_1.json" --mode regression || note_fail
fi

say "SUMMARY"
if [ $RC -eq 0 ]; then
  echo "  ✅ 全部通過。results: $WORK/cadc1075/results_1.json"
else
  echo "  ❌ 有 FAIL，逐條看上面。不要在沒解釋清楚每一條之前上傳。"
fi
echo "  總分請以本腳本自己算的為準，不要採信任何外部宣稱值。"
exit $RC
