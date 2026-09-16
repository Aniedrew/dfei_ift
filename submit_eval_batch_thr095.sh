#!/bin/bash
#
# 批量阈值复评 (thr0.95): 对"曾判无提升"的版本统一以 thr0.95 重新评估,
# 检验此前的 REJECT / "不如基线" 是否为评估口径(剪枝阈值 0.9 次优)所致。
#
# 原理: evaluate.py -> adjust_config_evaluation 会读取各版本自己的
#   LHCb_logs/DFEI/version_<V>/input_config.yaml 作为架构基底,
#   再被本脚本生成的最小 eval config 覆盖 (阈值/评估设置)。
#   故无需为每版重写完整 DFEI 架构, 只要改 settings.model 与阈值即可。
#
# 幂等: 已产出 info_*__<OW>_reco.txt 的版本直接跳过 -> 中途失败重排可断点续跑。
#
# 用法:
#   hep_sub submit_eval_batch_thr095.sh -g ghigh -gpu 1 -cpu 4 -m 32000 -wt mid \
#           -o logs/eval_batch_thr095.out -e logs/eval_batch_thr095.err

source ~/.bashrc

export PATH=$HOME/miniconda3/envs/dfei/bin:$PATH
export PYTHONPATH=/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn:$PYTHONPATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn

# === 待复评版本列表 (均为"曾判无提升"或需对照的基线/正结果) ===
# 消融链: 500(基线) 501(B2) 502(cl2w) 503(hinge) 504(chainCE✅) 505(source) 506(mass)
# attention 线: 510(纯内容) 511(边bias✅) 515(v47+attn) 516(v47+attn续)
# v47 世代线: 518(v47+struct ep131) 520(v47+struct 长训)
# 容量/组合线: 48(三头) 53(asym) 512(升维继承) 514(GN宽256)
VERSIONS="48 53 500 501 502 503 504 505 506 510 511 512 514 515 516 518 520"
OW="batch_thr095"
NODE="${1:-}"   # 可选: 指定 worker 节点 (如 gpu03), 避开坏卡节点 gpu06 的活锁

MAX_RETRY=1000
RETRY_SLEEP=60
RETRY_COUNT_FILE=logs/eval_batch_thr095.retry_count
SUMMARY=logs/eval_batch_thr095_results.txt

retry_or_exit() {
  local MSG="$1"
  local RC="$2"
  echo "[RETRY] $MSG"
  N=0
  [ -f "$RETRY_COUNT_FILE" ] && N=$(cat "$RETRY_COUNT_FILE")
  if [ "$N" -lt "$MAX_RETRY" ]; then
    N=$((N+1))
    echo "$N" > "$RETRY_COUNT_FILE"
    echo "[RETRY] 第 $N/$MAX_RETRY 次, ${RETRY_SLEEP}s 后重排 (已完成版本会自动跳过)..."
    sleep $RETRY_SLEEP
    WN_ARGS=()
    ARGU_ARGS=()
    [ -n "$NODE" ] && WN_ARGS=(-wn "$NODE")
    [ -n "$NODE" ] && ARGU_ARGS=("$NODE")
    hep_sub submit_eval_batch_thr095.sh "${ARGU_ARGS[@]}" -g ghigh -gpu 1 -cpu 4 -m 32000 -wt mid \
      -o logs/eval_batch_thr095.out -e logs/eval_batch_thr095.err "${WN_ARGS[@]}"
    echo "[RETRY] 已重提, 本次退出"
    exit 0
  fi
  echo "[RETRY] 已达 ${MAX_RETRY} 次上限, 放弃"
  exit $RC
}

echo "========================================"
echo "JOB ID      : $_CONDOR_IHEP_JOB_ID"
echo "HOST        : $(hostname)"
echo "START TIME  : $(date)"
echo "GPU         : $CUDA_VISIBLE_DEVICES"
echo "VERSIONS    : $VERSIONS"
echo "========================================"

# === GPU 预检 ===
echo "[PREFLIGHT] GPU check at $(date), device=$CUDA_VISIBLE_DEVICES"
python3 -u -c "
import torch
if not torch.cuda.is_available():
    print('[PREFLIGHT] FAIL: torch.cuda.is_available()=False')
    raise SystemExit(77)
p = torch.cuda.get_device_properties(0)
print(f'[PREFLIGHT] OK: {p.name} (mem {p.total_memory/1024**3:.1f} GB)')
a = torch.randn(500,500,device='cuda')
print('[PREFLIGHT] matmul OK, sum=%.3f' % (a @ a).sum().item())
"
if [ $? -ne 0 ]; then
  retry_or_exit "GPU预检失败, 自动重试" 77
fi

mkdir -p logs/batch_cfg
touch "$SUMMARY"

for V in $VERSIONS; do
  OUT="LHCb_logs/DFEI/version_${V}/info_inclusive_00342442__${OW}_reco.txt"
  if [ -f "$OUT" ]; then
    echo "[skip] v$V 已有结果: $OUT"
    continue
  fi

  CFG="logs/batch_cfg/eval_batch_v${V}_thr095.yaml"
  cat > "$CFG" <<EOF
settings:
  data_dir: "/lzufs/user/guoqingxiang/DFEI_IFT_20260702/data/MC_normed"
  sample: [ "inclusive_00342442" ]
  nfiles: [ 20 ]
  model_arch: "DFEI"
  model: ${V}
  ngpu: 1
  ncpu: 4
  graph_mode: "None"
  node_sel: "true"
  pv_model: "None"
  calibration: False

inference:
  LCA: true
  LCA_weights: true
  node_prune: true
  node_prune_weights: true
  edge_prune: true
  edge_prune_weights: true
  pv_asso: true
  pv_asso_weights: true
  edge_prune_thr: 0.95
  node_prune_thr: 0.95

evaluate:
  sample: [ "inclusive_00342442" ]
  nfiles: [ 20 ]
  over_write: "${OW}"
EOF

  echo ""
  echo "########## [$(date)] evaluating v$V (thr0.95) ##########"
  python3 -u wmpgnn/analysis/evaluate.py --config "$CFG"
  RC=$?

  if [ $RC -ne 0 ]; then
    if grep -qiE "CUDA error|busy or unavailable|out of memory|unknown error" logs/eval_batch_thr095.err; then
      retry_or_exit "运行时CUDA错误 (v$V), 自动重试" $RC
    fi
    echo "[warn] v$V 评估失败 (rc=$RC), 继续下一个"
    echo "v$V FAILED rc=$RC $(date)" >> "$SUMMARY"
    continue
  fi

  # 汇总 AllParticles / PerfectReco
  if [ -f "$OUT" ]; then
    ALL=$(grep -m1 "all_particles" "$OUT" | grep -oE "[0-9]+\.[0-9]+" | head -1)
    PERF=$(grep -m1 "perfect_reco" "$OUT" | grep -oE "[0-9]+\.[0-9]+" | head -1)
    echo "v$V  All=$ALL  Perfect=$PERF" | tee -a "$SUMMARY"
  fi
done

echo ""
echo "========================================"
echo "全部完成  END TIME : $(date)"
echo "========================================"
echo "汇总: $SUMMARY"
cat "$SUMMARY"

exit 0
