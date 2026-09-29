#!/bin/bash
# v3 pruning benchmark 批跑: 在一个 GPU 名额里顺序跑多个版本。
# 指标: 阈值无关 AUC/AP + **分层池/工作点池 AP (主判据)** + 链级 AND 语义 + PV 关联组,
#       并对主判据给事件级 bootstrap 置信区间。汇总写 report_figs/bench_auc_v3.csv。
# 用法: hep_sub submit_bench_v3.sh -argu "[events] [ver:tag[:thr] ...]" -g ghigh -gpu 1 -cpu 4 \
#          -m 32000 -wt long -o logs/bench_v3.out -e logs/bench_v3.err
# 默认: 200 事件, 跑 v2 表里的那批 (pseudo 目录), 便于 bench_auc.csv(v2) vs v3 直接对照。
source ~/.bashrc
export PATH=$HOME/miniconda3/envs/dfei/bin:$PATH
export PYTHONPATH=/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn:$PYTHONPATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
cd /lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn

EVENTS=200
if [ $# -ge 1 ] && [[ "$1" =~ ^[0-9]+$ ]]; then EVENTS="$1"; shift; fi
if [ $# -eq 0 ]; then
  set -- 960119:v601_ep19 962319:v623_ep19 962409:v624_ep09 962813:v628_ep13 963110:v631_ep10
fi

echo "JOB ${_CONDOR_IHEP_JOB_ID:-?} HOST $(hostname) GPU $CUDA_VISIBLE_DEVICES START $(date)"
echo "EVENTS=$EVENTS  VERSIONS($#): $*"

python3 -u -c "
import torch
assert torch.cuda.is_available(), 'cuda not available'
print('[PREFLIGHT] OK', torch.cuda.get_device_name(0))
"
if [ $? -ne 0 ]; then
  N=0; F=logs/bench_v3.retry_count
  [ -f "$F" ] && N=$(cat "$F")
  if [ "$N" -lt 200 ]; then
    echo $((N + 1)) > "$F"
    echo "[PREFLIGHT] 失败, 60s 后重排 (第 $((N + 1)) 次)"
    sleep 60
    unset CUDA_VISIBLE_DEVICES NVIDIA_VISIBLE_DEVICES
    R=$(mktemp /tmp/resub_bench3_XXXXXX.sh)
    { echo '#!/bin/bash'
      printf 'cd %q || exit 1\n' "$PWD"
      printf 'exec hep_sub submit_bench_v3.sh -argu %q -g ghigh -gpu 1 -cpu 4 -m 32000 -wt long -o logs/bench_v3.out -e logs/bench_v3.err\n' "$*"
    } >"$R"
    env -i HOME="$HOME" USER="$USER" LOGNAME="$USER" SHELL=/bin/bash TERM=dumb /bin/bash -l "$R"
    rm -f "$R"
  fi
  exit 0
fi

FAIL=0
for spec in "$@"; do
  V="${spec%%:*}"
  rest="${spec#*:}"
  TAG="${rest%%:*}"
  THR="${rest#*:}"
  [ "$THR" = "$rest" ] && THR=0.9        # 未给第三段 -> 用 0.9 (与 v2 表一致)
  echo "==== [bench] $(date +%H:%M:%S) version=$V tag=$TAG thr=$THR"
  python3 -u analyze_prune_loss.py --config config_files/eval_bench_v3_n90.yaml \
    --version "$V" --tag "$TAG" --events "$EVENTS" --thr "$THR" \
    --out report_figs --csv_name bench_auc_v3.csv --n_boot 200
  rc=$?
  echo "==== [bench] $(date +%H:%M:%S) $TAG EXIT=$rc"
  [ $rc -ne 0 ] && FAIL=$((FAIL + 1))
done
echo "==== [bench] ALL DONE (失败 $FAIL 个) END $(date)"
exit 0
