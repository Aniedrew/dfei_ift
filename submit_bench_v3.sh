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
  # [2026-10-07] 1000 在集群过载时约 29h 就耗尽 -> 改为实际无限 (等卡而不是放弃)
  if [ "$N" -lt 100000 ]; then
    echo $((N + 1)) > "$F"
    echo "[PREFLIGHT] 失败, 60s 后重排 (第 $((N + 1)) 次)"
    sleep 60
    unset CUDA_VISIBLE_DEVICES NVIDIA_VISIBLE_DEVICES
    R=$(mktemp /tmp/resub_bench3_XXXXXX.sh)
    { echo '#!/bin/bash'
      printf 'cd %q || exit 1\n' "$PWD"
      # 注意: $* 已在上面 shift 掉 events -> 必须把 $EVENTS 显式放回首位,
      #       否则重排后 events 会静默退回默认 200 (踩过)
      printf 'exec hep_sub submit_bench_v3.sh -argu %q -g ghigh -gpu 1 -cpu 4 -m 32000 -wt long -o %q -e %q\n' \
        "$EVENTS $*" "logs/bench_v3.out" "logs/bench_v3.err"
    } >"$R"
    env -i HOME="$HOME" USER="$USER" LOGNAME="$USER" SHELL=/bin/bash TERM=dumb /bin/bash -l "$R"
    rm -f "$R"
  fi
  exit 0
fi

# 拿到可用 GPU 后立刻清零重试计数。否则计数跨作业累积 -> 一旦用满 200,
# 之后提交的 bench 作业会一提交就静默退出(什么都不跑)。别删这行。
echo 0 > logs/bench_v3.retry_count

FAIL=0
for spec in "$@"; do
  V="${spec%%:*}"
  TAG="${spec#*:}"
  echo "==== [bench] $(date +%H:%M:%S) version=$V tag=$TAG"
  # thr 不在这里传: 由 analyze_prune_loss 从配置读 (各版本在各自工作点上算)
  # csv 可用环境变量 BENCH_CSV 覆盖, 便于并发跑不同 events 的 bench 时不互相追加冲突
  python3 -u analyze_prune_loss.py --config config_files/eval_bench_v3_n90.yaml \
    --version "$V" --tag "$TAG" --events "$EVENTS" \
    --out report_figs --csv_name "${BENCH_CSV:-bench_auc_v3_seed.csv}" --n_boot 200
  rc=$?
  echo "==== [bench] $(date +%H:%M:%S) $TAG EXIT=$rc"
  [ $rc -ne 0 ] && FAIL=$((FAIL + 1))
done
echo "==== [bench] ALL DONE (失败 $FAIL 个) END $(date)"
exit 0
