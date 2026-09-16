#!/bin/bash
#
# 消融链 CPU watchdog: 检测"当前步已完成" → 从外部提交下一步。
#
# 背景 (2026-09-07): GPU 作业内 hep_sub 自重提交不可靠——env 有 64KB 上限,
#   而 hep_sub 组解析需要的 _CONDOR_*/HepJob_* 变量恰是 env 体积主力 (裁剪则报
#   "No resource serving for group", 保留则 env-too-big), 且调度器会静默丢弃
#   作业内重提。改为 CPU watchdog (干净 env) 从外部驱动链式推进。
#
# 驱动协议:
#   - 每 5 分钟: 读 STEP_FILE (已完成步数 = 下一步索引)
#   - 队列有 submit_ablation_chain.sh 在跑 → 等待 (记录 last_step)
#   - 队列空 & step 推进 → 提交下一步
#   - 队列空 & step 未推进 (job 崩/坏卡 exit77) → 超过 30min 重试提交
#   - step >= 9 → 全链完成, 退出
#
# 提交方式:
#   hep_sub ablation_watchdog.sh -g ghigh -cpu 1 -m 4000 -wt long \
#       -o logs/ablation_watchdog.out -e logs/ablation_watchdog.err
#

source ~/.bashrc

export PATH=$HOME/miniconda3/envs/dfei/bin:$PATH
export PYTHONUNBUFFERED=1

cd /lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn

STEP_FILE=/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn/logs/ablation_chain.step
STATE_FILE=/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn/logs/ablation_watchdog.state   # "last_step last_submit_epoch"
FAIL_FILE=/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn/logs/ablation_chain.failed      # 主脚本坏卡 exit77 时 touch
CONFIGS=(ab01_rebal ab02_b2 ab03_cl2w ab04_hinge ab05_ce ab06_source ab07_mass ab08_struct ab09_mom)
N_STEPS=${#CONFIGS[@]}
SUBMIT_ARGS="-g ghigh -gpu 1 -cpu 4 -m 64000 -wt long -o logs/ablation_chain.out -e logs/ablation_chain.err"
RETRY_WINDOW=1800          # 队列空且 step 未推进超过该秒数才重试提交 (兜底)

echo "========================================"
echo "JOB ID      : $_CONDOR_IHEP_JOB_ID"
echo "START TIME  : $(date)"
echo "WATCHDOG    : ablation_watchdog 启动 (共 $N_STEPS 步)"
echo "========================================"

[ -f "$STEP_FILE" ] || echo "0" > "$STEP_FILE"

last_step=-1
last_ts=0
[ -f "$STATE_FILE" ] && read -r last_step last_ts < "$STATE_FILE"

while :; do
  T=$(cat "$STEP_FILE" 2>/dev/null || echo 0)
  NOW=$(date +%s)

  if [ "$T" -ge "$N_STEPS" ]; then
    echo "[watchdog] $(date +%H:%M) 全部 $N_STEPS 步完成 (step=$T), 退出"
    rm -f "$STATE_FILE"
    exit 0
  fi

  if hep_q 2>/dev/null | grep "submit_ablation_chain.sh" | grep -qE "^[0-9]+\.[0-9]+\s+guoqingxiang"; then
    echo "[watchdog] $(date +%H:%M) step=$T (${CONFIGS[$T]}) job 在队列/运行, 等待"
    last_step=$T
  else
    NEED=0
    if [ "$T" -ne "$last_step" ]; then
      NEED=1                                                  # step 推进 → 提交下一步
    elif [ -f "$FAIL_FILE" ] && [ "$(stat -c %Y "$FAIL_FILE" 2>/dev/null || echo 0)" -gt "$last_ts" ]; then
      NEED=1                                                  # 坏卡 exit77 → 立即重试
      rm -f "$FAIL_FILE"
    elif [ $((NOW - last_ts)) -gt "$RETRY_WINDOW" ]; then
      NEED=1                                                  # 兜底: 长期无进展
    fi
    if [ "$NEED" -eq 1 ]; then
      echo "[watchdog] $(date +%H:%M) 提交 step=$T/${N_STEPS} (${CONFIGS[$T]})  [last=$last_step @$last_ts]"
      hep_sub submit_ablation_chain.sh $SUBMIT_ARGS
      echo "$T $NOW" > "$STATE_FILE"
      last_step=$T
      last_ts=$NOW
    else
      echo "[watchdog] $(date +%H:%M) step=$T 队列空但无需动作 (等推进/失败标记/超时)"
    fi
  fi
  sleep 300
done
