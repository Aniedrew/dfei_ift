#!/bin/bash
#
# watch_cern_retry.sh - CERN训练自动看护/重提交
#
# 背景: gpu02 上 GPU-a2db080d 是坏的, 但调度器不知道, 会把新GPU作业分给它。
#       CERN训练因此反复以退出码77失败(预检失败)。本脚本循环重提交,
#       直到训练真正开始(出现 logs/cern_training_started.flag)为止。
#
# 用法:
#   nohup bash watch_cern_retry.sh > logs/watch_cern_retry.log 2>&1 &
#
# 退出条件:
#   - 训练开始(flag出现) -> 正常退出并微信通知
#   - 超过 MAX_ATTEMPTS 次仍没开始 -> 退出并微信通知(需人工介入)
#

BASE=/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn
LOG_DIR=$BASE/logs
SCRIPT=$BASE/submit_train_cern_new.sh
FLAG=$LOG_DIR/cern_training_started.flag
RETRY_WAIT=180        # 失败后等待秒数再重提
CHECK_INTERVAL=60     # 检查间隔秒数
MAX_ATTEMPTS=240      # 最多重提次数(约 240*3min=12h 窗口)
SEND_KEY="SCT387631TDiuLj6UNUsFTaDRjkaSWcdPv"

notify() {
  local title="$1" desc="$2"
  curl -s --connect-timeout 10 -X POST "https://sctapi.ftqq.com/${SEND_KEY}.send" \
    -d "title=${title}" -d "desp=${desc}" > /dev/null 2>&1
}

# 清理旧标志(全新一轮看护)
rm -f "$FLAG"

JOBID=""
attempt=0

submit_job() {
  attempt=$((attempt+1))
  local out
  out=$(hep_sub "$SCRIPT" -g ghigh -gpu 1 -cpu 4 -m 32000 -wt long \
        -o "$LOG_DIR/train_cern_new.out" -e "$LOG_DIR/train_cern_new.err" 2>&1)
  JOBID=$(echo "$out" | grep -oP 'cluster \K[0-9]+')
  echo "[$(date)] attempt=$attempt submit -> job=$JOBID ($out)"
  if [ -z "$JOBID" ]; then
    echo "[$(date)] ERROR: 未能解析作业ID, 30秒后重试"
    sleep 30
    submit_job
  fi
}

job_running() {
  [ -z "$JOBID" ] && return 1
  hep_q -u guoqingxiang 2>/dev/null | grep -q "$JOBID"
}

echo "[$(date)] watch_cern_retry started (max attempts=$MAX_ATTEMPTS, retry wait=${RETRY_WAIT}s)"
submit_job

while true; do
  sleep "$CHECK_INTERVAL"

  if [ -f "$FLAG" ]; then
    echo "[$(date)] 训练已真正开始(flag出现), 看护结束"
    notify "[DFEI] ✅ CERN训练已开始" "## CERN训练已成功拿到好的GPU并开始训练

| 项目 | 值 |
|------|-----|
| **作业ID** | ${JOBID} |
| **重提次数** | ${attempt} |
| **时间** | $(date) |

日志: \`logs/train_cern_new.out\`"
    exit 0
  fi

  if ! job_running; then
    echo "[$(date)] job $JOBID 已结束且训练未开始, 等待${RETRY_WAIT}s后重提交"
    sleep "$RETRY_WAIT"
    if [ -f "$FLAG" ]; then
      echo "[$(date)] 训练已真正开始(flag出现), 看护结束"
      notify "[DFEI] ✅ CERN训练已开始" "## CERN训练已成功拿到好的GPU并开始训练

| 项目 | 值 |
|------|-----|
| **作业ID** | ${JOBID} |
| **重提次数** | ${attempt} |
| **时间** | $(date) |

日志: \`logs/train_cern_new.out\`"
      exit 0
    fi
    if [ "$attempt" -ge "$MAX_ATTEMPTS" ]; then
      echo "[$(date)] 已达最大重提次数 $MAX_ATTEMPTS, 放弃。请检查GPU集群状态。"
      notify "[DFEI] ❌ CERN训练看护超时" "## 已达最大重提次数(${MAX_ATTEMPTS})仍未开始训练

gpu02 上的坏GPU(GPU-a2db080d)一直未修复或没有好GPU空出。
建议联系管理员禁用该GPU，或手动检查 \`hep_q -u guoqingxiang\`。"
      exit 1
    fi
    submit_job
  fi
done
