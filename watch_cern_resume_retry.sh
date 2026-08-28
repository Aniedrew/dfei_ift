#!/bin/bash
#
# watch_cern_resume_retry.sh - CERN续训自动看护/重提交 (应对坏GPU调度)
# 用法: nohup bash watch_cern_resume_retry.sh > logs/watch_cern_resume_retry.log 2>&1 &
# 退出条件: 续训真正开始(flag出现) 或 超过 MAX_ATTEMPTS 次仍失败
#

BASE=/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn
LOG_DIR=$BASE/logs
SCRIPT=$BASE/submit_train_cern_resume.sh
FLAG=$LOG_DIR/cern_resume_started.flag
RETRY_WAIT=180
CHECK_INTERVAL=60
MAX_ATTEMPTS=240
SEND_KEY="SCT387631TDiuLj6UNUsFTaDRjkaSWcdPv"

notify() {
  local title="$1" desc="$2"
  curl -s --connect-timeout 10 -X POST "https://sctapi.ftqq.com/${SEND_KEY}.send" \
    -d "title=${title}" -d "desp=${desc}" > /dev/null 2>&1
}

rm -f "$FLAG"

JOBID=""
attempt=0

submit_job() {
  attempt=$((attempt+1))
  local out
  out=$(hep_sub "$SCRIPT" -g ghigh -gpu 1 -cpu 4 -m 32000 -wt long \
        -o "$LOG_DIR/train_cern_resume.out" -e "$LOG_DIR/train_cern_resume.err" 2>&1)
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

echo "[$(date)] watch_cern_resume_retry started (max attempts=$MAX_ATTEMPTS, retry wait=${RETRY_WAIT}s)"
submit_job

while true; do
  sleep "$CHECK_INTERVAL"

  if [ -f "$FLAG" ]; then
    echo "[$(date)] 续训已真正开始(flag出现), 看护结束"
    notify "[DFEI] ✅ CERN续训已开始" "## CERN续训已成功拿到好的GPU并开始训练

| 项目 | 值 |
|------|-----|
| **作业ID** | ${JOBID} |
| **重提次数** | ${attempt} |
| **时间** | $(date) |

日志: \`logs/train_cern_resume.out\`"
    exit 0
  fi

  if ! job_running; then
    echo "[$(date)] job $JOBID 已结束且训练未开始, 等待${RETRY_WAIT}s后重提交"
    sleep "$RETRY_WAIT"
    if [ -f "$FLAG" ]; then
      echo "[$(date)] 续训已真正开始(flag出现), 看护结束"
      notify "[DFEI] ✅ CERN续训已开始" "## CERN续训已成功拿到好的GPU并开始训练

| 项目 | 值 |
|------|-----|
| **作业ID** | ${JOBID} |
| **重提次数** | ${attempt} |
| **时间** | $(date) |

日志: \`logs/train_cern_resume.out\`"
      exit 0
    fi
    if [ "$attempt" -ge "$MAX_ATTEMPTS" ]; then
      echo "[$(date)] 已达最大重提次数 $MAX_ATTEMPTS, 放弃。请检查GPU集群状态。"
      notify "[DFEI] ❌ CERN续训看护超时" "## 已达最大重提次数(${MAX_ATTEMPTS})仍未开始续训

gpu02 上的坏GPU(GPU-a2db080d)一直未修复或没有好GPU空出。
建议联系管理员禁用该GPU。" > /dev/null 2>&1
      exit 1
    fi
    submit_job
  fi
done
