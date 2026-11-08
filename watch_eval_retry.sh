#!/bin/bash
#
# watch_eval_retry.sh - 评估作业自动重提交 (应对坏GPU调度)
# 用法: nohup bash watch_eval_retry.sh <config.yaml> > logs/watch_eval_retry.log 2>&1 &
# 当评估作业因GPU预检失败(exit 77)时自动重提, 直到评估真正开始(作业运行超过3分钟)为止
#

BASE=/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn
CONFIG="${1:?usage: watch_eval_retry.sh <config.yaml>}"
CHECK_INTERVAL=60
RETRY_WAIT=60
MAX_ATTEMPTS=120

submit_job() {
  local out
  out=$(hep_sub "$BASE/submit_eval.sh" -argu "$CONFIG" -g ghigh -gpu 1 -cpu 4 -m 32000 -wt mid \
        -o "$BASE/logs/eval_$(basename $CONFIG .yaml).out" -e "$BASE/logs/eval_$(basename $CONFIG .yaml).err" 2>&1)
  JOBID=$(echo "$out" | grep -oP 'cluster \K[0-9]+')
  echo "[$(date)] attempt=$((++attempt)) submit -> job=$JOBID"
  SUBMIT_TIME=$(date +%s)
}

job_running() {
  [ -z "$JOBID" ] && return 1
  hep_q -u guoqingxiang 2>/dev/null | grep -q "$JOBID"
}

echo "[$(date)] watch_eval_retry started for $CONFIG"
submit_job

while true; do
  sleep "$CHECK_INTERVAL"

  if job_running; then
    # 作业运行超过3分钟 -> 说明已通过预检开始评估
    now=$(date +%s)
    if [ $((now - SUBMIT_TIME)) -gt 180 ]; then
      echo "[$(date)] job $JOBID 已运行超过3分钟, 评估进行中。看护结束。"
      exit 0
    fi
    continue
  fi

  echo "[$(date)] job $JOBID 已结束, 重提交"
  if [ "$attempt" -ge "$MAX_ATTEMPTS" ]; then
    echo "[$(date)] 达到最大重提次数, 放弃"
    exit 1
  fi
  sleep "$RETRY_WAIT"
  submit_job
done
