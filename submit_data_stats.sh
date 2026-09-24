#!/bin/bash
#
# 0904 真实数据(无真值) 推理行为统计 提交脚本
# 用法: hep_sub submit_data_stats.sh -argu "<config文件名> [版本号列表] [节点名]" \
#            -g ghigh -gpu 1 -cpu 4 -m 32000 -wt mid -o logs/stats.out -e logs/stats.err
#   $1 = config 文件名 (位于 config_files/ 下), 例如 eval_0904real.yaml
#   $2 = (可选) 逗号分隔的版本号, 默认 31,38,47,557
#   $3 = (可选) 指定 worker 节点 (重试时固定同一节点)
#
# 重试机制与 submit_eval.sh 一致: PREFLIGHT 失败或运行时 CUDA 错误时立即重排,
# 并用 env -i 干净登录 shell 重提 (避免 env 逐代累积越过 64KB 上限把重排链打断)。

CONFIG_FILE="${1:?usage: submit_data_stats.sh <config.yaml> [versions] [node]}"
VERSIONS="${2:-31,38,47,557}"
NODE="${3:-}"

source ~/.bashrc

export PATH=$HOME/miniconda3/envs/dfei/bin:$PATH
export PYTHONPATH=/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn:$PYTHONPATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn

echo "========================================"
echo "JOB ID      : $_CONDOR_IHEP_JOB_ID"
echo "HOST        : $(hostname)"
echo "START TIME  : $(date)"
echo "GPU         : $CUDA_VISIBLE_DEVICES"
echo "CONFIG      : $CONFIG_FILE"
echo "VERSIONS    : $VERSIONS"
echo "========================================"

MAX_RETRY=1000
RETRY_SLEEP=60
RETRY_COUNT_FILE=/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn/logs/stats_${CONFIG_FILE%.yaml}.retry_count

retry_or_exit() {
  local MSG="$1"
  local RC="$2"
  echo "[RETRY] $MSG"
  N=0
  [ -f "$RETRY_COUNT_FILE" ] && N=$(cat "$RETRY_COUNT_FILE")
  if [ "$N" -lt "$MAX_RETRY" ]; then
    N=$((N+1))
    echo "$N" > "$RETRY_COUNT_FILE"
    echo "[RETRY] 第 $N/$MAX_RETRY 次, ${RETRY_SLEEP}s 后重排..."
    sleep $RETRY_SLEEP
    echo "[RETRY] 重新提交 submit_data_stats.sh $CONFIG_FILE $VERSIONS ${NODE} (原作业 ${_CONDOR_IHEP_JOB_ID:-unknown}, GPU ${CUDA_VISIBLE_DEVICES})"
    unset CUDA_VISIBLE_DEVICES NVIDIA_VISIBLE_DEVICES
    RESUB=$(mktemp /tmp/resub_stats_XXXXXX.sh)
    {
      echo '#!/bin/bash'
      printf 'cd %q || exit 1\n' "$PWD"
      printf 'exec hep_sub submit_data_stats.sh -argu %q %q' "$CONFIG_FILE" "$VERSIONS"
      [ -n "$NODE" ] && printf ' %q' "$NODE"
      printf ' -g ghigh -gpu 1 -cpu 4 -m 32000 -wt mid -o %q -e %q' \
        "logs/stats_${CONFIG_FILE%.yaml}.out" "logs/stats_${CONFIG_FILE%.yaml}.err"
      [ -n "$NODE" ] && printf ' -wn %q' "$NODE"
      echo
    } > "$RESUB"
    env -i HOME="$HOME" USER="$USER" LOGNAME="$USER" SHELL=/bin/bash TERM=dumb /bin/bash -l "$RESUB"
    RC_SUB=$?
    rm -f "$RESUB"
    [ "$RC_SUB" -ne 0 ] && {
      echo "[RETRY] ⚠️ 重提失败 (rc=$RC_SUB), 重排链已断"
      curl -s --connect-timeout 10 -X POST https://sctapi.ftqq.com/SCT387631TDiuLj6UNUsFTaDRjkaSWcdPv.send \
        -d "title=[DFEI] ⚠️ 统计 ${CONFIG_FILE} 重排链断了" \
        -d "desp=## hep_sub 重提失败 (rc=$RC_SUB), 该作业不会再自动重排" > /dev/null 2>&1
    }
    echo "[RETRY] 已重提, 本次退出"
    exit 0
  fi
  echo "[RETRY] 已达 ${MAX_RETRY} 次上限, 放弃"
  curl -s --connect-timeout 10 -X POST https://sctapi.ftqq.com/SCT387631TDiuLj6UNUsFTaDRjkaSWcdPv.send \
    -d "title=[DFEI] ⚠️ 统计作业 ${_CONDOR_IHEP_JOB_ID:-unknown} 分到坏GPU" \
    -d "desp=## GPU不可用(重试${MAX_RETRY}次后仍失败)" > /dev/null 2>&1
  exit $RC
}

# === GPU 预检 (快速失败, 避免分到坏GPU白跑; 含 matmul 测试以真正触发显存分配) ===
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
PREFLIGHT_RC=$?
if [ $PREFLIGHT_RC -ne 0 ]; then
  echo "[PREFLIGHT] FAIL (rc=$PREFLIGHT_RC): 分配的GPU不可用"
  retry_or_exit "GPU预检失败, 自动重试" $PREFLIGHT_RC
fi

# 运行推理行为统计 (无真值, 不写 LHCb_logs; 只出 CSV)
python3 -u analyze_data_inference.py --config config_files/${CONFIG_FILE} --versions ${VERSIONS}

EXIT_CODE=$?
echo "========================================"
echo "EXIT CODE   : $EXIT_CODE"
echo "END TIME    : $(date)"
echo "========================================"

if [ $EXIT_CODE -ne 0 ]; then
  if grep -qiE "CUDA error|busy or unavailable|out of memory|unknown error" logs/stats_${CONFIG_FILE%.yaml}.err; then
    retry_or_exit "运行时CUDA错误, 自动重试" $EXIT_CODE
  fi
fi

JOB_ID="${_CONDOR_IHEP_JOB_ID:-unknown}"
STATUS="✅ 完成"
[ $EXIT_CODE -ne 0 ] && STATUS="❌ 失败"
curl -s --connect-timeout 10 -X POST https://sctapi.ftqq.com/SCT387631TDiuLj6UNUsFTaDRjkaSWcdPv.send \
  -d "title=[DFEI] ${STATUS} 数据统计 Job ${JOB_ID}" \
  -d "desp=## 数据推理统计 ${JOB_ID} ${STATUS}
| 配置 | ${CONFIG_FILE} |
| 版本 | ${VERSIONS} |
| 退出码 | ${EXIT_CODE} |" > /dev/null 2>&1

exit $EXIT_CODE
