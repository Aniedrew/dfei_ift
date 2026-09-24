#!/bin/bash
#
# 通用 GPU 评估提交脚本
# 用法: hep_sub submit_eval.sh -argu "<config文件名> [节点名]" -g ghigh -gpu 1 -cpu 4 -m 32000 -wt mid -o logs/eval.out -e logs/eval.err
#   $1 = config 文件名 (位于 config_files/ 下), 例如 eval_CERN_normed.yaml
#   $2 = (可选) 指定 worker 节点 (如 gpu09), 通过 -wn 提交; 重试时固定同一节点
#
# 重试机制: PREFLIGHT 失败或运行时 CUDA 错误(忙卡/显存不足)时,
#   立即重新 hep_sub, 几乎无限次 (直到拿到可用GPU跑起来)。

CONFIG_FILE="${1:?usage: submit_eval.sh <config.yaml> [node]}"
NODE="${2:-}"

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
echo "========================================"

# === 自动重试: 失败后立即重排, 几乎无限次 (直到拿到可用GPU跑起来) ===
MAX_RETRY=1000
RETRY_SLEEP=60
RETRY_COUNT_FILE=/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn/logs/eval_${CONFIG_FILE%.yaml}.retry_count

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
    echo "[RETRY] 重新提交 submit_eval.sh $CONFIG_FILE ${NODE} (原作业 ${_CONDOR_IHEP_JOB_ID:-unknown}, GPU ${CUDA_VISIBLE_DEVICES})"
    # 清掉父作业继承的 GPU 绑定, 否则重提会被"钉"回同一张坏卡 (死循环)
    unset CUDA_VISIBLE_DEVICES NVIDIA_VISIBLE_DEVICES
    # 用 env -i 起一个干净登录 shell 再提交。自重排会把环境变量逐代累积,
    # 约 200 代后越过 64KB 上限, hep_sub 直接拒绝:
    #   ERROR: The total size of your environment variables exceeds the
    #   system's 64KB limit (currently 65748 bytes)
    # 这条错误发生在重提瞬间, 重排链就此静默断掉 (2026-09-17 踩到:
    # 6 条链在 181-216 代之间集体阵亡, 队列变空)。干净登录 shell 只有 ~2.3KB。
    RESUB=$(mktemp /tmp/resub_eval_XXXXXX.sh)
    {
      echo '#!/bin/bash'
      printf 'cd %q || exit 1\n' "$PWD"
      printf 'exec hep_sub submit_eval.sh -argu %q' "$CONFIG_FILE"
      [ -n "$NODE" ] && printf ' %q' "$NODE"
      printf ' -g ghigh -gpu 1 -cpu 4 -m 32000 -wt mid -o %q -e %q' \
        "logs/eval_${CONFIG_FILE%.yaml}.out" "logs/eval_${CONFIG_FILE%.yaml}.err"
      [ -n "$NODE" ] && printf ' -wn %q' "$NODE"
      echo
    } > "$RESUB"
    env -i HOME="$HOME" USER="$USER" LOGNAME="$USER" SHELL=/bin/bash TERM=dumb /bin/bash -l "$RESUB"
    RC_SUB=$?
    rm -f "$RESUB"
    [ "$RC_SUB" -ne 0 ] && {
      echo "[RETRY] ⚠️ 重提失败 (rc=$RC_SUB), 重排链已断"
      curl -s --connect-timeout 10 -X POST https://sctapi.ftqq.com/SCT387631TDiuLj6UNUsFTaDRjkaSWcdPv.send \
        -d "title=[DFEI] ⚠️ 评估 ${CONFIG_FILE} 重排链断了" \
        -d "desp=## hep_sub 重提失败 (rc=$RC_SUB), 该作业不会再自动重排
| 配置 | ${CONFIG_FILE} |
| 原作业 | ${_CONDOR_IHEP_JOB_ID:-unknown} |
| 第几次 | ${N}/${MAX_RETRY} |
| 主机 | $(hostname) |" > /dev/null 2>&1
    }
    echo "[RETRY] 已重提, 本次退出"
    exit 0
  fi
  echo "[RETRY] 已达 ${MAX_RETRY} 次上限, 放弃"
  curl -s --connect-timeout 10 -X POST https://sctapi.ftqq.com/SCT387631TDiuLj6UNUsFTaDRjkaSWcdPv.send \
    -d "title=[DFEI] ⚠️ 评估作业 ${_CONDOR_IHEP_JOB_ID:-unknown} 分到坏GPU" \
    -d "desp=## GPU不可用(重试${MAX_RETRY}次后仍失败)
| 作业 | ${_CONDOR_IHEP_JOB_ID:-unknown} |
| 主机 | $(hostname) |
| 配置 | ${CONFIG_FILE} |
| GPU | ${CUDA_VISIBLE_DEVICES} |" > /dev/null 2>&1
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
b = (a @ a).sum().item()
print('[PREFLIGHT] matmul OK, sum=%.3f' % b)
"
PREFLIGHT_RC=$?
if [ $PREFLIGHT_RC -ne 0 ]; then
  echo "[PREFLIGHT] FAIL (rc=$PREFLIGHT_RC): 分配的GPU不可用"
  retry_or_exit "GPU预检失败, 自动重试" $PREFLIGHT_RC
fi

# 运行评估
python3 -u wmpgnn/analysis/evaluate.py --config config_files/${CONFIG_FILE}

EXIT_CODE=$?
echo "========================================"
echo "EXIT CODE   : $EXIT_CODE"
echo "END TIME    : $(date)"
echo "========================================"

# 运行时 CUDA 错误 (加载 checkpoint / 前向时 GPU busy 或显存不足) 也自动重试
if [ $EXIT_CODE -ne 0 ]; then
  if grep -qiE "CUDA error|busy or unavailable|out of memory|unknown error" logs/eval_${CONFIG_FILE%.yaml}.err; then
    retry_or_exit "运行时CUDA错误, 自动重试" $EXIT_CODE
  fi
fi

# Server酱微信通知
JOB_ID="${_CONDOR_IHEP_JOB_ID:-unknown}"
STATUS="✅ 完成"
[ $EXIT_CODE -ne 0 ] && STATUS="❌ 失败"
curl -s --connect-timeout 10 -X POST https://sctapi.ftqq.com/SCT387631TDiuLj6UNUsFTaDRjkaSWcdPv.send \
  -d "title=[DFEI] ${STATUS} 评估 Job ${JOB_ID}" \
  -d "desp=## 评估作业 ${JOB_ID} ${STATUS}

| 项目 | 值 |
|------|-----|
| **作业ID** | ${JOB_ID} |
| **状态** | ${STATUS} |
| **主机** | $(hostname) |
| **配置** | ${CONFIG_FILE} |
| **退出码** | ${EXIT_CODE} |

### 查看
\`\`\`bash
tail -30 /lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn/logs/eval_${CONFIG_FILE%.yaml}.out
\`\`\`" > /dev/null 2>&1

exit $EXIT_CODE
