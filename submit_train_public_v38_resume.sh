#!/bin/bash
#
# DFEI CERN public_v38 训练 (PV 分簇: 可训练 cluster 头 + 温度退火 + 课程过渡 + val 对齐)
# v40 失败修复: ①val 也切子图 ②truth->cluster 课程式过渡 ③训练侧子图数上限
#
# 提交方式:
#   hep_sub submit_train_public_v38.sh -g ghigh -gpu 1 -cpu 4 -m 32000 -wt long \
#       -o logs/train_public_v38.out -e logs/train_public_v38.err
#

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
echo "PYTHON      : $(which python3)"
echo "GPU         : $CUDA_VISIBLE_DEVICES"
echo "CONFIG      : config_files/train_public_v38_resume.yaml"
echo "========================================"

# === 前置检查 1: resume_ckpt 是否已填 ===
if grep -qE 'resume_ckpt:\s*".*TODO' config_files/train_public_v38_resume.yaml; then
  echo "[CHECK] FAIL: train_public_v38_resume.yaml 的 resume_ckpt 仍是 TODO"
  exit 1
fi
echo "[CHECK] resume_ckpt 已填"

# === GPU 预检 (含 matmul, 失败自动重排, 不指定节点) ===
echo "[PREFLIGHT] GPU check at $(date), device=$CUDA_VISIBLE_DEVICES"
nvidia-smi -L 2>&1 | head -3
python3 -u -c "
import torch
if not torch.cuda.is_available():
    print('[PREFLIGHT] FAIL: torch.cuda.is_available()=False')
    raise SystemExit(77)
p = torch.cuda.get_device_properties(0)
print(f'[PREFLIGHT] OK: {p.name} (cap {p.major}.{p.minor}, mem {p.total_memory/1024**3:.1f} GB)')
a = torch.randn(500,500,device='cuda')
b = (a @ a).sum().item()
print('[PREFLIGHT] matmul OK, sum=%.3f' % b)
"
PREFLIGHT_RC=$?
if [ $PREFLIGHT_RC -ne 0 ]; then
  echo "[PREFLIGHT] FAIL (rc=$PREFLIGHT_RC): 分配的GPU不可用"
  # === 自动重试: 失败后立即重排, 几乎无限次 ===
  MAX_RETRY=1000
  RETRY_SLEEP=60
  RETRY_COUNT_FILE=/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn/logs/public_v38_resume.retry_count
  N=0
  [ -f "$RETRY_COUNT_FILE" ] && N=$(cat "$RETRY_COUNT_FILE")
  if [ "$N" -lt "$MAX_RETRY" ]; then
    N=$((N+1))
    echo "$N" > "$RETRY_COUNT_FILE"
    echo "[RETRY] 第 $N/$MAX_RETRY 次, sleep ${RETRY_SLEEP}s 后重排..."
    sleep $RETRY_SLEEP
    echo "[RETRY] 重新提交 submit_train_public_v38_resume.sh (原作业 ${_CONDOR_IHEP_JOB_ID:-unknown}, GPU ${CUDA_VISIBLE_DEVICES})"
    hep_sub submit_train_public_v38_resume.sh -g ghigh -gpu 1 -cpu 4 -m 32000 -wt long -o logs/train_public_v38_resume.out -e logs/train_public_v38_resume.err
    echo "[RETRY] 已重提, 本次退出"
    exit 0
  fi
  echo "[RETRY] 已达 ${MAX_RETRY} 次上限, 放弃"
  echo "========================================"
  echo "EXIT CODE   : 77"
  echo "END TIME    : $(date)"
  echo "========================================"
  JOB_ID="${_CONDOR_IHEP_JOB_ID:-unknown}"
  curl -s --connect-timeout 10 -X POST https://sctapi.ftqq.com/SCT387631TDiuLj6UNUsFTaDRjkaSWcdPv.send \
    -d "title=[DFEI] ⚠️ public_v38训练 Job ${JOB_ID} 分到坏GPU" \
    -d "desp=## public_v38训练作业 ${JOB_ID} GPU预检失败(重试${MAX_RETRY}次后仍失败)
| 作业 | ${JOB_ID} |
| 主机 | $(hostname) |
| GPU | ${CUDA_VISIBLE_DEVICES} |
| 时间 | $(date) |" > /dev/null 2>&1
  exit 77
fi

# 标记训练已真正开始
touch /lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn/logs/public_v38_resume_started.flag

python3 -u wmpgnn/analysis/trainer.py --config config_files/train_public_v38_resume.yaml

EXIT_CODE=$?
echo "========================================"
echo "EXIT CODE   : $EXIT_CODE"
echo "END TIME    : $(date)"
echo "========================================"

# Server酱微信通知
JOB_ID="${_CONDOR_IHEP_JOB_ID:-unknown}"
STATUS="✅ 完成"
[ $EXIT_CODE -ne 0 ] && STATUS="❌ 失败"
curl -s --connect-timeout 10 -X POST https://sctapi.ftqq.com/SCT387631TDiuLj6UNUsFTaDRjkaSWcdPv.send \
  -d "title=[DFEI] ${STATUS} public_v38训练 Job ${JOB_ID}" \
  -d "desp=## public_v38训练作业 ${JOB_ID} ${STATUS}

| 项目 | 值 |
|------|-----|
| **作业ID** | ${JOB_ID} |
| **状态** | ${STATUS} |
| **主机** | $(hostname) |
| **配置** | train_public_v38_resume.yaml (ep66续训到100, 公开数据) |
| **结束时间** | $(date) |
| **退出码** | ${EXIT_CODE} |

### 查看日志
\`\`\`bash
tail -50 /lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn/logs/train_public_v38.out
\`\`\`" > /dev/null 2>&1

exit $EXIT_CODE
