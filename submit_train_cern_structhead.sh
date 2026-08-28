#!/bin/bash
#
# DFEI CERN struct_head 最小实验训练 (第8个监督头: 节点结构监督, depth 主 + RC 辅)
# 用户提议: source_head 只预测"根"(1 bit) 信息量低; 连续结构监督让主干学
# "节点在衰变树中的位置/层级", 与 LCA 边分类形成一致性约束。
# 从 v38 best (ep105) 续训 20 epoch, 与 mass-only 实验同起点对比。
# ⚠️ 前置:
#   1. 已实现 truth_chain_structure (topk_selection.py) + _struct_loss (dfei_lightning_module.py)
#   2. resume_ckpt 已填 v38 best checkpoint (config_files/train_CERN_v38_structhead.yaml)
#
# 提交方式:
#   hep_sub submit_train_cern_structhead.sh -g ghigh -gpu 1 -cpu 4 -m 32000 -wt long \
#       -o logs/train_cern_structhead.out -e logs/train_cern_structhead.err
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
echo "CONFIG      : config_files/train_CERN_v38_structhead.yaml"
echo "========================================"

# === 前置检查 1: truth_chain_structure 是否已实现 ===
if ! grep -q "def truth_chain_structure" wmpgnn/reconstruction/topk_selection.py; then
  echo "[CHECK] FAIL: truth_chain_structure 未实现 (topk_selection.py), 拒绝提交"
  exit 1
fi
echo "[CHECK] truth_chain_structure 代码已实现"

# === 前置检查 2: _struct_loss 是否已实现 ===
if ! grep -q "def _struct_loss" wmpgnn/lightning_module/dfei_lightning_module.py; then
  echo "[CHECK] FAIL: _struct_loss 未实现 (dfei_lightning_module.py), 拒绝提交"
  exit 1
fi
echo "[CHECK] _struct_loss 代码已实现"

# === 前置检查 3: resume_ckpt 是否已填 ===
if grep -qE 'resume_ckpt:\s*".*TODO' config_files/train_CERN_v38_structhead.yaml; then
  echo "[CHECK] FAIL: train_CERN_v38_structhead.yaml 的 resume_ckpt 仍是 TODO"
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
  MAX_RETRY=1000
  RETRY_SLEEP=60
  RETRY_COUNT_FILE=/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn/logs/structhead.retry_count
  N=0
  [ -f "$RETRY_COUNT_FILE" ] && N=$(cat "$RETRY_COUNT_FILE")
  if [ "$N" -lt "$MAX_RETRY" ]; then
    N=$((N+1))
    echo "$N" > "$RETRY_COUNT_FILE"
    echo "[RETRY] 第 $N/$MAX_RETRY 次, sleep ${RETRY_SLEEP}s 后重排..."
    sleep $RETRY_SLEEP
    echo "[RETRY] 重新提交 submit_train_cern_structhead.sh (原作业 ${_CONDOR_IHEP_JOB_ID:-unknown}, GPU ${CUDA_VISIBLE_DEVICES})"
    hep_sub submit_train_cern_structhead.sh -g ghigh -gpu 1 -cpu 4 -m 32000 -wt long -o logs/train_cern_structhead.out -e logs/train_cern_structhead.err
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
    -d "title=[DFEI] ⚠️ struct_head训练 Job ${JOB_ID} 分到坏GPU" \
    -d "desp=## struct_head训练作业 ${JOB_ID} GPU预检失败(重试${MAX_RETRY}次后仍失败)
| 作业 | ${JOB_ID} |
| 主机 | $(hostname) |
| GPU | ${CUDA_VISIBLE_DEVICES} |
| 时间 | $(date) |" > /dev/null 2>&1
  exit 77
fi

# 标记训练已真正开始
touch /lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn/logs/structhead_started.flag

python3 -u wmpgnn/analysis/trainer.py --config config_files/train_CERN_v38_structhead.yaml

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
  -d "title=[DFEI] ${STATUS} struct_head训练 Job ${JOB_ID}" \
  -d "desp=## struct_head训练作业 ${JOB_ID} ${STATUS}

| 项目 | 值 |
|------|-----|
| **作业ID** | ${JOB_ID} |
| **状态** | ${STATUS} |
| **主机** | $(hostname) |
| **配置** | train_CERN_v38_structhead.yaml (节点结构监督: depth主+RC辅) |
| **结束时间** | $(date) |
| **退出码** | ${EXIT_CODE} |

### 查看日志
\`\`\`bash
tail -50 /lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn/logs/train_cern_structhead.out
\`\`\`" > /dev/null 2>&1

exit $EXIT_CODE
