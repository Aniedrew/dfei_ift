#!/bin/bash
#
# DFEI CERN v39 训练 (下下轮: PV 分簇分层重建)
# 方向: 事件内 tracks 按 PV 分簇 -> 簇内独立链重建, 降低高连通图复杂度
# ⚠️ 前置:
#   1. 需实现 reconstruction.py 的 pv_cluster 分簇逻辑, 否则本作业会失败:
#      - reconstruct_heavyhadrons: 事件级剪枝后, 按 pv_cluster_assign 将 tracks 分组
#      - 每簇独立调用链重建 (reconstruct_decay), 汇总所有簇结果
#      - 参考 pv_des["pred"]/pred_pv_track_level (推理侧) 与 y_pv (truth 侧) 做分配
#   2. resume_ckpt 需填 v38 best checkpoint 路径 (config_files/train_CERN_v39.yaml)
#
# 提交方式:
#   hep_sub submit_train_cern_v39.sh -g ghigh -gpu 1 -cpu 4 -m 32000 -wt long \
#       -o logs/train_cern_v39.out -e logs/train_cern_v39.err
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
echo "CONFIG      : config_files/train_CERN_v39.yaml"
echo "========================================"

# === 前置检查 1: pv_cluster 分簇代码是否已实现 (reconstruction.py) ===
if ! grep -q "pv_cluster" wmpgnn/reconstruction/reconstruction.py; then
  echo "[CHECK] FAIL: pv_cluster 分簇逻辑未实现 (reconstruction.py 无 pv_cluster), 拒绝提交"
  echo "[CHECK] 接入点: reconstruct_heavyhadrons 事件级剪枝后, 按 pv_cluster_assign 分簇,"
  echo "[CHECK]          每簇独立链重建后汇总。实现后再提交。"
  exit 1
fi
echo "[CHECK] pv_cluster 代码已实现"

# === 前置检查 2: resume_ckpt 是否已填 (精确匹配 resume_ckpt 行, 避免误匹配注释里的 TODO) ===
if grep -qE 'resume_ckpt:\s*".*TODO' config_files/train_CERN_v39.yaml; then
  echo "[CHECK] FAIL: train_CERN_v39.yaml 的 resume_ckpt 仍是 TODO, 先填 v38 best checkpoint"
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
  RETRY_COUNT_FILE=/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn/logs/v39.retry_count
  N=0
  [ -f "$RETRY_COUNT_FILE" ] && N=$(cat "$RETRY_COUNT_FILE")
  if [ "$N" -lt "$MAX_RETRY" ]; then
    N=$((N+1))
    echo "$N" > "$RETRY_COUNT_FILE"
    echo "[RETRY] 第 $N/$MAX_RETRY 次, sleep ${RETRY_SLEEP}s 后重排..."
    sleep $RETRY_SLEEP
    echo "[RETRY] 重新提交 submit_train_cern_v39.sh (原作业 ${_CONDOR_IHEP_JOB_ID:-unknown}, GPU ${CUDA_VISIBLE_DEVICES})"
    hep_sub submit_train_cern_v39.sh -g ghigh -gpu 1 -cpu 4 -m 32000 -wt long -o logs/train_cern_v39.out -e logs/train_cern_v39.err
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
    -d "title=[DFEI] ⚠️ v39训练 Job ${JOB_ID} 分到坏GPU" \
    -d "desp=## v39训练作业 ${JOB_ID} GPU预检失败(重试${MAX_RETRY}次后仍失败)
| 作业 | ${JOB_ID} |
| 主机 | $(hostname) |
| GPU | ${CUDA_VISIBLE_DEVICES} |
| 时间 | $(date) |" > /dev/null 2>&1
  exit 77
fi

# 标记训练已真正开始
touch /lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn/logs/v39_started.flag

python3 -u wmpgnn/analysis/trainer.py --config config_files/train_CERN_v39.yaml

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
  -d "title=[DFEI] ${STATUS} v39训练 Job ${JOB_ID}" \
  -d "desp=## v39训练作业 ${JOB_ID} ${STATUS}

| 项目 | 值 |
|------|-----|
| **作业ID** | ${JOB_ID} |
| **状态** | ${STATUS} |
| **主机** | $(hostname) |
| **配置** | train_CERN_v39.yaml (PV分簇分层重建) |
| **结束时间** | $(date) |
| **退出码** | ${EXIT_CODE} |

### 查看日志
\`\`\`bash
tail -50 /lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn/logs/train_cern_v39.out
\`\`\`" > /dev/null 2>&1

exit $EXIT_CODE
