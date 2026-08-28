#!/bin/bash
#
# DFEI CERN 下一次训练 (B2 可微剪枝 + 源检测头 + Loss 再平衡 + LCA clip)
# 前置:
#   1. 已实现 B2 软掩码 (hetero_graph_network.py) + 源检测头 (dfei_lightning_module.py)
#   2. 方案F 已弃用 (CERN 评估证明扩展失控), B2 参数为确定默认, 无需等 F 结果
#   3. resume_ckpt 已指向 v32 训练完成后的最终 best
#
# 提交方式:
#   hep_sub submit_train_cern_next.sh -g ghigh -gpu 1 -cpu 4 -m 32000 -wt long \
#       -o logs/train_cern_next.out -e logs/train_cern_next.err
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
echo "CONFIG      : config_files/train_CERN_next.yaml"
echo "========================================"

# === 前置检查 1: B2 软掩码是否已实现 (防止提交了跑不了的作业) ===
if ! grep -q "b2" wmpgnn/model/gnn/hetero_graph_network.py; then
  echo "[CHECK] FAIL: B2 软掩码未实现 (hetero_graph_network.py 无 b2), 拒绝提交"
  exit 1
fi
if ! grep -q "source_head" wmpgnn/lightning_module/dfei_lightning_module.py; then
  echo "[CHECK] FAIL: 源检测头未实现 (dfei_lightning_module.py 无 source_head), 拒绝提交"
  exit 1
fi
echo "[CHECK] B2 + 源检测头代码已实现"

# === 前置检查 2: 待定参数是否已填 (B2 参数应为确定默认, 不应再含 '待定') ===
if grep -q "待定" config_files/train_CERN_next.yaml; then
  echo "[CHECK] WARN: train_CERN_next.yaml 仍含 '待定' 注释, 请确认参数已确定"
fi

# === GPU 预检 (快速失败, 避免分到坏GPU白跑) ===
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
  echo "[PREFLIGHT] FAIL (rc=$PREFLIGHT_RC): 分配的GPU不可用，退出"
  echo "========================================"
  echo "EXIT CODE   : 77"
  echo "END TIME    : $(date)"
  echo "========================================"
  JOB_ID="${_CONDOR_IHEP_JOB_ID:-unknown}"
  curl -s --connect-timeout 10 -X POST https://sctapi.ftqq.com/SCT387631TDiuLj6UNUsFTaDRjkaSWcdPv.send \
    -d "title=[DFEI] ⚠️ Next训练 Job ${JOB_ID} 分到坏GPU" \
    -d "desp=## Next训练作业 ${JOB_ID} GPU预检失败
| 作业 | ${JOB_ID} |
| 主机 | $(hostname) |
| GPU | ${CUDA_VISIBLE_DEVICES} |
| 时间 | $(date) |" > /dev/null 2>&1
  exit 77
fi

# 标记训练已真正开始
touch /lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn/logs/cern_next_started.flag

python3 -u wmpgnn/analysis/trainer.py --config config_files/train_CERN_next.yaml

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
  -d "title=[DFEI] ${STATUS} Next训练 Job ${JOB_ID}" \
  -d "desp=## Next训练作业 ${JOB_ID} ${STATUS}

| 项目 | 值 |
|------|-----|
| **作业ID** | ${JOB_ID} |
| **状态** | ${STATUS} |
| **主机** | $(hostname) |
| **配置** | train_CERN_next.yaml (B2+源检测头+loss再平衡) |
| **结束时间** | $(date) |
| **退出码** | ${EXIT_CODE} |

### 查看日志
\`\`\`bash
tail -50 /lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn/logs/train_cern_next.out
\`\`\`" > /dev/null 2>&1

exit $EXIT_CODE
