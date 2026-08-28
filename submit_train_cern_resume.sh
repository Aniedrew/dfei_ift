#!/bin/bash
#
# DFEI CERN 续训 (从 v32 best checkpoint 继续, 降LR + loss防爆)
#
# 提交方式:
#   hep_sub submit_train_cern_resume.sh -g ghigh -gpu 1 -cpu 4 -m 32000 -wt long -o logs/train_cern_resume.out -e logs/train_cern_resume.err
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
echo "CONFIG      : config_files/train_CERN_normed_resume.yaml"
echo "========================================"

# === GPU 预检 (gpu02 上 GPU-a2db080d 是坏的, 快速失败避免白跑) ===
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
    -d "title=[DFEI] ⚠️ 续训 Job ${JOB_ID} 分到坏GPU" \
    -d "desp=## 续训作业 ${JOB_ID} GPU预检失败
| 作业 | ${JOB_ID} |
| 主机 | $(hostname) |
| GPU | ${CUDA_VISIBLE_DEVICES} |
| 时间 | $(date) |

需要人工重提, 或等管理员禁用 GPU-a2db080d。" > /dev/null 2>&1
  exit 77
fi

# 标记训练已真正开始(供看护脚本判断)
touch /lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn/logs/cern_resume_started.flag

python3 -u wmpgnn/analysis/trainer.py --config config_files/train_CERN_normed_resume.yaml

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
  -d "title=[DFEI] ${STATUS} 续训 Job ${JOB_ID}" \
  -d "desp=## 续训作业 ${JOB_ID} ${STATUS}

| 项目 | 值 |
|------|-----|
| **作业ID** | ${JOB_ID} |
| **状态** | ${STATUS} |
| **主机** | $(hostname) |
| **配置** | train_CERN_normed_resume.yaml |
| **结束时间** | $(date) |
| **退出码** | ${EXIT_CODE} |

### 查看日志
\`\`\`bash
tail -50 /lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn/logs/train_cern_resume.out
\`\`\`" > /dev/null 2>&1

exit $EXIT_CODE
