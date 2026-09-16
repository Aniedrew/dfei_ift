#!/bin/bash
#
# DFEI v47 (masshead2, 最强 32.7/55.9) + track 级自注意力完整版 (edge-bias)
# 20ep 微调探针, 对照 v47
#
# 提交方式:
#   hep_sub submit_train_cern_v47_attn.sh -g ghigh -gpu 1 -cpu 4 -m 64000 -wt long \
#       -o logs/v47_attn.out -e logs/v47_attn.err
#

source ~/.bashrc

export PATH=$HOME/miniconda3/envs/dfei/bin:$PATH
export PYTHONPATH=/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn:$PYTHONPATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn

CFG=config_files/train_CERN_v47_attn.yaml

echo "========================================"
echo "JOB ID      : $_CONDOR_IHEP_JOB_ID"
echo "HOST        : $(hostname)"
echo "START TIME  : $(date)"
echo "GPU         : $CUDA_VISIBLE_DEVICES"
echo "CONFIG      : $CFG"
echo "========================================"

# === GPU 预检: 坏卡直接退出 (单发 job, 无自重提交) ===
echo "[PREFLIGHT] GPU check at $(date), device=$CUDA_VISIBLE_DEVICES"
nvidia-smi -L 2>&1 | head -3
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
  echo "[PREFLIGHT] FAIL (rc=$PREFLIGHT_RC): 分配的GPU不可用, 退出 (稍后人工/定时重提)"
  exit 77
fi

python3 -u wmpgnn/analysis/trainer.py --config "$CFG"
EXIT_CODE=$?
echo "========================================"
echo "EXIT CODE   : $EXIT_CODE"
echo "END TIME    : $(date)"
echo "========================================"

JOB_ID="${_CONDOR_IHEP_JOB_ID:-unknown}"
STATUS="✅ 完成"; [ $EXIT_CODE -ne 0 ] && STATUS="❌ 失败"
curl -s --connect-timeout 10 -X POST https://sctapi.ftqq.com/SCT387631TDiuLj6UNUsFTaDRjkaSWcdPv.send \
  -d "title=[DFEI] ${STATUS} v47+attn Job ${JOB_ID}" \
  -d "desp=## v47 + attention(edge-bias) ${STATUS}
| 作业ID | ${JOB_ID} |
| 状态 | ${STATUS} |
| 配置 | train_CERN_v47_attn.yaml |
| 结束时间 | $(date) |
| 退出码 | ${EXIT_CODE} |

\`\`\`bash
tail -40 /lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn/logs/v47_attn.out
\`\`\`" > /dev/null 2>&1

exit $EXIT_CODE
