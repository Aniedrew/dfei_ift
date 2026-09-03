#!/bin/bash
#
# DFEI v38 + track 级自注意力 (试验, 单发): 从 v38 best 加载, 20ep 微调 + thr0.9 评估
#
# 提交方式:
#   hep_sub submit_train_cern_v38_attn.sh -g ghigh -gpu 1 -cpu 4 -m 64000 -wt long \
#       -o logs/v38_attn.out -e logs/v38_attn.err
#

source ~/.bashrc

export PATH=$HOME/miniconda3/envs/dfei/bin:$PATH
export PYTHONPATH=/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn:$PYTHONPATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn

CFG=config_files/train_CERN_v38_attn.yaml

echo "========================================"
echo "JOB ID      : $_CONDOR_IHEP_JOB_ID"
echo "HOST        : $(hostname)"
echo "START TIME  : $(date)"
echo "GPU         : $CUDA_VISIBLE_DEVICES"
echo "CONFIG      : $CFG"
echo "========================================"

# === GPU 预检 (失败自动重排) ===
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
  echo "[PREFLIGHT] FAIL (rc=$PREFLIGHT_RC): 分配的GPU不可用"
  RETRY_COUNT_FILE=/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn/logs/v38_attn.retry_count
  M=0
  [ -f "$RETRY_COUNT_FILE" ] && M=$(cat "$RETRY_COUNT_FILE")
  M=$((M+1)); echo "$M" > "$RETRY_COUNT_FILE"
  echo "[RETRY] 第 $M 次, sleep 120s 后重排..."
  sleep 120
  for v in $(env | cut -d= -f1); do
    echo " HOME PATH USER LOGNAME HOSTNAME SHELL PWD OMP_NUM_THREADS PYTORCH_CUDA_ALLOC_CONF CUDA_VISIBLE_DEVICES PYTHONPATH _CONDOR_IHEP_JOB_ID " | grep -q " $v " || unset "$v" 2>/dev/null
  done
  hep_sub submit_train_cern_v38_attn.sh -g ghigh -gpu 1 -cpu 4 -m 64000 -wt long \
      -o logs/v38_attn.out -e logs/v38_attn.err
  echo "[RETRY] 已重提, 本次退出"
  exit 0
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
  -d "title=[DFEI] ${STATUS} v38+attention Job ${JOB_ID}" \
  -d "desp=## v38+attention 试验 ${STATUS}
| 作业ID | ${JOB_ID} |
| 状态 | ${STATUS} |
| 配置 | train_CERN_v38_attn.yaml |
| 结束时间 | $(date) |
| 退出码 | ${EXIT_CODE} |

\`\`\`bash
tail -40 /lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn/logs/v38_attn.out
\`\`\`" > /dev/null 2>&1

exit $EXIT_CODE
