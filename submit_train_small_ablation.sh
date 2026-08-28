#!/bin/bash
#
# 小规模消融训练提交脚本 (参数化): mass-only / +struct / +mom
# 用法: hep_sub submit_train_small_ablation.sh -argu "<config.yaml>" -g ghigh -gpu 1 -cpu 4 -m 32000 -wt long -o logs/small_<name>.out -e logs/small_<name>.err
#   $1 = config 文件名 (config_files/ 下), 如 train_CERN_small_struct.yaml
#
# 重试机制: PREFLIGHT 失败自动重排 (几乎无限次)

CONFIG_FILE="${1:?usage: submit_train_small_ablation.sh <config.yaml>}"

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
echo "CONFIG      : config_files/$CONFIG_FILE"
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
  MAX_RETRY=1000
  RETRY_SLEEP=60
  RETRY_COUNT_FILE=/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn/logs/small_${CONFIG_FILE%.yaml}.retry_count
  N=0
  [ -f "$RETRY_COUNT_FILE" ] && N=$(cat "$RETRY_COUNT_FILE")
  if [ "$N" -lt "$MAX_RETRY" ]; then
    N=$((N+1))
    echo "$N" > "$RETRY_COUNT_FILE"
    echo "[RETRY] 第 $N/$MAX_RETRY 次, sleep ${RETRY_SLEEP}s 后重排..."
    sleep $RETRY_SLEEP
    echo "[RETRY] 重新提交 submit_train_small_ablation.sh $CONFIG_FILE (原作业 ${_CONDOR_IHEP_JOB_ID:-unknown}, GPU ${CUDA_VISIBLE_DEVICES})"
    hep_sub submit_train_small_ablation.sh -argu "$CONFIG_FILE" -g ghigh -gpu 1 -cpu 4 -m 32000 -wt long -o logs/small_${CONFIG_FILE%.yaml}.out -e logs/small_${CONFIG_FILE%.yaml}.err
    echo "[RETRY] 已重提, 本次退出"
    exit 0
  fi
  echo "[RETRY] 已达 ${MAX_RETRY} 次上限, 放弃"
  echo "========================================"
  echo "EXIT CODE   : 77"
  echo "END TIME    : $(date)"
  echo "========================================"
  exit 77
fi

python3 -u wmpgnn/analysis/trainer.py --config config_files/$CONFIG_FILE

EXIT_CODE=$?
echo "========================================"
echo "EXIT CODE   : $EXIT_CODE"
echo "END TIME    : $(date)"
echo "========================================"
exit $EXIT_CODE
