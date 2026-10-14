#!/bin/bash
# 纯 CPU 批量评估: 与 submit_eval_batch.sh 相同，但**不要求 GPU**。
# 用途: 不需要模型算力、或 GPU 名额紧张时的评估(如 noprune 真值统计)。
# 用法: hep_sub submit_eval_cpu.sh -argu "cfg1.yaml ..." -g ghigh -cpu 16 -m 32000 -wt long \
#          -o logs/eval_cpu.out -e logs/eval_cpu.err
[ $# -ge 1 ] || { echo "usage: submit_eval_cpu.sh <cfg1.yaml> [cfg2.yaml ...]"; exit 2; }
source ~/.bashrc
export PATH=$HOME/miniconda3/envs/dfei/bin:$PATH
export PYTHONPATH=/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn:$PYTHONPATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=""
export NVIDIA_VISIBLE_DEVICES=""
cd /lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn || exit 1

echo "JOB ${_CONDOR_IHEP_JOB_ID:-?} HOST $(hostname) NCPU $(nproc) START $(date)"
echo "CONFIGS($#): $*"
python3 -c "import torch,torch_geometric;print('[PREFLIGHT] torch',torch.__version__,'cpu-ok')" \
  || { echo "[PREFLIGHT] FAIL"; exit 1; }

FAIL=0
for cf in "$@"; do
  echo "==== [batch] $(date +%H:%M:%S) -> $cf"
  python3 -u wmpgnn/analysis/evaluate.py --config "config_files/$cf"
  rc=$?
  echo "==== [batch] $(date +%H:%M:%S) $cf EXIT=$rc"
  [ $rc -ne 0 ] && FAIL=$((FAIL+1))
done
echo "==== [batch] ALL DONE (失败 $FAIL 个) END $(date)"
exit 0
