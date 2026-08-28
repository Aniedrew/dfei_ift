#!/bin/bash
#
# DFEI 完整训练 (GPU)
# 使用全部 50 个训练 chunk (39519 events) 进行 100 epoch 训练
#
# 提交方式:
#   hep_sub submit_train_dfei.sh -g lzuhep -gpu 1 -cpu 8 -m 32000 -wt long -o logs/train_dfei.out -e logs/train_dfei.err
#

source ~/.bashrc

export PATH=$HOME/miniconda3/envs/dfei/bin:$PATH
export PYTHONPATH=/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn:$PYTHONPATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

cd /lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn

echo "========================================"
echo "JOB ID      : $_CONDOR_IHEP_JOB_ID"
echo "HOST        : $(hostname)"
echo "START TIME  : $(date)"
echo "PYTHON      : $(which python3)"
echo "GPU         : $CUDA_VISIBLE_DEVICES"
echo "CONFIG      : config_files/train_DFEI.yaml"
echo "========================================"

python3 -u wmpgnn/analysis/trainer.py --config config_files/train_DFEI.yaml

EXIT_CODE=$?
echo "========================================"
echo "EXIT CODE   : $EXIT_CODE"
echo "END TIME    : $(date)"
echo "========================================"
exit $EXIT_CODE
