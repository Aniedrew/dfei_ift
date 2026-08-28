#!/bin/bash
#
# DFEI 小规模训练测试
# 使用少量数据 (2 chunks, ~1600 events) + 2 epochs 验证训练流程
#
# 提交方式 (GPU):
#   hep_sub submit_train_dfei_test.sh -g lzuhep -gpu 1 -cpu 8 -m 16000 -o logs/train_dfei_test.out -e logs/train_dfei_test.err
#
# 提交方式 (CPU only, 仅用于调试加载逻辑):
#   hep_sub submit_train_dfei_test.sh -g lzuhep -cpu 8 -m 16000 -o logs/train_dfei_test.out -e logs/train_dfei_test.err
#

source ~/.bashrc

export PATH=$HOME/miniconda3/envs/dfei/bin:$PATH
export PYTHONPATH=/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn:$PYTHONPATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

# 切换到项目根目录
cd /lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn

echo "========================================"
echo "JOB ID      : $_CONDOR_IHEP_JOB_ID"
echo "HOST        : $(hostname)"
echo "START TIME  : $(date)"
echo "PYTHON      : $(which python3)"
echo "GPU         : $CUDA_VISIBLE_DEVICES"
echo "CONFIG      : config_files/train_DFEI_small.yaml"
echo "========================================"

python3 -u wmpgnn/analysis/trainer.py --config config_files/train_DFEI_small.yaml

EXIT_CODE=$?
echo "========================================"
echo "EXIT CODE   : $EXIT_CODE"
echo "END TIME    : $(date)"
echo "========================================"
exit $EXIT_CODE
