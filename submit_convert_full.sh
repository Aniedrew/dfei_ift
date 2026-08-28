#!/bin/bash
#
# 数据转换作业 - 完整转换
# 将所有 npy 逐事件数据转换为 zst chunk 格式
#
# 提交方式:
#   hep_sub submit_convert_full.sh -g lzuhep -cpu 8 -m 8000 -wt mid -o logs/ -e logs/
#

source ~/.bashrc

export PATH=$HOME/miniconda3/envs/dfei/bin:$PATH
export PYTHONPATH=/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn:$PYTHONPATH

cd /lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn

echo "========================================"
echo "JOB ID      : $_CONDOR_IHEP_JOB_ID"
echo "HOST        : $(hostname)"
echo "START TIME  : $(date)"
echo "PYTHON      : $(which python3 2>/dev/null || echo 'not found')"
echo "========================================"

export PYTHONUNBUFFERED=1

$HOME/miniconda3/envs/dfei/bin/python3 -u convert_all_data.py --threads 8

EXIT_CODE=$?
echo "========================================"
echo "EXIT CODE   : $EXIT_CODE"
echo "END TIME    : $(date)"
echo "========================================"
exit $EXIT_CODE
