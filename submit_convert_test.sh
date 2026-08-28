#!/bin/bash
#
# 数据转换作业 - 快速测试模式
# 仅转换少量事件，验证 HTCondor 作业提交 + 转换流程
#
# 提交方式:
#   hep_sub submit_convert_test.sh -g lzuhep -cpu 4 -m 4000 -o logs/ -e logs/
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
echo "CONDA ENV   : $($HOME/miniconda3/envs/dfei/bin/python3 --version 2>/dev/null)"
echo "========================================"

export PYTHONUNBUFFERED=1

# 测试模式 (每个数据集只转换前4个事件)
$HOME/miniconda3/envs/dfei/bin/python3 -u convert_all_data.py --test --threads 4

EXIT_CODE=$?
echo "EXIT CODE   : $EXIT_CODE"
echo "========================================"
echo "END TIME    : $(date)"
echo "========================================"
exit $EXIT_CODE
