#!/bin/bash
# 
# 数据转换作业 - 将 npy 逐事件数据转换为 zst chunk 格式
# 提交方式:
#   hep_sub submit_convert.sh -g lzuhep -cpu 8 -m 8000 -wt mid -o logs/ -e logs/
#

source ~/.bashrc

# 激活 conda 环境
# conda activate dfei 在某些 HTCondor 节点上可能不 work，使用全路径
export PATH=$HOME/miniconda3/envs/dfei/bin:$PATH
export PYTHONPATH=/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn:$PYTHONPATH

cd /lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn

echo "========================================"
echo "Job ID     : $_CONDOR_IHEP_JOB_ID"
echo "Host       : $(hostname)"
echo "Start time : $(date)"
echo "Python     : $(which python3)"
echo "========================================"

python3 convert_all_data.py --threads 8

echo "========================================"
echo "End time   : $(date)"
echo "========================================"
