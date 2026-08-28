#!/bin/bash
# 非贪心重建分析 (v23 模型 + 论文数据, 多阈值剪枝流失)
source ~/.bashrc
export PATH=$HOME/miniconda3/envs/dfei/bin:$PATH
export PYTHONPATH=/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn:$PYTHONPATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
cd /lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn
echo "========================================"
echo "JOB ID      : $_CONDOR_IHEP_JOB_ID"
echo "HOST        : $(hostname)"
echo "GPU         : $CUDA_VISIBLE_DEVICES"
echo "========================================"
# GPU 预检
python3 -u -c "
import torch
if not torch.cuda.is_available():
    print('[PREFLIGHT] FAIL: no cuda')
    raise SystemExit(77)
print(f'[PREFLIGHT] OK: {torch.cuda.get_device_properties(0).name}')
"
[ $? -ne 0 ] && exit 77
MAX_SCAN=1000 python3 -u analyze_non_greedy.py
echo "EXIT CODE   : $?"
echo "========================================"
