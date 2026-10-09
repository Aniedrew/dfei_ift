#!/bin/bash
# 秒级 GPU 探针: 看某一台节点当前能否真正初始化 CUDA。
# 用法: hep_sub probe_gpu.sh -g ghigh -gpu 1 -cpu 1 -m 2000 -wt test -wn <node> -o logs/probe_<node>.out -e logs/probe_<node>.err
source ~/.bashrc
export PATH=$HOME/miniconda3/envs/dfei/bin:$PATH
echo "PROBE host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?} time=$(date)"
nvidia-smi -L 2>&1 | head -3
python3 -u -c "
import torch
print('PROBE cuda_available =', torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print('PROBE OK', p.name, round(p.total_memory/1024**3,1), 'GB')
    a = torch.randn(300,300,device='cuda'); print('PROBE matmul', float((a@a).sum()))
" 2>&1 | tail -4
echo "PROBE done"
