#!/bin/bash
# 快速GPU探测作业：打印节点GPU布局，用于排查"No CUDA GPUs are available"
source ~/.bashrc
export PATH=$HOME/miniconda3/envs/dfei/bin:$PATH

echo "JOB ID      : $_CONDOR_IHEP_JOB_ID"
echo "HOST        : $(hostname)"
echo "CUDA_VISIBLE_DEVICES : $CUDA_VISIBLE_DEVICES"
echo "nvidia-smi -L:"
nvidia-smi -L 2>&1
echo "nvidia-smi (summary):"
nvidia-smi 2>&1 | head -25
echo "torch test:"
python3 -u -c "
import torch
print('torch', torch.__version__, 'cuda_available=', torch.cuda.is_available())
print('visible devices=', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f'  device {i}: {p.name} cap={p.major}.{p.minor} mem={p.total_memory/1024**3:.1f}GB')
if torch.cuda.is_available():
    a = torch.randn(1000,1000,device='cuda')
    b = (a @ a).sum().item()
    print('matmul OK, sum=', b)
"
echo "PROBE DONE rc=$?"
