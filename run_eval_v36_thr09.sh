#!/bin/bash
# v36 B2 thr0.9 评估 (同口径对比 v31 基线)
source ~/.bashrc
export PATH=$HOME/miniconda3/envs/dfei/bin:$PATH
export PYTHONPATH=/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn:$PYTHONPATH
cd /lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn
python3 -u wmpgnn/analysis/evaluate.py --config config_files/eval_CERN_v36_b2_thr09.yaml
