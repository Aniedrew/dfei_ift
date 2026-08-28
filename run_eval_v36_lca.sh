#!/bin/bash
# v36 B2 thr0.7 + chain_lca_filter 评估 (组合方案验证)
source ~/.bashrc
export PATH=$HOME/miniconda3/envs/dfei/bin:$PATH
export PYTHONPATH=/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn:$PYTHONPATH
cd /lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn
python3 -u wmpgnn/analysis/evaluate.py --config config_files/eval_CERN_v36_b2_thr07_lca.yaml
