#!/bin/bash
# v36 B2 thr0.9 + chain_lca_record 评估 (记录链LCA置信度, 供多阈值后处理)
source ~/.bashrc
export PATH=$HOME/miniconda3/envs/dfei/bin:$PATH
export PYTHONPATH=/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn:$PYTHONPATH
cd /lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn
python3 -u wmpgnn/analysis/evaluate.py --config config_files/eval_CERN_v36_b2_thr09_lca_record.yaml
