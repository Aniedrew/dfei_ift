#!/bin/bash
#
# 次级顶点归属 — **最优配置多种子钉死** (2026-10-05)
#
# 动机: lr 扫描已确认 lr=3e-4 在 3 种子上稳定优于 1e-3 (均值 0.8106 vs 0.8018, 三种子全为正)。
#   现在把它当"当前最优配置"用 7 个新种子 (3..9) 再钉一遍, 目的是:
#     (1) 把均值的标准误压到 ~0.005 以内, 作为之后所有比较的基准线;
#     (2) 看 0.8210 这个最好种子是不是尾部运气。
#   同时给 lr=1e-4 (次优) 补 3 个种子, 确认两者是否有实质差距。
#
# 最优配置: attn softmax geom layers=4 feats=node+geo+ctx lr=3e-4
#   已知 (seed 0/1/2): 0.8210 / 0.8025 / 0.8082  -> 均值 0.8106
#
# 用法:
#   hep_sub submit_vertex_assoc_best.sh -argu "report_figs/feat_ceiling_big2.npz" \
#       -g ghigh -gpu 1 -cpu 4 -m 16000 -wt long \
#       -o logs/vertex_best.out -e logs/vertex_best.err
#
source ~/.bashrc
export PATH=$HOME/miniconda3/envs/dfei/bin:$PATH
export PYTHONPATH=/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn:$PYTHONPATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
cd /lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn

NPZ="${1:-report_figs/feat_ceiling_big2.npz}"
export PF_SCRIPT="submit_vertex_assoc_best.sh"
export PF_ARGS="$NPZ"
export PF_FLAGS="-g ghigh -gpu 1 -cpu 4 -m 16000 -wt long -o logs/vertex_best.out -e logs/vertex_best.err"
export PF_TAG="vertex_best"
source ./gpu_preflight.sh

echo "JOB ${_CONDOR_IHEP_JOB_ID:-?} HOST $(hostname) GPU ${CUDA_VISIBLE_DEVICES:-?} START $(date)"
echo "NPZ=$NPZ  (已知 seed0/1/2 lr=3e-4: 0.8210/0.8025/0.8082 均值 0.8106)"

FAIL=0
run() {
  SD="$1"; shift
  echo "==== [best] $(date +%H:%M:%S) seed=$SD $*"
  python3 -u analyze_vertex_assoc.py train --npz "$NPZ" --epochs 60 --patience 12 --seed "$SD" "$@" \
      2>&1 | tail -6
  rc=${PIPESTATUS[0]}
  echo "==== [best] EXIT=$rc seed=$SD ($*)"
  [ $rc -ne 0 ] && FAIL=$((FAIL + 1))
}

BASE="--model attn --feats node+geo+ctx --attn_fn softmax --attn_bias geom --layers 4"

# ---- A. 最优配置 lr=3e-4 x 新种子 3..9 ----
for SD in 3 4 5 6 7 8 9; do
  run $SD $BASE --lr 3e-4
done

# ---- B. 次优 lr=1e-4 补种子 3/4/5 ----
for SD in 3 4 5; do
  run $SD $BASE --lr 1e-4
done

echo "==== [best] ALL DONE (失败 $FAIL 个) END $(date)"
exit 0
