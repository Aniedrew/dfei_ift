#!/bin/bash
#
# 次级顶点归属 — 注意力**结构**第二波 (统一挂 4 层骨干, 2026-10-04)
#
# 动机: 第一轮 (10485484) 的结构比较都在 2 层下做, 且多种子把 2 层下的差异全抹平了。
#   既然 4 层才是最强骨干, 就应把"稀疏化 / 头数 / bias 列"这些结构旋钮在 4 层下重扫一遍,
#   免得"没用"只是因为骨干太弱盖住了效果。
#
# 与 deep 作业不重复: deep 已覆盖 layers4 x 3种子 / all47 / heads8 / layers6 /
#   bias none / bias geom4(seed0) / topk8 x 3种子。这里只补剩下的。
#
# 基线: attn softmax geom layers=4, 单种子 AP=0.8146 / p@r90=0.6698。
#
# 用法:
#   hep_sub submit_vertex_assoc_variants2.sh -argu "report_figs/feat_ceiling_big2.npz" \
#       -g ghigh -gpu 1 -cpu 4 -m 16000 -wt long \
#       -o logs/vertex_variants2.out -e logs/vertex_variants2.err
#
source ~/.bashrc
export PATH=$HOME/miniconda3/envs/dfei/bin:$PATH
export PYTHONPATH=/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn:$PYTHONPATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
cd /lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn

NPZ="${1:-report_figs/feat_ceiling_big2.npz}"
export PF_SCRIPT="submit_vertex_assoc_variants2.sh"
export PF_ARGS="$NPZ"
export PF_FLAGS="-g ghigh -gpu 1 -cpu 4 -m 16000 -wt long -o logs/vertex_variants2.out -e logs/vertex_variants2.err"
export PF_TAG="vertex_variants2"
source ./gpu_preflight.sh

echo "JOB ${_CONDOR_IHEP_JOB_ID:-?} HOST $(hostname) GPU ${CUDA_VISIBLE_DEVICES:-?} START $(date)"
echo "NPZ=$NPZ  (基线: softmax geom L4 AP=0.8146 / p@r90=0.6698)"

FAIL=0
run() {
  SD="$1"; shift
  echo "==== [var2] $(date +%H:%M:%S) seed=$SD $*"
  python3 -u analyze_vertex_assoc.py train --npz "$NPZ" --epochs 60 --patience 12 --seed "$SD" "$@" \
      2>&1 | tail -6
  rc=${PIPESTATUS[0]}
  echo "==== [var2] EXIT=$rc seed=$SD ($*)"
  [ $rc -ne 0 ] && FAIL=$((FAIL + 1))
}

BASE="--model attn --feats node+geo+ctx --attn_bias geom --layers 4"

# ---- A. sparsemax (欧氏投影到单纯形) x 3 种子 ----
for SD in 0 1 2; do
  run $SD $BASE --attn_fn sparsemax
done

# ---- B. 模型内 top-k 稀疏注意力: k 扫描 (单种子; k=8 已在 deep 作业里跑 3 种子) ----
run 0 $BASE --attn_fn topk --attn_topk 4
run 0 $BASE --attn_fn topk --attn_topk 16
run 0 $BASE --attn_fn topk --attn_topk 32

# ---- C. 头数 (4 层下 2 头 vs 8 头; 8 头已在 deep 作业 seed0) ----
run 0 $BASE --attn_fn softmax --heads 2

# ---- D. bias 列扩展 (geom4) 补种子 1/2 ----
for SD in 1 2; do
  run $SD $BASE --attn_fn softmax --attn_bias geom4
done

echo "==== [var2] ALL DONE (失败 $FAIL 个) END $(date)"
exit 0
