#!/bin/bash
#
# 次级顶点归属小题 — 注意力模型**训练配方**扫描 (2026-10-03 第二波, 与 deep 作业正交)
#
# 动机: 目前所有 attn 结果都固定在 hidden=256 / dropout=0.1 / lr=1e-3 / bs=16, 且
#   layers=4 的 run 在 ep37 才到最好、ep49 早停 -> 怀疑"还没训够 / 容量与正则没调"。
#   本作业只动训练配方, 不动结构, 看还能不能再挤出 AP。
#
# 基线: layers=4 单种子 AP=0.8146 / p@r90=0.6698 (30 维 node+geo+ctx, softmax+geom)。
#
# 用法:
#   hep_sub submit_vertex_assoc_recipe.sh -argu "report_figs/feat_ceiling_big2.npz" \
#       -g ghigh -gpu 1 -cpu 4 -m 16000 -wt long \
#       -o logs/vertex_recipe.out -e logs/vertex_recipe.err
#
source ~/.bashrc
export PATH=$HOME/miniconda3/envs/dfei/bin:$PATH
export PYTHONPATH=/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn:$PYTHONPATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
cd /lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn

NPZ="${1:-report_figs/feat_ceiling_big2.npz}"
export PF_SCRIPT="submit_vertex_assoc_recipe.sh"
export PF_ARGS="$NPZ"
export PF_FLAGS="-g ghigh -gpu 1 -cpu 4 -m 16000 -wt long -o logs/vertex_recipe.out -e logs/vertex_recipe.err"
export PF_TAG="vertex_recipe"
source ./gpu_preflight.sh

echo "JOB ${_CONDOR_IHEP_JOB_ID:-?} HOST $(hostname) GPU ${CUDA_VISIBLE_DEVICES:-?} START $(date)"
echo "NPZ=$NPZ  (基线: layers=4 单种子 AP=0.8146 / p@r90=0.6698)"

FAIL=0
run() {
  SD="$1"; shift
  echo "==== [recipe] $(date +%H:%M:%S) seed=$SD $*"
  python3 -u analyze_vertex_assoc.py train --npz "$NPZ" --epochs 60 --patience 12 --seed "$SD" "$@" \
      2>&1 | tail -6
  rc=${PIPESTATUS[0]}
  echo "==== [recipe] EXIT=$rc seed=$SD ($*)"
  [ $rc -ne 0 ] && FAIL=$((FAIL + 1))
}

BASE="--model attn --feats node+geo+ctx --attn_fn softmax --attn_bias geom --layers 4"

# ---- A. 更长训练 (epochs 150 / patience 30) x 3 种子 ----
for SD in 0 1 2; do
  run $SD $BASE --epochs 150 --patience 30
done

# ---- B. 容量与正则 (单种子) ----
run 0 $BASE --hidden 512
run 0 $BASE --dropout 0.0
run 0 $BASE --dropout 0.3
run 0 $BASE --lr 3e-4
run 0 $BASE --bs 8
run 0 $BASE --bs 32

echo "==== [recipe] ALL DONE (失败 $FAIL 个) END $(date)"
exit 0
