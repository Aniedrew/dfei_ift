#!/bin/bash
#
# 次级顶点归属 — **学习率**确认 (2026-10-04)
#
# 动机: 配方作业里唯一一个正向的信号是 lr=3e-4 (单种子 AP=0.8210, 而 lr=1e-3 是 0.8146)。
#   其余旋钮 (epochs/hidden/dropout/bs) 都没挤出来。所以把 lr 单独拎出来, 多种子确认。
#
# 基线: attn softmax geom layers=4 feats=node+geo+ctx, lr=1e-3 三种子 = 0.8146/0.7904/0.8003 (均值 0.8018)。
#
# 用法:
#   hep_sub submit_vertex_assoc_lr.sh -argu "report_figs/feat_ceiling_big2.npz" \
#       -g ghigh -gpu 1 -cpu 4 -m 16000 -wt long \
#       -o logs/vertex_lr.out -e logs/vertex_lr.err
#
source ~/.bashrc
export PATH=$HOME/miniconda3/envs/dfei/bin:$PATH
export PYTHONPATH=/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn:$PYTHONPATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
cd /lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn

NPZ="${1:-report_figs/feat_ceiling_big2.npz}"
export PF_SCRIPT="submit_vertex_assoc_lr.sh"
export PF_ARGS="$NPZ"
export PF_FLAGS="-g ghigh -gpu 1 -cpu 4 -m 16000 -wt long -o logs/vertex_lr.out -e logs/vertex_lr.err"
export PF_TAG="vertex_lr"
source ./gpu_preflight.sh

echo "JOB ${_CONDOR_IHEP_JOB_ID:-?} HOST $(hostname) GPU ${CUDA_VISIBLE_DEVICES:-?} START $(date)"
echo "NPZ=$NPZ  (基线 lr=1e-3 三种子 0.8146/0.7904/0.8003)"

FAIL=0
run() {
  SD="$1"; shift
  echo "==== [lr] $(date +%H:%M:%S) seed=$SD $*"
  python3 -u analyze_vertex_assoc.py train --npz "$NPZ" --epochs 60 --patience 12 --seed "$SD" "$@" \
      2>&1 | tail -6
  rc=${PIPESTATUS[0]}
  echo "==== [lr] EXIT=$rc seed=$SD ($*)"
  [ $rc -ne 0 ] && FAIL=$((FAIL + 1))
}

BASE="--model attn --feats node+geo+ctx --attn_fn softmax --attn_bias geom --layers 4"

# ---- A. lr x 3 种子 ----
for LR in 3e-4 5e-4 1e-4; do
  for SD in 0 1 2; do
    run $SD $BASE --lr $LR
  done
done

# ---- B. lr=3e-4 + 更长训练 (看是否还能再涨) ----
run 0 $BASE --lr 3e-4 --epochs 150 --patience 30

echo "==== [lr] ALL DONE (失败 $FAIL 个) END $(date)"
exit 0
