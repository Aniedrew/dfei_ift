#!/bin/bash
#
# 多 B 专用探针矩阵 (纯 CPU, 不需要 GPU)
#
# 动机 (2026-10-08): DFEI 的目标场景是"一个事件里有多条 B", 但这类事件只占 ~24%,
#   容易被大量单 B 事件淹没。本脚本把**整个 train/val/test 流程**限制到指定 nb 子集
#   (nb = 该事件真值链数), 在子集上比较各种优化, 找出"只对多 B 有效"的那些。
#
# 用法:
#   bash run_multib_probe.sh <npz> <min_nb> <max_nb> <tag>
#   # 多 B: bash run_multib_probe.sh report_figs/feat_ceiling_nb.npz 2 999999 multib
#   # 单 B: bash run_multib_probe.sh report_figs/feat_ceiling_nb.npz 1 1 singleb
#
# 纯 CPU -> 也可以提到 lzuhep (CPU 池) 跑, 完全绕开 GPU 排队:
#   hep_sub run_multib_probe.sh -argu "<npz> 2 999999 multib" -g lzuhep -cpu 8 -m 16000 -wt long \
#       -o logs/multib_probe.out -e logs/multib_probe.err
#
source ~/.bashrc
export PATH=$HOME/miniconda3/envs/dfei/bin:$PATH
export PYTHONPATH=/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn:$PYTHONPATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
cd /lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn

NPZ="${1:-report_figs/feat_ceiling_nb.npz}"
MIN_NB="${2:-2}"
MAX_NB="${3:-999999}"
TAG="${4:-multib}"
SUB="--min_nb $MIN_NB --max_nb $MAX_NB"

echo "==== [multib] 子集 nb in [$MIN_NB,$MAX_NB]  npz=$NPZ  开始 $(date)"
FAIL=0
run() {   # run <seed> <args...>
  SD="$1"; shift
  echo "---- $TAG seed=$SD $*"
  python3 -u analyze_vertex_assoc.py train --npz "$NPZ" $SUB --epochs 60 --patience 12 --seed "$SD" "$@" \
      2>&1 | grep -E "^\[train\]|多 B|子集" | tail -3
  rc=${PIPESTATUS[0]}
  [ $rc -ne 0 ] && FAIL=$((FAIL + 1))
  echo "     EXIT=$rc"
}

BASE="--model attn --attn_fn softmax --attn_bias geom --layers 4 --lr 3e-4"

# 1) 特征组对多 B 的作用 (核心: 修复 D1 后 raw/vrt 是否终于有用)
run 0 $BASE --feats node+geo+ctx
run 0 $BASE --feats node+geo+raw+der+vrt
run 0 $BASE --feats node+geo+raw+der+vrt+ctx
# 2) 与无上下文基线对照
run 0 --model mlp --feats node+geo+raw+der+vrt+ctx
# 3) 机制 (在这条子集上重测)
run 0 $BASE --feats node+geo+raw+der+vrt+ctx --attn_fn topk --attn_topk 8
run 0 $BASE --feats node+geo+raw+der+vrt+ctx --attn_fn sparsemax

echo "==== [multib] ALL DONE (失败 $FAIL) END $(date)"
exit 0
