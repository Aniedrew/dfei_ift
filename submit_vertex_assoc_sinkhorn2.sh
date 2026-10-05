#!/bin/bash
#
# 次级顶点归属 — Sinkhorn 第二波: 挂到**更强的 4 层骨干**上 + 超参敏感性 (2026-10-04)
#
# 动机: 第一波 (10486118) 是 2 层骨干 + match none/sinkhorn(tau 0.25/0.5/1.0) x 3 种子。
#   但 seeds 作业已证明 layers=4 明显更强 (0.8146 vs 0.7997), 所以"排他约束是否有用"应在
#   最强骨干上重测; 同时把 match_w / dust 这两个没扫过的旋钮补上, 免得"没用"其实只是没调对。
#
# 基线: attn softmax geom layers=4, match=none, 单种子 AP=0.8146 / p@r90=0.6698。
#
# 用法:
#   hep_sub submit_vertex_assoc_sinkhorn2.sh -argu "report_figs/feat_ceiling_big2.npz" \
#       -g ghigh -gpu 1 -cpu 4 -m 16000 -wt long \
#       -o logs/vertex_sinkhorn2.out -e logs/vertex_sinkhorn2.err
#
source ~/.bashrc
export PATH=$HOME/miniconda3/envs/dfei/bin:$PATH
export PYTHONPATH=/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn:$PYTHONPATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
cd /lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn

NPZ="${1:-report_figs/feat_ceiling_big2.npz}"
export PF_SCRIPT="submit_vertex_assoc_sinkhorn2.sh"
export PF_ARGS="$NPZ"
export PF_FLAGS="-g ghigh -gpu 1 -cpu 4 -m 16000 -wt long -o logs/vertex_sinkhorn2.out -e logs/vertex_sinkhorn2.err"
export PF_TAG="vertex_sinkhorn2"
source ./gpu_preflight.sh

echo "JOB ${_CONDOR_IHEP_JOB_ID:-?} HOST $(hostname) GPU ${CUDA_VISIBLE_DEVICES:-?} START $(date)"
echo "NPZ=$NPZ  (基线: 4 层, match=none, AP=0.8146 / p@r90=0.6698)"

FAIL=0
run() {
  SD="$1"; shift
  echo "==== [sink2] $(date +%H:%M:%S) seed=$SD $*"
  python3 -u analyze_vertex_assoc.py train --npz "$NPZ" --epochs 60 --patience 12 --seed "$SD" "$@" \
      2>&1 | tail -6
  rc=${PIPESTATUS[0]}
  echo "==== [sink2] EXIT=$rc seed=$SD ($*)"
  [ $rc -ne 0 ] && FAIL=$((FAIL + 1))
}

BASE="--model attn --feats node+geo+ctx --attn_fn softmax --attn_bias geom --layers 4"

# ---- A. tau 扫描 x 3 种子 (在 4 层骨干上) ----
for TAU in 0.25 0.5 1.0; do
  for SD in 0 1 2; do
    run $SD $BASE --match sinkhorn --match_tau $TAU
  done
done

# ---- B. match_w 扫描 (默认 0.5; 看排他约束该给多大权重) ----
run 0 $BASE --match sinkhorn --match_tau 0.5 --match_w 0.1
run 1 $BASE --match sinkhorn --match_tau 0.5 --match_w 0.1
run 0 $BASE --match sinkhorn --match_tau 0.5 --match_w 2.0
run 1 $BASE --match sinkhorn --match_tau 0.5 --match_w 2.0

# ---- C. dustbin 大小 (允许多少未匹配) ----
run 0 $BASE --match sinkhorn --match_tau 0.5 --match_dust 0.01
run 0 $BASE --match sinkhorn --match_tau 0.5 --match_dust 0.5

echo "==== [sink2] ALL DONE (失败 $FAIL 个) END $(date)"
exit 0
