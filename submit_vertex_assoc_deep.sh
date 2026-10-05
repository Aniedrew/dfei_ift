#!/bin/bash
#
# 次级顶点归属小题 — 注意力**深度/容量** + **全特征公平对照** (2026-10-03 第二波)
#
# 背景 (基于 10485484 的 3 种子结果, report_figs/feat_ceiling_big.npz):
#   1) 单种子曾显示 "几何 bias (geom) 比 none 好 +0.008" -> **3 种子打平** (geom 0.7937 vs none 0.7937)。
#      结论修正: pair-bias 本身无增益; 单种子差异是噪声。
#   2) MLP 0.766 vs attn 0.794 -> 注意力 +0.028, 3 种子一致 (唯一稳的结论)。
#   3) 单种子消融里 **layers=4 明显最好** (AP 0.8146 vs layers=2 的 0.7997) -> 本作业多种子确认。
#   4) 之前 attn 只用 30 维 (node+geo+ctx), 而 GBDT 用全部 47 维拿到 0.7824 -> 不公平。
#      本作业补上 attn + 全 47 维的公平对照。
#
# 判据: 测试集 pairwise AP / p@r90。参照: 单种子 layers=4 AP=0.8146 / p@r90=0.6698;
#       GBDT(47维) AP=0.7824; 单种子 attn(30维,2层) AP=0.8042。
#
# 用法:
#   hep_sub submit_vertex_assoc_deep.sh -argu "report_figs/feat_ceiling_big2.npz" \
#       -g ghigh -gpu 1 -cpu 4 -m 16000 -wt long \
#       -o logs/vertex_deep.out -e logs/vertex_deep.err
#
source ~/.bashrc
export PATH=$HOME/miniconda3/envs/dfei/bin:$PATH
export PYTHONPATH=/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn:$PYTHONPATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
cd /lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn

NPZ="${1:-report_figs/feat_ceiling_big2.npz}"
export PF_SCRIPT="submit_vertex_assoc_deep.sh"
export PF_ARGS="$NPZ"
export PF_FLAGS="-g ghigh -gpu 1 -cpu 4 -m 16000 -wt long -o logs/vertex_deep.out -e logs/vertex_deep.err"
export PF_TAG="vertex_deep"
source ./gpu_preflight.sh

echo "JOB ${_CONDOR_IHEP_JOB_ID:-?} HOST $(hostname) GPU ${CUDA_VISIBLE_DEVICES:-?} START $(date)"
echo "NPZ=$NPZ  (参照: layers=4 单种子 AP=0.8146/p@r90=0.6698; GBDT47 AP=0.7824)"

FAIL=0
run() {   # run <seed> <args...>
  SD="$1"; shift
  echo "==== [deep] $(date +%H:%M:%S) seed=$SD $*"
  # 注意: `cmd | tail` 的 $? 是 tail 的 -> 必须用 PIPESTATUS, 否则崩溃会被记成成功 (踩过)
  python3 -u analyze_vertex_assoc.py train --npz "$NPZ" --epochs 60 --patience 12 --seed "$SD" "$@" \
      2>&1 | tail -6
  rc=${PIPESTATUS[0]}
  echo "==== [deep] EXIT=$rc seed=$SD ($*)"
  [ $rc -ne 0 ] && FAIL=$((FAIL + 1))
}

BASE="--model attn --feats node+geo+ctx --attn_fn softmax --attn_bias geom"

# ---- A. 深度确认: layers=4 x 3 种子 (单种子显示 +0.015) ----
for SD in 0 1 2; do
  run $SD $BASE --layers 4
done

# ---- B. 全 47 维公平对照 vs GBDT x 3 种子 ----
for SD in 0 1 2; do
  run $SD --model attn --feats node+geo+raw+der+vrt+ctx --attn_fn softmax --attn_bias geom --layers 4
done

# ---- C. 单种子扩展 ----
run 0 $BASE --layers 4 --heads 8
run 0 $BASE --layers 6
run 0 $BASE --layers 4 --attn_bias none
run 0 $BASE --layers 4 --attn_bias geom4

# ---- D. top-k 稀疏注意力在 4 层下的表现 x 3 种子 ----
for SD in 0 1 2; do
  run $SD $BASE --layers 4 --attn_fn topk --attn_topk 8
done

echo "==== [deep] ALL DONE (失败 $FAIL 个) END $(date)"
exit 0
