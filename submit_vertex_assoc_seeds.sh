#!/bin/bash
#
# 次级顶点归属小题 — 3-seed 复现 + 消融 (本轮第 1-2 项)
#
# 动机: 上一轮 (10479341) 的结论是"注意力 > MLP ≈ GBDT", 但**只有单种子**, 且
#   softmax-geom / topk4/8/16 / 无手工ctx 这一簇差异 ≤0.003 -> 必须多种子确认哪些差异是真的。
#   同时做两组消融: bias 列 (none / geom=doca+dstart / geom4=再加 dz0,logDOCA)、层数与头数。
#
# 用法:
#   hep_sub submit_vertex_assoc_seeds.sh -argu "<npz>" -g ghigh -gpu 1 -cpu 4 -m 16000 -wt long \
#       -o logs/vertex_seeds.out -e logs/vertex_seeds.err
#
source ~/.bashrc
export PATH=$HOME/miniconda3/envs/dfei/bin:$PATH
export PYTHONPATH=/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn:$PYTHONPATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
cd /lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn

NPZ="${1:-report_figs/feat_ceiling_big.npz}"
echo "JOB ${_CONDOR_IHEP_JOB_ID:-?} HOST $(hostname) GPU $CUDA_VISIBLE_DEVICES START $(date)"
echo "NPZ=$NPZ   (参考线: GBDT AP=0.7824; MLP 单种子 0.7777; attn softmax geom 单种子 0.8042)"
python3 -u -c "import torch; print('[PREFLIGHT] cuda', torch.cuda.is_available())"

FAIL=0
run() {   # run <seed> <args...>
  SD="$1"; shift
  echo "==== [seed] $(date +%H:%M:%S) seed=$SD $*"
  # 注意: `cmd | tail` 的 $? 是 tail 的 -> 必须用 PIPESTATUS, 否则崩溃会被记成成功 (踩过)
  python3 -u analyze_vertex_assoc.py train --npz "$NPZ" --epochs 60 --patience 12 --seed "$SD" "$@" \
      2>&1 | tail -6
  rc=${PIPESTATUS[0]}
  echo "==== [seed] EXIT=$rc seed=$SD ($*)"
  [ $rc -ne 0 ] && FAIL=$((FAIL + 1))
}

# ---- A. 关键 4 组 × 3 种子 (确认哪些差异是真的) ----
for SD in 0 1 2; do
  run $SD --model mlp  --feats node+geo+ctx
  run $SD --model attn --feats node+geo+ctx --attn_fn softmax --attn_bias geom
  run $SD --model attn --feats node+geo+ctx --attn_fn topk --attn_topk 8 --attn_bias geom
  run $SD --model attn --feats node+geo+ctx --attn_fn softmax --attn_bias none
done

# ---- B. 消融 (单种子): bias 列扩展 / 层数 / 头数 / 无手工ctx ----
run 0 --model attn --feats node+geo+ctx --attn_fn softmax --attn_bias geom4
run 0 --model attn --feats node+geo+ctx --attn_fn softmax --attn_bias geom --layers 1
run 0 --model attn --feats node+geo+ctx --attn_fn softmax --attn_bias geom --layers 4
run 0 --model attn --feats node+geo+ctx --attn_fn softmax --attn_bias geom --heads 8
run 0 --model attn --feats node+geo      --attn_fn softmax --attn_bias geom

echo "==== [seed] ALL DONE (失败 $FAIL 个) END $(date)"
exit 0
