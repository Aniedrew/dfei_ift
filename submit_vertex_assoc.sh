#!/bin/bash
#
# 次级顶点归属小题 — **注意力机制矩阵** (本轮主线)
#
# 目的: 在独立小模型上, 干净地回答"注意力/top-k 这些方法到底有没有用"。
#   判据: 能否超过"GBDT + 全部 47 维特征"的 AP = 0.782 (大样本探针, report_figs/feat_ceiling_big_probe.json)。
#
# 矩阵 (全部 feats=node+geo+ctx, 除非注明):
#   1) mlp                       —— 无上下文基线 (只有局部特征)
#   2) attn softmax, bias=none   —— 纯自注意力 (上下文来自 token 间交互)
#   3) attn softmax, bias=geom   —— + 物理偏置: 用 doca/|Δ起点| 导出 attention bias (Graphormer 思路)
#   4) attn sparsemax, bias=geom —— 稀疏注意力 (欧氏投影到单纯形, 可产生精确 0 权重)
#   5) attn topk4,  bias=geom    —— 模型内**可学习稀疏化**: 每个 query 只保留 top-k 个 key
#   6) attn topk8,  bias=geom
#   7) attn topk16, bias=geom
#   8) attn softmax, bias=geom, feats=node+geo —— 不给手工 ctx, 看注意力能否自己把上下文挖出来
#
# 注意与旧做法的区别: 旧的 edge_topk 是**推理时截断**(已证明只是工作点旋钮); 这里 5-7 是
#   **模型内部的稀疏注意力**, 让模型自己学"该看哪些邻居"。
#
# 用法:
#   hep_sub submit_vertex_assoc.sh -argu "<npz路径>" -g ghigh -gpu 1 -cpu 4 -m 16000 -wt long \
#       -o logs/vertex_assoc_attn.out -e logs/vertex_assoc_attn.err
#
source ~/.bashrc
export PATH=$HOME/miniconda3/envs/dfei/bin:$PATH
export PYTHONPATH=/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn:$PYTHONPATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
cd /lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn

NPZ="${1:-report_figs/feat_ceiling_big.npz}"
echo "JOB ${_CONDOR_IHEP_JOB_ID:-?} HOST $(hostname) GPU $CUDA_VISIBLE_DEVICES START $(date)"
echo "NPZ=$NPZ  (GBDT 参考线: AP=0.782 / AUC=0.772)"
python3 -u -c "import torch; print('[PREFLIGHT] cuda', torch.cuda.is_available())"

FAIL=0
run() {
  echo "==== [va-attn] $(date +%H:%M:%S) $*"
  # 注意: `cmd | tail` 的 $? 是 tail 的退出码 -> 会把 python 的崩溃吞掉 (实测踩过)。
  python3 -u analyze_vertex_assoc.py train --npz "$NPZ" --epochs 60 --patience 12 --seed 0 "$@" 2>&1 | tail -8
  rc=${PIPESTATUS[0]}
  echo "==== [va-attn] EXIT=$rc  ($*)"
  [ $rc -ne 0 ] && FAIL=$((FAIL + 1))
}

run --model mlp  --feats node+geo+ctx
run --model attn --feats node+geo+ctx --attn_fn softmax   --attn_bias none
run --model attn --feats node+geo+ctx --attn_fn softmax   --attn_bias geom
run --model attn --feats node+geo+ctx --attn_fn sparsemax --attn_bias geom
run --model attn --feats node+geo+ctx --attn_fn topk --attn_topk 4  --attn_bias geom
run --model attn --feats node+geo+ctx --attn_fn topk --attn_topk 8  --attn_bias geom
run --model attn --feats node+geo+ctx --attn_fn topk --attn_topk 16 --attn_bias geom
run --model attn --feats node+geo      --attn_fn softmax --attn_bias geom

echo "==== [va-attn] ALL DONE (失败 $FAIL 个) END $(date)"
exit 0
