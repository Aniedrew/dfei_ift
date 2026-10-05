#!/bin/bash
#
# 次级顶点归属小题 — 全局匹配 (Sinkhorn) 对照实验
#
# 动机: 逐对分类只看"这一对像不像真边", 而物理约束是**排他** —— 一条径迹只能属于一条链。
#   Sinkhorn 把事件的边分数放进 (n+1)x(n+1) 矩阵 (末行/列 dustbin) 做 log-domain 双随机归一,
#   得到"在全局可分配方案下该边被选中的 log 概率" -> 让边与边**竞争**。可微, 端到端训练。
#
# 数据: report_figs/feat_ceiling_big2.npz  (比 big.npz 多 t0/t1/ntr 三个端点 id 数组;
#       X/y/grp 与 big.npz **逐位一致**, 所以与正在跑的 seeds job (10485484, 用 big.npz) 可直接配对)。
#   含 4827 事件 / 86690 边 (真边 45036, 难负例 41654); 测试集 16958 边。
#
# 参照线 (seeds job 单种子): attn softmax geom  pairwise AP=0.8042, p@r90=0.6604
#
# 设计 (base = attn softmax bias=geom, feats=node+geo+ctx):
#   A. match=none      x seed{0,1,2}  —— 配对基线 (应逐位复现 seeds job, 兼作确定性校验)
#   B. match=sinkhorn  x seed{0,1,2}  tau=0.5  (默认)
#   C. match=sinkhorn  x seed{0,1,2}  tau=0.25 (更尖锐的分配)
#   D. match=sinkhorn  x seed{0,1,2}  tau=1.0  (更软的分配)
#
# 判读: 看测试集 **comb** (pairwise+match 相加) 与 **match** 单项的 AP / p@r90 能否超过 none 的
#   pairwise 参照 (0.8042 / 0.6604); 多种子看 Δ 是否稳定超过种子噪声。
#
# 用法:
#   hep_sub submit_vertex_assoc_sinkhorn.sh -argu "report_figs/feat_ceiling_big2.npz" \
#       -g ghigh -gpu 1 -cpu 4 -m 16000 -wt long \
#       -o logs/vertex_sinkhorn.out -e logs/vertex_sinkhorn.err
#
source ~/.bashrc
export PATH=$HOME/miniconda3/envs/dfei/bin:$PATH
export PYTHONPATH=/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn:$PYTHONPATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
cd /lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn

NPZ="${1:-report_figs/feat_ceiling_big2.npz}"
echo "JOB ${_CONDOR_IHEP_JOB_ID:-?} HOST $(hostname) GPU $CUDA_VISIBLE_DEVICES START $(date)"
echo "NPZ=$NPZ  (参照: 单种子 attn softmax geom AP=0.8042 / p@r90=0.6604; GBDT AP=0.7824)"
python3 -u -c "import torch; print('[PREFLIGHT] cuda', torch.cuda.is_available())"

FAIL=0
run() {   # run <seed> <args...>
  SD="$1"; shift
  echo "==== [sinkhorn] $(date +%H:%M:%S) seed=$SD $*"
  # 注意: `cmd | tail` 的 $? 是 tail 的退出码 -> 必须用 PIPESTATUS, 否则崩溃会被记成成功 (踩过)
  python3 -u analyze_vertex_assoc.py train --npz "$NPZ" --epochs 60 --patience 12 --seed "$SD" "$@" \
      2>&1 | tail -6
  rc=${PIPESTATUS[0]}
  echo "==== [sinkhorn] EXIT=$rc seed=$SD ($*)"
  [ $rc -ne 0 ] && FAIL=$((FAIL + 1))
}

BASE="--model attn --feats node+geo+ctx --attn_fn softmax --attn_bias geom"

# ---- A. 配对基线 (match=none) x 3 种子 ----
for SD in 0 1 2; do
  run $SD $BASE --match none
done

# ---- B. Sinkhorn tau=0.5 x 3 种子 ----
for SD in 0 1 2; do
  run $SD $BASE --match sinkhorn --match_tau 0.5
done

# ---- C. Sinkhorn tau=0.25 x 3 种子 ----
for SD in 0 1 2; do
  run $SD $BASE --match sinkhorn --match_tau 0.25
done

# ---- D. Sinkhorn tau=1.0 x 3 种子 ----
for SD in 0 1 2; do
  run $SD $BASE --match sinkhorn --match_tau 1.0
done

echo "==== [sinkhorn] ALL DONE (失败 $FAIL 个) END $(date)"
exit 0
