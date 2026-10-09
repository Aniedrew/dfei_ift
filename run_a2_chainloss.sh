#!/bin/bash
#
# A2: 链级存活损失 (softmin/prod) 在多 B 子集上的对照实验 (纯 CPU)。
#
# 背景: 物理判据是"整条链每个环节都要对"(AND, 一失毁全链), 训练却一直是逐边 BCE。
#   本脚本比较 chain_loss_w=0(基线) / softmin(罚最弱环) / prod(几何平均, ≈逐边 BCE 的对照),
#   判据是 **链存活@边精度90%**(AND 语义) 而不是只看边级 AP。
#
# 用法: bash run_a2_chainloss.sh [seeds...]  (默认 0 1 2; 每个 seed 跑 4 组配置 × 2 子集)
source ~/.bashrc
export PATH=$HOME/miniconda3/envs/dfei/bin:$PATH
export PYTHONPATH=/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn:$PYTHONPATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
cd /lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn

SEEDS="${*:-0 1 2}"
BASE="--npz report_figs/feat_ceiling_nb.npz --epochs 60 --patience 12 --model attn --attn_fn softmax
      --attn_bias geom --layers 4 --lr 3e-4 --feats node+geo+ctx"

for SUB in "2 999999 multib" "1 1 singleb"; do
  LO="${SUB%% *}"; REST="${SUB#* }"; HI="${REST%% *}"; NM="${REST#* }"
  echo "################ $NM (nb in [$LO,$HI]) ################"
  for SD in $SEEDS; do
    for CW in "0.0 softmin" "1.0 softmin" "1.0 prod" "0.3 softmin"; do
      W="${CW%% *}"; PL="${CW##* }"
      R=$(python3 -u analyze_vertex_assoc.py train $BASE --min_nb "$LO" --max_nb "$HI" \
            --seed "$SD" --chain_loss_w "$W" --chain_pool "$PL" 2>&1 | grep "测试集")
      echo "  seed=$SD w=$W pool=$PL  $R"
    done
  done
done
echo "######## A2 DONE $(date) ########"
