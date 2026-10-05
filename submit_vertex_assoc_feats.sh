#!/bin/bash
#
# 次级顶点归属 — **特征组贡献**扫描 (统一 4 层骨干, 2026-10-04)
#
# 动机: 之前"边信息 vs 点信息"之争靠 GBDT 逐步累加回答, 但那是在**逐对分类器**上做的。
#   现在有了注意力, 上下文可以由结构自己生成 -> 应重问一次: 哪些特征组真的还有边际价值?
#   特别是 ctx (手工竞争上下文) 在 2 层下已显示"不加更好"(0.8054 vs 0.7997), 4 层下再确认。
#
# 组定义 (见 analyze_vertex_assoc.py GROUPS):
#   node=16 维两端节点, geo=3 维几何, raw=5 维原始量, der=9 维导出量, vrt=3 维次级顶点, ctx=11 维手工上下文
#
# 用法:
#   hep_sub submit_vertex_assoc_feats.sh -argu "report_figs/feat_ceiling_big2.npz" \
#       -g ghigh -gpu 1 -cpu 4 -m 16000 -wt long \
#       -o logs/vertex_feats.out -e logs/vertex_feats.err
#
source ~/.bashrc
export PATH=$HOME/miniconda3/envs/dfei/bin:$PATH
export PYTHONPATH=/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn:$PYTHONPATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
cd /lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn

NPZ="${1:-report_figs/feat_ceiling_big2.npz}"
export PF_SCRIPT="submit_vertex_assoc_feats.sh"
export PF_ARGS="$NPZ"
export PF_FLAGS="-g ghigh -gpu 1 -cpu 4 -m 16000 -wt long -o logs/vertex_feats.out -e logs/vertex_feats.err"
export PF_TAG="vertex_feats"
source ./gpu_preflight.sh

echo "JOB ${_CONDOR_IHEP_JOB_ID:-?} HOST $(hostname) GPU ${CUDA_VISIBLE_DEVICES:-?} START $(date)"
echo "NPZ=$NPZ  (参照: L4 node+geo+ctx AP=0.8146; all47 见 deep 作业)"

FAIL=0
run() {
  SD="$1"; shift
  echo "==== [feats] $(date +%H:%M:%S) seed=$SD $*"
  python3 -u analyze_vertex_assoc.py train --npz "$NPZ" --epochs 60 --patience 12 --seed "$SD" "$@" \
      2>&1 | tail -6
  rc=${PIPESTATUS[0]}
  echo "==== [feats] EXIT=$rc seed=$SD ($*)"
  [ $rc -ne 0 ] && FAIL=$((FAIL + 1))
}

AT="--model attn --attn_fn softmax --attn_bias geom --layers 4"

# ---- A. 逐步累加 (seed 0) ----
run 0 $AT --feats node
run 0 $AT --feats node+geo
run 0 $AT --feats node+geo+raw
run 0 $AT --feats node+geo+raw+der
run 0 $AT --feats node+geo+raw+der+vrt
run 0 $AT --feats node+geo+ctx

# ---- B. 只有上下文 / 只有几何 (信息来自哪里) ----
run 0 $AT --feats geo
run 0 $AT --feats ctx
run 0 $AT --feats ctx+geo

# ---- C. 去掉 ctx 但给全几何: node+geo+raw+der+vrt x 3 种子 (最重要的一项, 多给种子) ----
for SD in 0 1 2; do
  run $SD $AT --feats node+geo+raw+der+vrt
done

echo "==== [feats] ALL DONE (失败 $FAIL 个) END $(date)"
exit 0
