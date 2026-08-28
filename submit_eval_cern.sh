#!/bin/bash
#
# DFEI 模型评估 (CERN数据训练, version_8)
# 评估40个测试文件
#
# 提交方式:
#   hep_sub submit_eval_cern.sh -g lzuhep -gpu 1 -cpu 4 -m 16000 -wt short -o logs/eval_cern.out -e logs/eval_cern.err
#

source ~/.bashrc

export PATH=$HOME/miniconda3/envs/dfei/bin:$PATH
export PYTHONPATH=/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn:$PYTHONPATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn

echo "========================================"
echo "JOB ID      : $_CONDOR_IHEP_JOB_ID"
echo "HOST        : $(hostname)"
echo "START TIME  : $(date)"
echo "PYTHON      : $(which python3)"
echo "GPU         : $CUDA_VISIBLE_DEVICES"
echo "CONFIG      : config_files/eval_CERN_DFEI.yaml"
echo "========================================"

python3 -u wmpgnn/analysis/evaluate.py --config config_files/eval_CERN_DFEI.yaml

EXIT_CODE=$?
echo "========================================"
echo "EXIT CODE   : $EXIT_CODE"
echo "END TIME    : $(date)"
echo "========================================"

# 邮件通知
JOB_ID="${_CONDOR_IHEP_JOB_ID:-unknown}"
echo "DFEI 评估作业 $JOB_ID 已完成。
HOST: $(hostname)
END TIME: $(date)
EXIT CODE: $EXIT_CODE
日志: tail -50 /lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn/logs/eval_cern.out" | \
  mail -s "[DFEI] Eval $JOB_ID finished (exit=$EXIT_CODE)" guoqingxiang21@mails.ucas.ac.cn

exit $EXIT_CODE
