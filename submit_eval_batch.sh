#!/bin/bash
# 批量评估: 在一个 GPU 名额里顺序跑多个 eval 配置。
# 动机 (2026-09-24): 集群按用户限并发 GPU 名额 (~3), 单个 eval 只有几分钟却各占一个 job 号,
# 结果 8 个 eval 排了 15h 还没跑。把 N 个配置合成 1 个 job -> 1 个名额就能全跑完。
# 用法: hep_sub submit_eval_batch.sh -argu "cfg1.yaml cfg2.yaml ..." -g ghigh -gpu 1 -cpu 4 -m 32000 -wt long \
#          -o logs/eval_batch.out -e logs/eval_batch.err
[ $# -ge 1 ] || { echo "usage: submit_eval_batch.sh <cfg1.yaml> [cfg2.yaml ...]"; exit 2; }
source ~/.bashrc
export PATH=$HOME/miniconda3/envs/dfei/bin:$PATH
export PYTHONPATH=/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn:$PYTHONPATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn

echo "JOB ${_CONDOR_IHEP_JOB_ID:-?} HOST $(hostname) GPU $CUDA_VISIBLE_DEVICES START $(date)"
echo "CONFIGS($#): $*"

python3 -u -c "
import torch
assert torch.cuda.is_available(), 'cuda not available'
p = torch.cuda.get_device_properties(0)
print('[PREFLIGHT] OK', p.name)
a = torch.randn(500,500,device='cuda'); print('[PREFLIGHT] matmul OK', float((a@a).sum()))
"
if [ $? -ne 0 ]; then
  N=0; F=logs/eval_batch.retry_count
  [ -f "$F" ] && N=$(cat "$F")
  if [ "$N" -lt 200 ]; then
    echo $((N+1)) > "$F"
    echo "[PREFLIGHT] 失败, 60s 后重排 (第 $((N+1)) 次)"
    sleep 60
    unset CUDA_VISIBLE_DEVICES NVIDIA_VISIBLE_DEVICES
    R=$(mktemp /tmp/resub_batch_XXXXXX.sh)
    { echo '#!/bin/bash'
      printf 'cd %q || exit 1\n' "$PWD"
      printf 'exec hep_sub submit_eval_batch.sh -argu %q -g ghigh -gpu 1 -cpu 4 -m 32000 -wt long -o logs/eval_batch.out -e logs/eval_batch.err\n' "$*"
    } > "$R"
    env -i HOME="$HOME" USER="$USER" LOGNAME="$USER" SHELL=/bin/bash TERM=dumb /bin/bash -l "$R"
    rm -f "$R"
  fi
  exit 0
fi

FAIL=0
for cf in "$@"; do
  echo "==== [batch] $(date +%H:%M:%S) -> $cf"
  python3 -u wmpgnn/analysis/evaluate.py --config "config_files/$cf"
  rc=$?
  echo "==== [batch] $(date +%H:%M:%S) $cf EXIT=$rc"
  [ $rc -ne 0 ] && FAIL=$((FAIL+1))
done
echo "==== [batch] ALL DONE (失败 $FAIL 个) END $(date)"
exit 0
