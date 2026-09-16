#!/bin/bash
#
# DFEI v38 栈 + GN 中间宽度 128→256 (轻量容量测试, latent 16 不变)
# 对照 v38 (Perfect 29.26/All ~50.6); v53/v512 显示 latent 升维是破坏性改动,
# 本实验只加宽 GN 更新网络, encoder/decoder/下游头全继承
#
# 提交方式:
#   hep_sub submit_train_cern_v38_wide.sh -g ghigh -gpu 1 -cpu 4 -m 64000 -wt long \
#       -o logs/v38_wide.out -e logs/v38_wide.err
#

source ~/.bashrc

export PATH=$HOME/miniconda3/envs/dfei/bin:$PATH
export PYTHONPATH=/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn:$PYTHONPATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn

CFG=config_files/train_CERN_v38_wide.yaml

echo "========================================"
echo "JOB ID      : $_CONDOR_IHEP_JOB_ID"
echo "HOST        : $(hostname)"
echo "START TIME  : $(date)"
echo "GPU         : $CUDA_VISIBLE_DEVICES"
echo "CONFIG      : $CFG"
echo "========================================"

# === GPU 预检 (失败自动重排) ===
echo "[PREFLIGHT] GPU check at $(date), device=$CUDA_VISIBLE_DEVICES"
nvidia-smi -L 2>&1 | head -3
python3 -u -c "
import torch
if not torch.cuda.is_available():
    print('[PREFLIGHT] FAIL: torch.cuda.is_available()=False')
    raise SystemExit(77)
p = torch.cuda.get_device_properties(0)
print(f'[PREFLIGHT] OK: {p.name} (mem {p.total_memory/1024**3:.1f} GB)')
a = torch.randn(500,500,device='cuda')
b = (a @ a).sum().item()
print('[PREFLIGHT] matmul OK, sum=%.3f' % b)
"
PREFLIGHT_RC=$?
if [ $PREFLIGHT_RC -ne 0 ]; then
  echo "[PREFLIGHT] FAIL (rc=$PREFLIGHT_RC): 分配的GPU不可用"
  RETRY_COUNT_FILE=/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn/logs/v38_wide.retry_count
  M=0
  [ -f "$RETRY_COUNT_FILE" ] && M=$(cat "$RETRY_COUNT_FILE")
  M=$((M+1)); echo "$M" > "$RETRY_COUNT_FILE"
  echo "[RETRY] 第 $M 次, sleep 120s 后重排..."
  sleep 120
  # 自重提交有 64KB env 限制: 只裁剪最大非保留变量 (勿 unset 白名单, 会破坏 hep_sub 组解析)
  KEEP=" HOME PATH USER LOGNAME HOSTNAME SHELL PWD OMP_NUM_THREADS PYTORCH_CUDA_ALLOC_CONF CUDA_VISIBLE_DEVICES PYTHONPATH _CONDOR_IHEP_JOB_ID "
  for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30; do
    SZ=$(env | wc -c)
    [ "$SZ" -le 64000 ] && break
    BIG=""
    while read -r vn; do
      case " $KEEP " in *" $vn "*) continue ;; esac
      BIG="$vn"; break
    done < <(env | awk -F= '{if (length($1)>0) print length($0), $1}' | sort -rn | cut -d' ' -f2)
    if [ -z "$BIG" ]; then echo "[trim_env] 无可裁剪的非保留变量, 停止"; break; fi
    echo "[trim_env] env ${SZ}B -> unset $BIG"
    unset "$BIG" 2>/dev/null
  done
  echo "[trim_env] env size now $(env | wc -c) B"
  hep_sub submit_train_cern_v38_wide.sh -g ghigh -gpu 1 -cpu 4 -m 64000 -wt long \
      -o logs/v38_wide.out -e logs/v38_wide.err
  echo "[RETRY] 已重提, 本次退出"
  exit 0
fi

python3 -u wmpgnn/analysis/trainer.py --config "$CFG"
EXIT_CODE=$?
echo "========================================"
echo "EXIT CODE   : $EXIT_CODE"
echo "END TIME    : $(date)"
echo "========================================"

JOB_ID="${_CONDOR_IHEP_JOB_ID:-unknown}"
STATUS="✅ 完成"; [ $EXIT_CODE -ne 0 ] && STATUS="❌ 失败"
curl -s --connect-timeout 10 -X POST https://sctapi.ftqq.com/SCT387631TDiuLj6UNUsFTaDRjkaSWcdPv.send \
  -d "title=[DFEI] ${STATUS} v38宽GN(256) Job ${JOB_ID}" \
  -d "desp=## v38 栈 + GN 宽度 256 ${STATUS}
| 作业ID | ${JOB_ID} |
| 状态 | ${STATUS} |
| 配置 | train_CERN_v38_wide.yaml |
| 结束时间 | $(date) |
| 退出码 | ${EXIT_CODE} |

\`\`\`bash
tail -40 /lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn/logs/v38_wide.out
\`\`\`" > /dev/null 2>&1

exit $EXIT_CODE
