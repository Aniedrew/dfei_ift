#!/bin/bash
#
# DFEI CERN v53 (asym 宽隐空间) 链式自动续训
# 背景: v53 每次作业都被墙钟限制在 ~ep74 截断 (metrics 0-74, 目标 150)。
# 本脚本每段:
#   1) 自动把 config 的 resume_ckpt 指向 version_53 里最新的 epoch_epoch=*.ckpt
#   2) 训练 (max_epochs=150, PL 会从 checkpoint 的 epoch 继续)
#   3) 结束后若最新 epoch < 149, 自动重提交自己 (链式续训), 直至跑满
#
# 提交方式:
#   hep_sub submit_train_cern_v53_resume.sh -g ghigh -gpu 1 -cpu 4 -m 64000 -wt long \
#       -o logs/v53_resume.out -e logs/v53_resume.err
#

source ~/.bashrc

export PATH=$HOME/miniconda3/envs/dfei/bin:$PATH
export PYTHONPATH=/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn:$PYTHONPATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn

CONFIG=config_files/train_CERN_v38_masshead2_asym_resume.yaml
# 从配置读固定续训目录 (log_version), 避免再查错目录
CKPT_DIR=$(grep -oE 'log_version: [0-9]+' "$CONFIG" | grep -oE '[0-9]+$')
CKPT_DIR=/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn/LHCb_logs/DFEI/version_${CKPT_DIR}/checkpoints
TARGET_EPOCH=149                      # 最后一个 epoch (epochs=150 -> 0..149)
MAX_CHAIN=12
CHAIN_FILE=/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn/logs/v53_resume.chain_count

echo "========================================"
echo "JOB ID      : $_CONDOR_IHEP_JOB_ID"
echo "HOST        : $(hostname)"
echo "START TIME  : $(date)"
echo "GPU         : $CUDA_VISIBLE_DEVICES"
echo "CONFIG      : $CONFIG"
echo "========================================"

# === 找 version_53 里最新的 epoch checkpoint ===
latest_ckpt=$(ls "$CKPT_DIR"/epoch_epoch=*.ckpt 2>/dev/null | \
  sed -E 's/.*epoch=([0-9]+).*/\1 &/' | sort -n | tail -1 | cut -d' ' -f2-)
if [ -z "$latest_ckpt" ]; then
  echo "[FAIL] $CKPT_DIR 里没有 epoch_epoch=*.ckpt"
  exit 1
fi
cur_epoch=$(basename "$latest_ckpt" | grep -oE 'epoch=[0-9]+' | cut -d= -f2)
echo "[PRE] 最新 checkpoint: $(basename "$latest_ckpt")  (epoch $cur_epoch/$TARGET_EPOCH)"

# === 已跑满则直接结束 ===
if [ "$cur_epoch" -ge "$TARGET_EPOCH" ]; then
  echo "[DONE] 已达到目标 epoch $TARGET_EPOCH, 无需续训"
  exit 0
fi

# === 链式计数: 每段 +1, 防止死循环 ===
N=0; [ -f "$CHAIN_FILE" ] && N=$(cat "$CHAIN_FILE")
N=$((N+1)); echo "$N" > "$CHAIN_FILE"
if [ "$N" -gt "$MAX_CHAIN" ]; then
  echo "[FAIL] 链式续训超过 ${MAX_CHAIN} 段, 停止"
  exit 1
fi
echo "[CHAIN] 第 $N/$MAX_CHAIN 段续训"

# === 把 resume_ckpt 改写为最新 checkpoint ===
sed -i "s|^  resume_ckpt:.*|  resume_ckpt: \"$latest_ckpt\"|" "$CONFIG"
echo "[PRE] resume_ckpt -> $(grep 'resume_ckpt' "$CONFIG")"

# === GPU 预检 (失败自动重排, 不指定节点) ===
echo "[PREFLIGHT] GPU check at $(date), device=$CUDA_VISIBLE_DEVICES"
nvidia-smi -L 2>&1 | head -3
python3 -u -c "
import torch
if not torch.cuda.is_available():
    print('[PREFLIGHT] FAIL: torch.cuda.is_available()=False')
    raise SystemExit(77)
p = torch.cuda.get_device_properties(0)
print(f'[PREFLIGHT] OK: {p.name} (cap {p.major}.{p.minor}, mem {p.total_memory/1024**3:.1f} GB)')
a = torch.randn(500,500,device='cuda')
b = (a @ a).sum().item()
print('[PREFLIGHT] matmul OK, sum=%.3f' % b)
"
PREFLIGHT_RC=$?
if [ $PREFLIGHT_RC -ne 0 ]; then
  echo "[PREFLIGHT] FAIL (rc=$PREFLIGHT_RC): 分配的GPU不可用"
  MAX_RETRY=1000
  RETRY_SLEEP=60
  RETRY_COUNT_FILE=/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn/logs/v53_resume.retry_count
  M=0
  [ -f "$RETRY_COUNT_FILE" ] && M=$(cat "$RETRY_COUNT_FILE")
  if [ "$M" -lt "$MAX_RETRY" ]; then
    M=$((M+1))
    echo "$M" > "$RETRY_COUNT_FILE"
    echo "[RETRY] 第 $M/$MAX_RETRY 次, sleep ${RETRY_SLEEP}s 后重排..."
    sleep $RETRY_SLEEP
    # 教训(ab02/v38_attn_full): 不要 unset 环境再自重提交——会误删 hep_sub 组解析
    # 变量, 报 "No resource serving for group 'ghigh'"。直接重提即可。
    hep_sub submit_train_cern_v53_resume.sh -g ghigh -gpu 1 -cpu 4 -m 64000 -wt long \
        -o logs/v53_resume.out -e logs/v53_resume.err
    echo "[RETRY] 已重提, 本次退出"
    exit 0
  fi
  echo "[RETRY] 已达 ${MAX_RETRY} 次上限, 放弃"
  exit 77
fi

# === 训练 (trainer.py 结束后还会自动跑一次评估) ===
python3 -u wmpgnn/analysis/trainer.py --config "$CONFIG"
EXIT_CODE=$?
echo "========================================"
echo "EXIT CODE   : $EXIT_CODE"
echo "END TIME    : $(date)"
echo "========================================"

# === 训练后: 检查是否已跑满, 否则链式续训 ===
newest_ckpt=$(ls "$CKPT_DIR"/epoch_epoch=*.ckpt 2>/dev/null | \
  sed -E 's/.*epoch=([0-9]+).*/\1 &/' | sort -n | tail -1 | cut -d' ' -f2-)
new_epoch=$(basename "$newest_ckpt" | grep -oE 'epoch=[0-9]+' | cut -d= -f2)
echo "[POST] 训练结束, 最新 epoch = $new_epoch/$TARGET_EPOCH (exit=$EXIT_CODE)"

if [ "$new_epoch" -lt "$TARGET_EPOCH" ]; then
  if [ "$new_epoch" -le "$cur_epoch" ]; then
    echo "[FAIL] 最新 epoch 没有前进 ($cur_epoch -> $new_epoch), 停止链式续训, 避免死循环"
    exit 1
  fi
  if [ "$N" -lt "$MAX_CHAIN" ]; then
    echo "[CHAIN] 尚未到 $TARGET_EPOCH (当前 $new_epoch), 重提交续训 ($N/$MAX_CHAIN)"
    hep_sub submit_train_cern_v53_resume.sh -g ghigh -gpu 1 -cpu 4 -m 64000 -wt long \
        -o logs/v53_resume.out -e logs/v53_resume.err
    echo "[CHAIN] 已重提, 本次退出"
    exit 0
  fi
fi

# === 完成通知 (Server酱) ===
JOB_ID="${_CONDOR_IHEP_JOB_ID:-unknown}"
STATUS="✅ 完成"
[ "$new_epoch" -lt "$TARGET_EPOCH" ] && STATUS="❌ 未跑满($new_epoch/$TARGET_EPOCH)"
curl -s --connect-timeout 10 -X POST https://sctapi.ftqq.com/SCT387631TDiuLj6UNUsFTaDRjkaSWcdPv.send \
  -d "title=[DFEI] ${STATUS} v53续训 Job ${JOB_ID}" \
  -d "desp=## v53 (asym) 续训作业 ${JOB_ID} ${STATUS}

| 项目 | 值 |
|------|-----|
| **作业ID** | ${JOB_ID} |
| **状态** | ${STATUS} |
| **主机** | $(hostname) |
| **最新 epoch** | ${new_epoch}/${TARGET_EPOCH} |
| **配置** | train_CERN_v38_masshead2_asym_resume.yaml |
| **结束时间** | $(date) |
| **退出码** | ${EXIT_CODE} |

\`\`\`bash
tail -50 /lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn/logs/v53_resume.out
\`\`\`" > /dev/null 2>&1

exit 0
