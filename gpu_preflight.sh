#!/bin/bash
# 共享 GPU 预检 + 换卡自重排 (被 vertex 类提交脚本 source)
#
# 动机 (2026-10-03): vertex 作业反复落在同一张坏卡上 (cuda False) 却静默退化成 CPU,
#   单次 60 epoch 要 5 分钟; 而训练作业 (submit_train_cern_one.sh) 因为带预检+重排,
#   总能换到好卡。这里复用那套经验: 预检失败 -> 清掉 GPU 绑定 -> 干净 shell 自重排。
#   与训练脚本的区别: 达重试上限后**退化为 CPU 继续跑**(而不是放弃作业), 避免白丢。
#
# 调用方需在 source 之前设置:
#   PF_SCRIPT : 本提交脚本文件名 (用于自重排)
#   PF_ARGS   : -argu 的参数字符串
#   PF_FLAGS  : hep_sub 其余参数 (含 -o/-e)
#   PF_TAG    : 重试计数文件名 (logs/<PF_TAG>.retry_count)
# 若 GPU 可用则直接返回并继续; 否则自重排并 exit 0。

PF_MAX_RETRY="${PF_MAX_RETRY:-60}"

if python3 -u -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 77)" 2>/dev/null; then
  echo "[PREFLIGHT] OK: $(python3 -c 'import torch;print(torch.cuda.get_device_name(0))')"
  return 0 2>/dev/null || exit 0
fi

echo "[PREFLIGHT] GPU 不可用 (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-?})"
nvidia-smi -L 2>&1 | head -2

PF_N=0
PF_F="logs/${PF_TAG}.retry_count"
[ -f "$PF_F" ] && PF_N=$(cat "$PF_F")

if [ "$PF_N" -lt "$PF_MAX_RETRY" ]; then
  echo $((PF_N + 1)) > "$PF_F"
  echo "[PREFLIGHT] 第 $((PF_N + 1))/$PF_MAX_RETRY 次重排, 45s 后重提 (换卡)"
  sleep 45
  # 关键: 清掉父作业继承的 GPU 绑定, 否则重提会被钉回同一张坏卡
  unset CUDA_VISIBLE_DEVICES NVIDIA_VISIBLE_DEVICES
  PF_R=$(mktemp /tmp/resub_vertex_XXXXXX.sh)
  {
    echo '#!/bin/bash'
    printf 'cd %q || exit 1\n' "$PWD"
    printf 'exec hep_sub %q -argu %q %s\n' "$PF_SCRIPT" "$PF_ARGS" "$PF_FLAGS"
  } > "$PF_R"
  # 干净登录 shell (~2.3KB env): 自重排会逐代累积环境变量, 约 200 代后越过 64KB 上限
  env -i HOME="$HOME" USER="$USER" LOGNAME="$USER" SHELL=/bin/bash TERM=dumb /bin/bash -l "$PF_R"
  rm -f "$PF_R"
  echo "[PREFLIGHT] 已重提, 本次退出"
  exit 0
fi

echo "[PREFLIGHT] 已达 $PF_MAX_RETRY 次上限, 退化为 CPU 运行 (结果仍有效, 只是慢)"
return 0 2>/dev/null || exit 0
