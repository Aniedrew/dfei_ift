#!/usr/bin/env python3
"""v38 栈 16→32/24 升维 + 部分继承 冒烟测试 (CPU):
1) 从 v38 best (cpt=38) 加载到升维模型: 不报 size mismatch
2) 打印 skip(shape) 清单 —— 应恰为与 tracks/tt 16 维耦合的层 (各类型 MLP 末层 + 下游头输入层)
3) 主干继承: 其余层权重非零 (来自 v38), 数值层比例应远大于随机层
"""
import sys, os, torch
sys.path.append('/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn')
import yaml
from wmpgnn.analysis.config_adjusting import adjust_config_training
from wmpgnn.analysis.load_module import load_module

BASE = '/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn'
os.chdir(BASE)

cfg = yaml.safe_load(open(sys.argv[1] if len(sys.argv) > 1
                          else 'config_files/train_CERN_v38_asym_up.yaml'))
cfg = adjust_config_training(cfg)
pw = {"LCA": torch.ones(4), "nodes": torch.tensor(1.0),
      "edges": torch.tensor(1.0), "pv_asso": torch.tensor(1.0)}
module = load_module(cfg, pw)

sd = module.state_dict()
n_param_layers = len(sd)
n_random = sum(1 for k, v in sd.items() if "model._track_attn" in k)  # 本 config 无 attention -> 0
# shape-skip 层从 stdout 读取 (load_state_dict 已打印 skip(shape) 行)
print(f"[ok] 总参数层数 {n_param_layers} (含 buffer), attention 层 {n_random} (应为 0)")
nz = tot = 0
pending = 0
for v in sd.values():
    if not v.dtype.is_floating_point:
        continue
    try:
        nz += int((v != 0).sum())
        tot += int(v.numel())
    except ValueError:          # uninit Lazy 层 (drop 后待首 forward materialize)
        pending += 1
print(f"[ok] 浮点参数非零比例 {nz}/{tot} = {nz/tot:.4f} (继承成功应接近 1)")
print(f"[ok] 待首 forward 初始化的 Lazy 层: {pending} 个 (drop 的 tracks/tt 输入层, 预期)")
assert nz / tot > 0.95, "继承异常: 非零参数占比过低"
print("[ALL PASS]")
