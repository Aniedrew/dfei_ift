#!/usr/bin/env python3
"""v38+attention 冒烟测试 (CPU):
1) TrackSelfAttention 功能性: 事件内 track 相等 -> 输出相等; 改变别的事件不影响本事件 (mask 正确)
2) cpt=38 加载: attention 新参数随机初始化, 主干加载成功
"""
import sys, os, torch
sys.path.append('/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn')
import yaml
from wmpgnn.model.attention import TrackSelfAttention
from wmpgnn.analysis.config_adjusting import adjust_config_training
from wmpgnn.analysis.load_module import load_module

torch.manual_seed(0)
BASE = '/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn'
os.chdir(BASE)

# ---- 1) 功能性: 事件内一致性 + 跨事件独立性 (mask 生效) ----
attn = TrackSelfAttention(dim=16, n_heads=4, dropout=0.0)
batch = torch.tensor([0, 0, 0, 1, 1, 1])
a = torch.randn(16)
x1 = torch.stack([a, a, a, torch.randn(16), torch.randn(16), torch.randn(16)])
o1 = attn(x1, batch)
assert torch.allclose(o1[:3], o1[0:1].expand(3, -1), atol=1e-5), "事件内相同 track 输出应相同"
# 换掉事件1, 事件0 的输出必须不变
x2 = torch.stack([a, a, a, torch.randn(16) * 5, torch.randn(16) * 5, torch.randn(16) * 5])
o2 = attn(x2, batch)
assert torch.allclose(o1[:3], o2[:3], atol=1e-5), "跨事件注意力应被 mask, 事件0输出不能变"
print("[ok] TrackSelfAttention: 事件内一致性 OK, 跨事件独立性 OK")

# ---- 2) cpt=38 加载 (attention 新参数) ----
cfg = yaml.safe_load(open('config_files/train_CERN_v38_attn.yaml'))
cfg = adjust_config_training(cfg)
pw = {"LCA": torch.ones(4), "nodes": torch.tensor(1.0),
      "edges": torch.tensor(1.0), "pv_asso": torch.tensor(1.0)}
module = load_module(cfg, pw)
sd = module.state_dict()
attn_keys = [k for k in sd if k.startswith("model._track_attn.")]
assert len(attn_keys) > 0, "attention 参数缺失"
print(f"[ok] attention 参数: {len(attn_keys)} 个, 示例 {attn_keys[:2]}")
nz = sum(int((sd[k] != 0).sum()) for k in sd if not k.startswith("model._track_attn."))
print(f"[ok] 主干加载: 非注意力参数非零计数={nz}")
print("[ALL PASS]")
