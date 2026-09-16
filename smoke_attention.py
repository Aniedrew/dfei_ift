#!/usr/bin/env python3
"""v38+attention 冒烟测试 (CPU):
1) TrackSelfAttention 纯内容版: 事件内 track 相等 -> 输出相等; 改变别的事件不影响本事件 (mask 正确)
2) TrackSelfAttention edge-bias 版 (ParT 完全体):
   a) 同一事件内对称边 + 均匀边特征 -> 同内容 track 输出仍相等; 跨事件掩码仍生效
   b) 非零边特征能改变被连 track 的输出 (bias 梯度路径存活, scatter 生效)
3) 完整版 config 加载 (node_attention_edge_bias=true): _edge_bias 参数存在, 输入维=16
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

# ---- 1) 纯内容版: 事件内一致性 + 跨事件独立性 (mask 生效) ----
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
print("[ok] TrackSelfAttention 纯内容版: 事件内一致性 OK, 跨事件独立性 OK")

# ---- 2a) edge-bias 版: 对称边+均匀边特征下事件内一致性 & 跨事件掩码 ----
def complete_edges(m, off=0):
    """事件内 m 条 track 的完全图 (含双向+自环), 返回全局 index 的 rows/cols。

    自环需包含: 均匀 bias 下要让同内容 track 输出相等, 对角(自注意力位)也必须
    有同值 bias, 否则各行因"对角无 bias 而对称位有 bias"不再全等。
    """
    r, c = [], []
    for i in range(m):
        for j in range(m):
            r.append(off + i); c.append(off + j)
    return torch.tensor(r), torch.tensor(c)

attn_eb = TrackSelfAttention(dim=16, n_heads=4, dropout=0.0, edge_dim=16)
a = torch.randn(16)
x_eb1 = torch.stack([a, a, a, torch.randn(16), torch.randn(16), torch.randn(16)])
# 事件0: 完全图; 事件1: 完全图; 特征统一用同一向量 -> 对称偏置
rows0, cols0 = complete_edges(3, 0)
rows1, cols1 = complete_edges(3, 3)
rows = torch.cat([rows0, rows1]); cols = torch.cat([cols0, cols1])
feat = torch.randn(1, 16).expand(len(rows), -1).clone()  # 所有边同一特征 -> 偏置对称均匀
o_eb1 = attn_eb(x_eb1, batch, edge_index=torch.stack([rows, cols]), edge_feat=feat)
assert torch.allclose(o_eb1[:3], o_eb1[0:1].expand(3, -1), atol=1e-4), \
    "edge-bias: 同内容+对称边 track 输出应相同"
# 换掉事件1 内容, 事件0 输出不变
x_eb2 = torch.stack([a, a, a, torch.randn(16) * 5, torch.randn(16) * 5, torch.randn(16) * 5])
o_eb2 = attn_eb(x_eb2, batch, edge_index=torch.stack([rows, cols]), edge_feat=feat)
assert torch.allclose(o_eb1[:3], o_eb2[:3], atol=1e-4), "edge-bias: 跨事件掩码失效"
print("[ok] TrackSelfAttention edge-bias: 事件内一致性 OK, 跨事件独立性 OK")

# ---- 2b) 非零边特征能改变被连 track 的输出 (bias 通路存活) ----
attn_eb2 = TrackSelfAttention(dim=16, n_heads=4, dropout=0.0, edge_dim=16)
x_s = torch.randn(4, 16)
b_s = torch.tensor([0, 0, 0, 0])
rows_s = torch.tensor([0, 1, 2, 3]); cols_s = torch.tensor([1, 0, 3, 2])
zero_feat = torch.zeros(4, 16)
strong = torch.zeros(4, 16); strong[:2] = 5.0  # 只强化 (0,1) 这对
o_off = attn_eb2(x_s, b_s, edge_index=torch.stack([rows_s, cols_s]), edge_feat=zero_feat)
o_on = attn_eb2(x_s, b_s, edge_index=torch.stack([rows_s, cols_s]), edge_feat=strong)
assert not torch.allclose(o_off, o_on, atol=1e-4), "edge-bias 特征应改变输出 (scatter/bias 通路失效?)"
assert torch.allclose(o_off[2:], o_on[2:], atol=1e-4), "无强偏置的 pair 输出不应被强偏置影响"
print("[ok] TrackSelfAttention edge-bias: 边特征 -> 注意力 bias 生效 (仅作用于被连 pair)")

# ---- 3) 完整版 config 加载 (edge_bias 参数随机初始化, 主干 cpt=38) ----
cfg = yaml.safe_load(open('config_files/train_CERN_v38_attn_full.yaml'))
cfg = adjust_config_training(cfg)
pw = {"LCA": torch.ones(4), "nodes": torch.tensor(1.0),
      "edges": torch.tensor(1.0), "pv_asso": torch.tensor(1.0)}
module = load_module(cfg, pw)
sd = module.state_dict()
eb_keys = [k for k in sd if k.startswith("model._track_attn._edge_bias.")]
assert eb_keys, "edge_bias 参数缺失 (node_attention_edge_bias 未生效?)"
w = sd["model._track_attn._edge_bias.weight"]
assert tuple(w.shape) == (4, 16), f"_edge_bias.weight 形状 {tuple(w.shape)} != (4,16)"
print(f"[ok] edge_bias 参数存在: {len(eb_keys)} 个, weight 形状 {tuple(w.shape)}")
nz = sum(int((sd[k] != 0).sum()) for k in sd if not k.startswith("model._track_attn."))
print(f"[ok] 主干加载: 非注意力参数非零计数={nz}")
print("[ALL PASS]")
