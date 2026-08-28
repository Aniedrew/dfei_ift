#!/usr/bin/env python3
"""本地验证 _split_by_pv: 用真实事件拼 batch, 检查拆分后结构正确"""
import sys, io
import zstandard as zstd
import torch
from torch_geometric.data import Batch

sys.path.insert(0, "/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn")
from wmpgnn.lightning_module.dfei_lightning_module import DFEILightningModule

DATA_FILE = "/lzufs/user/guoqingxiang/DFEI_IFT_20260702/data/MC_normed/inclusive_00342442/tst_data_00231000_00231999.pt.zst"
IDXS = [747, 748, 749]

dctx = zstd.ZstdDecompressor()
with open(DATA_FILE, "rb") as f:
    with dctx.stream_reader(f) as r:
        data = torch.load(io.BytesIO(r.read()), weights_only=False)

evts = [data[i] for i in IDXS]
b = Batch.from_data_list(evts)

print(f"原 batch: {len(evts)} 事件, tracks={b['tracks'].x.shape[0]}, pvs={b['pvs'].x.shape[0]}, "
      f"tt 边={b[('tracks','to','tracks')].edge_index.shape[1]}")

# 绑定 _split_by_pv 到轻量对象
class T:
    configs = {'pv_cluster': True}
m = T()
m._split_by_pv = DFEILightningModule._split_by_pv.__get__(m, T)
m._build_pv_subgraph = DFEILightningModule._build_pv_subgraph.__get__(m, T)

nb = m._split_by_pv(b)
print(f"拆分后 batch: {nb['tracks'].batch.max().item()+1} 个子图, "
      f"tracks={nb['tracks'].x.shape[0]}, pvs={nb['pvs'].x.shape[0]}, "
      f"tt 边={nb[('tracks','to','tracks')].edge_index.shape[1]}")

# 验证: 子图数 = 每事件 (PV数 + 无PV簇)
tb = nb['tracks'].batch
pb = nb['pvs'].batch
print(f"子图 tracks 分布: {torch.bincount(tb).tolist()}")
print(f"子图 pvs 分布: {torch.bincount(pb).tolist()}")

# 验证: 每个子图至少 1 track; 簇内 tt 边两端同子图 (batch 属性一致)
ei = nb[('tracks','to','tracks')].edge_index
same = (tb[ei[0]] == tb[ei[1]]).float().mean().item()
print(f"tt 边两端同子图比例: {same:.3f}")

# 验证: 拆分前后 truth 链完整性 (同链同簇): 检查 y>0 的 tt 边两端同簇
y = nb[('tracks','to','tracks')].y
pos = y > 0
same_pos = (tb[ei[0][pos]] == tb[ei[1][pos]]).float().mean().item()
print(f"y>0 (truth 链内边) 两端同簇比例: {same_pos:.3f}")

# 验证 pv_asso 边保留
trpv = nb[('tracks','to','pvs')]
print(f"拆分后 tr-pv 边: {trpv.edge_index.shape[1]}, y 分布: {torch.bincount(trpv.y.long(), minlength=2).tolist()}")
