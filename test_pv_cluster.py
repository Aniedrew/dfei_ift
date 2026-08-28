#!/usr/bin/env python3
"""本地复现 pv_cluster IndexError: 用真实事件 + 伪 lca 调 _reconstruct_pv_clustered"""
import sys, os, io, traceback
import zstandard as zstd
import torch
import yaml

sys.path.insert(0, "/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn")

DATA_FILE = "/lzufs/user/guoqingxiang/DFEI_IFT_20260702/data/MC_normed/inclusive_00342442/tst_data_00231000_00231999.pt.zst"
EVT_IDX = 747

dctx = zstd.ZstdDecompressor()
with open(DATA_FILE, "rb") as f:
    with dctx.stream_reader(f) as r:
        data = torch.load(io.BytesIO(r.read()), weights_only=False)
evt = data[EVT_IDX]

# 伪 lca: 用 y 的 one-hot
tt = evt[("tracks", "to", "tracks")]
yy = tt.y
if yy.dim() > 1:
    yy = yy.argmax(dim=-1)
lca = torch.nn.functional.one_hot(yy.clamp(0, 3).long(), num_classes=4).float()
tt.lca = lca

n = evt["tracks"].x.shape[0]
trpv = evt[("tracks", "to", "pvs")]
ei, y = trpv.edge_index, trpv.y
npvs = evt["pvs"].x.shape[0]
y_pv = torch.full((n,), -1, dtype=torch.long)
for t, p, lab in zip(ei[0].tolist(), ei[1].tolist(), y.tolist()):
    if lab == 1:
        y_pv[t] = p
pred_pv = torch.zeros(n, npvs)
pred_pv[ei[0], ei[1]] = (y > 0).float()

from wmpgnn.reconstruction.reconstruction import EventReconstruction
cfg = yaml.safe_load(open("/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn/config_files/eval_CERN_v38_pvcluster_smoke.yaml"))
rec = EventReconstruction(cfg)

print(f"事件: {n} tracks, {npvs} PV, y_pv 分布: {torch.unique(y_pv, return_counts=True)}")
for assign in ["true", "pred"]:
    pv_des = {"true": y_pv, "pred": pred_pv, "ip": pred_pv, "npvs": npvs}
    rec.configs["pv_cluster_assign"] = assign
    try:
        rc, nclust = rec._reconstruct_pv_clustered(evt, pv_des)
        print(f"[{assign}] OK: 链数={len(rc)}, 各簇链统计={nclust}")
    except Exception:
        print(f"[{assign}] 崩溃:")
        traceback.print_exc()
