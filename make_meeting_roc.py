"""生成剪枝 ROC (edge + node) 对比 CERN(v25) vs 公开(v23)
此任务受 weight bug 影响较小 (二分类, 负样本主导), 反映数据差异
CPU 推理采样, 输出到 meeting_20260811_figs/
"""
import os
import sys
import glob
import io
import numpy as np
import torch
import zstandard as zstd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

BASE = "/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn"
sys.path.insert(0, BASE)
OUT = f"{BASE}/meeting_20260811_figs"
os.makedirs(OUT, exist_ok=True)

import analyze_reco_visual as A

N_EVENTS = int(os.environ.get("N_EVENTS", "250"))

def tt_binary(y):
    y = torch.as_tensor(y)
    return (y.argmax(dim=-1) if y.dim() == 2 and y.shape[-1] > 1 else y) > 0

def scan(version, data_dir, sample):
    A.VERSION = version
    A.DATA_DIR = data_dir
    A.SAMPLE = sample
    import yaml
    with open(f"{BASE}/LHCb_logs/DFEI/version_{version}/hparams.yaml") as fh:
        use_pid = yaml.safe_load(fh).get("DFEI", {}).get("use_pid", "None")
    module, _ = A.load_module(None)
    module.eval()
    files = sorted(glob.glob(f"{data_dir}/{sample}/tst_data_*"))
    dctx = zstd.ZstdDecompressor()
    ew, et, nw, nt = [], [], [], []
    n = 0
    for f in files:
        with open(f, "rb") as fh:
            with dctx.stream_reader(fh) as r:
                data = torch.load(io.BytesIO(r.read()), weights_only=False)
        for evt in data:
            n += 1
            if n > N_EVENTS:
                break
            etr = ("tracks", "to", "tracks")
            st = evt[etr]
            st.edge_index = torch.cat([st.edge_index, st.edge_index.flip(0)], dim=1)
            st.edges = st.edges.repeat(2, 1)
            st.y = st.y.repeat(2)
            outputs, batch = A.predict(module, evt, use_pid=use_pid)
            e_w = outputs["edge_weights"].cpu().numpy()
            n_w = outputs["node_weights"].cpu().numpy()
            ew.append(e_w); et.append(tt_binary(st.y).cpu().numpy())
            # 节点 truth
            if "ft" in evt["tracks"]:
                n_true = (evt["tracks"].ft != 1).cpu().numpy().astype(np.float64)
            else:
                part_keys = evt["tracks"].part_keys.numpy()
                sig = set(evt["tracks"].sig_keys.numpy().tolist())
                n_true = np.isin(part_keys, list(sig)).astype(np.float64)
            nw.append(n_w); nt.append(n_true)
        if n > N_EVENTS:
            break
    ew = np.concatenate(ew); et = np.concatenate(et)
    nw = np.concatenate(nw); nt = np.concatenate(nt)
    print(f"  v{version}: {n} events, edges={len(ew)} (pos {et.sum()}), nodes={len(nw)} (pos {nt.sum()})", flush=True)
    return ew, et, nw, nt

def plot_roc(ax, score, y_true, label, color, ls="-"):
    fpr, tpr, _ = roc_curve(y_true, score)
    a = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, ls=ls, lw=1.8, label=f"{label} (AUC={a:.3f})")
    return a

print("Scanning v25 CERN...", flush=True)
cern = scan(25, "/lzufs/user/guoqingxiang/DFEI_IFT_20260702/data/MC_normed", "inclusive_00342442")
print("Scanning v23 public...", flush=True)
pub = scan(23, "/lzufs/user/guoqingxiang/DFEI_data/converted_LHCbcollision", "00342442_inclusive")

fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
axes[0].plot([0, 1], [0, 1], ls=":", color="gray", lw=1)
a1 = plot_roc(axes[0], cern[0], cern[1], "CERN (v25)", "#1f77b4", "-")
a2 = plot_roc(axes[0], pub[0], pub[1], "public (v23)", "#2ca02c", "--")
axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
axes[0].set_title("Edge pruning", fontsize=12, fontweight="bold")
axes[0].legend(fontsize=9)
axes[1].plot([0, 1], [0, 1], ls=":", color="gray", lw=1)
b1 = plot_roc(axes[1], cern[2], cern[3], "CERN (v25)", "#1f77b4", "-")
b2 = plot_roc(axes[1], pub[2], pub[3], "public (v23)", "#2ca02c", "--")
axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR")
axes[1].set_title("Node pruning", fontsize=12, fontweight="bold")
axes[1].legend(fontsize=9)
fig.suptitle(f"Pruning heads ROC: CERN vs public data (robust to weight bug; N≈{N_EVENTS} events/model)", fontsize=13, y=1.01)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(f"{OUT}/fig04_05_pruning_roc.png", dpi=150)
plt.close(fig)
print("AUC edges: CERN=%.3f public=%.3f | nodes: CERN=%.3f public=%.3f" % (a1, a2, b1, b2))
print("done ROC")
