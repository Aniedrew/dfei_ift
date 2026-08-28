"""生成 DFEI 组会报告所需的数据驱动图 (无需跑模型)
输出到 meeting_20260811_figs/
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE = "/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn"
LOG = f"{BASE}/LHCb_logs/DFEI"
OUT = f"{BASE}/meeting_20260811_figs"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})

def load_metrics(v):
    return pd.read_csv(f"{LOG}/version_{v.lstrip('v')}/metrics.csv")

def per_class_acc(m, split="val"):
    # metrics.csv 中 pred_classN 本身是比例 (0-1), 不是计数
    out = {}
    for c in range(4):
        out[c] = m[f"{split}_LCA_class{c}_pred_class{c}"]
    return out

# ============ fig02 训练损失 vs epoch ============
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
style = {"v8": ("#888888", "o"), "v23": ("#1f77b4", "o"), "v25": ("#2ca02c", "s"),
         "v30": ("#ff7f0e", "^"), "v31": ("#d62728", "v")}
label = {"v8": "v8 CERN-old (bug)", "v23": "v23 public (bug)", "v25": "v25 CERN (bug)",
         "v30": "v30 public (weights fixed)", "v31": "v31 CERN (weights fixed, running)"}
for v in ["v8", "v23", "v25"]:
    m = load_metrics(v)
    ep = m[m["val_combined_loss"].notna()]
    col, mk = style[v]
    axes[0].plot(ep["epoch"], ep["val_combined_loss"], color=col, marker=mk,
                 ms=4, lw=1.3, label=label[v])
    bi = ep["val_combined_loss"].idxmin()
    axes[0].scatter([ep.loc[bi, "epoch"]], [ep.loc[bi, "val_combined_loss"]], color=col, s=60,
                    zorder=5, edgecolor="k")
axes[0].set_xlabel("epoch"); axes[0].set_ylabel("val combined loss")
axes[0].set_title("Bug models (weights broken): combined loss")
axes[0].legend(fontsize=8)
for v in ["v30", "v31"]:
    m = load_metrics(v)
    ep = m[m["val_combined_loss"].notna()]
    col, mk = style[v]
    axes[1].plot(ep["epoch"], ep["val_combined_loss"], color=col, marker=mk,
                 ms=4, lw=1.3, label=label[v])
    bi = ep["val_combined_loss"].idxmin()
    axes[1].scatter([ep.loc[bi, "epoch"]], [ep.loc[bi, "val_combined_loss"]], color=col, s=60,
                    zorder=5, edgecolor="k")
axes[1].set_xlabel("epoch"); axes[1].set_ylabel("val combined loss")
axes[1].set_title("Weight-fixed retrains (loss inflated by class weights)")
axes[1].legend(fontsize=8)
fig.suptitle("Training curves: validation combined loss (dot = best epoch)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(f"{OUT}/fig02_training_loss.png", dpi=150)
plt.close(fig)

# ============ fig03 LCA 每类准确率 vs epoch (bug vs fix) ============
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for v, ax in [("v25", axes[0]), ("v31", axes[1])]:
    m = load_metrics(v)
    # 过滤掉 NaN 行（与 fig02 一致）
    valid = m[m["val_combined_loss"].notna()]
    accs = per_class_acc(valid)
    for c in range(4):
        ax.plot(valid["epoch"], accs[c] * 100, lw=1.6,
                label=f"class {c} (LCA={['bkg','sisters','same-B','two-B'][c]})")
    ax.set_xlabel("epoch"); ax.set_ylabel("per-class accuracy (%)")
    ax.set_title(f"{v}: LCA per-class accuracy ({'bug model' if v=='v25' else 'weights fixed'})")
    ax.legend(fontsize=8); ax.set_ylim(-2, 102)
fig.suptitle("LCAG class accuracy: class-1 (sisters) recovers after weight fix", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(f"{OUT}/fig03_lca_class_acc.png", dpi=150)
plt.close(fig)

# ============ fig06 剪枝数据流失 ============
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
thr = [0.2, 0.5, 0.9]
track = [78.9, 65.3, 38.3]
edge = [79.5, 64.8, 32.5]
chain = [68.9, 49.1, 15.1]
x = np.arange(len(thr))
axes[0].bar(x - 0.25, track, 0.25, label="signal tracks survive", color="#1f77b4")
axes[0].bar(x, edge, 0.25, label="true positive edges survive", color="#2ca02c")
axes[0].bar(x + 0.25, chain, 0.25, label="complete chains survive", color="#d62728")
for xi, (t, e, c) in enumerate(zip(track, edge, chain)):
    axes[0].annotate(f"{t:.0f}%", (xi - 0.25, t + 2), ha="center", fontsize=8)
    axes[0].annotate(f"{e:.0f}%", (xi, e + 2), ha="center", fontsize=8)
    axes[0].annotate(f"{c:.0f}%", (xi + 0.25, c + 2), ha="center", fontsize=8)
axes[0].set_xticks(x); axes[0].set_xticklabels([f"thr={t}" for t in thr])
axes[0].set_ylim(0, 110); axes[0].set_ylabel("survival (%)")
axes[0].set_title("v25 CERN: pruning data loss (bug model)")
axes[0].legend(fontsize=9)
# 链死亡分解
cats = ["both\npruned", "edges kept\n(nodes pruned)", "nodes kept\n(edges pruned)"]
vals = [69.5, 73.7, 69.5]
bars = axes[1].bar(cats, vals, color=["#d62728", "#2ca02c", "#1f77b4"], alpha=0.85)
for b, v in zip(bars, vals):
    axes[1].annotate(f"{v:.1f}%", (b.get_x() + b.get_width()/2, v + 1.5), ha="center", fontsize=10)
axes[1].axhline(95, color="k", ls="--", lw=1.2)
axes[1].annotate("E+B2 target ~95%", (0.05, 96), fontsize=9, color="k")
axes[1].set_ylim(0, 105); axes[1].set_ylabel("complete-chain survival (%)")
axes[1].set_title("v25 CERN: chain death decomposition @thr 0.2")
axes[1].tick_params(axis="x", labelsize=8)
fig.suptitle("Why hard threshold pruning loses chains (and what E+B2 recovers)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(f"{OUT}/fig06_chain_survival.png", dpi=150)
plt.close(fig)

# ============ fig07 重建分类对比 ============
fig, ax = plt.subplots(figsize=(10, 5.5))
models = ["v8 CERN-old*", "v23 public (bug)", "v25 CERN (bug)"]
perfect = [18.66, 19.76, 27.68]
allp_only = [35.24-18.66, 55.25-19.76, 55.57-27.68]
noniso = [0.97, 12.56, 6.03]
part = [17.46, 21.65, 15.90]
nf = [100-(35.24+0.97+17.46), 100-(55.25+12.56+21.65), 100-(55.57+6.03+15.90)]
x = np.arange(len(models))
ax.bar(x, perfect, 0.55, label="Perfect", color="#2ca02c")
ax.bar(x, allp_only, 0.55, bottom=perfect, label="AllParticles (structure wrong)", color="#1f77b4")
b2 = [p + a for p, a in zip(perfect, allp_only)]
ax.bar(x, noniso, 0.55, bottom=b2, label="NoneIso (bkg mixed in)", color="#ff7f0e")
b3 = [a + b for a, b in zip(b2, noniso)]
ax.bar(x, part, 0.55, bottom=b3, label="PartReco", color="#9467bd")
b4 = [a + b for a, b in zip(b3, part)]
ax.bar(x, nf, 0.55, bottom=b4, label="NotFound", color="#d62728")
for xi, p in enumerate(perfect):
    ax.annotate(f"{p:.1f}%", (xi, p/2), ha="center", color="w", fontsize=10, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(models)
ax.set_ylabel("fraction of chains (%)"); ax.set_ylim(0, 105)
ax.set_title("Reconstruction quality per chain (thr 0.2; *v8 uses old thr 0.9, not comparable)")
ax.legend(ncol=3, fontsize=8, loc="upper left", bbox_to_anchor=(0, 1.02))
fig.tight_layout()
fig.savefig(f"{OUT}/fig07_reco_categories.png", dpi=150)
plt.close(fig)

# ============ fig08 预期收益 ============
fig, ax = plt.subplots(figsize=(9, 5.5))
scen = ["Current\n(bug, thr 0.2)", "Fix weight bug\n(retrain only)", "Fix bug + E + B2\n(top-k edges, Gumbel nodes)"]
lo = [27.7, 42, 55]
hi = [27.7, 50, 65]
mid = [(a+b)/2 for a, b in zip(lo, hi)]
cols = ["#d62728", "#1f77b4", "#2ca02c"]
for i, (m, l, h, c) in enumerate(zip(mid, lo, hi, cols)):
    ax.bar(i, m, 0.5, color=c, alpha=0.85)
    ax.errorbar(i, m, yerr=[[m-l], [h-m]], fmt="none", ecolor="k", capsize=6, lw=1.5)
    ax.annotate(f"~{m:.0f}%", (i, m + 2), ha="center", fontsize=12, fontweight="bold")
ax.axhline(21.5, color="gray", ls=":", lw=1.5)
ax.annotate("paper WHGNN per-event 21.5% (different metric def. / harder test set)", (0.02, 22.8), fontsize=8, color="gray")
ax.set_xticks(range(3)); ax.set_xticklabels(scen, fontsize=10)
ax.set_ylabel("estimated PerfectReco per chain (%)")
ax.set_title("Estimated impact of planned changes (v25 CERN, per chain)")
ax.set_ylim(0, 75)
fig.tight_layout()
fig.savefig(f"{OUT}/fig08_expected_gain.png", dpi=150)
plt.close(fig)

# ============ fig09 类别准确率: bug vs fix vs paper ============
def best_acc(v):
    m = load_metrics(v)
    ep = m[m["val_combined_loss"].notna()]
    bi = ep["val_combined_loss"].idxmin()
    return {c: ep.loc[bi, f"val_LCA_class{c}_pred_class{c}"] * 100 for c in range(4)}
v25_c = best_acc("v25")
v31_c = best_acc("v31")
paper = {0: 99.3, 1: 75.9, 2: 61.3, 3: 84.0}
fig, ax = plt.subplots(figsize=(9.5, 5))
x = np.arange(4); w = 0.26
ax.bar(x - w, [v25_c[c] for c in range(4)], w, label="v25 CERN bug model (best ep96)", color="#d62728", alpha=0.85)
ax.bar(x, [v31_c[c] for c in range(4)], w, label="v31 CERN weights-fixed (best ep56, running)", color="#2ca02c", alpha=0.85)
ax.bar(x + w, [paper[c] for c in range(4)], w, label="paper WHGNN target", color="#1f77b4", alpha=0.85)
for i, c in enumerate(range(4)):
    ax.annotate(f"{v25_c[c]:.0f}%", (i - w, v25_c[c] + 2), ha="center", fontsize=8)
    ax.annotate(f"{v31_c[c]:.0f}%", (i, v31_c[c] + 2), ha="center", fontsize=8)
    ax.annotate(f"{paper[c]:.0f}%", (i + w, paper[c] + 2), ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(["class 0\nbackground", "class 1\nsisters", "class 2\nsame B", "class 3\ndifferent B"])
ax.set_ylabel("accuracy (%)"); ax.set_ylim(0, 112)
ax.set_title("Weight-bug impact & fix: LCAG per-class accuracy (best-epoch val)")
ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig(f"{OUT}/fig09_class_acc_bug_vs_paper.png", dpi=150)
plt.close(fig)

# ============ fig10 每个epoch耗时 (慢训练证据) ============
def epoch_durations(v):
    ck = sorted(glob.glob(f"{LOG}/version_{v.lstrip('v')}/checkpoints/epoch_epoch*.ckpt"))
    times = sorted(os.path.getmtime(c) for c in ck)
    if len(times) < 3:
        return None
    dt = np.diff(times) / 60.0  # minutes
    return dt
fig, ax = plt.subplots(figsize=(9, 5))
for v, col, lab in [("v25", "#2ca02c", "v25 CERN (bug)"), ("v31", "#d62728", "v31 CERN (weights fixed)"),
                    ("v30", "#ff7f0e", "v30 public (weights fixed)")]:
    dt = epoch_durations(v)
    if dt is None:
        continue
    ax.plot(range(1, len(dt) + 1), dt, lw=1.4, color=col, label=lab)
    ax.axhline(dt.mean(), color=col, ls="--", lw=0.9, alpha=0.6)
    ax.annotate(f"mean {dt.mean():.0f} min", (1, dt.mean()), fontsize=8, color=col, xytext=(3, 0),
                textcoords="offset points")
ax.set_xlabel("epoch index"); ax.set_ylabel("wall time per epoch (min)")
ax.set_title("Slow training: wall-clock per epoch (from checkpoint timestamps)")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(f"{OUT}/fig10_epoch_time.png", dpi=150)
plt.close(fig)

# ============ fig11 CERN vs 公开数据对比 ============
import torch, io, zstandard as zstd
def load_sample_events(data_dir, sample, n_files=3, max_evt=120):
    files = sorted(glob.glob(f"{data_dir}/{sample}/tst_data_*"))[:n_files]
    dctx = zstd.ZstdDecompressor()
    evts = []
    for f in files:
        with open(f, "rb") as fh:
            with dctx.stream_reader(fh) as r:
                data = torch.load(io.BytesIO(r.read()), weights_only=False)
        for e in data:
            evts.append(e)
            if len(evts) >= max_evt:
                return evts
    return evts

def evt_stats(evt):
    et = ("tracks", "to", "tracks")
    n_trk = evt["tracks"].x.shape[0]
    y = evt[et].y
    if y.dim() == 2 and y.shape[-1] > 1:
        y = y.argmax(dim=-1)
    y = y.numpy().flatten()
    n_edges = len(y)
    cls = np.bincount(y, minlength=4)[:4] / max(n_edges, 1) * 100
    return n_trk, n_edges, cls

cern = load_sample_events("/lzufs/user/guoqingxiang/DFEI_IFT_20260702/data/MC_normed", "inclusive_00342442")
pub = load_sample_events("/lzufs/user/guoqingxiang/DFEI_data/converted_LHCbcollision", "00342442_inclusive")
def agg(evts):
    trk = np.array([evt_stats(e)[0] for e in evts])
    cls = np.stack([evt_stats(e)[2] for e in evts]).mean(axis=0)
    return trk, cls
trk_c, cls_c = agg(cern)
trk_p, cls_p = agg(pub)
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
axes[0].hist(trk_c, bins=30, alpha=0.6, label=f"CERN (mean {trk_c.mean():.0f})", color="#1f77b4")
axes[0].hist(trk_p, bins=30, alpha=0.6, label=f"public (mean {trk_p.mean():.0f})", color="#2ca02c")
axes[0].set_xlabel("tracks per event"); axes[0].set_ylabel("count")
axes[0].set_title("Event complexity: CERN vs public data"); axes[0].legend()
x = np.arange(4)
# 右图：小类别 (<1%) 用对数刻度以便可见
axes[1].bar(x - 0.2, np.clip(cls_c, 0.01, None), 0.4, label="CERN", color="#1f77b4")
axes[1].bar(x + 0.2, np.clip(cls_p, 0.01, None), 0.4, label="public", color="#2ca02c")
for i in range(4):
    axes[1].annotate(f"{cls_c[i]:.1f}%", (i - 0.2, cls_c[i] * 1.5), ha="center", fontsize=7, color="#1f77b4")
    axes[1].annotate(f"{cls_p[i]:.1f}%", (i + 0.2, cls_p[i] * 1.5), ha="center", fontsize=7, color="#2ca02c")
axes[1].set_xticks(x); axes[1].set_xticklabels(["class 0", "class 1", "class 2", "class 3"])
axes[1].set_ylabel("edge class fraction (%)")
axes[1].set_yscale("log")
axes[1].set_ylim(0.005, 200)
axes[1].set_title("LCAG class distribution on edges (all pairs, log scale)")
axes[1].legend()
fig.suptitle("Data comparison: CERN official MC vs public dataset", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(f"{OUT}/fig11_data_compare.png", dpi=150)
plt.close(fig)

# ============ fig12 修改方案示意图 (无重叠版) ============
fig, ax = plt.subplots(figsize=(12, 6.2))
ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")
def box(x, y, w, h, text, fc="#eaf2fb", ec="#1f77b4", fs=10.5, bold=False):
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                 fc=fc, ec=ec, lw=1.5))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal")
def arrow(x1, y1, x2, y2):
    ax.annotate("", (x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color="gray", lw=1.6))

# 主流水线 (一行, 无重叠)
box(0.2, 3.8, 2.0, 1.4, "Event graph\n~150 tracks\n~23k edges", fc="#f5f5f5", ec="#666666")
box(2.6, 3.8, 2.0, 1.4, "HGNN\nmulti-task\n(LCA + heads)", fc="#eaf2fb", ec="#1f77b4")
box(5.0, 3.8, 2.0, 1.4, "node pruning\nsmooth mask\n(in training)", fc="#fdf3e4", ec="#ff7f0e", fs=9.5)
box(7.4, 3.8, 2.0, 1.4, "edge candidates\ntop-k per track", fc="#fdf3e4", ec="#ff7f0e", fs=9.5)
box(9.8, 3.8, 2.0, 1.4, "decay-tree\nreconstruction", fc="#eaf2fb", ec="#1f77b4", fs=10)
arrow(2.2, 4.5, 2.6, 4.5)
arrow(4.6, 4.5, 5.0, 4.5)
arrow(7.0, 4.5, 7.4, 4.5)
arrow(9.4, 4.5, 9.8, 4.5)

# 下行: 选择MLP
box(0.2, 2.2, 2.6, 1.2, "several candidate\ndecay trees", fc="#fff8dc", ec="#d62728", fs=9.5)
box(3.6, 2.2, 4.0, 1.2, "NEW: MLP scores each candidate\n& selects the most plausible", fc="#ffe4e1", ec="#d62728", fs=9.5, bold=True)
box(8.6, 2.2, 2.2, 1.2, "final\ndecay tree", fc="#e6ffe6", ec="#2ca02c", fs=10)
# reconstruction -> candidate trees: 去掉箭头，底部文字已说明关系
# (greedy reconstruction can return several trees)
arrow(2.8, 2.8, 3.6, 2.8)       # candidates -> MLP
arrow(7.6, 2.8, 8.6, 2.8)       # MLP -> final tree

# 底部说明
ax.text(6, 1.1, "Node: smooth mask (Gumbel-style) lets the model experience pruning during training — vectorized, GPU-parallel.\n"
               "Edge: top-k per track (<= N*k candidates, hard compute cap) instead of a global threshold.\n"
               "Selection: the greedy reconstruction can return several trees — a small MLP ranks them and picks the best.",
        ha="center", va="center", fontsize=10, color="#333333",
        bbox=dict(boxstyle="round", fc="#fafafa", ec="#bbbbbb"))
ax.set_title("Proposed changes: node soft-mask training, edge top-k, + selection MLP", fontsize=13)
fig.tight_layout()
fig.savefig(f"{OUT}/fig12_eb2_schematic.png", dpi=150)
plt.close(fig)

print("done ->", OUT)
print(sorted(os.path.basename(f) for f in glob.glob(f"{OUT}/*.png")))
