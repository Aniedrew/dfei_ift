"""阶段对比图: 4 个面板 (ε_found / ε_clean / PV 关联 / LCAG class-1)。

数据: report_figs/stage_comparison.csv
注意: 0702 与 0904 两条线之间画断点, 不连线(数据集变更 + 泄漏修复)。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

CSV = "report_figs/stage_comparison.csv"
OUT = "report_figs/stage_comparison.png"

df = pd.read_csv(CSV)
df = df[df["status"] == "ok"].reset_index(drop=True)
# 断点位置: 第一个 0904 版本之前
first0904 = df.index[df["line"] == "0904"]
brk = int(first0904[0]) - 0.5 if len(first0904) else None
x = range(len(df))

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
                     "legend.fontsize": 8.5, "figure.dpi": 200, "savefig.dpi": 200,
                     "savefig.bbox": "tight"})
fig, axes = plt.subplots(2, 2, figsize=(14.5, 8.4))

panels = [
    ("found_pct", "epsilon_found [%]  (all truth tracks grouped; trigger-relevant)", "#1f77b4"),
    ("all_fix_pct", "epsilon_clean [%]  (exact chain, fixed denominator)", "#2ca02c"),
    ("lcag_c1", "LCAG class-1 diagonal [%]  (truth tt-edge class)", "#9467bd"),
]
for ax, (col, title, c) in zip(axes.flat[:3], panels):
    d = df[df[col].notna()]
    ax.plot(list(d.index), d[col], "o-", ms=6, lw=1.6, color=c)
    for i, r in d.iterrows():
        ax.annotate(f"v{int(r['ver'])}", (list(d.index).index(i) if False else i, r[col]),
                    textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7.5)
    if brk is not None:
        ax.axvline(brk, color="#d62728", ls="--", lw=1.3)
        ax.text(brk, ax.get_ylim()[1], " 0702 -> 0904\n (leak fix)", color="#d62728",
                fontsize=8, va="top")
    ax.set_title(title)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"v{int(v)}" for v in df["ver"]], rotation=90, fontsize=8)
    ax.grid(alpha=0.25)

# 面板 4: PV 关联 HGNN vs minIP (错误率, 越低越好)
ax = axes.flat[3]
d = df[df["pv_hgnn"].notna()]
ax.plot(list(d.index), d["pv_hgnn"], "o-", ms=6, lw=1.6, color="#ff7f0e", label="HGNN association")
ax.plot(list(d.index), d["pv_minip"], "s--", ms=5, lw=1.3, color="0.55", label="minIP (traditional)")
if brk is not None:
    ax.axvline(brk, color="#d62728", ls="--", lw=1.3)
ax.set_title("PV association error rate [%]  (lower is better)")
ax.set_xticks(list(x)); ax.set_xticklabels([f"v{int(v)}" for v in df["ver"]], rotation=90, fontsize=8)
ax.legend(); ax.grid(alpha=0.25)

stage_of = {}
for s in df["stage"].unique():
    idx = df.index[df["stage"] == s]
    stage_of[s] = (int(idx[0]), int(idx[-1]))
# 顶部阶段色带
for ax in axes.flat:
    for s, (a, b) in stage_of.items():
        ax.axvspan(a - 0.45, b + 0.45, color="0.9", alpha=0.35, zorder=0) if list(stage_of).index(s) % 2 == 0 else None
fig.suptitle("Stage-by-stage comparison (representative version per stage).  "
             "Red dashed line = dataset switch 0702->0904 + delta_z0 leak fix: do NOT compare across it.",
             fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(OUT)
print("written:", OUT)
