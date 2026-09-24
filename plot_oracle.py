"""把 oracle 分解画成图: 各组干预的 All_fix / 覆盖率, 以及"剪枝天花板 vs 解码损失"的瀑布图。
输入: report_figs/oracle_decompose_<tag>.csv
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = "/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn/report_figs"
TAG = sys.argv[1] if len(sys.argv) > 1 else "v601_0904"
OUTNAME = sys.argv[2] if len(sys.argv) > 2 else "oracle_decompose_v601_0904.png"

d = pd.read_csv(f"{FIG}/oracle_decompose_{TAG}.csv")
short = {"baseline": "baseline\n(model)", "node_add_tp": "node\n+recall", "node_remove_fp": "node\n+precision",
         "node_both": "node\nperfect", "edge_add_tp": "edge\n+recall", "edge_remove_fp": "edge\n+precision",
         "edge_both": "edge\nperfect", "both_both": "node+edge\nperfect", "node_drop_tp": "reverse ctrl\n(drop true)"}
x = np.arange(len(d))
fig, axes = plt.subplots(1, 2, figsize=(16.5, 6))

ax = axes[0]
ax.bar(x, d["All_fix"], 0.6, color=["tab:grey" if a == "baseline" else ("tab:red" if a == "node_drop_tp" else "tab:blue")
                                    for a in d["arm"]])
ax.plot(x, d["coverage"], "o--", color="tab:orange", lw=1.6, ms=6, label="coverage (of N_total)")
for i, (v, c) in enumerate(zip(d["All_fix"], d["coverage"])):
    ax.text(i, v + 1.0, f"{v:.1f}", ha="center", fontsize=8.5)
ax.set_xticks(x); ax.set_xticklabels([short.get(a, a) for a in d["arm"]], fontsize=9)
ax.set_ylabel("All_fix %  (fixed denominator)"); ax.set_ylim(0, max(d["coverage"]) * 1.15)
ax.set_title("oracle interventions: All_fix and coverage"); ax.grid(alpha=.25, axis="y")
ax.legend(fontsize=9)

ax = axes[1]
base = float(d.loc[d["arm"] == "baseline", "All_fix"].iloc[0])
labels = ["baseline"]; vals = [base]
for arm, lab in [("node_add_tp", "+ node recall"), ("node_remove_fp", "+ node precision"),
                 ("edge_both", "+ edge (both)"), ("both_both", "= pruning ceiling")]:
    if (d["arm"] == arm).any():
        labels.append(lab); vals.append(float(d.loc[d["arm"] == arm, "All_fix"].iloc[0]))
cov_both = float(d.loc[d["arm"] == "both_both", "coverage"].iloc[0]) if (d["arm"] == "both_both").any() else np.nan
xs = np.arange(len(labels))
ax.bar(xs, vals, 0.55, color=["tab:grey"] + ["tab:blue"] * (len(vals) - 1))
for i, v in enumerate(vals):
    ax.text(i, v + 0.8, f"{v:.1f}", ha="center", fontsize=9)
if np.isfinite(cov_both):
    ax.bar([xs[-1] + 0.42], [cov_both - vals[-1]], 0.32, bottom=vals[-1], color="tab:orange", alpha=.75,
           label="decode loss (coverage - All at pruning ceiling)")
    ax.text(xs[-1] + 0.42, cov_both + 0.8, f"{cov_both:.1f}", ha="center", fontsize=9)
ax.set_xticks(list(xs) + ([xs[-1] + 0.42] if np.isfinite(cov_both) else []))
ax.set_xticklabels(labels + (["decode\nloss"] if np.isfinite(cov_both) else []), fontsize=9)
ax.set_ylabel("All_fix %"); ax.set_ylim(0, max(100, cov_both * 1.1))
ax.set_title("headroom attribution: how far can fixes to pruning take us?")
ax.grid(alpha=.25, axis="y"); ax.legend(fontsize=9)

fig.suptitle(f"Oracle decomposition on the fixed production ({TAG})", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(f"{FIG}/{OUTNAME}", dpi=130)
print("写出", f"{FIG}/{OUTNAME}")
print(d.to_string(index=False))
