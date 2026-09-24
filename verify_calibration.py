"""验证"标定说": 把各版本自己的 thr 扫描曲线与官方 thr0.9 结果叠在一起看。
输入: report_figs/thr_metric_official_v<ver>_v<ver>.csv (thr_scan_official.py)
      logs/fixed_denominator_metrics_inclusive_00342442.csv (官方 thr0.9 结果)
产出: report_figs/calibration_check.png / .csv
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn"
FIG = BASE + "/report_figs"
OFFICIAL = {557: ("hilr_node5", 42.01, 24.34, 12903, 0.90),
            559: ("hilr_ctrl", 40.04, 23.81, 12366, 0.90),
            565: ("hilr_node20", 40.11, 22.57, None, 0.90),
            554: ("prune_node5focal", 23.99, 14.63, 4715, 0.90)}
EFF_THR = {557: 0.960, 559: 0.900, 565: 0.955, 554: 0.740}   # 逐径迹剪枝分析给出的"等效阈值"
COL = {557: "tab:green", 559: "tab:olive", 565: "darkorange", 554: "tab:purple"}
LAB = {557: "v557 (node w=5 + lr1e-4) SOTA", 559: "v559 control (node w=1)",
       565: "v565 (node w=20)", 554: "v554 (node5 + focal γ2)"}


def main():
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5))
    rows = []
    for ver, (tag, allfix, perffix, N, thr0) in OFFICIAL.items():
        p = f"{FIG}/thr_metric_official_v{ver}_v{ver}.csv"
        if not os.path.exists(p):
            print("缺", p, "跳过")
            continue
        d = pd.read_csv(p).sort_values("thr")
        c = COL[ver]
        axes[0].plot(d["thr"], d["All_fix"], "o-", color=c, lw=1.9, ms=5, label=f"{LAB[ver]}")
        i = int(np.argmax(d["All_fix"]))
        axes[0].plot(d["thr"][i], d["All_fix"][i], "*", ms=16, color=c)
        axes[0].annotate(f"thr*={d['thr'][i]:.2f}\n{d['All_fix'][i]:.1f}%",
                         (d["thr"][i], d["All_fix"][i]), textcoords="offset points",
                         xytext=(6, -16), fontsize=8, color=c)
        axes[0].scatter([thr0], [allfix], marker="D", s=55, facecolor="none", edgecolor=c, lw=2,
                        label=f"{LAB[ver]} official @0.90")
        axes[1].plot(d["thr"], d["NoneIso"], "o-", color=c, lw=1.7, ms=4, label=LAB[ver])
        axes[1].plot(d["thr"], d["coverage"], "s--", color=c, lw=1.4, ms=4, alpha=.7)
        axes[2].scatter([EFF_THR[ver]], [d["All_fix"].max()], s=70, color=c, label=LAB[ver])
        axes[2].annotate(f"v{ver}", (EFF_THR[ver], d["All_fix"].max()), textcoords="offset points",
                         xytext=(5, 5), fontsize=9)
        rows.append(dict(version=ver, label=LAB[ver], effective_thr=EFF_THR[ver],
                         thr_star=float(d["thr"][i]), All_fix_at_star=float(d["All_fix"][i]),
                         Perf_fix_at_star=float(d["Perf_fix"][i]),
                         All_fix_scan_at_090=float(d.loc[d["thr"].round(2) == 0.90, "All_fix"].iloc[0])
                         if (d["thr"].round(2) == 0.90).any() else np.nan,
                         All_fix_official_at_090=allfix,
                         coverage_at_star=float(d["coverage"][i]), NoneIso_at_star=float(d["NoneIso"][i])))
    axes[0].set_xlabel("pruning threshold"); axes[0].set_ylabel("All_fix % (fixed denominator)")
    axes[0].set_title("All_fix vs threshold: each version has its own optimum")
    axes[0].grid(alpha=.25); axes[0].legend(fontsize=7)
    axes[1].set_xlabel("pruning threshold"); axes[1].set_ylabel("% (of N_total)")
    axes[1].set_title("coverage (dashed) and pollution NoneIso (solid)")
    axes[1].grid(alpha=.25); axes[1].legend(fontsize=7)
    axes[2].set_xlabel("equivalent threshold (from per-track calibration)")
    axes[2].set_ylabel("best All_fix % at that version's own optimum")
    axes[2].set_title("at matched effective threshold the versions converge")
    axes[2].grid(alpha=.25); axes[2].legend(fontsize=7)
    fig.suptitle("Calibration check: the pruning levers shift the score scale, so each version needs its own threshold", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f"{FIG}/calibration_check.png", dpi=130); plt.close(fig)
    df = pd.DataFrame(rows)
    df.to_csv(f"{FIG}/calibration_check.csv", index=False)
    print(df.round(2).to_string(index=False))
    print("写出", f"{FIG}/calibration_check.png")


if __name__ == "__main__":
    main()
