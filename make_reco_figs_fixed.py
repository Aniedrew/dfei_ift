"""旧 benchmark（all_particles / perfect_reco）演变图 —— 固定分母版本。

只用**非泄漏**数据（0904 线, sample 00342451）出结论图；0702 线（含 delta_z0 泄漏）不进入结论。
同时把"原始可变分母口径"以浅色虚线叠在同一张图上，直观显示口径差异有多大。

输入: report_figs/unconditional_reco_00342451.csv  (由 recompute_unconditional_reco.py 生成)
输出: report_figs/reco_fixed/F1_fixed_denominator.png, F2_by_nB.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

CSV = "report_figs/unconditional_reco_00342451.csv"
OUT = "report_figs/reco_fixed"

# 口径断点（仅作标注；0904 线内部）
BREAKS = {
    632: ("thr 0.9->0.95", "0.45", (0, (1, 2)), 1.0),
    642: ("eval seed fixed", "#ff7f0e", ":", 1.4),
    643: ("local DOCA fix", "#2ca02c", ":", 1.4),
}


def load():
    d = pd.read_csv(CSV)
    d = d[~d["tag"].str.contains("ckpt-scan|smoke|paper|cpubench|cpub_", na=False)]
    # 每版本取"代表性评估": 优先 tag 以 v<版本>_ 开头者, 否则取评估内真值链最多者
    keep = []
    for v, g in d.groupby("version"):
        m = g[g["tag"].str.match(rf"v{v}_")]
        keep.append((m.iloc[0] if len(m) else g.sort_values("rows(=评估内真值链)").iloc[-1]))
    d = pd.DataFrame(keep).sort_values("version").reset_index(drop=True)
    return d


def deco(ax, d):
    for v, (lab, col, ls, lw) in BREAKS.items():
        m = d.index[d["version"] == v]
        if len(m):
            ax.axvline(int(m[0]), color=col, ls=ls, lw=lw, alpha=0.85, label=lab)
    n = len(d)
    step = max(1, n // 12)
    t = list(range(0, n, step))
    ax.set_xticks(t)
    ax.set_xticklabels([f"v{int(d['version'].iloc[i])}" for i in t], rotation=90, fontsize=8.5)
    ax.grid(alpha=0.25)
    ax.set_xlabel("version (chronological)")


def main():
    os.makedirs(OUT, exist_ok=True)
    d = load()
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
                         "legend.fontsize": 9, "figure.dpi": 200, "savefig.dpi": 200,
                         "savefig.bbox": "tight"})

    # ---- F1: 固定分母 vs 原始(可变分母) ----
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4))
    for ax, (fixc, origc, name) in zip(axes, [("All_fix%", "All_orig%", "all_particles"),
                                              ("Perf_fix%", "Perf_orig%", "perfect_reco")]):
        ax.plot(d.index, d[origc], "o--", ms=3, lw=0.9, color="0.65",
                label=f"{name}: original protocol (variable denominator, inflated)")
        ax.plot(d.index, d[fixc], "o-", ms=4.5, lw=1.6, color="#1f77b4",
                label=f"{name}: fixed denominator N_total (unconditional)")
        i = d[fixc].idxmax()
        ax.annotate(f"best v{int(d['version'][i])}\n{d[fixc][i]:.1f}%",
                    (i, d[fixc][i]), textcoords="offset points", xytext=(6, 6), fontsize=9)
        deco(ax, d)
        ax.set_ylabel(f"{name} [%]")
        ax.legend(loc="upper left")
    axes[0].set_title("F1  Old benchmarks, 0904 (leak-free) line, 3000-event style eval\n"
                      "solid = fixed denominator N_total (unconditional);  dashed = original variable denominator")
    fig.tight_layout()
    fig.savefig(f"{OUT}/F1_fixed_denominator.png")
    plt.close(fig)

    # ---- F2: 固定分母, 按 n_B 分层 ----
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    for col, lab, c in [("All_fix%_1B", "1 B in event", "#1f77b4"),
                        ("All_fix%_2B", "2 B in event", "#d62728"),
                        ("All_fix%_>2B", ">2 B in event", "#2ca02c")]:
        if col in d:
            ax.plot(d.index, d[col], "o-", ms=4, lw=1.5, color=c, label=lab)
    deco(ax, d)
    ax.set_ylabel("all_particles [%]  (fixed denominator)")
    ax.set_title("F2  Old benchmark by event B-multiplicity, fixed denominator, 0904 (leak-free)\n"
                 "note: >2B stratum is tiny (very few truth chains) -> large statistical error")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(f"{OUT}/F2_by_nB.png")
    plt.close(fig)

    # ---- F3: 真值径迹存活率 vs 无条件整链恢复率（揭示"保留越多反而越差"） ----
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    ax.plot(d.index, d["All_fix%"], "o-", ms=4.5, lw=1.6, color="#1f77b4",
            label="all_particles, fixed denominator (unconditional)")
    ax.set_ylabel("all_particles [%]  (fixed denominator)")
    ax2 = ax.twinx()
    ax2.plot(d.index, d["track_surv%"], "s--", ms=4, lw=1.3, color="#d62728",
             label="truth-track survival after pruning [%]")
    ax2.set_ylabel("truth-track survival [%]", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    deco(ax, d)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left")
    ax.set_title("F3  Keep more truth content does NOT mean recover more chains\n"
                 "0904 (leak-free); e.g. the chain-loss versions keep the most tracks but recover the fewest chains")
    fig.tight_layout()
    fig.savefig(f"{OUT}/F3_survival_vs_recovery.png")
    plt.close(fig)
    print("已写出:", f"{OUT}/F1_fixed_denominator.png", f"{OUT}/F2_by_nB.png",
          f"{OUT}/F3_survival_vs_recovery.png")


if __name__ == "__main__":
    main()
