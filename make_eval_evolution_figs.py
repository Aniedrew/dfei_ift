"""画"各 benchmark 随版本演变"的 7 张多面板图 (matplotlib Agg, DPI>=200, 英文标签).

输入: report_figs/eval_matrix_all_versions.csv (由 build_eval_matrix.py 生成)
输出: report_figs/eval_evolution/E1..E7_*.png

口径与数据说明全部写在图内 (轴标签 / 图例 / 标题); 缺口留空不连线; 三个断点画竖线.
用法: export PATH=$HOME/miniconda3/envs/dfei/bin:$PATH; python make_eval_evolution_figs.py
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.abspath(__file__))
MATRIX = os.path.join(REPO, "report_figs", "eval_matrix_all_versions.csv")
E2E = os.path.join(REPO, "report_figs", "seqrec_offline_v654_thr0.9_all.csv")
OUTDIR = os.path.join(REPO, "report_figs", "eval_evolution")
os.makedirs(OUTDIR, exist_ok=True)

plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
                     "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
                     "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight"})

C_B1 = "#d62728"   # dataset switch
C_B2 = "#ff7f0e"   # eval seed fix
C_B3 = "#2ca02c"   # DOCA geom fix
BP_COLORS = {"B1": C_B1, "B2": C_B2, "B3": C_B3, "T": "0.45"}
BP_LABELS = {
    "B1": "B1 dataset switch 0702->0904 (delta_z0 leak fix)",
    "B2": "B2 eval event-sampling seed fix",
    "B3": "B3 local corrected DOCA geometry",
    "T": "sub-switch: training inference thr 0.9 -> 0.95 (at v632)",
}


def load():
    df = pd.read_csv(MATRIX)
    df = df.sort_values(["completion_date", "version"]).reset_index(drop=True).copy()
    df["idx"] = df.index
    return df


def bp_positions(df):
    pos = {}
    # B1: 第一个 0904 版本
    m = df.index[df["line"] == "0904 (post-leakfix)"]
    if len(m):
        pos["B1"] = int(m[0])
    for name, ver in (("B2", 642), ("B3", 643), ("T", 632)):
        m = df.index[df["version"] == ver]
        if len(m):
            pos[name] = int(m[0])
    return pos


def add_breakpoints(ax, df, pos, label=True):
    for name, x in pos.items():
        ls = {"B1": "--", "B2": ":", "B3": ":", "T": (0, (1, 2))}.get(name, ":")
        lw = {"B1": 1.7, "B2": 1.3, "B3": 1.3, "T": 1.1}.get(name, 1.3)
        ax.axvline(x, color=BP_COLORS[name], ls=ls, lw=lw, alpha=0.8,
                   label=BP_LABELS[name] if label else None)


def deco_xaxis(ax, df):
    """x 轴: 版本序号; 只标注一部分版本号."""
    n = len(df)
    step = max(1, n // 12)
    ticks = list(range(0, n, step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"v{int(df['version'].iloc[i])}" for i in ticks],
                       rotation=90, fontsize=8.5)


def yerr(df, vcol, ecol):
    v = df[vcol]
    e = df[ecol]
    return v, e


# ---------------------------------------------------------------------------
def E1(df, pos):
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    metrics = [("all_particles", "all_particles_err", "all_particles [%]"),
               ("perfect_reco", "perfect_reco_err", "perfect_reco [%]")]
    for ax, (vcol, ecol, ylab) in zip(axes, metrics):
        d = df[df[vcol].notna()]
        ax.errorbar(d["idx"], d[vcol], yerr=d[ecol].fillna(0), fmt="o-",
                    ms=3.5, lw=1.0, capsize=2, color="#1f77b4")
        add_breakpoints(ax, df, pos, label=(ax is axes[0]))
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.25)
    axes[0].set_title("E1  Old (paper-protocol) benchmarks vs version - does the paper metric 'improve'?\n"
                      "vertical lines: B1/B2/B3 protocol breakpoints + training-thr sub-switch")
    axes[1].set_xlabel("training version (chronological by completion date); labels = version number")
    deco_xaxis(axes[1], df)
    axes[0].legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "E1_old_benchmarks.png")); plt.close(fig)


def E2(df, pos):
    tiers = [("1B", "only 1 B / event"), ("2B", "only 2 B / event"),
             ("gt2B", "more than 2 B / event")]
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    for ax, (t, tname) in zip(axes, tiers):
        va, ea = f"all_particles_{t}", f"all_particles_err_{t}"
        vp, ep = f"perfect_reco_{t}", f"perfect_reco_err_{t}"
        d = df[df[va].notna()]
        ax.errorbar(d["idx"], d[va], yerr=d[ea].fillna(0), fmt="o-", ms=3.2, lw=0.9,
                    capsize=2, color="#1f77b4", label="all_particles")
        d2 = df[df[vp].notna()]
        ax.errorbar(d2["idx"], d2[vp], yerr=d2[ep].fillna(0), fmt="s-", ms=3.0, lw=0.9,
                    capsize=2, color="#ff7f0e", label="perfect_reco")
        add_breakpoints(ax, df, pos, label=(ax is axes[0]))
        # 标注该层的事件数 (取该层 n 的中位数)
        ncol = f"n_{t}"
        if ncol in df:
            nn = df[ncol].dropna()
            if len(nn):
                ax.text(0.01, 0.86, f"tier N per version ~ median {int(nn.median())} "
                        f"(min {int(nn.min())}, max {int(nn.max())})",
                        transform=ax.transAxes, fontsize=9,
                        bbox=dict(fc="white", ec="0.7", alpha=0.8))
        ax.set_ylabel("efficiency [%]")
        ax.set_title(f"E2  Old benchmark by number of B's per event - {tname} "
                     "(>2B has very few events -> huge error bars)", fontsize=12)
        ax.grid(alpha=0.25)
    axes[0].legend(loc="lower right", fontsize=9)
    axes[2].set_xlabel("training version (chronological by completion date)")
    deco_xaxis(axes[2], df)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "E2_old_benchmarks_by_nB.png")); plt.close(fig)


def E3(df, pos):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8.5), sharex=True)
    styles = {200: dict(marker="o", color="#1f77b4"),
              500: dict(marker="s", color="#ff7f0e"),
              3000: dict(marker="^", color="#2ca02c")}
    haveb = df[df[[f"ap_hardpool_{e}" for e in (200, 500, 3000)]].notna().any(axis=1)]
    xlo, xhi = int(haveb["idx"].min()) - 2, int(haveb["idx"].max()) + 2
    for ax, (vbase, ylab) in zip(axes, [("ap_hardpool", "hard-pool AP (threshold-free)"),
                                        ("chain_surv", "chain survival [%] (thr = 0.9)")]):
        any_ = False
        for ev, st in styles.items():
            vc = f"{vbase}_{ev}"
            if vc not in df:
                continue
            d = df[df[vc].notna()]
            if not len(d):
                continue
            yerr_v = None
            if vbase == "ap_hardpool":
                lo, hi = df.loc[d.index, f"ap_hardpool_lo_{ev}"], df.loc[d.index, f"ap_hardpool_hi_{ev}"]
                if lo.notna().all():
                    yerr_v = np.vstack([d[vc] - lo, hi - d[vc]])
            ax.errorbar(d["idx"], d[vc], yerr=yerr_v, fmt=st["marker"] + "-", ms=5, lw=1.2,
                        capsize=2, color=st["color"], label=f"{ev} events")
            any_ = True
        add_breakpoints(ax, df, pos, label=(ax is axes[0]))
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.25)
        ax.set_xlim(xlo, xhi)
        if not any_:
            ax.text(0.3, 0.5, "no data", transform=ax.transAxes)
    axes[0].set_title("E3  New (threshold-free) benchmark vs version - event counts are DIFFERENT rulers, "
                      "so never connect across event-count tiers (different markers)")
    axes[0].legend(loc="lower right", fontsize=9)
    axes[1].set_xlabel("training version (chronological by completion date)")
    deco_xaxis(axes[1], df)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "E3_new_benchmarks.png")); plt.close(fig)


def E4(df, pos):
    d = pd.read_csv(E2E)
    d = d.sort_values("chain_full", ascending=False)
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#7f7f7f" if s == "cc" else
              ("#d62728" if s.startswith("learned") else
               ("#2ca02c" if s.startswith("mix") else "#1f77b4"))
              for s in d["decoder"]]
    ax.bar(range(len(d)), d["chain_full"], color=colors)
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels(d["decoder"], rotation=45, ha="right", fontsize=10)
    for i, v in enumerate(d["chain_full"]):
        ax.text(i, v + 0.3, f"{v:.1f}", ha="center", fontsize=8)
    ax.set_ylabel("chain_full [%] - fraction of truth chains EXACTLY reproduced")
    ax.set_title("E4  Strict end-to-end (chain_full) - only v654 has an offline end-to-end eval.\n"
                 "grey = connected-component baseline; red = learned seqrec; green = mixed metric; "
                 "blue = pure-geometry. 600 events, thr 0.9.")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "E4_end_to_end.png")); plt.close(fig)


def E5(df, pos):
    fig, axes = plt.subplots(2, 1, figsize=(13, 8.5), sharex=True,
                             gridspec_kw=dict(height_ratios=[2, 1]))
    d = df[df["pv_hgnn_all_tracks"].notna()]
    axes[0].errorbar(d["idx"], d["pv_hgnn_all_tracks"], yerr=d["pv_hgnn_all_tracks_err"].fillna(0),
                     fmt="o-", ms=3.5, lw=1.0, capsize=2, color="#1f77b4", label="HGNN association")
    axes[0].errorbar(d["idx"], d["pv_minip_all_tracks"], yerr=d["pv_minip_all_tracks_err"].fillna(0),
                     fmt="s-", ms=3.5, lw=1.0, capsize=2, color="#d62728", label="minIP association (baseline)")
    add_breakpoints(axes[0], df, pos, label=True)
    axes[0].set_ylabel("PV association all_tracks [%]")
    axes[0].set_title("E5  PV association: HGNN vs traditional minIP baseline vs version "
                      "(lower is better here; see protocol doc for units)")
    axes[0].legend(loc="upper left"); axes[0].grid(alpha=0.25)

    dd = df[df["pv_diff_all_tracks"].notna()]
    axes[1].axhline(0, color="k", lw=1)
    axes[1].plot(dd["idx"], dd["pv_diff_all_tracks"], "D-", ms=3.5, lw=1.0, color="#2ca02c")
    add_breakpoints(axes[1], df, pos, label=False)
    axes[1].set_ylabel("HGNN - minIP [pp]")
    axes[1].set_xlabel("training version (chronological by completion date)")
    axes[1].grid(alpha=0.25)
    deco_xaxis(axes[0], df)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "E5_pv_association.png")); plt.close(fig)


def E6(df, pos):
    """slope / dumbbell: 同一批版本在三把尺子下的排名是否一致."""
    d = df[(df["all_particles"].notna()) & (df["ap_hardpool_200"].notna() | df["ap_hardpool_500"].notna())].copy()
    # 新 bench 用 500 档优先, 否则 200 档
    ap = []
    cs = []
    for _, r in d.iterrows():
        a = r["ap_hardpool_500"] if pd.notna(r["ap_hardpool_500"]) else r["ap_hardpool_200"]
        c = r["chain_surv_500"] if pd.notna(r["chain_surv_500"]) else r["chain_surv_200"]
        ap.append(a); cs.append(c)
    d["r_new_ap"] = ap
    d["r_new_cs"] = cs

    rulers = [("old\nall_particles", "all_particles", "#1f77b4"),
              ("new\nap_hardpool", "r_new_ap", "#ff7f0e"),
              ("new\nchain_surv", "r_new_cs", "#2ca02c")]
    ranks = {}
    for _, col, _c in rulers:
        s = d.set_index("version")[col]
        ranks[col] = s.rank(ascending=True).to_dict()

    fig, ax = plt.subplots(figsize=(11, 9))
    xs = [0, 1, 2]
    for _, r in d.iterrows():
        v = int(r["version"])
        ys = [ranks[col][v] for _, col, _ in rulers]
        hl = v in (623, 632, 633, 648, 654)
        ax.plot(xs, ys, "-o", ms=5, lw=2.2 if hl else 0.7,
                alpha=0.95 if hl else 0.4,
                color="#000000" if hl else "#888888",
                zorder=3 if hl else 1)
        ax.text(-0.06, ys[0], f"v{v}", ha="right", va="center", fontsize=8.5,
                color="k" if hl else "0.4")
        ax.text(2.06, ys[2], f"v{v}", ha="left", va="center", fontsize=8.5, color="0.4")
    ax.set_xticks(xs); ax.set_xticklabels([nm for nm, _c, _col in rulers], fontsize=12)
    ax.set_ylabel("rank within the common set (1 = best under that ruler)")
    ax.set_xlim(-0.5, 2.5)
    ax.invert_yaxis()
    ax.set_title("E6  Ruler divergence: same versions, three rulers - do the rankings agree?\n"
                 "Straight parallel lines = agreement; crossings = the ruler decides the winner. "
                 "(strict end-to-end chain_full exists only for v654 -> not rankable, see report)")
    for i in range(3):
        ax.axvline(i, color="0.85", zorder=0)
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "E6_ruler_divergence.png")); plt.close(fig)


def E7(df, pos):
    fig, ax = plt.subplots(figsize=(13, 6))
    for cls, col in ((1, "#1f77b4"), (2, "#ff7f0e"), (3, "#2ca02c")):
        c = f"lcag_acc_class{cls}"
        d = df[df[c].notna()]
        ax.plot(d["idx"], d[c], "o-", ms=3.5, lw=1.0, color=col,
                label=f"class{cls} diagonal accuracy")
    add_breakpoints(ax, df, pos, label=True)
    ax.set_ylabel("LCAG confusion diagonal accuracy [%]")
    ax.set_xlabel("training version (chronological by completion date)")
    ax.set_title("E7  LCAG structural-class accuracy (diagonal of the 4x4 confusion table) vs version.\n"
                 "class0 (background, ~90%) omitted; classes 1/2/3 are the structural classes.")
    deco_xaxis(ax, df)
    ax.legend(loc="center left"); ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "E7_lcag_confusion.png")); plt.close(fig)


def main():
    df = load()
    pos = bp_positions(df)
    print("breakpoint x positions:", pos)
    E1(df, pos); E2(df, pos); E3(df, pos); E4(df, pos)
    E5(df, pos); E6(df, pos); E7(df, pos)
    print("wrote figures to", OUTDIR)


if __name__ == "__main__":
    main()
