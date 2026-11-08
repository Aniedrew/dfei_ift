"""Extra explanatory figures for the deck (geometry, ranking, context, line graph)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle

OUT = "report_figs/deck_20261023"
GREY = "#9AA3B2"; GREEN = "#2E8B57"; NAVY = "#1F4E79"; PURPLE = "#8E44AD"
GOLD = "#B8860B"; TEAL = "#2C7A7B"; RED = "#C0392B"; LIGHT = "#C9D0DD"
plt.rcParams.update({"font.size": 12, "font.family": "DejaVu Sans"})


def _save(fig, name):
    p = f"{OUT}/{name}.png"
    fig.savefig(p, dpi=200, transparent=True, bbox_inches="tight")
    plt.close(fig)
    print("saved", p)


def fig_geo():
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    A1 = np.array([0.8, 1.05]); A2 = np.array([2.2, 2.35])
    u = np.array([1.0, 0.45]); u = u / np.linalg.norm(u)
    t = np.linspace(-0.4, 5.6, 10)
    L1 = A1[:, None] + u[:, None] * t
    L2 = A2[:, None] + u[:, None] * t
    ax.plot(L1[0], L1[1], color=NAVY, lw=3)
    ax.plot(L2[0], L2[1], color=GOLD, lw=3)
    P1 = A1 + 2.6 * u
    w = A2 - A1
    perp = w - (w @ u) * u
    P2 = P1 + perp
    ax.plot([P1[0], P2[0]], [P1[1], P2[1]], color=RED, lw=2.4, ls="--")
    ax.plot(*P1, "o", ms=10, color=NAVY, zorder=6)
    ax.plot(*P2, "o", ms=10, color=GOLD, zorder=6)
    ax.annotate("DOCA", xy=((P1[0] + P2[0]) / 2 + 0.12, (P1[1] + P2[1]) / 2 + 0.12),
                color=RED, fontsize=16, fontweight="bold")
    ax.plot(*A1, "s", ms=10, color=NAVY, zorder=6)
    ax.plot(*A2, "s", ms=10, color=GOLD, zorder=6)
    ax.plot([A1[0], A2[0]], [A1[1], A2[1]], color=PURPLE, lw=2.2, ls=":")
    ax.annotate("|Δstart|", xy=((A1[0] + A2[0]) / 2 - 0.95, (A1[1] + A2[1]) / 2 + 0.18),
                color=PURPLE, fontsize=16, fontweight="bold")
    ax.text(A1[0] - 0.20, A1[1] - 0.34, "A₁", color=NAVY, fontsize=15, fontweight="bold")
    ax.text(A2[0] + 0.10, A2[1] + 0.12, "A₂", color=GOLD, fontsize=15, fontweight="bold")
    ax.text(L1[0, -1] + 0.05, L1[1, -1] + 0.10, "track 1", color=NAVY, fontsize=14)
    ax.text(L2[0, -1] + 0.05, L2[1, -1] - 0.30, "track 2", color=GOLD, fontsize=14)
    ax.set_xlim(0, 6.4); ax.set_ylim(0.2, 5.2); ax.axis("off")
    _save(fig, "fig_geo")


def fig_rank():
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    rng = np.random.default_rng(3)
    neg = rng.uniform(0.10, 0.92, 40)
    hard = np.sort(neg)[-6:]
    pos = rng.uniform(0.80, 0.99, 5)
    ax.scatter(neg, np.zeros_like(neg) + 0.32, s=55, color=GREY, alpha=0.65, label="background")
    ax.scatter(hard, np.zeros_like(hard) + 0.32, s=90, color=RED, label="64 hardest")
    ax.scatter(pos, np.zeros_like(pos) + 0.60, s=110, color=GREEN,
               marker="D", label="signal")
    ax.annotate("", xy=(0.94, 0.46), xytext=(0.79, 0.46),
                arrowprops=dict(arrowstyle="<->", color=NAVY, lw=2))
    ax.text(0.865, 0.505, "m", color=NAVY, fontsize=17, ha="center", fontweight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(0.15, 0.78)
    ax.set_yticks([]); ax.set_xlabel("logit score  s", fontsize=14)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=11.5, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, 1.16), handletextpad=0.4, columnspacing=1.2)
    fig.tight_layout()
    _save(fig, "fig_rank_pair")


def fig_ctx():
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    i = np.array([2.6, 2.0]); j = np.array([4.4, 2.0])
    ax.plot([i[0], j[0]], [i[1], j[1]], color=GREEN, lw=3.4)
    ax.add_patch(Circle(i, 0.26, color=NAVY, zorder=5))
    ax.add_patch(Circle(j, 0.26, color=NAVY, zorder=5))
    ax.text(i[0] - 0.08, i[1] - 0.09, "i", color="white", fontsize=14, fontweight="bold", zorder=6)
    ax.text(j[0] - 0.08, j[1] - 0.09, "j", color="white", fontsize=14, fontweight="bold", zorder=6)
    left = [(1.15, 1.10), (1.05, 2.45), (1.55, 3.10)]
    right = [(5.75, 1.20), (5.85, 2.70)]
    for (x, y), a in zip(left, ["A=0.95", "A=0.60", "A=0.10"]):
        ax.plot([i[0], x], [i[1], y], color=GREY, lw=1.6)
        ax.plot(x, y, "o", ms=7, color=GREY)
        ax.text(x - 0.05, y + 0.12, a, fontsize=10.5, color="#5A6270", ha="center")
    for (x, y), a in zip(right, ["A=0.55", "A=0.20"]):
        ax.plot([j[0], x], [j[1], y], color=GREY, lw=1.6)
        ax.plot(x, y, "o", ms=7, color=GREY)
        ax.text(x, y + 0.12, a, fontsize=10.5, color="#5A6270", ha="center")
    ax.text(1.05, 3.55, "other pairs at i", color=GREY, fontsize=12)
    ax.text(5.30, 3.15, "other pairs at j", color=GREY, fontsize=12)
    ax.annotate("r, d, c at each end", xy=(3.5, 1.55), ha="center", color=GREEN,
                fontsize=14, fontweight="bold")
    ax.set_xlim(0.5, 6.4); ax.set_ylim(0.6, 3.9); ax.axis("off")
    _save(fig, "fig_ctx")


def fig_line():
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), gridspec_kw={"width_ratios": [1, 1.15]})
    ax = axes[0]
    P = {"t1": (1.0, 2.6), "t2": (2.9, 2.9), "t3": (1.1, 1.2), "t4": (3.0, 1.1)}
    for a, b in [("t1", "t2"), ("t2", "t3"), ("t2", "t4"), ("t3", "t4")]:
        ax.plot(*zip(P[a], P[b]), color=LIGHT, lw=1.8)
    for a, b in [("t1", "t2"), ("t1", "t3"), ("t1", "t4")]:
        ax.plot(*zip(P[a], P[b]), color=PURPLE, lw=3.0)
    for k, (x, y) in P.items():
        ax.add_patch(Circle((x, y), 0.20, color=NAVY, zorder=5))
        ax.text(x - 0.13, y - 0.08, k, color="white", fontsize=10, fontweight="bold", zorder=6)
    ax.set_title("tracks are nodes", fontsize=13)
    ax.set_xlim(0.4, 3.7); ax.set_ylim(0.7, 3.4); ax.axis("off")

    ax = axes[1]
    Q = {"e1": (1.3, 2.8), "e2": (0.8, 1.6), "e3": (2.2, 2.1), "e4": (3.1, 1.4), "e5": (1.9, 0.9)}
    for a, b, col in [("e1", "e2", PURPLE), ("e1", "e3", PURPLE), ("e2", "e3", PURPLE),
                      ("e1", "e4", TEAL), ("e2", "e4", GOLD), ("e4", "e5", GOLD),
                      ("e2", "e5", GOLD), ("e3", "e5", GREY)]:
        ax.plot(*zip(Q[a], Q[b]), color=col, lw=2.4)
    for k, (x, y) in Q.items():
        ax.add_patch(Circle((x, y), 0.20, color=PURPLE, zorder=5))
        ax.text(x - 0.13, y - 0.08, k, color="white", fontsize=10, fontweight="bold", zorder=6)
    ax.set_title("pairs are nodes", fontsize=13)
    ax.set_xlim(0.4, 3.7); ax.set_ylim(0.6, 3.4); ax.axis("off")
    fig.tight_layout()
    _save(fig, "fig_line")




def fig_dist_pair(fb, fa, name, tb, ta, cut_b=0.90, cut_a=0.90):
    """Two-panel edge-score distribution, same style as the paper figure."""
    import numpy as np
    bins = np.linspace(0.85, 1.0, 17)
    ctr = 0.5 * (bins[1:] + bins[:-1])
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.85))
    hs = []
    ymax = 0.0
    for f in (fb, fa):
        z = np.load(f)
        sc = z["edge_scores"]; yy = z["edge_labels"].astype(bool)
        m = sc >= 0.85
        hb, _ = np.histogram(sc[m & ~yy], bins=bins)
        ht, _ = np.histogram(sc[m & yy], bins=bins)
        hs.append((hb, ht)); ymax = max(ymax, hb.max())
    for ax, (hb, ht), lab, cut in zip(axes, hs, (tb, ta), (cut_b, cut_a)):
        ax.step(ctr, np.maximum(hb, 0.4), where="mid", color=GREY, lw=1.8)
        ax.step(ctr, np.maximum(ht, 0.4), where="mid", color=GREEN, lw=1.8)
        ax.set_yscale("log"); ax.set_ylim(0.8, ymax * 2.4)
        ax.axvline(cut, color=RED, ls="--", lw=1.3)
        ax.set_title(lab, fontsize=13, pad=6)
        ax.set_xlim(0.85, 1.01); ax.set_xticks([0.85, 0.90, 0.95, 1.00])
        ax.set_xlabel("logit score", fontsize=11, labelpad=2)
        ax.set_ylabel("edges / bin", fontsize=11, labelpad=2)
        ax.tick_params(labelsize=10)
    fig.text(0.5, 0.015, "grey = background · green = signal", ha="center", fontsize=10, color="#5A6270")
    fig.subplots_adjust(left=0.085, right=0.985, top=0.87, bottom=0.22, wspace=0.14)
    _save(fig, name)


# [2026-10-30] cut lines follow the cut that produced the numbers on each slide:
#   dist_wp   : the working-point scan itself (0.90 -> 0.95)
#   dist_rank/vert/ctx/att : both panels at 0.95 (every result from the working point on)
fig_dist_pair("report_figs/pruning_scores_0904/v601.npz",
              "report_figs/pruning_scores_0904/v601.npz", "dist_wp", "cut 0.90", "cut 0.95", 0.90, 0.95)
fig_dist_pair("report_figs/pruning_scores_0904/v601.npz",
              "report_figs/pruning_scores_0904/v631.npz", "dist_rank", "before", "after separation loss", 0.95, 0.95)
fig_dist_pair("report_figs/pruning_scores_0904/v631.npz",
              "report_figs/pruning_scores_0904/v643.npz", "dist_vert", "before (24.1%)", "after vertex (26.0%)", 0.95, 0.95)
fig_dist_pair("report_figs/pruning_scores_0904/v643.npz",
              "report_figs/pruning_scores_0904/v648.npz", "dist_ctx", "before", "after context", 0.95, 0.95)
fig_dist_pair("report_figs/pruning_scores_0904/v648.npz",
              "report_figs/pruning_scores_0904/v651.npz", "dist_att", "before", "after attention", 0.95, 0.95)




def fig_vertex_eff():
    import numpy as np
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    x = np.arange(2); w = 0.34
    before = [22.6, 14.8]; after = [26.0, 17.8]
    b1 = ax.bar(x - w/2, before, w, color=GREY, label="before")
    b2 = ax.bar(x + w/2, after, w, color=PURPLE, label="after")
    for bars, vals in ((b1, before), (b2, after)):
        for r, v in zip(bars, vals):
            ax.text(r.get_x() + r.get_width()/2, v + 0.5, f"{v:.1f}", ha="center", fontsize=13)
    ax.set_xticks(x); ax.set_xticklabels(["AllParticles", "PerfectReco"], fontsize=14)
    ax.set_ylabel("% of signal chains", fontsize=13)
    ax.set_ylim(0, 33); ax.legend(frameon=False, fontsize=12, loc="upper right")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout(); _save(fig, "fig_vertex_eff")


fig_vertex_eff()




def fig_linegraph():
    import numpy as np
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))
    P = {"i": (0.5, 0.92), "j": (0.10, 0.12), "k": (0.90, 0.12)}
    E = {"ij": ("i", "j"), "ik": ("i", "k"), "jk": ("j", "k")}
    ax = axes[0]
    for a, b in E.values():
        ax.plot([P[a][0], P[b][0]], [P[a][1], P[b][1]], color="#9AA3B2", lw=2.0, zorder=1)
    for n, (x, y) in P.items():
        ax.add_patch(Circle((x, y), 0.075, color=NAVY, zorder=3))
        ax.text(x, y, n, color="white", ha="center", va="center", fontsize=12, fontweight="bold", zorder=4)
    ax.set_title("track graph: tracks are nodes", fontsize=12)
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05); ax.axis("off")

    ax = axes[1]
    EP = {"ij": (0.5, 0.86), "ik": (0.16, 0.20), "jk": (0.84, 0.20)}
    for a in EP:
        for b in EP:
            if a < b:
                ax.plot([EP[a][0], EP[b][0]], [EP[a][1], EP[b][1]], color=PURPLE, lw=2.0, alpha=0.75, zorder=1)
    for n, (x, y) in EP.items():
        ax.add_patch(Circle((x, y), 0.095, color=PURPLE, zorder=3))
        ax.text(x, y, n, color="white", ha="center", va="center", fontsize=11, fontweight="bold", zorder=4)
    ax.set_title("line graph: pairs are nodes", fontsize=12)
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05); ax.axis("off")
    fig.text(0.5, 0.03, "two pair-nodes are adjacent when they share a track", ha="center", fontsize=11, color="#5A6270")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _save(fig, "fig_linegraph")


fig_linegraph()




def fig_progress():
    import numpy as np
    vals = [14.5, 17.9, 22.6, 26.0, 27.0, 28.0]
    labs = ["baseline", "working\npoint", "separation\n+ small", "vertex\ngeometry",
            "edge\ncontext", "attention"]
    fig, ax = plt.subplots(figsize=(10.4, 3.8), dpi=180)
    x = np.arange(len(vals))
    cols = [GREY] + [PURPLE] * 5
    bars = ax.bar(x, vals, 0.62, color=cols)
    for r, v in zip(bars, vals):
        ax.text(r.get_x() + r.get_width()/2, v + 0.6, f"{v:.1f}", ha="center", fontsize=14)
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=11)
    ax.set_ylabel("% of signal chains", fontsize=12); ax.set_ylim(0, 33)
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout(); _save(fig, "progress")


fig_progress()


def fig_sat():
    import numpy as np
    xs = [0, 1, 2, 3]
    ys = [22.5, 33.6, 36.7, 36.7]
    labs = ["baseline", "+ graph structure\nand features", "+ heads, width,\nattention", "+ more\ncapacity"]
    fig, ax = plt.subplots(figsize=(6.6, 4.3), dpi=180)
    ax.axhspan(36.0, 37.4, color=GREY, alpha=0.14, lw=0)
    ax.plot(xs[:3], ys[:3], "-o", color=PURPLE, lw=2.8, ms=10, zorder=3)
    ax.plot(xs[2:], ys[2:], "--o", color=GREY, lw=2.4, ms=10, mfc="white", mew=2, zorder=3)
    for x, y in zip(xs, ys):
        ax.text(x, y + 1.5, f"{y:.1f}", ha="center", fontsize=15, fontweight="bold", color=PURPLE if x < 3 else GREY)
    ax.annotate("no further gain", xy=(2.5, 36.7), xytext=(1.35, 41.2), fontsize=13,
                color=GREY, arrowprops=dict(arrowstyle="->", color=GREY, lw=1.6))
    ax.set_xticks(xs); ax.set_xticklabels(labs, fontsize=12)
    ax.set_ylabel("AllParticles [%]", fontsize=13)
    ax.set_ylim(18, 44); ax.set_xlim(-0.35, 3.4)
    ax.grid(axis="y", alpha=0.18)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.tight_layout(); _save(fig, "sat")


fig_sat()


if __name__ == "__main__":
    fig_geo(); fig_rank(); fig_ctx(); fig_line()
