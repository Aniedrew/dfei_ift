"""Deck v4 — large-type redesign: body >= 25 pt, one idea per slide, attention explained in plain words."""
import os
from io import BytesIO
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import mathtext
from matplotlib.font_manager import FontProperties
from PIL import Image, ImageDraw
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR

OUT = "DFEI_bugfix_pruning_20261023_EN.pptx"
FIGS = "report_figs/deck_20261023"
os.makedirs(FIGS, exist_ok=True)

NAVY = RGBColor(0x1F, 0x4E, 0x79); BLUE = RGBColor(0x1F, 0x4E, 0x79)
TEAL = RGBColor(0x2C, 0x7A, 0x7B); MAGENTA = RGBColor(0xC2, 0x18, 0x5B)
ICE = RGBColor(0xDC, 0xE6, 0xF1); GREY = RGBColor(0x77, 0x77, 0x77)
LIGHT = RGBColor(0xF7, 0xF7, 0xF7); WHITE = RGBColor(0xFF, 0xFF, 0xFF)
INK = RGBColor(0x2B, 0x2B, 0x2B); RED = RGBColor(0xC0, 0x39, 0x2B)
GREEN = RGBColor(0x2E, 0x8B, 0x57); GOLD = RGBColor(0xB8, 0x86, 0x0B)
PURPLE = RGBColor(0x8E, 0x44, 0xAD)
BORDER = RGBColor(0xD9, 0xD9, 0xD9); PALE = RGBColor(0xF5, 0xF8, 0xFA)
CODEBG = RGBColor(0xF7, 0xF7, 0xF7)
C = dict(p="#1F4E79", s="#2E8B57", ice="#DCE6F1", grey="#777777", red="#C0392B", light="#F7F7F7")

plt.rcParams.update({
    "axes.facecolor": "none", "figure.facecolor": "none", "axes.edgecolor": "#9AA3B2",
    "axes.labelcolor": C["p"], "text.color": C["p"], "xtick.color": C["p"], "ytick.color": C["p"],
    "font.size": 13, "axes.titlesize": 15, "xtick.labelsize": 13, "ytick.labelsize": 13,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.16, "grid.color": "#9AA3B2",
})

VERS = ["v601", "v623", "v631", "v633", "v640", "v643", "v648", "v651"]
NAME = {"v601": "baseline", "v623": "input fix", "v631": "ranking term", "v633": "cleanup",
        "v640": "cut + small items", "v643": "DOCA fix", "v648": "edge context", "v651": "attention"}
NAME_ROC = dict(NAME, v631="earlier bundle")
RAW_ALL = [14.45, 14.31, 19.77, 21.83, 22.60, 26.04, 27.00, 27.96]
RAW_PERF = [9.54, 9.57, 13.30, 14.46, 14.79, 17.75, 18.12, 19.00]
AUC = [0.9716, 0.9737, 0.9352, 0.9242, None, 0.9236, 0.9324, 0.9327]
PRUNED = [43.05, 39.12, 42.26, 45.71, 49.97, 47.45, 52.12, 54.06]
CONTAM = [42.50, 46.57, 37.97, 32.46, 27.43, 26.51, 20.88, 17.98]
CLEAN = [14.45, 14.31, 19.77, 21.83, 22.60, 26.04, 27.00, 27.96]


def save(fig, name, w, h):
    p = f"{FIGS}/{name}_{int(w*180)}x{int(h*180)}.png"
    fig.savefig(p, dpi=180, transparent=True); plt.close(fig); return p


def eq_block(name, lines, size=18, pad=14, gap=8, bgpad=6):
    """Typeset a block of LaTeX-like equations as one transparent PNG (300 dpi).

    lines: list of (tex, colour, background-colour-or-None).
    Returns (path, native_width_in, native_height_in); insert at native size
    to keep the requested point size.
    """
    imgs = []
    for tex, col, bg in lines:
        buf = BytesIO()
        mathtext.math_to_image(f"${tex}$", buf, prop=FontProperties(size=size),
                               dpi=300, format="png", color=col)
        buf.seek(0)
        imgs.append(Image.open(buf).convert("RGBA"))
    W = max(im.width for im in imgs) + 2 * pad
    H = sum(im.height for im in imgs) + gap * (len(imgs) - 1) + 2 * pad
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(canvas)
    y = pad
    for (tex, col, bg), im in zip(lines, imgs):
        if bg is not None:
            d.rounded_rectangle([pad - bgpad, y - bgpad, W - pad + bgpad, y + im.height + bgpad],
                                radius=6, fill=bg)
        canvas.alpha_composite(im, (pad, y))
        y += im.height + gap
    p = f"{FIGS}/eq_{name}.png"
    canvas.save(p)
    return p, W / 300.0, H / 300.0


def fig_pv():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 3.5), dpi=180, gridspec_kw={"width_ratios": [1.5, 1]})
    lab = ["old 1", "old 2", "old 3", "old 4", "baseline", "input fix", "final"]
    allv = [13.80, 13.40, 13.07, 14.02, 8.61, 8.61, 8.64]
    a1.bar(lab, allv, color=[C["grey"]] * 4 + [C["s"]] * 3, width=0.62)
    a1.axhline(10.15, color=C["red"], ls="--", lw=2)
    a1.text(6.5, 10.6, "minIP 10.15", color=C["red"], ha="right", fontsize=12)
    for i, v in enumerate(allv): a1.text(i, v + 0.28, f"{v:.1f}", ha="center", fontsize=12)
    a1.set_ylim(0, 16.5); a1.set_ylabel("misassociation [%]", fontsize=12)
    a1.tick_params(axis="x", labelsize=9)
    a1.set_title("all tracks", fontsize=14, loc="left")
    a2.bar(["baseline", "final", "minIP"], [16.50, 14.86, 19.99], color=[C["s"], C["s"], C["red"]], width=0.55)
    for i, v in enumerate([16.50, 14.86, 19.99]): a2.text(i, v + 0.35, f"{v:.1f}", ha="center", fontsize=12)
    a2.set_ylim(0, 23); a2.set_title("signal", fontsize=14, loc="left")
    fig.tight_layout(); return save(fig, "pv2", 10.4, 3.5)


def fig_decomp():
    pr = [43.05, 52.88, 51.99, 47.45, 52.12, 54.06]
    co = [42.50, 29.23, 23.89, 26.51, 20.88, 17.98]
    cl = [14.45, 17.90, 24.12, 26.04, 27.00, 27.96]
    pf = [9.5, 11.7, 15.9, 17.8, 18.1, 19.0]
    labels = ["baseline\n(cut 0.90)", "working\npoint", "separation\nloss", "DOCA\nfix (Part 2)",
              "edge\ncontext", "attention"]
    fig, ax = plt.subplots(figsize=(10.4, 3.6), dpi=180)
    x = np.arange(len(labels))
    ax.bar(x, pr, 0.46, color="#C9D0DD", label="pruned away")
    ax.bar(x, co, 0.46, bottom=pr, color=C["s"], label="merged with background")
    ax.bar(x, cl, 0.46, bottom=np.array(pr) + np.array(co), color=C["p"], label="exact")
    plt.rcParams["hatch.linewidth"] = 0.9
    ax.bar(x, pf, 0.46, bottom=np.array(pr) + np.array(co), facecolor="none",
           edgecolor="white", linewidth=0.0, hatch="///", label="perfect (hatched)")
    for i in range(len(labels)):
        _b = pr[i] + co[i]
        ax.text(i + 0.27, pr[i] / 2, f"{pr[i]:.0f}", ha="left", va="center",
                fontsize=10, color="#4A5568")
        ax.text(i + 0.27, pr[i] + co[i] / 2, f"{co[i]:.0f}", ha="left", va="center",
                fontsize=10, color=C["s"])
        ax.text(i + 0.27, _b + pf[i] / 2, f"{pf[i]:.1f}", ha="left", va="center",
                fontsize=10, color=C["p"])
        ax.text(i + 0.27, _b + cl[i] - 0.8, f"{cl[i]:.1f}", ha="left", va="center",
                fontsize=10.5, fontweight="bold", color=C["p"])
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10.5); ax.set_ylim(0, 100)
    ax.set_xlim(-0.55, 5.75)
    ax.set_ylabel("% of signal chains", fontsize=12)
    ax.legend(frameon=False, fontsize=11, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout(); return save(fig, "decomp", 10.4, 3.6)


def fig_scatter():
    fig, ax = plt.subplots(figsize=(6.0, 3.4), dpi=180)
    a = [(AUC[i], RAW_ALL[i], VERS[i]) for i in range(len(VERS)) if AUC[i]]
    ax.scatter([t[0] for t in a], [t[1] for t in a], s=90, color=C["s"], zorder=3)
    off = {"v601": (-46, 5), "v623": (8, 5), "v631": (8, -14), "v633": (-44, 4),
           "v643": (-40, -4), "v648": (-8, -17), "v651": (8, 4)}
    for x, y, v in a:
        ax.annotate(v, (x, y), textcoords="offset points", xytext=off.get(v, (7, 5)), fontsize=11)
    ax.set_xlabel("edge AUC", fontsize=12); ax.set_ylabel("AllParticles [%]", fontsize=12)
    ax.set_xlim(0.918, 0.982); ax.set_ylim(10, 33)
    fig.tight_layout(); return save(fig, "scatter", 6.0, 3.4)


def fig_ladder():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 3.3), dpi=180)
    a1.plot(VERS, RAW_ALL, "o-", color=C["p"], lw=2.6, ms=7, label="AllParticles")
    a1.plot(VERS, RAW_PERF, "s-", color=C["s"], lw=2.6, ms=7, label="PerfectReco")
    a1.set_ylim(5, 33); a1.set_ylabel("[%]", fontsize=12)
    a1.legend(frameon=False, fontsize=11, loc="upper left")
    a1.set_title("reconstruction", fontsize=14, loc="left")
    idx = [t for t in range(len(VERS)) if AUC[t]]
    a2.plot([VERS[i] for i in idx], [AUC[i] for i in idx], "o-", color=C["grey"], lw=2.6, ms=7)
    a2.set_ylim(0.915, 0.985); a2.set_ylabel("edge AUC", fontsize=12)
    a2.set_title("ranking quality", fontsize=14, loc="left")
    fig.tight_layout(); return save(fig, "ladder2", 10.6, 3.3)


def fig_delta():
    steps = ["input fix", "M1 point", "M1 cleanup", "M1 select", "DOCA fix", "M2 context", "M3 attn"]
    da = [RAW_ALL[i + 1] - RAW_ALL[i] for i in range(7)]
    dp = [RAW_PERF[i + 1] - RAW_PERF[i] for i in range(7)]
    fig, ax = plt.subplots(figsize=(9.6, 3.3), dpi=180)
    y = np.arange(len(steps)); h = 0.36
    ax.barh(y + h / 2, da, h, color=C["p"], label="AllParticles")
    ax.barh(y - h / 2, dp, h, color=C["s"], label="PerfectReco")
    for i, (u, v) in enumerate(zip(da, dp)):
        ax.text(u + (0.3 if u >= 0 else -0.3), i + h / 2, f"{u:+.1f}", va="center", ha="left" if u >= 0 else "right", fontsize=11)
        ax.text(v + (0.3 if v >= 0 else -0.3), i - h / 2, f"{v:+.1f}", va="center", ha="left" if v >= 0 else "right", fontsize=11)
    ax.set_yticks(y); ax.set_yticklabels(steps, fontsize=11); ax.invert_yaxis()
    ax.set_xlim(-0.9, 6.2)
    ax.axvline(0, color="#9AA3B2", lw=1.2)
    ax.set_xlabel("change vs previous version [pp]", fontsize=12)
    ax.legend(frameon=False, fontsize=12, loc="lower right")
    fig.subplots_adjust(left=0.20, right=0.98, top=0.96, bottom=0.16); return save(fig, "delta", 9.6, 3.3)


def fig_affinity():
    fig, ax = plt.subplots(figsize=(5.6, 2.5), dpi=180)
    x = np.linspace(0, 200, 400); y = np.exp(-x / 50)
    ax.plot(x, y, color=C["p"], lw=2.8)
    ax.axhline(0.5, ls="--", color=C["red"], lw=1.2)
    ax.axvline(50 * np.log(2), ls="--", color=C["red"], lw=1.2)
    ax.text(38, 0.56, "d₃ ≈ 35 mm", color=C["red"], fontsize=11)
    ax.text(120, 0.56, "A = 0.5", color=C["red"], fontsize=11)
    ax.set_xlabel("d₃  [mm]", fontsize=12); ax.set_ylabel("A$_e$", fontsize=12)
    ax.set_xlim(0, 200); ax.set_ylim(0, 1.05)
    fig.tight_layout(); return save(fig, "affinity", 5.6, 2.5)


P_AFF = fig_affinity()


def fig_neg():
    names = ["attention\nalone", "node-pool\nmask", "geom.\ncandidates", "slot\ndecode"]
    vals = [0.0, -0.82, -7.39, -23.92]
    fig, ax = plt.subplots(figsize=(6.6, 3.0), dpi=180)
    ax.barh(names, vals, color=[C["grey"]] + [C["red"]] * 3, height=0.55)
    for i, v in enumerate(vals):
        ax.text(v - 0.6 if v < 0 else v + 0.4, i, f"{v:+.1f}", va="center", ha="right" if v < 0 else "left", fontsize=12)
    ax.axvline(0, color="#9AA3B2", lw=1.2); ax.invert_yaxis(); ax.set_xlim(-28, 3)
    ax.set_xlabel("Δ vs frontier", fontsize=12)
    fig.tight_layout(); return save(fig, "neg", 6.6, 3.0)


def fig_doca():
    fig, ax = plt.subplots(figsize=(4.8, 3.1), dpi=180)
    x = [0, 1]; wd = 0.32
    a = [22.60, 26.04]; p = [14.79, 17.75]
    ax.bar([i - wd / 2 for i in x], a, wd, color=C["p"], label="AllParticles")
    ax.bar([i + wd / 2 for i in x], p, wd, color=C["s"], label="PerfectReco")
    for i in range(2):
        ax.text(i - wd / 2, a[i] + 0.5, f"{a[i]+1e-9:.1f}", ha="center", fontsize=12)
        ax.text(i + wd / 2, p[i] + 0.5, f"{p[i]+1e-9:.1f}", ha="center", fontsize=12)
    ax.set_xticks(x); ax.set_xticklabels(["before", "after"], fontsize=13)
    ax.set_ylim(0, 33); ax.set_ylabel("[%]", fontsize=12)
    ax.legend(frameon=False, fontsize=11, loc="upper left")
    fig.tight_layout(); return save(fig, "doca", 4.8, 3.1)


def fig_doca_fix():
    """The stored DOCA column alone: fixing it does not move the chain yield."""
    fig, ax = plt.subplots(figsize=(4.8, 3.1), dpi=180)
    x = [0, 1]; wd = 0.32
    a = [14.5, 14.3]; p = [9.5, 9.6]
    ax.bar([i - wd / 2 for i in x], a, wd, color=C["p"], label="AllParticles")
    ax.bar([i + wd / 2 for i in x], p, wd, color=C["s"], label="PerfectReco")
    for i in range(2):
        ax.text(i - wd / 2, a[i] + 0.35, f"{a[i]:.1f}", ha="center", fontsize=12)
        ax.text(i + wd / 2, p[i] + 0.35, f"{p[i]:.1f}", ha="center", fontsize=12)
    ax.set_xticks(x); ax.set_xticklabels(["stored column", "recomputed"], fontsize=12)
    ax.set_ylim(0, 20); ax.set_ylabel("[%]", fontsize=12)
    ax.legend(frameon=False, fontsize=10, loc="upper center", ncol=2)
    fig.tight_layout(); return save(fig, "docafix", 4.8, 3.1)


def fig_threshold():
    """Same v601 weights, only the node/edge working point changes."""
    fig, ax = plt.subplots(figsize=(6.2, 3.2), dpi=180)
    x = np.arange(3)
    a = [14.5, 17.9, 18.5]; p = [9.5, 11.7, 11.9]
    w = 0.34
    ax.bar(x - w / 2, a, w, color=C["p"], label="AllParticles")
    ax.bar(x + w / 2, p, w, color=C["s"], label="PerfectReco")
    for i in range(3):
        ax.text(x[i] - w / 2, a[i] + 0.35, f"{a[i]:.1f}", ha="center", fontsize=11)
        ax.text(x[i] + w / 2, p[i] + 0.35, f"{p[i]:.1f}", ha="center", fontsize=11)
    ax.set_xticks(x); ax.set_xticklabels(["0.90", "0.95", "0.97"], fontsize=13)
    ax.set_xlabel("node / edge pruning threshold", fontsize=11)
    ax.set_ylim(0, 22); ax.set_ylabel("[%]", fontsize=11)
    ax.legend(frameon=False, fontsize=10.5, loc="upper left")
    fig.tight_layout(); return save(fig, "thr", 6.2, 3.2)


def fig_rank_effect():
    """AllParticles for baseline and with the ranking term, at two working points."""
    fig, ax = plt.subplots(figsize=(6.2, 3.2), dpi=180)
    x = np.arange(2); w = 0.34
    base = [14.5, 17.9]; rank = [16.3, 20.5]
    ax.bar(x - w / 2, base, w, color="#9AA3B2", label="baseline")
    ax.bar(x + w / 2, rank, w, color=C["p"], label="+ ranking loss")
    for i in range(2):
        ax.text(x[i] - w / 2, base[i] + 0.35, f"{base[i]:.1f}", ha="center", fontsize=11)
        ax.text(x[i] + w / 2, rank[i] + 0.35, f"{rank[i]:.1f}", ha="center", fontsize=11)
    ax.set_xticks(x); ax.set_xticklabels(["0.90", "0.95"], fontsize=13)
    ax.set_xlabel("working point", fontsize=11)
    ax.set_ylim(0, 24); ax.set_ylabel("AllParticles [%]", fontsize=11)
    ax.legend(frameon=False, fontsize=10.5, loc="upper left")
    fig.tight_layout(); return save(fig, "rankeff", 6.2, 3.2)


def fig_rank():
    fig, ax = plt.subplots(figsize=(5.4, 2.8), dpi=180)
    x = np.linspace(0, 1, 300)
    neg = 0.9 * np.exp(-((x - 0.42) ** 2) / 0.020)
    pos = 0.9 * np.exp(-((x - 0.76) ** 2) / 0.016)
    ax.plot(x, neg, color="#9AA3B2", lw=2); ax.fill_between(x, neg, color="#9AA3B2", alpha=0.22)
    ax.plot(x, pos, color=C["s"], lw=2); ax.fill_between(x, pos, color=C["s"], alpha=0.22)
    ax.annotate("", xy=(0.76, 0.62), xytext=(0.42, 0.62), arrowprops=dict(arrowstyle="<->", color=C["p"], lw=2))
    ax.text(0.59, 0.68, "margin", ha="center", fontsize=12, color=C["p"])
    ax.set_yticks([]); ax.set_xlim(0, 1); ax.set_xlabel("logit score", fontsize=12)
    fig.tight_layout(); return save(fig, "rank", 5.4, 2.8)


def fig_wp_combo():
    """Working-point page: one score distribution + the efficiency/purity trade-off.

    Top:    the same edge scores once, with both cuts drawn on it (they differ only by
            where the dashed line sits, so two panels were redundant).
    Bottom: signal efficiency and purity as a function of the cut, same x-axis.
    """
    GREY, GREEN, RED, BLUE = "#9AA3B2", "#2E8B57", "#C0392B", "#1F4E79"
    z = np.load("report_figs/pruning_scores_0904/v601.npz")
    s, y = z["edge_scores"], z["edge_labels"].astype(bool)
    ntot = max(int(y.sum()), 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.7, 5.15), dpi=180, sharex=True,
                                   gridspec_kw={"height_ratios": [1.0, 0.95], "hspace": 0.12})

    # ---- top: score distribution ----
    bins = np.linspace(0.85, 1.0, 17)
    ctr = 0.5 * (bins[1:] + bins[:-1])
    m = s >= 0.85
    hb, _ = np.histogram(s[m & ~y], bins=bins)
    ht, _ = np.histogram(s[m & y], bins=bins)
    ax1.step(ctr, np.maximum(hb, 0.4), where="mid", color=GREY, lw=1.8, label="background")
    ax1.step(ctr, np.maximum(ht, 0.4), where="mid", color=GREEN, lw=1.8, label="signal")
    ax1.set_yscale("log"); ax1.set_ylim(0.8, hb.max() * 3.0)
    ax1.set_ylabel("edges / bin", fontsize=11)
    for c in (0.90, 0.95):
        ax1.axvline(c, color=RED, ls="--", lw=1.4)
    ax1.text(0.898, hb.max() * 0.22, "0.90", color=RED, fontsize=10.5,
             ha="right", fontweight="bold")
    ax1.text(0.958, hb.max() * 0.22, "0.95", color=RED, fontsize=10.5,
             ha="left", fontweight="bold")
    ax1.legend(frameon=False, fontsize=10, loc="upper left", handlelength=1.4)
    ax1.tick_params(labelsize=9.5)

    # ---- bottom: efficiency / purity vs cut ----
    cuts = np.linspace(0.86, 0.99, 80)
    eff, pur = [], []
    for c in cuts:
        k = s >= c
        nt = float((k & y).sum()); nb = float((k & ~y).sum())
        eff.append(100.0 * nt / ntot)
        pur.append(100.0 * nt / max(nt + nb, 1.0))
    ax2.plot(cuts, eff, color=GREEN, lw=2.4, label="signal efficiency")
    ax2.set_ylabel("efficiency [%]", fontsize=11, color=GREEN)
    ax2.tick_params(axis="y", labelsize=9.5, colors=GREEN)
    ax2.set_ylim(0, 75)
    ax2b = ax2.twinx()
    ax2b.plot(cuts, pur, color=BLUE, lw=2.4, ls="-", label="purity")
    ax2b.set_ylabel("purity [%]", fontsize=11, color=BLUE)
    ax2b.tick_params(axis="y", labelsize=9.5, colors=BLUE)
    ax2b.set_ylim(0, 26)
    ax2b.spines["top"].set_visible(False)

    for c in (0.90, 0.95):
        ax2.axvline(c, color=RED, ls="--", lw=1.4)
    # mark the chosen point (exact cut, not the nearest grid sample)
    k = s >= 0.95
    nt = float((k & y).sum()); nb = float((k & ~y).sum())
    eff95 = 100.0 * nt / ntot
    pur95 = 100.0 * nt / max(nt + nb, 1.0)
    ax2.plot([0.95], [eff95], "o", ms=9, color=GREEN, zorder=5)
    ax2b.plot([0.95], [pur95], "o", ms=9, color=BLUE, zorder=5)
    ax2.annotate(f"{eff95:.1f}", (0.95, eff95), textcoords="offset points",
                 xytext=(-6, -22), ha="right", fontsize=10.5, color=GREEN, fontweight="bold")
    ax2b.annotate(f"{pur95:.1f}", (0.95, pur95), textcoords="offset points",
                  xytext=(9, -17), ha="left", fontsize=10.5, color=BLUE, fontweight="bold")
    ax2.set_xlabel("cut on the pruning score", fontsize=11)
    ax2.set_xlim(0.85, 1.00)
    ax2.set_xticks([0.85, 0.90, 0.95, 1.00])
    ax2.tick_params(labelsize=9.5)

    fig.subplots_adjust(left=0.135, right=0.865, top=0.975, bottom=0.105)
    return save(fig, "wp_combo", 5.7, 5.15)


P_WP_COMBO = fig_wp_combo()


def fig_dist(vb, va, name):
    """Same before/after score distributions for every optimisation step."""
    GREY, GREEN, RED = "#9AA3B2", "#2E8B57", "#C0392B"
    bins = np.linspace(0.85, 1.0, 17)
    ctr = 0.5 * (bins[1:] + bins[:-1])
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.85), dpi=180)
    ymax = 0.0
    hs = []
    for v in (vb, va):
        z = np.load(f"report_figs/pruning_scores_0904/{v}.npz")
        s, y = z["edge_scores"], z["edge_labels"].astype(bool)
        m = s >= 0.85
        hb, _ = np.histogram(s[m & ~y], bins=bins)
        ht, _ = np.histogram(s[m & y], bins=bins)
        hs.append((hb, ht))
        ymax = max(ymax, hb.max())

    for i, (ax, v, (hb, ht)) in enumerate(zip(axes, (vb, va), hs)):
        ax.step(ctr, np.maximum(hb, 0.4), where="mid", color=GREY, lw=1.8)
        ax.step(ctr, np.maximum(ht, 0.4), where="mid", color=GREEN, lw=1.8)
        ax.set_yscale("log"); ax.set_ylim(0.8, ymax * 2.4)
        ax.axvline(0.90, color=RED, ls="--", lw=1.3)
        ax.set_title(("before · " if i == 0 else "after · ") + NAME[v], fontsize=11, pad=7)
        ax.set_xlim(0.85, 1.01)
        ax.set_xticks([0.85, 0.90, 0.95, 1.00])
        ax.set_xlabel("logit score", fontsize=10, labelpad=2)
        ax.set_ylabel("edges / bin", fontsize=10, labelpad=2)
        ax.tick_params(labelsize=9)

    fig.text(0.5, 0.015,
             "grey = background edges · green = signal edges",
             ha="center", fontsize=9.4, color="#5A6270")
    fig.subplots_adjust(left=0.085, right=0.985, top=0.885, bottom=0.215, wspace=0.14)
    return save(fig, name, 7.6, 3.85)


P_DIST_M1 = fig_dist("v601", "v631", "dist_m1")
P_DIST_M2 = fig_dist("v643", "v648", "dist_m2")
P_DIST_M3 = fig_dist("v648", "v651", "dist_m3")


def bench_rows(vb, va, cuts=(0.90, 0.97, 0.99)):
    """Retained background, retained signal edges, efficiency and purity."""
    out = []
    for v in (vb, va):
        z = np.load(f"report_figs/pruning_scores_0904/{v}.npz")
        s, y = z["edge_scores"], z["edge_labels"].astype(bool)
        out.append((s, y, int(y.sum())))
    rows = []
    for c in cuts:
        cells = []
        for s, y, nt_tot in out:
            k = s >= c
            nt = int((k & y).sum()); nb = int((k & ~y).sum())
            cells.append((nb, nt, nt / nt_tot, nt / max(nt + nb, 1)))
        rows.append((c, cells[0], cells[1]))
    return rows


def fig_bias():
    """The cleanest ablation: same idea, with and without the geometric prior."""
    fig, ax = plt.subplots(figsize=(6.2, 3.05), dpi=180)
    names = ["attention only", "attention + geometry\ninside the score"]
    vals = [0.0, 4.5]
    ax.bar(names, vals, color=["#9AA3B2", "#1F4E79"], width=0.52)
    ax.text(0, 0.18, "0.0 — no gain", ha="center", fontsize=12.5, color="#C0392B", weight="bold")
    ax.text(1, 4.5 + 0.22, "+4.5 pp", ha="center", fontsize=15, color="#1F4E79", weight="bold")
    ax.set_ylim(-0.45, 5.85)
    ax.set_ylabel("Δ AllParticles [pp]", fontsize=11.5)
    ax.tick_params(labelsize=10.5)
    ax.axhline(0, color="#9AA3B2", lw=1.1)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.text(0.5, 5.45, "same backbone, same data, same schedule", ha="center", fontsize=9.5,
            color="#8A8F98", style="italic")
    fig.tight_layout(); return save(fig, "bias", 6.2, 3.05)


def fig_workpoint():
    """What the six inputs change at the working cut: fewer background edges, same truth."""
    fig, ax = plt.subplots(figsize=(5.7, 3.25), dpi=180)
    groups = ["background kept", "truth kept"]
    before, after = [4965, 649], [4177, 629]
    x = np.arange(2); w = 0.33
    b1 = ax.bar(x - w / 2, before, w, color="#9AA3B2", label="before")
    b2 = ax.bar(x + w / 2, after, w, color="#1F4E79", label="+6 inputs")
    for rect, v in list(zip(b1, before)) + list(zip(b2, after)):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 95, f"{v}", ha="center", fontsize=11)
    ax.text(x[0], 5500, "−16%", ha="center", fontsize=18, color="#1F4E79", weight="bold")
    ax.text(x[1], 1150, "−3%", ha="center", fontsize=18, color="#1F4E79", weight="bold")
    ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=11.5)
    ax.set_ylim(0, 6100); ax.set_ylabel("edges passing cut 0.90", fontsize=11)
    ax.tick_params(labelsize=10)
    ax.legend(frameon=False, fontsize=10.2, loc="upper right")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout(); return save(fig, "work", 5.7, 3.25)


def fig_roc():
    """Edge ROC for the versions kept in the deck, dumped on the same 200 events."""
    cols = {"v601": "#9AA3B2", "v631": "#2C7A7B", "v643": "#B8860B",
            "v648": "#8E44AD", "v651": "#1F4E79"}
    fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=180)
    got = 0
    for v in ["v601", "v631", "v643", "v648", "v651"]:
        p = f"report_figs/pruning_scores_0904/{v}.npz"
        if not os.path.exists(p):
            continue
        z = np.load(p)
        s = z["edge_scores"]; y = z["edge_labels"].astype(bool)
        order = np.argsort(-s)
        y = y[order]
        tp = np.cumsum(y); fp = np.cumsum(~y)
        tpr = tp / max(tp[-1], 1); fpr = fp / max(fp[-1], 1)
        auc = np.trapezoid(tpr, fpr)
        keep = fpr > 0
        ax.plot(fpr[keep], tpr[keep], lw=2.2, color=cols[v], label=f"{NAME_ROC.get(v, NAME[v])}   AUC {auc:.3f}")
        got += 1
    if got < 2:
        plt.close(fig); return None
    ax.set_xscale("log"); ax.set_xlim(4e-4, 1); ax.set_ylim(0, 1.02)
    ax.axvline(5e-3, color="#999999", ls="--", lw=1.2)
    ax.text(5.6e-3, 0.12, "operating region", rotation=90, fontsize=10, color="#777777")
    ax.set_xlabel("false-positive rate (log scale)", fontsize=12)
    ax.set_ylabel("true-positive rate", fontsize=12)
    ax.legend(frameon=False, fontsize=11, loc="lower right", title="AUC (200-event subset)")
    fig.tight_layout(); return save(fig, "roc", 7.0, 4.6)


P_PV, P_DECOMP, P_SCAT = fig_pv(), fig_decomp(), fig_scatter()
P_LADDER, P_DELTA, P_NEG, P_DOCA = fig_ladder(), fig_delta(), fig_neg(), fig_doca()
P_DOCAFIX, P_THR, P_RANKEFF = fig_doca_fix(), fig_threshold(), fig_rank_effect()
P_RANK, P_BIAS, P_WORK = fig_rank(), fig_bias(), fig_workpoint()
P_ROC = fig_roc()

INKH, GREENH, REDH = "#2B2B2B", "#2E8B57", "#C0392B"
GREENBG, REDBG = (232, 245, 233, 255), (253, 236, 234, 255)

P_EQ_DOCA_C, W_DOCA_C, H_DOCA_C = eq_block("doca_correct", [
    (r"\min\,|w+t_1u_1-t_2u_2|^2", INKH, None),
    (r"u_1=p_1/|p_1|,\quad u_2=p_2/|p_2|", INKH, None),
    (r"w=A_1-A_2", INKH, None),
    (r"c=u_1\cdot u_2,\quad d=w\cdot u_1,\quad e=w\cdot u_2", INKH, None),
    (r"t_1=\frac{c\,e-d}{1-c^2}", GREENH, GREENBG),
    (r"t_2=\frac{e-c\,d}{1-c^2}", INKH, None),
    (r"\mathrm{DOCA}=|w+t_1u_1-t_2u_2|", INKH, None),
], size=20)

P_EQ_DOCA_W, W_DOCA_W, H_DOCA_W = eq_block("doca_wrong", [
    (r"\min\,|w+t_1u_1-t_2u_2|^2", INKH, None),
    (r"u_1=p_1/|p_1|,\quad u_2=p_2/|p_2|", INKH, None),
    (r"w=A_1-A_2", INKH, None),
    (r"c=u_1\cdot u_2,\quad d=w\cdot u_1,\quad e=w\cdot u_2", INKH, None),
    (r"t_1=-\frac{d+c\,e}{1-c^2}", REDH, REDBG),
    (r"t_2=\frac{e-c\,d}{1-c^2}", INKH, None),
    (r"\mathrm{DOCA}=|w+t_1u_1-t_2u_2|", INKH, None),
], size=20)

P_EQ_M2, W_EQ_M2, H_EQ_M2 = eq_block("m2", [
    (r"d_3=|r_i-r_j|,\qquad A_e=\exp(-d_3/50)", INKH, None),
    (r"r_t=\frac{A_e}{\max_{f\in C_t}A_f}", INKH, None),
    (r"d_t=\frac{\mathrm{count}\{f\in C_t:\ A_f>0.5\}}{10}", INKH, None),
    (r"c_t=\frac{|C_t|}{50}", INKH, None),
    (r"z_e=[\min(r_i,r_j),\ \max(r_i,r_j),\ \min(d_i,d_j),\ \max(d_i,d_j),", INKH, None),
    (r"\qquad\ \min(c_i,c_j),\ \max(c_i,c_j)]", INKH, None),
], size=17)

P_EQ_LOSS, W_EQ_LOSS, H_EQ_LOSS = eq_block("loss", [
    (r"L_{\mathrm{rank}}=\mathrm{softplus}\left(-\frac{s_{\mathrm{truth}}-s_{\mathrm{fake}}}{\tau}\right)", INKH, None),
    (r"L_{\mathrm{total}}=L_{\mathrm{BCE}}+\lambda\,L_{\mathrm{rank}},\qquad \lambda=10", "#1F4E79", None),
], size=20)

P_EQ_ATTN, W_EQ_ATTN, H_EQ_ATTN = eq_block("attn", [
    (r"h_e\in\mathbb{R}^{16}", INKH, None),
    (r"q_e=W_qh_e,\quad k_f=W_kh_f,\quad v_f=W_vh_f\quad\in\mathbb{R}^{8}", INKH, None),
    (r"s_{ef}=\frac{q_e\cdot k_f}{\sqrt{8}}+b_{ef}", INKH, None),
    (r"\alpha_{ef}=\frac{\exp(s_{ef})}{\sum_{f\in N(e)}\exp(s_{ef})}", INKH, None),
    (r"c_e=\sum_{f\in N(e)}\alpha_{ef}v_f", INKH, None),
    (r"h'_e=h_e+W_o\left[c_e^{(1)};\ldots;c_e^{(4)}\right]", INKH, None),
], size=16)

P_EQ_BIAS, W_EQ_BIAS, H_EQ_BIAS = eq_block("bias", [
    (r"s_{ef}=\frac{q_e\cdot k_f}{\sqrt{8}}+b_{ef}", INKH, None),
], size=20)

P_EQ_BDEF, W_EQ_BDEF, H_EQ_BDEF = eq_block("bdef", [
    (r"b_e=(\Delta R,\ \Delta z,\ \mathrm{DOCA})", INKH, None),
    (r"b_{ef}=\mathrm{MLP}\left([b_e,\ b_f,\ |b_e-b_f|,\ b_e\cdot b_f]\right)", INKH, None),
], size=14)

P_EQ_EX, W_EQ_EX, H_EQ_EX = eq_block("example", [
    (r"c_1=0.63\,v_2+0.26\,v_3+0.11\,v_4", INKH, None),
    (r"h'_1=h_1+W_o(c_1)", INKH, None),
], size=18)

P_EQ_Z, W_EQ_Z, H_EQ_Z = eq_block("zvec", [
    (r"z_e=[\min(r_i,r_j),\ \max(r_i,r_j),\ \min(d_i,d_j),\ \max(d_i,d_j),", INKH, None),
    (r"\qquad\ \min(c_i,c_j),\ \max(c_i,c_j)]", INKH, None),
], size=19)

P_EQ_A, W_EQ_A, H_EQ_A = eq_block("affinity", [
    (r"d_3=|r_i-r_j|,\qquad A_e=\exp(-d_3/50)", INKH, None),
], size=22)

P_EQ_R, W_EQ_R, H_EQ_R = eq_block("r", [
    (r"r_t=\frac{A_e}{\max_{f\in C(t)}A_f}", INKH, None),
], size=22)

P_EQ_D, W_EQ_D, H_EQ_D = eq_block("d", [
    (r"d_t=\frac{\mathrm{count}\{f\in C(t):\ A_f>0.5\}}{10}", INKH, None),
], size=19)

P_EQ_C, W_EQ_C, H_EQ_C = eq_block("c", [
    (r"c_t=\frac{|C(t)|}{50}", INKH, None),
], size=22)

print("eq sizes:",
      round(W_DOCA_C, 2), round(H_DOCA_C, 2),
      round(W_EQ_M2, 2), round(H_EQ_M2, 2),
      round(W_EQ_ATTN, 2), round(H_EQ_ATTN, 2))

prs = Presentation(); prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
W, H, MX = 13.333, 7.5, 0.6

BODY = 25      # body text size, per request
SMALL = 15


def _rgb(v):
    if isinstance(v, str):
        v = v.lstrip("#"); return RGBColor(int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16))
    return v


def _lines(text, width, size=17, factor=144):
    cpl = max(18, int(width * factor / size))
    n = 0
    for part in str(text).split("\n"):
        n += max(1, -(-len(part) // cpl))
    return n


def tb(s, x, y, w, h, text, size=BODY, bold=False, color=INK, align=PP_ALIGN.LEFT,
       font="Calibri", italic=False, line=1.0, anchor=MSO_ANCHOR.TOP):
    box = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, m, 0)
    for i, ln in enumerate(str(text).split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align; p.line_spacing = line
        r = p.add_run(); r.text = ln
        r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = color
        r.font.name = font; r.font.italic = italic
    return box


def rect(s, x, y, w, h, fill, line=None, shape=MSO_SHAPE.RECTANGLE, radius=None, lw=1.5):
    sh = s.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb = _rgb(fill)
    if line is None: sh.line.fill.background()
    else:
        sh.line.color.rgb = _rgb(line); sh.line.width = Pt(lw)
    sh.shadow.inherit = False
    if radius is not None and shape == MSO_SHAPE.ROUNDED_RECTANGLE: sh.adjustments[0] = radius
    return sh


def panel(s, x, y, w, h, fill=WHITE, line=BORDER, lw=1.2):
    return rect(s, x, y, w, h, fill, line=line, shape=MSO_SHAPE.ROUNDED_RECTANGLE,
                radius=0.035, lw=lw)


def bullets(s, x, y, w, h, items, size=BODY, color=INK, gap=12):
    box = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame; tf.word_wrap = True
    for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, m, 0)
    for i, it in enumerate(items):
        assert len(it) <= 78, f"bullet too long for {size}pt: {it}"
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(gap)
        r = p.add_run(); r.text = "—  " + it
        r.font.size = Pt(size); r.font.color.rgb = color; r.font.name = "Calibri"
    return box


def formula(s, x, y, w, h, lines, size=19):
    tb(s, x, y, w, h, lines, size, False, NAVY, font="Consolas", line=1.35)


def _section_color(tag):
    t = tag.upper()
    if t.startswith("SUMMARY"): return GREEN
    if t.startswith("PART 1"): return NAVY
    if t.startswith("PART 2"): return MAGENTA
    if t.startswith("PART 3"): return GOLD
    if t.startswith("PART 3"): return PURPLE
    if t.startswith("PIPELINE"): return TEAL
    if t.startswith("BACKUP"): return GREY
    return NAVY


def header(s, tag, title, sub=None, chip=None):
    """Colour-tab header, matching DFEI_meeting_20260915_group_EN.pptx."""
    color = _section_color(tag) if tag else NAVY
    pw = 0.60 + 0.25 * len(tag)
    if tag:
        rect(s, 0.42, 0.16, pw, 0.50, color, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.18)
        tb(s, 0.42, 0.23, pw, 0.36, tag, 19, True, WHITE, align=PP_ALIGN.CENTER)
    tx = 0.42 + (pw + 0.20 if tag else 0.13)
    size = 34 if len(title) <= 34 else (30 if len(title) <= 46 else 26)
    tb(s, tx, 0.20, 12.9 - tx, 0.52, title, size, True, NAVY)
    rect(s, 0.55, 1.10, 12.3, 0.04, NAVY)
    if sub: tb(s, 0.55, 1.16, 12.3, 0.34, sub, 24, False, GREY)


def footer(s, idx=None):
    tb(s, 0.55, 7.12, 12.3, 0.30,
       "Qingxiang Guo · UCAS · DFEI group meeting · 2026-10-30          "
       + str(idx if idx is not None else len(prs.slides)),
       13, False, GREY)


def pic(s, path, x, y, w=None, h=None):
    if w: s.shapes.add_picture(path, Inches(x), Inches(y), width=Inches(w))
    else: s.shapes.add_picture(path, Inches(x), Inches(y), height=Inches(h))


def note(s, text, y=6.35, size=16, *extra):
    """Plain grey note, as under the figures in the group-meeting deck."""
    if isinstance(y, str):
        text = text + " " + y
        y, size = size, (extra[0] if extra else 16)
    h = _lines(text, 12.2, size) * size * 1.25 / 72
    tb(s, MX, y, 12.2, h + 0.05, text, size, False, GREY, italic=True, line=1.25)


# ======================= DECK BODY v5 · large type =======================
# Rule: body text >= 24 pt; one figure/formula + one short paragraph per page.

FIGS5 = "report_figs/deck_20261023"
P_GEO = f"{FIGS5}/fig_geo.png"
P_RANKPAIR = f"{FIGS5}/fig_rank_pair.png"
P_CTX = f"{FIGS5}/fig_ctx.png"
P_LINE = f"{FIGS5}/fig_line.png"
P_LINEGRAPH = f"{FIGS5}/fig_linegraph.png"
P_ATTN_FIG = f"{FIGS5}/attn_vaswani.png"
P_EQ_SMALL, W_EQ_SMALL, H_EQ_SMALL = eq_block("small", [
    (r"\tilde{s}_e=s_e+b_{\mathrm{evt}}", INKH, None),
    (r"f_{ij}=\min(\Delta z_{ij}\,\mathrm{IP}_{ij},\ \Delta z_{ji}\,\mathrm{IP}_{ji})", "#1F4E79", None)], size=24)

P_DIST_WP = f"{FIGS5}/dist_wp.png"
P_DIST_RANK = f"{FIGS5}/dist_rank.png"
P_DIST_CTX = f"{FIGS5}/dist_ctx.png"
P_DIST_ATT = f"{FIGS5}/dist_att.png"

P_EQ_PV5, W_EQ_PV5, H_EQ_PV5 = eq_block("pv5", [
    (r"L_{\mathrm{PV}}=-\log p(\mathrm{true\ PV})", INKH, None)], size=26)
P_EQ_DOCA_DEF, W_EQ_DOCA_DEF, H_EQ_DOCA_DEF = eq_block("doca_def", [
    (r"|\mathbf{w}+t_1\mathbf{u}_1-t_2\mathbf{u}_2|,\quad c=\mathbf{u}_1\cdot\mathbf{u}_2,\ d=\mathbf{w}\cdot\mathbf{u}_1,\ e=\mathbf{w}\cdot\mathbf{u}_2", INKH, None)], size=24)
P_EQ_DOCA_BAD, W_EQ_DOCA_BAD, H_EQ_DOCA_BAD = eq_block("doca_bad", [
    (r"t_1=-(d+ce)/(1-c^2),\qquad t_2=(e-cd)/(1-c^2)", "#B03A2E", None)], size=28)
P_EQ_DOCA_OK, W_EQ_DOCA_OK, H_EQ_DOCA_OK = eq_block("doca_ok", [
    (r"t_1=(ce-d)/(1-c^2),\qquad t_2=(e-cd)/(1-c^2)", "#1F4E79", None)], size=28)
P_PROGRESS = f"{FIGS5}/progress.png"
P_DIST_VERT = f"{FIGS5}/dist_vert.png"
P_VERT_EFF = f"{FIGS5}/fig_vertex_eff.png"
P_DOCA_OK = f"{FIGS5}/doca_correct.png"
P_DOCA_BAD = f"{FIGS5}/doca_wrong.png"
P_EQ_RANK5, W_EQ_RANK5, H_EQ_RANK5 = eq_block("rank5", [
    (r"s_e=\mathrm{logit}(e)", INKH, None),
    (r"L_{\mathrm{sep}}=\left\langle\max(0,\ m+s_{\mathrm{bkg}}-s_{\mathrm{sig}})\right\rangle", INKH, None),
    (r"L=L_{\mathrm{BCE}}+\lambda\,L_{\mathrm{sep}}", "#1F4E79", None)], size=30)
P_EQ_CTX5, W_EQ_CTX5, H_EQ_CTX5 = eq_block("ctx5", [
    (r"A=\exp(-d_3/\tau)", INKH, None),
    (r"r_t=\frac{A_e}{\max_{f\in C(t)}A_f}", INKH, None),
    (r"d_t=\frac{\#\{f\in C(t):A_f>\vartheta\}}{10},\quad c_t=\frac{|C(t)|}{50}", INKH, None)], size=24)
P_EQ_ATTN2, W_EQ_ATTN2, H_EQ_ATTN2 = eq_block("attn2", [
    (r"\alpha_{ef}=\frac{e^{s_{ef}}}{\sum_{g\in N(e)}e^{s_{eg}}}", INKH, None),
    (r"h'_e=h_e+W_o\,\mathrm{concat}_{c=1}^{H}\sum_{f}\alpha^{c}_{ef}V^{c}_{f}", "#1F4E79", None)], size=26)
P_EQ_CKPT1, W_EQ_CKPT1, H_EQ_CKPT1 = eq_block("ckpt1", [
    (r"\hat{\theta}=\arg\max_{\theta}\ \mathrm{AP}_{\mathrm{val}}(\theta)", INKH, None)], size=30)
P_EQ_CKPT, W_EQ_CKPT, H_EQ_CKPT = eq_block("ckpt", [
    (r"\hat{\theta}=\arg\max_{\theta}\ \mathrm{AP}_{\mathrm{val}}(\theta)", INKH, None),
    (r"\tilde{s}_e=s_e+b_{\mathrm{evt}}", "#1F4E79", None)], size=26)
P_EQ_OCC, W_EQ_OCC, H_EQ_OCC = eq_block("occ", [
    (r"b_{\mathrm{evt}}=w\cdot\bar{h}_{\mathrm{evt}}+c", INKH, None),
    (r"\tilde{s}_e=s_e+b_{\mathrm{evt}}", "#1F4E79", None)], size=26)
P_EQ_ATTN5, W_EQ_ATTN5, H_EQ_ATTN5 = eq_block("attn5", [
    (r"s_{ef}=\frac{q_e\cdot k_f}{\sqrt{8}}+b_{ef}", INKH, None)], size=26)


def fig_pairbias():
    """Schematic: two pair-nodes sharing a track, and the geometric bias they exchange."""
    from matplotlib.patches import FancyBboxPatch, Circle
    fig, ax = plt.subplots(figsize=(6.6, 3.2), dpi=180)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.2); ax.axis("off"); ax.grid(False)
    trk = {"i": (1.0, 4.15), "j": (5.0, 4.15), "k": (9.0, 4.15)}
    for name, (x, y) in trk.items():
        ax.add_patch(Circle((x, y), 0.42, fc="#DCE6F1", ec=C["p"], lw=2, zorder=3))
        ax.text(x, y, name, ha="center", va="center", fontsize=14, color=C["p"], zorder=4)
    for key, (x, lab) in {"e": (1.55, "(i, j)"), "f": (8.45, "(j, k)")}.items():
        ax.add_patch(FancyBboxPatch((x - 1.00, 1.58), 2.00, 0.84,
                                    boxstyle="round,pad=0.06", fc="#F5F8FA",
                                    ec=C["s"], lw=2, zorder=3))
        ax.text(x, 2.20, f"pair {lab}", ha="center", va="center", fontsize=11.5,
                color=C["p"], zorder=4)
        ax.text(x, 1.83, rf"$b_{{{key}}}$", ha="center", va="center", fontsize=13,
                color=C["s"], zorder=4)
    for (px, targets) in [(1.55, ["i", "j"]), (8.45, ["j", "k"])]:
        for t in targets:
            tx, ty = trk[t]
            ax.plot([px, tx], [2.46, ty - 0.42], color="#9AA3B2", lw=1.4, zorder=2)
    ax.annotate("", xy=(7.30, 2.00), xytext=(2.70, 2.00),
                arrowprops=dict(arrowstyle="-|>", color=C["s"], lw=2.2, ls="--"))
    ax.text(5.0, 2.30, "share track j: neighbours", ha="center",
            fontsize=11.5, color=C["s"])
    ax.text(5.0, 0.55, r"attention logit  $s_{ef}=q_e\cdot k_f/\sqrt{d}+b_{ef}$",
            ha="center", fontsize=12, color=C["p"])
    fig.tight_layout(); return save(fig, "pairbias", 6.6, 3.2)


P_PAIRBIAS = fig_pairbias()
P_SAT = f"{FIGS5}/sat.png"
P_EQ_PB, W_EQ_PB, H_EQ_PB = eq_block("pb", [
    (r"b_{ef}=\mathrm{MLP}([\,b_e,\ b_f,\ |b_e-b_f|,\ b_e\odot b_f\,])", INKH, None),
    (r"s_{ef}=q_e\cdot k_f/\sqrt{d}+b_{ef}", "#1F4E79", None)], size=24)


def _p(s, x, y, w, text, size=26, h=3.6, color=INK):
    tb(s, x, y, w, h, text, size, False, color, line=1.30)


def _c(s, text, size=26):
    tb(s, 0.55, 6.45, 12.25, 0.55, text, size, True, NAVY)
    footer(s)


def _tbl(s, x, y, w, rows, rel, fs=22, rowh=0.62):
    n_r, n_c = len(rows), len(rows[0])
    t = s.shapes.add_table(n_r, n_c, Inches(x), Inches(y), Inches(w), Inches(rowh * n_r)).table
    for i, r in enumerate(rel):
        t.columns[i].width = Inches(w * r)
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cell = t.cell(r, c); cell.text = val; cell.fill.solid()
            cell.fill.fore_color.rgb = NAVY if r == 0 else (WHITE if r % 2 else PALE)
            for p in cell.text_frame.paragraphs:
                p.alignment = PP_ALIGN.CENTER if (c == 0 or c == n_c - 1) else PP_ALIGN.LEFT
                for run in p.runs:
                    run.font.size = Pt(fs); run.font.bold = (r == 0)
                    run.font.name = "Calibri"
                    run.font.color.rgb = WHITE if r == 0 else INK
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE


# ---------------- 1 cover ----------------
s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
tb(s, 0.85, 2.05, 11.7, 1.90, "DFEI: bug fixes and\npruning optimization", 44, True, NAVY, line=1.18)
tb(s, 0.85, 4.05, 11.6, 1.0, "0904 sample · measured one change at a time", 26, False, INK, line=1.22)
tb(s, 0.85, 6.10, 11.6, 0.5, "Qingxiang Guo · UCAS · 30 October 2026", 22, False, GREY)
footer(s)

# ---------------- 2 contents ----------------
s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "CONTENTS", "Three parts")
rows = [("PART 1", "PV association", NAVY),
        ("PART 2", "the DOCA fix", MAGENTA),
        ("PART 3", "pruning optimization", GOLD),
        ("SUMMARY", "results and next steps", GREEN)]
y = 1.85
for tag, desc, col in rows:
    rect(s, 0.80, y, 2.10, 0.62, col, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.18)
    tb(s, 0.80, y + 0.10, 2.10, 0.42, tag, 20, True, WHITE, align=PP_ALIGN.CENTER)
    tb(s, 3.20, y + 0.04, 9.3, 0.55, desc, 28, False, col)
    y += 1.05
footer(s)

# ---------------- 3 PV association ----------------
s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 1", "PV association")
pic(s, P_PV, 1.10, 1.32, w=11.1)
bullets(s, 0.85, 5.10, 11.7, 1.7, [
    "y: PV misassociation rate [%], 0904 sample",
    "old four models: 13.6% (mean)",
    "retrained: 8.6%",
    "minIP (IP to closest PV): 10.15%"], size=23, gap=5)

# ---------------- 4 PV head ----------------
s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 1", "PV-aware head")
tb(s, 0.75, 1.22, 5.6, 0.40, "p(PV | track)", 22, True, PURPLE)
for yy, lab, col in [(2.05, "PV1 (true)", GREEN), (2.95, "PV2", GREY), (3.85, "PV3", GREY)]:
    rect(s, 0.90, yy - 0.28, 1.85, 0.56, PALE, line=col, lw=2.0,
         shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.15)
    tb(s, 0.98, yy - 0.16, 1.70, 0.34, lab, 18, True, col, align=PP_ALIGN.CENTER)
trk = (4.55, 2.95)
rect(s, trk[0] - 0.45, trk[1] - 0.45, 0.90, 0.90, NAVY, shape=MSO_SHAPE.OVAL)
tb(s, trk[0] - 0.45, trk[1] - 0.19, 0.90, 0.36, "track", 18, True, WHITE, align=PP_ALIGN.CENTER)
for yy, prob, col, dash in [(2.05, "0.85", GREEN, None), (2.95, "0.10", GREY, 4), (3.85, "0.05", GREY, 4)]:
    ln = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(trk[0] - 0.45), Inches(trk[1]),
                                Inches(2.80), Inches(yy))
    ln.line.color.rgb = _rgb(col); ln.line.width = Pt(3.0 if prob == "0.85" else 1.6)
    if dash is not None:
        ln.line.dash_style = dash
    lab = tb(s, 3.35, yy - 0.30, 0.85, 0.32, prob, 18, True, col, align=PP_ALIGN.CENTER)
    lab.fill.solid(); lab.fill.fore_color.rgb = WHITE
pic(s, P_EQ_PV5, 0.75, 4.70, w=min(W_EQ_PV5, 4.60))
bullets(s, 7.30, 1.55, 5.40, 3.6, [
    "PV = primary vertex",
    "auxiliary head: p(PV | track)",
    "shares the trunk",
    "dropped at inference"], size=24, gap=12)

# ---------------- PART 2: the DOCA bug ----------------
s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 2", "The DOCA bug")
tb(s, 0.55, 1.42, 5.95, 0.36, "CORRECT: closest approach", 22, True, GREEN)
tb(s, 6.95, 1.42, 5.95, 0.36, "WHAT THE CODE DID", 22, True, RED)
pic(s, P_DOCA_OK, 0.55, 1.88, w=5.95)
pic(s, P_DOCA_BAD, 6.95, 1.88, w=5.95)
tb(s, 0.55, 5.25, 12.30, 0.50,
   "Before the fix, the stored DOCA correlated with the truth at 0.02 — effectively noise.",
   20, False, RED, line=1.20)

s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 2", "Result: the DOCA fix",
       "the score is compared before and after the fix, at the same cut")
_tbl(s, 0.65, 1.64, 6.10,
     [["", "before", "after"],
      ["pruned away", "52.0", "47.5"],
      ["merged with background", "23.9", "26.5"],
      ["AllParticles", "24.1", "26.0"],
      ["PerfectReco", "15.9", "17.8"]], [0.46, 0.27, 0.27], fs=22, rowh=0.58)
pic(s, P_DIST_VERT, 7.05, 1.64, w=5.60)

# ---- timeline: this step is number four of six, not the first ----
tb(s, 0.60, 4.92, 12.10, 0.34,
   "Where this step sits in the chain — Part 3 continues from the value it produces",
   17, False, GREY)
_chips = [("baseline", "14.5"), ("working point", "17.9"), ("separation", "24.1"),
          ("DOCA fix", "26.0"), ("edge context", "27.0"), ("attention", "28.0")]
_cw, _gap, _y = 1.78, 0.28, 5.38
for _i, (_lab, _val) in enumerate(_chips):
    _x = 0.60 + _i * (_cw + _gap)
    _hit = _lab == "DOCA fix"
    rect(s, _x, _y, _cw, 0.52, MAGENTA if _hit else PALE,
         line=MAGENTA if _hit else RGBColor(0xC9, 0xD0, 0xDD),
         shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.14, lw=2.0 if _hit else 1.2)
    tb(s, _x, _y + 0.08, _cw, 0.36, _lab, 15, True,
       WHITE if _hit else INK, align=PP_ALIGN.CENTER)
    tb(s, _x, _y + 0.58, _cw, 0.30, _val, 15, _hit,
       MAGENTA if _hit else GREY, align=PP_ALIGN.CENTER)
    if _i < len(_chips) - 1:
        tb(s, _x + _cw - 0.02, _y + 0.10, _gap + 0.04, 0.32, "→", 14, True,
           GREY, align=PP_ALIGN.CENTER)

# ---------------- 7 0702 saturation ----------------
s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 3", "GNN side: saturation", "AllParticles, 0702 sample (N = 17 561)")
bullets(s, 0.65, 1.62, 6.20, 3.9, [
    "nine physics heads",
    "wider layers, attention",
    "event restructuring",
    "22.5 → 36.7%",
    "then: no further gain"], size=26, gap=13)
pic(s, P_SAT, 7.05, 1.62, w=5.70)

# ---------------- 8 agenda ----------------
s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 3", "What we changed")

_items = [("1", "Working point", "pre-fix"),
          ("2", "Separation loss", "pre-fix"),
          ("3", "Edge-context inputs", "post-fix"),
          ("4", "Attention", "post-fix")]
y = 1.55
for n, name, when in _items:
    tb(s, 0.65, y, 0.45, 0.40, n, 22, True, GREY)
    tb(s, 1.15, y, 3.15, 0.40, name, 22, False, INK)
    tb(s, 4.32, y + 0.07, 1.00, 0.30, when, 13, False, GREY)
    y += 0.68

# ---- pipeline: scoring and cutting both live inside the GN blocks ----
rect(s, 5.45, 1.50, 6.40, 2.42, WHITE, line=NAVY, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.05)
tb(s, 5.68, 1.57, 5.90, 0.34, "GNN: four blocks", 19, True, GREY)
_rows = [("node score", "node pruning", 2.02), ("edge score", "edge pruning", 2.90)]
for _s, _p, _y in _rows:
    rect(s, 5.68, _y, 2.55, 0.62, RGBColor(0xE3, 0xF2, 0xE3),
         line=GREEN, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.12)
    tb(s, 5.68, _y + 0.12, 2.55, 0.38, _s, 19, True, NAVY, align=PP_ALIGN.CENTER)
    tb(s, 8.20, _y + 0.12, 0.42, 0.36, "→", 17, True, GREY, align=PP_ALIGN.CENTER)
    rect(s, 8.62, _y, 2.55, 0.62, WHITE, line=GREEN, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.12)
    tb(s, 8.62, _y + 0.12, 2.55, 0.38, _p, 19, True, NAVY, align=PP_ALIGN.CENTER)

_flow = [("LCAG 0–3", 4.25), ("chains", 5.20)]
for k, (t, _y) in enumerate(_flow):
    rect(s, 6.55, _y, 4.20, 0.64, WHITE, line=NAVY, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.12)
    tb(s, 6.55, _y + 0.13, 4.20, 0.38, t, 21, True, NAVY, align=PP_ALIGN.CENTER)
    if k < len(_flow) - 1:
        tb(s, 8.45, _y + 0.62, 0.40, 0.32, "↓", 18, True, GREY, align=PP_ALIGN.CENTER)
tb(s, 8.45, 3.92, 0.40, 0.32, "↓", 18, True, GREY, align=PP_ALIGN.CENTER)

# ---------------- 9 ledger ----------------
s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 3", "From 14.5 to 28.0%",
       "AllParticles: exact chains / 13 255   ·   PerfectReco: no fakes")
pic(s, P_DECOMP, 0.75, 1.58, w=11.60)

# ---------------- 10 working point ----------------
s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 3", "Working point")
_tbl(s, 0.65, 1.45, 6.20,
     [["", "0.90", "0.95", "0.97"],
      ["remaining bkg", "14 081", "7 222", "4 223"],
      ["signal efficiency", "69.0%", "62.7%", "56.6%"],
      ["purity", "5.8%", "9.8%", "14.3%"],
      ["AllParticles", "14.5", "17.9", "18.5"],
      ["PerfectReco", "9.5", "11.7", "11.9"]], [0.34, 0.22, 0.22, 0.22], fs=21, rowh=0.60)
pic(s, P_WP_COMBO, 6.95, 1.22, w=5.75)

# ---------------- 13 ranking ----------------
s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 3", "Separation loss")
pic(s, P_EQ_RANK5, 0.55, 1.22, w=min(W_EQ_RANK5, 6.35))
pic(s, P_RANKPAIR, 7.05, 1.20, w=5.60)
bullets(s, 0.55, 3.95, 6.10, 2.8, [
    "logit: the score before the sigmoid",
    "cut c = 0.95 ⇔ logit 2.9",
    "L_BCE: pair cross-entropy"], size=23, gap=12)
bullets(s, 6.95, 3.95, 5.70, 2.9, [
    "all true pairs × k hardest background pairs",
    "mean over ≤ k × n_sig pairs, then over events",
    "per pair: zero once the gap ≥ m",
    "current: λ = 10, m = 1, k = 64"], size=23, gap=11)

s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 3", "Result: separation loss")
_tbl(s, 0.65, 1.55, 6.10,
     [["", "before", "after"],
      ["pruned away", "52.9", "52.0"],
      ["merged with background", "29.2", "23.9"],
      ["AllParticles", "17.9", "24.1"],
      ["PerfectReco", "11.7", "15.9"]
], [0.46, 0.27, 0.27], fs=22, rowh=0.52)
pic(s, P_DIST_RANK, 7.05, 1.55, w=5.60)


# ---------------- 14 context ----------------
s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 3", "Edge-context inputs")
pic(s, P_CTX, 0.55, 1.22, w=6.10)
pic(s, P_EQ_CTX5, 0.55, 4.72, w=6.05)
bullets(s, 6.95, 1.35, 5.75, 5.3, [
    "d₃: 3D vertex separation [mm]",
    "τ = 50 mm: decay length inside A",
    "ϑ = 0.5: what counts as a partner",
    "C(t): candidate partners of track t",
    "r_t: score relative to its best partner in C(t)",
    "d_t: near-equal alternatives",
    "c_t: number of partners",
    "sees the pairs at both ends",
    "3 per end × 2 ends = six numbers"], size=22, gap=9)

s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 3", "Result: edge-context inputs")
_tbl(s, 0.65, 1.55, 6.10,
     [["", "before", "after"],
      ["pruned away", "47.5", "52.1"],
      ["merged with background", "26.5", "20.9"],
      ["AllParticles", "26.0", "27.0"],
      ["PerfectReco", "17.8", "18.1"]
], [0.46, 0.27, 0.27], fs=22, rowh=0.52)
pic(s, P_DIST_CTX, 7.05, 1.55, w=5.60)


# ---------------- 15 attention ----------------
s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 3", "From graph to line graph")
pic(s, P_LINE, 0.55, 1.35, w=7.55)
bullets(s, 8.35, 1.55, 4.45, 4.2, [
    "pair = node",
    "adjacent: share a track",
    "no cross-event messages",
    "neighbour cap k = 32",
    "cost linear in pairs"], size=24, gap=12)

s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 3", "Only the attention block changes")
tb(s, 0.85, 1.35, 4.60, 0.40, "before", 24, True, GREY)
tb(s, 7.95, 1.35, 4.60, 0.40, "after", 24, True, GREY)

def _flow(x, y, text, fill, line, fs=18, h=0.62):
    rect(s, x, y, 4.60, h, fill, line=line, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.14)
    tb(s, x, y + 0.12, 4.60, 0.38, text, fs, True, NAVY, align=PP_ALIGN.CENTER)

_flow(0.85, 1.90, "edge vector h_e  (16-d)", WHITE, NAVY)
_flow(0.85, 2.86, "pruning MLP", WHITE, NAVY)
_flow(0.85, 3.82, "score", WHITE, NAVY)

_flow(7.95, 1.90, "edge vector h_e  (16-d)", WHITE, NAVY)
_flow(7.95, 2.86, "line-graph attention  H = 4, d = 8", GREEN, GREEN)
_flow(7.95, 3.82, "h_e + Δ  (attention update)", WHITE, NAVY)
_flow(7.95, 4.78, "pruning MLP", WHITE, NAVY)
_flow(7.95, 5.74, "score", WHITE, NAVY)

for x in (3.15, 10.25):
    for y in ([2.52, 3.48] if x < 5 else [2.52, 3.48, 4.44, 5.40]):
        tb(s, x, y, 0.40, 0.28, "↓", 18, True, GREY, align=PP_ALIGN.CENTER)

# ---------------- 15b geometric pair bias ----------------
s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 3", "Geometry enters the attention weight")
pic(s, P_PAIRBIAS, 0.55, 1.40, w=6.55)
pic(s, P_EQ_PB, 7.30, 1.45, w=min(W_EQ_PB, 5.40))
bullets(s, 7.30, 3.30, 5.40, 3.5, [
    "e, f: two pairs sharing a track",
    "b: DOCA, log DOCA, |Δstart|",
    "s_ef: attention logit of e onto f",
    "b_ef: one scalar per head, added to s_ef",
    "H = 4 heads, d = 8 per head",
    "symmetric under e ↔ f",
    "zero-init: starts at 0"], size=21, gap=7)

s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 3", "Result: attention")
_tbl(s, 0.65, 1.55, 6.10,
     [["", "before", "after"],
      ["pruned away", "52.1", "54.1"],
      ["merged with background", "20.9", "18.0"],
      ["AllParticles", "27.0", "28.0"],
      ["PerfectReco", "18.1", "19.0"]
], [0.46, 0.27, 0.27], fs=22, rowh=0.52)
pic(s, P_DIST_ATT, 7.05, 1.55, w=5.60)


# ---------------- 18 summary ----------------
s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "SUMMARY", "Results and next steps")
bullets(s, 0.85, 1.60, 11.60, 4.6, [
    "purity bought by pruning more true chains",
    "0.95: less background, lower efficiency",
    "working point is a choice, not a model property",
    "chain: 14.5 → 17.9 → 24.1 → 26.0 → 27.0 → 28.0",
    "AllParticles / PerfectReco: unweighted counts",
    "next: one physics process as benchmark"], size=28, gap=16)

# A1 attention 1
s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "APPENDIX A1", "Scaled dot-product attention", "what we use in each box")
pic(s, P_ATTN_FIG, 0.55, 1.52, h=5.05)
_rows = [("MatMul (bottom)", "Q = h_e (receiver), K = h_f (neighbour)"),
         ("Scale + bias", "q·k / √d, then + b_ef"),
         ("Mask", "same event, share one track"),
         ("SoftMax", "over N(e), cap k = 32"),
         ("MatMul (top)", "Σ α·V → H·d, then + h_e")]
y = 1.50
for name, txt in _rows:
    tb(s, 3.55, y, 2.55, 0.40, name, 21, True, NAVY)
    tb(s, 6.20, y, 6.65, 0.90, txt, 21, False, INK)
    y += 1.00
tb(s, 3.55, 6.70, 9.30, 0.35, "Figure: Vaswani et al., NeurIPS 2017, Fig. 2 (left). Green marks our choices.", 14, False, GREY)

# A1 attention 2
s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "APPENDIX A1", "Attention: the computation", "one round, four heads")
pic(s, P_EQ_ATTN2, 0.65, 1.55, w=min(W_EQ_ATTN2, 6.20))
bullets(s, 7.10, 1.50, 5.60, 5.0, [
    "h_e: pair vector, 16-d, last block",
    "q_e = W_Q h_e, k_f = W_K h_f",
    "s_ef = q_e·k_f / √d + b_ef",
    "H = 4 heads, d = 8 per head",
    "V = W_V h_f",
    "W_o zero-init: starts as identity"], size=23, gap=11)

# A3 metrics 2
if P_ROC:
    s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
    header(s, "APPENDIX A2", "Why edge AUC is not the target", "one cut, not an average")
    pic(s, P_ROC, 0.55, 1.45, w=6.5)
    bullets(s, 7.35, 1.50, 5.35, 5.0, [
        "signal pairs: 0.14% of candidates",
        "one cut decides whether a chain survives",
        "AUC averages all pairs and all cuts",
        "easy background dominates AUC",
        "baseline: best AUC, worst chain yield",
        "x = FPR, y = TPR (pairs above the cut)"], size=19, gap=8)




_all = list(prs.slides)
for _i, _s in enumerate(_all, start=1):
    _has = any(sh.has_text_frame and "DFEI group meeting" in sh.text_frame.text
               for sh in _s.shapes)
    if not _has:
        footer(_s, _i)

prs.save(OUT)
print("saved", OUT, os.path.getsize(OUT))
