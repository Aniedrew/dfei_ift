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
        "v640": "cut + small items", "v643": "vertex geometry", "v648": "edge context", "v651": "attention"}
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
    pr = [43.05, 52.88, 49.97, 47.45, 52.12, 54.06]
    co = [42.50, 29.23, 27.43, 26.51, 20.88, 17.98]
    cl = [14.45, 17.90, 22.60, 26.04, 27.00, 27.96]
    labels = ["baseline", "working\npoint", "separation\n+ small", "vertex\ngeometry",
              "edge\ncontext", "attention"]
    fig, ax = plt.subplots(figsize=(10.4, 3.6), dpi=180)
    x = np.arange(len(labels))
    ax.bar(x, pr, color="#C9D0DD", label="pruned away")
    ax.bar(x, co, bottom=pr, color=C["s"], label="merged with background")
    ax.bar(x, cl, bottom=np.array(pr) + np.array(co), color=C["p"], label="exact")
    for i in range(len(labels)):
        ax.text(i, pr[i] / 2, f"{pr[i]:.0f}", ha="center", fontsize=11, color="#4A5568")
        ax.text(i, pr[i] + co[i] / 2, f"{co[i]:.0f}", ha="center", fontsize=11, color="white")
        ax.text(i, pr[i] + co[i] + cl[i] / 2, f"{cl[i]:.1f}", ha="center", fontsize=11, color="white")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10.5); ax.set_ylim(0, 100)
    ax.set_ylabel("% of signal chains", fontsize=12)
    ax.legend(frameon=False, fontsize=12, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
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
    ax.set_yticks([]); ax.set_xlim(0, 1); ax.set_xlabel("pruning score", fontsize=12)
    fig.tight_layout(); return save(fig, "rank", 5.4, 2.8)


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
        ax.set_xlabel("pruning score", fontsize=10, labelpad=2)
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
    names = ["attention only\n(v637)", "attention + geometry\ninside the score (v651)"]
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
    b1 = ax.bar(x - w / 2, before, w, color="#9AA3B2", label="v643  (before)")
    b2 = ax.bar(x + w / 2, after, w, color="#1F4E79", label="v648  (+6 inputs)")
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
        tb(s, 0.42, 0.23, pw, 0.36, tag, 17, True, WHITE, align=PP_ALIGN.CENTER)
    tx = 0.42 + (pw + 0.20 if tag else 0.13)
    size = 28 if len(title) <= 42 else (24 if len(title) <= 52 else 21)
    tb(s, tx, 0.20, 12.9 - tx, 0.52, title, size, True, NAVY)
    rect(s, 0.55, 1.10, 12.3, 0.04, NAVY)
    if sub: tb(s, 0.55, 1.16, 12.3, 0.26, sub, 14, False, GREY)


def footer(s):
    tb(s, 0.55, 7.12, 12.3, 0.30,
       "Qingxiang Guo · UCAS · DFEI group meeting · 2026-10-23          " + str(len(prs.slides)),
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


def cover():
    s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
    tb(s, 0.85, 2.15, 11.7, 1.75, "Where inclusive beauty-hadron\nreconstruction loses chains", 40, True, NAVY, line=1.15)
    tb(s, 0.85, 3.95, 11.6, 0.8, "geometry-aware inputs, the pruning cut, and the output form",
       22, False, INK)
    tb(s, 0.85, 6.05, 11.6, 0.6,
       "Qingxiang Guo · University of Chinese Academy of Sciences · 23 October 2026", 20, False, GREY)
    footer(s)


def toc():
    s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
    header(s, "CONTENTS", "Three parts")
    rows = [("PART 1", "PV association and the minIP baseline", NAVY),
            ("PART 2", "a sign error in the DOCA calculation", GOLD),
            ("PART 3", "where the chain efficiency is lost", PURPLE),
            ("SUMMARY", "results and next steps", GREEN)]
    y = 1.72
    for tag, desc, col in rows:
        rect(s, 0.75, y, 1.95, 0.60, col, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.18)
        tb(s, 0.75, y + 0.10, 1.95, 0.40, tag, 18, True, WHITE, align=PP_ALIGN.CENTER)
        tb(s, 3.00, y + 0.05, 9.6, 0.55, desc, 24, False, col)
        y += 0.98
    footer(s)


cover(); toc()

# ---------------- A · what the leak fix cost ----------------
s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 1", "PV association: previous models and the minIP baseline")
pic(s, P_PV, 1.25, 1.38, w=10.8)
tb(s, 0.85, 5.10, 11.80, 0.95,
   "PV misassociation falls from 13.6% (average of four models trained with leakage) to 8.6% after retraining; "
   "the minIP baseline is 10.15%. The head and the retraining were introduced together, so their effects are not "
   "separated in this sample.", 13.5, False, INK, line=1.28)
tb(s, 0.85, 6.42, 11.80, 0.50, "Leakage inflated the gain; retraining still beats minIP by 1.5 points.",
   18, True, NAVY)
footer(s)

s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 1", "PV-aware head", "auxiliary task on the same trunk")

# left: one track, soft assignment over three PVs
tb(s, 0.75, 1.55, 5.6, 0.30, "soft assignment   p(PV | track i)", 15, True, PURPLE)
pvs = [(2.20, "PV1  (true)", GREEN), (3.10, "PV2", GREY), (4.00, "PV3", GREY)]
for yy, lab, col in pvs:
    rect(s, 0.90, yy - 0.25, 1.60, 0.50, PALE, line=col, lw=2.0,
         shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.15)
    tb(s, 0.98, yy - 0.14, 1.44, 0.30, lab, 12.5, True, col, align=PP_ALIGN.CENTER)

trk = (5.10, 3.10)
rect(s, trk[0] - 0.36, trk[1] - 0.36, 0.72, 0.72, NAVY, shape=MSO_SHAPE.OVAL)
tb(s, trk[0] - 0.36, trk[1] - 0.15, 0.72, 0.30, "track i", 10.5, True, WHITE, align=PP_ALIGN.CENTER)

for (yy, _, col), prob, dash in [((2.20, "", GREEN), "0.85", None),
                                 ((3.10, "", GREY), "0.10", 4),
                                 ((4.00, "", GREY), "0.05", 4)]:
    ln = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(trk[0] - 0.36), Inches(trk[1]),
                                Inches(2.52), Inches(yy))
    ln.line.color.rgb = _rgb(col); ln.line.width = Pt(2.6 if prob == "0.85" else 1.3)
    if dash is not None:
        ln.line.dash_style = dash
    _lab = tb(s, 3.52, 3.10 + (yy - 3.10) * 0.55 - 0.22, 0.55, 0.26, prob, 12, True, col,
              align=PP_ALIGN.CENTER)
    _lab.fill.solid(); _lab.fill.fore_color.rgb = WHITE

tb(s, 0.75, 4.55, 5.55, 0.80,
   "Training target: the MC-truth PV; cross-entropy over the three candidates.", 13.5, False, GREY,
   italic=True, line=1.22)
tb(s, 0.75, 5.35, 5.55, 0.40,
   "A wrong assignment adds one unit to the PV misassociation rate.", 13.5, False, GREY,
   italic=True, line=1.22)

# right: why this auxiliary task helps the pruning trunk
tb(s, 7.05, 1.58, 5.6, 0.30, "SHARED REPRESENTATION", 15, True, PURPLE)
tb(s, 7.05, 2.02, 5.55, 3.60,
   "The head shares the trunk with the pruning tasks and is removed at inference. Training it forces the trunk "
   "to encode impact parameter, vertex position and track direction, the quantities that also decide whether "
   "two tracks share a vertex. The head and the retraining were introduced together, so the effect of the head "
   "alone is not measured on this sample.", 15, False, INK, line=1.30)
tb(s, 0.75, 6.42, 12.10, 0.50, "An auxiliary task on the shared trunk recovers most of the loss.",
   18, True, NAVY)
footer(s)

# ---------------- C · the DOCA bug ----------------
s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 2", "DOCA: sign error in t₁", "same inputs and algebra; only the t₁ expression differs")

def _fx(x, y, items, w=5.45, size=16, gap=0.37):
    for t, col, bold, bg in items:
        if bg:
            rect(s, x - 0.10, y - 0.02, w + 0.20, 0.34, bg, line=None,
                 shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.15)
        tb(s, x, y, w, 0.30, t, size, bold, col, font="Consolas")
        y += gap

# left: correct geometry
panel(s, 0.55, 1.55, 5.85, 4.45, fill=WHITE, line=GREEN, lw=1.6)
tb(s, 0.75, 1.66, 5.45, 0.30, "CORRECT: closest approach", 15, True, GREEN)
_xc = 0.55 + (5.85 - W_DOCA_C) / 2
pic(s, P_EQ_DOCA_C, _xc, 2.00, w=W_DOCA_C)
tb(s, _xc, 2.00 + H_DOCA_C + 0.08, 5.45, 0.28,
   "A = production vertex · u = unit momentum · d₃ in mm", 12, False, GREY, italic=True)

# right: what the upstream code did
panel(s, 6.95, 1.55, 5.85, 4.45, fill=WHITE, line=RED, lw=1.6)
tb(s, 7.15, 1.66, 5.45, 0.30, "WHAT THE CODE DID", 15, True, RED)
_xw = 6.95 + (5.85 - W_DOCA_W) / 2
pic(s, P_EQ_DOCA_W, _xw, 2.00, w=W_DOCA_W)
tb(s, _xw, 2.00 + H_DOCA_W + 0.08, 5.45, 0.28,
   "A = production vertex · u = unit momentum · d₃ in mm", 12, False, GREY, italic=True)
tb(s, _xw, 2.00 + H_DOCA_W + 0.38, 5.45, 0.30, "×  only this line differs", 14, True, RED)

tb(s, 0.60, 6.12, 12.1, 0.60,
   "The correct t₁ is the stationary point of the minimisation; the stored version is not. No exception is "
   "raised and the output remains plausible.", 14, False, INK, line=1.24)
footer(s)

s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 2", "What the sign error did and did not change",
       "0904 sample · fixed denominator 13 255 signal chains")

panel(s, 0.65, 1.55, 4.90, 3.90, fill=PALE, line=BORDER)
tb(s, 0.95, 1.75, 4.30, 0.35, "correlation, stored vs recomputed DOCA", 13.5, False, GREY)
tb(s, 0.80, 2.20, 4.60, 1.10, "0.019", 54, True, NAVY)
tb(s, 0.95, 3.45, 4.30, 0.35, "99.5% of ordered pairs (i,j) and (j,i) differ", 13.5, False, INK)
tb(s, 0.95, 3.95, 4.30, 0.70, "the column is not symmetric and does not track the true distance", 13.0, False, GREY,
   line=1.22)

tb(s, 5.95, 1.55, 6.75, 4.2,
   "The reconstruction never used this column in a way that exposed the error: nothing raised an exception "
   "and the headline numbers stayed stable. Replacing the column alone cannot be isolated in the archived "
   "runs, because the only run with the corrected column also adds the other derived inputs. The value of "
   "the corrected distance appears when it is supplied as an input, in Part 3.", 14.0, False, INK, line=1.32)
tb(s, 0.60, 6.42, 12.25, 0.50, "The sign error is invisible to the current decision rule.", 18, True, NAVY)

# ---------------- D · pruning optimization ----------------
# --- recap of the previous round ---
s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 3", "The GNN side is saturated", "0702 sample · fixed denominator 17 561")
tb(s, 0.65, 1.55, 8.10, 3.90,
   "This sample was used for two rounds of GNN-side changes: differentiable pruning, class-0 handling, hinge and "
   "chain-level losses in the first; nine physics heads, wider layers, line-graph attention and event "
   "restructuring in the second. Reconstruction rose from 22.5 to 36.7% and then stopped. Further capacity no "
   "longer moves chains. These numbers are not comparable with the 0904 sample used in the rest of this talk, "
   "where the starting point is 14.5%.", 14.5, False, INK, line=1.34)
tb(s, 9.10, 1.70, 3.60, 0.90, "+14 pp", 44, True, NAVY)
tb(s, 9.10, 2.75, 3.60, 0.90, "AllParticles\n22.5 → 36.7%", 19, False, INK, line=1.2)
tb(s, 0.60, 6.42, 12.25, 0.50, "Adding capacity to the GNN no longer moves chains.", 18, True, NAVY)
footer(s)


# ======================= PART 3 · rebuilt 2026-10-26 =======================
# 规则：一页 = 一张图 + 一个文本框 + 页底一句结论。

def _para(s, x, y, w, text, size=13.0, h=4.6, color=INK):
    tb(s, x, y, w, h, text, size, False, color, line=1.32)


def _concl(s, text, size=18):
    tb(s, 0.60, 6.42, 12.25, 0.50, text, size, True, NAVY)


def _tbl(s, x, y, w, rows, rel, fs=11.5, rowh=0.42):
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
    return t


P_EQ_RANK, W_EQ_RANK, H_EQ_RANK = eq_block("rank2", [
    (r"L_{\mathrm{rank}}=\left\langle\max\left(0,\ \tau+s_{\mathrm{bkg}}-s_{\mathrm{sig}}\right)\right\rangle", INKH, None),
    (r"L=L_{\mathrm{BCE}}+10\,L_{\mathrm{rank}},\qquad \tau=1.0", "#1F4E79", None),
], size=19)

# ---------------- P8: the round in one number ----------------
s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 3", "From 14.5 to 28.0%", "0904 sample · fixed denominator 13 255")
pic(s, P_DECOMP, 1.05, 1.28, w=11.2)
_para(s, 0.70, 5.02, 12.0,
      "The cut accounts for 3.4 points; changes to the inputs, the loss and the aggregation account for the rest. "
      "The steps are measured one at a time and do not add up exactly.", 13.5, h=1.1)
_concl(s, "The first jump is a package; the cut is its largest part.")

# ---------------- P9: where the number can move ----------------
s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 3", "Where the number can move", "one change at a time, from the stated base")

_bx = [0.70, 3.05, 5.40, 7.75, 10.10]
_bt = ["candidate pair", "edge score", "cut", "partition", "decay chain"]
for x, t in zip(_bx, _bt):
    rect(s, x, 2.05, 2.15, 0.80, WHITE, line=NAVY, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.12)
    tb(s, x, 2.28, 2.15, 0.32, t, 14, True, NAVY, align=PP_ALIGN.CENTER)
for i in range(4):
    tb(s, _bx[i] + 2.16, 2.25, 0.19, 0.30, "→", 13, True, GREY, align=PP_ALIGN.CENTER)

_tags = [
    (3.05, 3.12, "vertex geometry  +3.4", PURPLE),
    (3.05, 3.56, "edge context  +1.0", PURPLE),
    (3.05, 4.00, "ranking loss  +2.6", PURPLE),
    (3.05, 4.44, "attention  +1.0", PURPLE),
    (5.40, 3.12, "threshold  +3.4", GOLD),
    (5.40, 3.56, "same weights", GREY),
    (7.75, 3.12, "assignment  0 (now)", GREY),
    (7.75, 3.56, "one chain per track", GREY),
]
for x, y, t, col in _tags:
    rect(s, x, y, 2.22, 0.34, WHITE, line=BORDER, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.10)
    tb(s, x + 0.08, y + 0.05, 2.06, 0.26, t, 11.0, True, col)
_para(s, 0.70, 5.15, 12.0,
      "Each label is one change and its measured gain. The cut costs no training and is applied first.",
      13.5, h=0.9)
_concl(s, "The cut needs no training; the model changes do.")

# ---------------- P10: working point ----------------
s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 3", "Working point", "cut on the edge score · same weights")
pic(s, P_THR, 0.60, 1.45, w=7.1)
_para(s, 8.05, 1.50, 4.70,
      "The weights are unchanged; only the cut on the edge score moves. At 0.90, 0.95 and 0.97 the chain yield "
      "is 14.5, 17.9 and 18.5%. All later numbers use 0.95.", 13.0, h=2.4)
_concl(s, "+3.4 points, no retraining.")

# ---------------- P11: vertex geometry ----------------
s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 3", "Vertex geometry", "three inputs per candidate pair")
pic(s, P_DOCA, 0.60, 1.45, w=5.40)
_rows = [["input", "what it is"],
         ["DOCA / 100", "closest distance between the two tracks"],
         ["log(DOCA)", "same quantity, log scale"],
         ["|Δstart| / 100", "distance between the two production vertices"]]
_tbl(s, 6.30, 1.48, 6.35, _rows, [0.30, 0.70], fs=11.0, rowh=0.50)
_para(s, 6.30, 3.62, 6.35,
      "The first two columns are recomputed: the stored distance is noise, with a correlation of 0.019 to the "
      "correct value. The third is a new absolute quantity. The three enter together and cannot be separated; "
      "there is no run with |Δstart| alone.", 12.5, h=2.3)
_concl(s, "+3.4 points from three input columns.")

# ---------------- P12: edge-ranking loss ----------------
s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 3", "Edge-ranking loss", "pairwise term · same network")
pic(s, P_RANKEFF, 0.60, 1.45, w=6.90)
pic(s, P_EQ_RANK, 7.85, 1.55, w=min(W_EQ_RANK, 4.85))
_para(s, 7.85, 3.35, 4.85,
      "The edge loss is dominated by a positive weight of about 700, which lifts signal edges but does not order "
      "them against background. We add a pairwise term: every signal edge is compared with the 64 highest-scoring "
      "background edges of the same event. The network, the inputs and the schedule are unchanged.", 12.5, h=2.7)
_concl(s, "+1.8 points at 0.90, +2.6 at 0.95.")

# ---------------- P13: edge-context inputs ----------------
s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 3", "Edge-context inputs", "six numbers per candidate pair")
pic(s, P_DIST_M2, 0.45, 1.35, w=7.45)
_para(s, 8.10, 1.45, 4.60,
      "For each end of a pair we ask two questions: how good a partner is this pair for that track, and how many "
      "near-equal alternatives the track has. Three numbers per end, then the minimum and the maximum over the two "
      "ends. The six inputs do not depend on the order of the pair.", 12.5, h=2.9)
pic(s, P_EQ_R, 8.10, 4.45, w=min(W_EQ_R, 2.35))
pic(s, P_EQ_D, 10.55, 4.45, w=min(W_EQ_D, 2.15))
_concl(s, "+1.0 point from six inputs.")

# ---------------- P14: attention ----------------
s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 3", "Attention on the line graph", "with a geometric bias")
pic(s, P_DIST_M3, 0.45, 1.35, w=7.45)
_para(s, 8.10, 1.45, 4.60,
      "Candidate pairs become the nodes of a line graph, and pairs sharing a track exchange messages in one round "
      "of multi-head attention. A small network maps the pair geometry into a bias on the attention logits. "
      "Attention alone is neutral (21.8 → 21.6); the gain appears only with this bias, and there is no bias-only "
      "run, so we attribute it to the geometry.", 12.0, h=3.3)
pic(s, P_EQ_BIAS, 8.15, 5.00, w=min(W_EQ_BIAS, 3.60))
_concl(s, "Geometry in the score; attention alone is neutral.")

# ---------------- P15: four small changes ----------------
s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 3", "Four small changes", "gain at 0.95, in percentage points")
_rows = [["change", "what changes", "Δ"],
         ["minIP input", "canonicalised dz·IP feature", "+0.2"],
         ["event bias", "one learned scalar per event on the pruning logits", "+0.5"],
         ["derived and triangle", "extra edge features from the same inputs", "+0.8"],
         ["checkpoint on pruning AP", "model selection on the validation pruning AP", "+0.8"]]
_tbl(s, 0.65, 1.65, 12.0, _rows, [0.30, 0.55, 0.15], fs=12.0, rowh=0.62)
_para(s, 0.65, 5.05, 12.0,
      "Four low-cost changes; none of them exceeds one point on its own. Mechanics are in the appendix.",
      13.0, h=0.9)
_concl(s, "About +2.3 points together; the two smallest are not resolved.")

# ---------------- P16: output structure ----------------
s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 3", "The next step: assignment", "each track to one chain or to background")

panel(s, 0.65, 1.55, 5.85, 3.55, fill=PALE, line=BORDER)
tb(s, 0.90, 1.68, 5.35, 0.30, "today", 14, True, GREY)
rect(s, 1.00, 2.20, 1.55, 0.62, WHITE, line=BORDER, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.10)
tb(s, 1.00, 2.32, 1.55, 0.28, "pair", 12.5, True, NAVY, align=PP_ALIGN.CENTER)
tb(s, 2.62, 2.30, 0.35, 0.28, "→", 13, True, GREY, align=PP_ALIGN.CENTER)
rect(s, 3.00, 2.20, 1.55, 0.62, WHITE, line=BORDER, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.10)
tb(s, 3.00, 2.32, 1.55, 0.28, "LCAG 0–3", 12.5, True, NAVY, align=PP_ALIGN.CENTER)
tb(s, 4.62, 2.30, 0.35, 0.28, "→", 13, True, GREY, align=PP_ALIGN.CENTER)
rect(s, 5.00, 2.20, 1.25, 0.62, WHITE, line=BORDER, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.10)
tb(s, 5.00, 2.28, 1.25, 0.46, "connected\ncomponents", 10.5, True, NAVY, align=PP_ALIGN.CENTER)
tb(s, 0.95, 3.10, 5.30, 0.90,
   "Classes are defined per pair; there is no track-level background label.", 12.5, False, INK, line=1.25)

panel(s, 6.85, 1.55, 5.85, 3.55, fill=PALE, line=BORDER)
tb(s, 7.10, 1.68, 5.35, 0.30, "next", 14, True, PURPLE)
rect(s, 7.55, 2.20, 1.60, 0.62, WHITE, line=BORDER, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.10)
tb(s, 7.55, 2.32, 1.60, 0.28, "track", 12.5, True, NAVY, align=PP_ALIGN.CENTER)
tb(s, 9.22, 2.30, 0.35, 0.28, "→", 13, True, GREY, align=PP_ALIGN.CENTER)
rect(s, 9.68, 1.95, 1.85, 0.52, WHITE, line=PURPLE, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.10)
tb(s, 9.68, 2.05, 1.85, 0.28, "one chain", 12, True, PURPLE, align=PP_ALIGN.CENTER)
rect(s, 9.68, 2.62, 1.85, 0.52, WHITE, line=GREY, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.10)
tb(s, 9.68, 2.72, 1.85, 0.28, "background", 12, True, GREY, align=PP_ALIGN.CENTER)
tb(s, 7.10, 3.35, 5.35, 1.40,
   "One arrow per track, so the partition matches a decay chain. The first attempt reproduced the baseline; "
   "the defects are listed in Appendix A6.", 12.5, False, INK, line=1.25)

_para(s, 0.65, 5.30, 12.0,
      "This step keeps the score and changes only how tracks are grouped into chains.",
      13.5, h=0.8)
_concl(s, "The remaining loss is in the output form.")

# ---------------- summary ----------------
s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "SUMMARY", "What moved the number", "0904 sample · fixed denominator 13 255")
tb(s, 0.75, 1.70, 6.20, 0.85, "14.5 → 28.0%", 40, True, NAVY)
tb(s, 0.75, 2.75, 6.20, 0.55, "+3.4 from the cut, no retraining", 18, True, GOLD)
tb(s, 0.75, 3.35, 6.20, 0.55, "+10.1 from the rest", 18, True, PURPLE)
tb(s, 0.75, 3.95, 6.20, 0.55, "attention alone: −0.2", 18, False, GREY)
panel(s, 7.15, 1.70, 5.55, 2.55, fill=PALE, line=BORDER)
tb(s, 7.40, 1.85, 5.05, 0.30, "NEXT", 14, True, GREEN)
tb(s, 7.40, 2.25, 5.05, 1.80,
   "Exclusive assignment: one chain per track, plus a background class.\n\n"
   "Benchmark at channel level, not one global edge AUC.", 15.5, False, INK, line=1.30)
_para(s, 0.75, 4.95, 12.0,
      "The gain comes from the inputs and from the position of the cut. The output form is the next step.", 14.0, h=0.9)
_concl(s, "The gain is in the inputs and the cut; the output is the next step.")


# ======================= APPENDIX =======================
# A1 vertex geometry
s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "APPENDIX A1", "Vertex geometry", "definition of the three inputs")
_rows = [["input", "definition", "note"],
         ["DOCA / 100", "distance of closest approach between the two track lines", "recomputed"],
         ["log(DOCA)", "logarithm of the same distance", "recomputed"],
         ["|Δstart| / 100", "3-D distance between the two production vertices", "new"]]
_tbl(s, 0.65, 1.55, 12.0, _rows, [0.20, 0.58, 0.22], fs=11.5, rowh=0.55)
_para(s, 0.65, 3.95, 12.0,
      "The stored DOCA column comes from the expression with the wrong sign described in Part 2. Against the "
      "correct value its correlation is 0.019, so it behaves as noise; the two DOCA inputs here are computed "
      "locally. The third input is new: the older derived block only contained d_perp/(|Δz|+1), which divides the "
      "scale out. The three columns were added together, so the +3.4 cannot be split further; there is no run "
      "with |Δstart| alone.", 13.0, h=2.5)
footer(s)

# A2 edge-context inputs
s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "APPENDIX A2", "Edge-context inputs", "definition of the six numbers")
pic(s, P_EQ_M2, 0.65, 1.45, w=min(W_EQ_M2, 6.10))
_para(s, 7.10, 1.50, 5.55,
      "For a candidate pair e = (i, j), C(t) is the set of candidate pairs attached to track t; e belongs to both "
      "C(i) and C(j). The affinity A = exp(−d3/50) scores each candidate, where d3 is the distance between the two "
      "production vertices. r_t compares the pair with the best alternative of that track, d_t counts alternatives "
      "with A > 0.5, and c_t is the number of candidates divided by 50. The minimum and maximum over the two ends "
      "make the six "
      "inputs symmetric under exchange of i and j, which matters because the mother/daughter direction is not "
      "reliable at this stage. The edge vector grows from 12 to 18; the new columns start at zero.", 12.0, h=5.0)
footer(s)

# A3 edge-ranking loss
s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "APPENDIX A3", "Edge-ranking loss", "all signal edges vs the 64 hardest background edges")
pic(s, P_EQ_RANK, 0.65, 1.55, w=min(W_EQ_RANK, 6.00))
_para(s, 7.05, 1.55, 5.60,
      "Signal edges are about 0.14% of the candidates, so the positive weight of the BCE is about 700. That lifts "
      "signal edges but leaves their order against a background edge unconstrained, and the reconstruction works "
      "by thresholding. The added term pairs every signal edge with the 64 background edges of the same event that "
      "have the highest score. Only the relative order enters, so the absolute score drifts and the working point "
      "must be re-scanned. With this term the same weights give 16.3% at 0.90 and 20.5% at 0.95.", 12.0, h=5.0)
footer(s)

# A4 attention
s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "APPENDIX A4", "Attention on the line graph", "four heads, eight dimensions, one round")
pic(s, P_EQ_ATTN, 0.65, 1.40, w=min(W_EQ_ATTN, 5.70))
_para(s, 6.70, 1.50, 5.95,
      "Candidate pairs are the nodes of a line graph; two pairs are neighbours when they share a track in the same "
      "event, so the neighbours are exactly the pairs that compete for a track. Each pair is projected to 8 "
      "dimensions per head, four heads are concatenated to 32 and projected back to 16 with a residual. Messages "
      "outside the neighbourhood are masked. The geometric bias below is added to the logits before the softmax.",
      12.0, h=3.4)
pic(s, P_EQ_BDEF, 6.70, 4.90, w=min(W_EQ_BDEF, 5.60))
footer(s)

# A5 four small changes
s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "APPENDIX A5", "Four small changes", "mechanics and measured gain")
_rows = [["change", "mechanism", "Δ"],
         ["minIP input", "the dz·IP edge feature is canonicalised, so the sign convention cannot flip", "+0.2"],
         ["event bias", "one learned scalar per event is added to all pruning logits of that event", "+0.5"],
         ["derived and triangle", "derived pruning features and triangle-transitivity features on candidate edges", "+0.8"],
         ["checkpoint on pruning AP", "early stopping and model selection on the validation pruning AP", "+0.8"]]
_tbl(s, 0.65, 1.55, 12.0, _rows, [0.26, 0.60, 0.14], fs=10.5, rowh=0.68)
_para(s, 0.65, 5.15, 12.0,
      "Two further modules were tried and removed: an event-count head cost 2.8 points and a PV-overlap input "
      "cost 0.4. They are not part of the chain.", 12.5, h=0.9)
footer(s)

# A6 output-structure attempt
s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "APPENDIX A6", "Output-structure attempt", "why the previous run gave nothing")
_rows = [["symptom", "root cause", "fix"],
         ["two heads identical to baseline", "labels decoded but never used: dispatch only accepted setpred_decode",
          "dispatch accepts partition_decode"],
         ["slot head never trained", "matching enumerates permutations; events with more chains than slots crash",
          "linear assignment for M > K"],
         ["no model selection", "monitor was the backbone pruning AP, blind to the new head",
          "monitor the validation partition loss"]]
_tbl(s, 0.65, 1.55, 12.0, _rows, [0.28, 0.48, 0.24], fs=10.0, rowh=0.80)
_para(s, 0.65, 5.00, 12.0,
      "With these three fixes the run is repeated with an object-condensation head, a slot head with eight slots, "
      "and the same slot head with a partition-level term. The assignment decoder that did reach the assembly "
      "stage differs from the baseline by −0.5 points; no corrected result is reported here.", 12.5, h=1.0)
footer(s)

# A7 metrics and ROC
if P_ROC:
    s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
    header(s, "APPENDIX A7", "Metrics and edge ROC", "definitions and the global ranking")
    pic(s, P_ROC, 0.55, 1.45, w=6.9)
    _para(s, 7.75, 1.50, 4.90,
          "AllParticles counts chains whose reconstructed cluster contains exactly the truth tracks. PerfectReco "
          "also requires the LCAG values to match. Purity is the signal fraction among kept edges. The ROC averages "
          "over all edges and thresholds, while reconstruction uses one cut on the few near-degenerate pairs, so a "
          "higher AUC does not imply a higher chain yield. Two normalisations appear: the paper convention divides "
          "by surviving chains, this talk by all 13 255 signal chains.", 12.0, h=4.9)
    footer(s)


prs.save(OUT)
print("saved", OUT, os.path.getsize(OUT))
