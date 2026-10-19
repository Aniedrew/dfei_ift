"""Deck v4 — large-type redesign: body >= 25 pt, one idea per slide, attention explained in plain words."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR

OUT = "DFEI_bugfix_pruning_20261023_EN.pptx"
FIGS = "report_figs/deck_20261023"
os.makedirs(FIGS, exist_ok=True)

NAVY = RGBColor(0x1E, 0x27, 0x61); TEAL = RGBColor(0x00, 0xA8, 0x96)
ICE = RGBColor(0xCA, 0xDC, 0xFC); GREY = RGBColor(0x5A, 0x62, 0x70)
LIGHT = RGBColor(0xF4, 0xF7, 0xFB); WHITE = RGBColor(0xFF, 0xFF, 0xFF)
RED = RGBColor(0xC0, 0x39, 0x2B); CODEBG = RGBColor(0xEC, 0xF0, 0xF7)
C = dict(p="#1E2761", s="#00A896", ice="#CADCFC", grey="#5A6270", red="#C0392B", light="#E8EDF6")

plt.rcParams.update({
    "axes.facecolor": "none", "figure.facecolor": "none", "axes.edgecolor": "#9AA3B2",
    "axes.labelcolor": C["p"], "text.color": C["p"], "xtick.color": C["p"], "ytick.color": C["p"],
    "font.size": 13, "axes.titlesize": 15, "xtick.labelsize": 13, "ytick.labelsize": 13,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.16, "grid.color": "#9AA3B2",
})

VERS = ["v601", "v623", "v631", "v633", "v640", "v643", "v648", "v651"]
RAW_ALL = [14.45, 14.31, 19.77, 21.83, 22.60, 26.04, 27.00, 27.96]
RAW_PERF = [9.54, 9.57, 13.30, 14.46, 14.79, 17.75, 18.12, 19.00]
AUC = [0.9716, 0.9737, 0.9352, 0.9242, None, 0.9236, 0.9324, 0.9327]
PRUNED = [43.05, 39.12, 42.26, 45.71, 49.97, 47.45, 52.12, 54.06]
CONTAM = [42.50, 46.57, 37.97, 32.46, 27.43, 26.51, 20.88, 17.98]
CLEAN = [14.45, 14.31, 19.77, 21.83, 22.60, 26.04, 27.00, 27.96]


def save(fig, name, w, h):
    p = f"{FIGS}/{name}_{int(w*180)}x{int(h*180)}.png"
    fig.savefig(p, dpi=180, transparent=True); plt.close(fig); return p


def fig_pv():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 3.5), dpi=180, gridspec_kw={"width_ratios": [1.5, 1]})
    lab = ["v31", "v38", "v47", "v557", "v601", "v623", "v651"]
    allv = [13.80, 13.40, 13.07, 14.02, 8.61, 8.61, 8.64]
    a1.bar(lab, allv, color=[C["grey"]] * 4 + [C["s"]] * 3, width=0.62)
    a1.axhline(10.15, color=C["red"], ls="--", lw=2)
    a1.text(6.5, 10.6, "minIP 10.15", color=C["red"], ha="right", fontsize=12)
    for i, v in enumerate(allv): a1.text(i, v + 0.28, f"{v:.1f}", ha="center", fontsize=12)
    a1.set_ylim(0, 16.5); a1.set_ylabel("misassociation [%]", fontsize=12)
    a1.set_title("all tracks", fontsize=14, loc="left")
    a2.bar(["v601", "v651", "minIP"], [16.50, 14.86, 19.99], color=[C["s"], C["s"], C["red"]], width=0.55)
    for i, v in enumerate([16.50, 14.86, 19.99]): a2.text(i, v + 0.35, f"{v:.1f}", ha="center", fontsize=12)
    a2.set_ylim(0, 23); a2.set_title("signal", fontsize=14, loc="left")
    fig.tight_layout(); return save(fig, "pv2", 10.4, 3.5)


def fig_decomp():
    fig, ax = plt.subplots(figsize=(10.4, 3.4), dpi=180)
    x = np.arange(len(VERS))
    ax.bar(x, PRUNED, color="#C9D0DD", label="pruned away")
    ax.bar(x, CONTAM, bottom=PRUNED, color=C["s"], label="merged with background")
    ax.bar(x, CLEAN, bottom=np.array(PRUNED) + np.array(CONTAM), color=C["p"], label="exact")
    for i in range(len(VERS)):
        ax.text(i, PRUNED[i] / 2, f"{PRUNED[i]:.0f}", ha="center", fontsize=11, color="#4A5568")
        ax.text(i, PRUNED[i] + CONTAM[i] / 2, f"{CONTAM[i]:.0f}", ha="center", fontsize=11, color="white")
        ax.text(i, PRUNED[i] + CONTAM[i] + CLEAN[i] / 2, f"{CLEAN[i]:.0f}", ha="center", fontsize=11, color="white")
    ax.set_xticks(x); ax.set_xticklabels(VERS, fontsize=12); ax.set_ylim(0, 100)
    ax.set_ylabel("% of truth chains", fontsize=12)
    ax.legend(frameon=False, fontsize=12, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.2))
    fig.tight_layout(); return save(fig, "decomp", 10.4, 3.4)


def fig_scatter():
    fig, ax = plt.subplots(figsize=(6.0, 3.4), dpi=180)
    a = [(AUC[i], RAW_ALL[i], VERS[i]) for i in range(len(VERS)) if AUC[i]]
    ax.scatter([t[0] for t in a], [t[1] for t in a], s=90, color=C["s"], zorder=3)
    for x, y, v in a: ax.annotate(v, (x, y), textcoords="offset points", xytext=(7, 5), fontsize=11)
    ax.set_xlabel("edge AUC", fontsize=12); ax.set_ylabel("AllParticles [%]", fontsize=12)
    ax.set_xlim(0.915, 0.985); ax.set_ylim(18, 66)
    fig.tight_layout(); return save(fig, "scatter", 6.0, 3.4)


def fig_ladder():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 3.3), dpi=180)
    a1.plot(VERS, RAW_ALL, "o-", color=C["p"], lw=2.6, ms=7, label="AllParticles")
    a1.plot(VERS, RAW_PERF, "s-", color=C["s"], lw=2.6, ms=7, label="PerfectReco")
    a1.set_ylim(10, 70); a1.set_ylabel("[%]", fontsize=12)
    a1.legend(frameon=False, fontsize=12, loc="lower right")
    a1.set_title("reconstruction", fontsize=14, loc="left")
    idx = [t for t in range(len(VERS)) if AUC[t]]
    a2.plot([VERS[i] for i in idx], [AUC[i] for i in idx], "o-", color=C["grey"], lw=2.6, ms=7)
    a2.set_ylim(0.915, 0.985); a2.set_ylabel("edge AUC", fontsize=12)
    a2.set_title("ranking quality", fontsize=14, loc="left")
    fig.tight_layout(); return save(fig, "ladder2", 10.6, 3.3)


def fig_delta():
    steps = ["features", "operating pt", "remove", "selection", "DOCA fix", "context", "attention"]
    da = [RAW_ALL[i + 1] - RAW_ALL[i] for i in range(7)]
    dp = [RAW_PERF[i + 1] - RAW_PERF[i] for i in range(7)]
    fig, ax = plt.subplots(figsize=(9.6, 3.3), dpi=180)
    y = np.arange(len(steps)); h = 0.36
    ax.barh(y + h / 2, da, h, color=C["p"], label="AllParticles")
    ax.barh(y - h / 2, dp, h, color=C["s"], label="PerfectReco")
    for i, (u, v) in enumerate(zip(da, dp)):
        ax.text(u + (0.3 if u >= 0 else -0.3), i + h / 2, f"{u:+.1f}", va="center", ha="left" if u >= 0 else "right", fontsize=11)
        ax.text(v + (0.3 if v >= 0 else -0.3), i - h / 2, f"{v:+.1f}", va="center", ha="left" if v >= 0 else "right", fontsize=11)
    ax.set_yticks(y); ax.set_yticklabels(steps, fontsize=12); ax.invert_yaxis()
    ax.axvline(0, color="#9AA3B2", lw=1.2)
    ax.set_xlabel("change vs previous version [pp]", fontsize=12)
    ax.legend(frameon=False, fontsize=12, loc="lower right")
    fig.tight_layout(); return save(fig, "delta", 9.6, 3.3)


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
    a = [40.22, 49.55]; p = [26.64, 33.78]
    ax.bar([i - wd / 2 for i in x], a, wd, color=C["p"], label="AllParticles")
    ax.bar([i + wd / 2 for i in x], p, wd, color=C["s"], label="PerfectReco")
    for i in range(2):
        ax.text(i - wd / 2, a[i] + 0.9, f"{a[i]:.1f}", ha="center", fontsize=12)
        ax.text(i + wd / 2, p[i] + 0.9, f"{p[i]:.1f}", ha="center", fontsize=12)
    ax.set_xticks(x); ax.set_xticklabels(["before", "after"], fontsize=13)
    ax.set_ylim(0, 58); ax.set_ylabel("[%]", fontsize=12)
    ax.legend(frameon=False, fontsize=11, loc="upper left")
    fig.tight_layout(); return save(fig, "doca", 4.8, 3.1)


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


def fig_thr():
    fig, ax = plt.subplots(figsize=(5.4, 2.8), dpi=180)
    x = np.linspace(0, 1, 400)
    bg = 1.0 * np.exp(-((x - 0.38) ** 2) / 0.016)
    sig = 0.55 * np.exp(-((x - 0.80) ** 2) / 0.018)
    ax.plot(x, bg, color="#9AA3B2", lw=2); ax.fill_between(x, bg, color="#9AA3B2", alpha=0.20)
    ax.plot(x, sig, color=C["s"], lw=2); ax.fill_between(x, sig, color=C["s"], alpha=0.22)
    ax.axvspan(0.90, 0.97, color=C["red"], alpha=0.16)
    ax.axvline(0.97, color=C["red"], ls=":", lw=2); ax.axvline(0.90, color=C["red"], ls="--", lw=2)
    ax.text(0.965, 0.95, "0.97", rotation=90, va="top", ha="right", fontsize=12, color=C["red"])
    ax.text(0.895, 0.95, "0.90", rotation=90, va="top", ha="right", fontsize=12, color=C["red"])
    ax.set_yticks([]); ax.set_xlim(0, 1); ax.set_xlabel("pruning score", fontsize=12)
    fig.tight_layout(); return save(fig, "thr", 5.4, 2.8)


def fig_bias():
    """The cleanest ablation: same idea, with and without the geometric prior."""
    fig, ax = plt.subplots(figsize=(6.2, 3.0), dpi=180)
    names = ["attention only\n(v637)", "attention +\ngeometry bias (v651)"]
    vals = [0.0, 4.5]
    ax.bar(names, vals, color=[C["grey"], C["s"]], width=0.5)
    for i, v in enumerate(vals): ax.text(i, v + 0.12, f"{v:+.1f} pp", ha="center", fontsize=13)
    ax.set_ylim(-0.6, 5.6); ax.set_ylabel("Δ AllParticles", fontsize=12)
    ax.axhline(0, color="#9AA3B2", lw=1.2)
    fig.tight_layout(); return save(fig, "bias", 6.2, 3.0)


P_PV, P_DECOMP, P_SCAT = fig_pv(), fig_decomp(), fig_scatter()
P_LADDER, P_DELTA, P_NEG, P_DOCA = fig_ladder(), fig_delta(), fig_neg(), fig_doca()
P_RANK, P_THR, P_BIAS = fig_rank(), fig_thr(), fig_bias()

prs = Presentation(); prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
W, H, MX = 13.333, 7.5, 0.6

BODY = 25      # body text size, per request
SMALL = 15


def _rgb(v):
    if isinstance(v, str):
        v = v.lstrip("#"); return RGBColor(int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16))
    return v


def tb(s, x, y, w, h, text, size=BODY, bold=False, color=NAVY, align=PP_ALIGN.LEFT,
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


def bullets(s, x, y, w, h, items, size=BODY, color=NAVY, gap=12):
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
    rect(s, x, y, w, h, CODEBG)
    tb(s, x + 0.2, y + 0.14, w - 0.35, h - 0.24, lines, size, False, NAVY, font="Consolas", line=1.35)


def header(s, kicker, title, sub=None):
    rect(s, 0, 0, W, 0.1, TEAL)
    tb(s, MX, 0.3, 11.5, 0.3, kicker, 13, True, TEAL)
    tb(s, MX, 0.6, 12.1, 0.62, title, 30, True, NAVY, font="Georgia")
    if sub: tb(s, MX, 1.26, 12.1, 0.36, sub, 17, False, GREY, italic=True)


def footer(s):
    tb(s, MX, H - 0.42, 8.0, 0.25, "DFEI · bug fixes and pruning optimization", 11, False, GREY)
    tb(s, W - 1.05, H - 0.42, 0.45, 0.25, str(len(prs.slides)), 11, True, GREY, align=PP_ALIGN.RIGHT)


def pic(s, path, x, y, w=None, h=None):
    if w: s.shapes.add_picture(path, Inches(x), Inches(y), width=Inches(w))
    else: s.shapes.add_picture(path, Inches(x), Inches(y), height=Inches(h))


def note(s, text, y=6.35, size=14, *extra):
    """读者注: 给无法参会的人看的 1-2 句说明.
    容忍 note(s, "a", "b", y, size) 这种被拆成两段的写法."""
    if isinstance(y, str):
        text = text + " " + y
        y, size = size, (extra[0] if extra else 14)
    rect(s, MX, y - 0.08, 12.1, 0.02, ICE)
    tb(s, MX, y, 12.1, 0.72, text, size, False, GREY, line=1.2)


def cover():
    s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, NAVY); rect(s, 0, 0, 0.16, H, TEAL)
    tb(s, 1.0, 1.75, 11.4, 0.35, "DFEI GROUP MEETING  ·  2026-10-23", 14, True, TEAL)
    tb(s, 1.0, 2.15, 11.3, 1.75, "DFEI: Bug Fixes and\nPruning Optimization", 44, True, WHITE, font="Georgia", line=1.12)
    tb(s, 1.0, 4.0, 10.8, 0.45, "From a corrected simulation sample to a stronger pruning stage", 18, False, ICE, italic=True)
    rect(s, 1.0, 4.7, 1.5, 0.05, TEAL)
    tb(s, 1.0, 4.95, 8.0, 0.35, "Qingxiang Guo  ·  UCAS", 16, False, WHITE)
    tb(s, 1.0, 6.6, 11.0, 0.3, "All numbers on the corrected sample unless stated otherwise", 12, False, ICE)


def toc():
    s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, LIGHT)
    header(s, "CONTENTS", "Four questions")
    rows = [("1", "What did the leak fix cost?", "PV association, and how we got it back"),
            ("2", "Where does the loss sit?", "two failure modes, one lever"),
            ("3", "What was wrong with DOCA?", "a silent feature corruption"),
            ("4", "How far did pruning get?", "four methods, explained and measured")]
    y = 2.0
    for i, (n, t, d) in enumerate(rows):
        rect(s, MX, y, 12.1, 1.05, WHITE); rect(s, MX, y, 0.07, 1.05, TEAL if i % 2 == 0 else NAVY)
        tb(s, MX + 0.28, y + 0.1, 0.7, 0.6, n, 30, True, TEAL if i % 2 == 0 else NAVY, font="Georgia")
        tb(s, MX + 1.15, y + 0.08, 10.6, 0.4, t, 24, True, NAVY)
        tb(s, MX + 1.15, y + 0.55, 10.6, 0.35, d, 16, False, GREY, italic=True)
        y += 1.2
    footer(s)


def divider(num, title, sub, items):
    s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, NAVY)
    tb(s, 1.0, 1.45, 3.0, 0.35, f"PART {num}", 14, True, TEAL)
    tb(s, 1.0, 1.85, 11.2, 1.0, title, 38, True, WHITE, font="Georgia")
    tb(s, 1.0, 2.95, 11.0, 0.45, sub, 18, False, ICE, italic=True)
    rect(s, 1.0, 3.6, 1.5, 0.05, TEAL)
    bullets(s, 1.0, 3.95, 11.0, 2.4, items, BODY, WHITE, 14)


cover(); toc()

# ---------------- PART 1 ----------------
divider(1, "What the leak fix cost", "One metric moved the wrong way",
        ["The corrected sample removes truth information from the inputs",
         "PV association lost the most",
         "A PV-aware head brought it back below minIP"])

s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 1  ·  CONTEXT", "Where pruning sits in DFEI", "Everything downstream inherits what pruning keeps")
labels = [("Graph", 0.75, GREY), ("Node\npruning", 3.15, TEAL), ("Edge\npruning", 5.55, TEAL),
          ("LCAG\nclasses", 7.95, NAVY), ("Chain\nassembly", 10.35, NAVY)]
for txt, x, col in labels:
    rect(s, x, 2.3, 2.2, 1.35, WHITE, line=col, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.12)
    tb(s, x + 0.1, 2.55, 2.0, 0.9, txt, 22, True, col, align=PP_ALIGN.CENTER)
for x in (2.98, 5.38, 7.78, 10.18):
    c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x), Inches(2.97), Inches(x + 0.16), Inches(2.97))
    c.line.color.rgb = GREY; c.line.width = Pt(1.8)
rect(s, 2.55, 3.95, 4.05, 0.65, RGBColor(0xE4, 0xF3, 0xF0))
tb(s, 2.68, 4.05, 3.85, 0.45, "our work", 18, True, TEAL)
rect(s, 9.45, 3.95, 3.05, 0.65, RGBColor(0xEC, 0xEC, 0xF5))
tb(s, 9.6, 4.05, 2.85, 0.45, "next round", 18, True, NAVY)
bullets(s, MX, 5.1, 12.1, 1.6, ["A chain lost here can never be recovered",
                                "PV association shares the same representation"], BODY, NAVY, 12)
note(s, "Pruning is the first stage of DFEI and the only one that can destroy a chain irreversibly. Everything downstream can",
        "only rearrange what survived.", 6.4)
footer(s)

s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 1  ·  PV ASSOCIATION", "Old models are worse than minIP", "misassociation rate, lower is better")
pic(s, P_PV, 0.7, 1.65, w=11.9)
rect(s, MX, 5.55, 12.1, 1.05, LIGHT); rect(s, MX, 5.55, 0.06, 1.05, TEAL)
tb(s, 0.85, 5.68, 11.7, 0.85, "13–14% before → 8.6% after retraining with a PV head (minIP 10.15)", BODY, True, NAVY)
note(s, "PV misassociation is the fraction of tracks attached to the wrong interaction point. Lower is better; minIP is the",
        "traditional LHCb baseline.", 6.75, 13)
footer(s)

s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 1  ·  THE FIX", "What the PV-aware head does", "one auxiliary target on the same representation")
rect(s, MX, 1.65, 6.2, 3.4, WHITE, line=ICE)
for i, yy in enumerate([2.2, 2.75, 3.3]):
    rect(s, MX + 0.35, yy, 0.2, 0.2, NAVY, shape=MSO_SHAPE.OVAL)
    tb(s, MX + 0.65, yy - 0.08, 0.9, 0.3, f"PV {i+1}", 16, False, GREY)
for x, y, col in [(3.6, 2.3, TEAL), (4.2, 2.62, TEAL), (4.8, 3.1, TEAL), (5.2, 2.45, "#9AA3B2")]:
    rect(s, x, y, 0.18, 0.18, col, shape=MSO_SHAPE.OVAL)
for x, y in [(3.6, 2.3), (4.2, 2.62), (4.8, 3.1)]:
    ln = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x + 0.18), Inches(y + 0.09), Inches(MX + 0.55), Inches(2.3))
    ln.line.color.rgb = TEAL; ln.line.width = Pt(1.4); ln.line.dash_style = 4
ln = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(5.38), Inches(2.54), Inches(MX + 0.55), Inches(2.85))
ln.line.color.rgb = RGBColor(0x9A, 0xA3, 0xB2); ln.line.width = Pt(1.4); ln.line.dash_style = 4
tb(s, MX + 0.3, 4.35, 5.7, 0.5, "each track is softly assigned to a PV", 16, False, GREY, italic=True)
rect(s, 7.05, 1.65, 5.65, 3.4, LIGHT); rect(s, 7.05, 1.65, 0.06, 3.4, TEAL)
bullets(s, 7.3, 1.95, 5.2, 3.0, ["PV is an event-level property", "Learning it forces geometry into the model", "Same features help judge edges"], BODY, NAVY, 14)
note(s, "The head is auxiliary: it is used only during training, to force the shared representation to encode where each track came from.", 6.4)
footer(s)

# ---------------- PART 2 ----------------
divider(2, "Where the loss actually sits", "Two failure modes, one lever",
        ["A chain is pruned away, or survives and is merged with background",
         "Assembly never fails on its own",
         "Pruning sets both terms"])

s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 2  ·  DIAGNOSIS", "Every step trades one failure for the other", "13 255 truth chains, fixed denominator")
pic(s, P_DECOMP, 0.7, 1.6, w=11.9)
bullets(s, MX, 5.55, 12.1, 1.3, ["v631 and v633 buy exact chains by dropping more", "No version shrinks both at once"], BODY, NAVY, 10)
footer(s)

s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 2  ·  THE PROXY TRAP", "Keeping truth content is not recovering chains",
       "track survival rises while exact reconstruction does not")
pic(s, "report_figs/reco_fixed/F3_survival_vs_recovery.png", 0.7, 1.55, w=11.9)
note(s, "Red: truth tracks surviving pruning. Blue: exact chains. Keeping more truth content does not recover more chains.", 5.9)
footer(s)

s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 2  ·  A WARNING", "Ranking and reconstruction come apart", "both measured on the corrected sample")
pic(s, P_SCAT, 0.7, 1.7, w=6.0)
rect(s, 7.1, 1.7, 5.6, 3.5, LIGHT); rect(s, 7.1, 1.7, 0.06, 3.5, TEAL)
bullets(s, 7.35, 2.0, 5.1, 3.0, ["v601: best AUC, poor chains", "v651: lower AUC, best chains", "So AUC is a diagnostic only"], BODY, NAVY, 14)
footer(s)

# ---------------- PART 3 ----------------
divider(3, "The DOCA bug", "A feature that was silently wrong",
        ["DOCA was inherited from an upstream quantity",
         "That quantity is not the geometric DOCA",
         "The MLP learned from a biased input"])

s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 3  ·  THE BUG", "What DOCA is", "no crash — just a wrong number")
rect(s, MX, 1.65, 6.3, 3.9, WHITE, line=ICE)
rect(s, 1.05, 4.75, 0.2, 0.2, NAVY, shape=MSO_SHAPE.OVAL)
tb(s, 1.32, 4.7, 0.8, 0.28, "PV", 15, False, GREY)
a = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(1.15), Inches(4.85), Inches(3.9), Inches(2.55))
b = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(1.15), Inches(4.85), Inches(5.2), Inches(3.35))
for ln in (a, b): ln.line.color.rgb = NAVY; ln.line.width = Pt(2.6)
rect(s, 3.9, 2.45, 0.18, 0.18, TEAL, shape=MSO_SHAPE.OVAL)
rect(s, 5.2, 3.25, 0.18, 0.18, TEAL, shape=MSO_SHAPE.OVAL)
ln = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(3.99), Inches(2.63), Inches(5.11), Inches(3.34))
ln.line.color.rgb = RED; ln.line.width = Pt(2.2); ln.line.dash_style = 4
tb(s, 3.3, 3.55, 3.3, 0.35, "DOCA", 18, True, RED)
rect(s, 7.15, 1.65, 5.55, 3.9, LIGHT); rect(s, 7.15, 1.65, 0.06, 3.9, RED)
bullets(s, 7.4, 1.95, 5.05, 3.4, ["We fed an upstream proxy, not the geometry", "The mapping learned was biased", "Fix: recompute it locally"], BODY, NAVY, 14)
footer(s)

s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 3  ·  THE EFFECT", "Correcting DOCA: +9.3 pp", "one variable changed")
pic(s, P_DOCA, 0.9, 1.9, w=5.2)
rect(s, 6.5, 1.9, 6.2, 3.2, LIGHT); rect(s, 6.5, 1.9, 0.06, 3.2, TEAL)
bullets(s, 6.75, 2.2, 5.7, 2.7, ["AllParticles 40.2 → 49.6%", "PerfectReco 26.6 → 33.8%", "Edge AUC flat: not a ranking gain"], BODY, NAVY, 14)
footer(s)

# ---------------- PART 4 ----------------
divider(4, "Pruning optimization", "Four methods, explained then measured",
        ["Physics features · ranking · competition · attention",
         "Each is one controlled change",
         "We report four benchmarks, never one"])

# --- recap of the previous round ---
s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "RECAP  ·  LAST ROUND", "Last round optimised the GNN, not the pruning",
       "September report; numbers on the sample in use at the time")
bullets(s, MX, 1.75, 8.3, 3.9,
        ["v31 → v38: differentiable pruning, class-0 fix, hinge, chain-CE",
         "v38 → v47: nine physics heads on one backbone — only mass paid",
         "Capacity and attention: wider layers, line-graph, edge bias — small",
         "Event restructuring: PV split, learned clustering, subgraph training",
         "v500–v509: one-variable ablation of the whole recipe"],
        20, NAVY, 9)
rect(s, 9.2, 1.75, 3.45, 3.9, LIGHT); rect(s, 9.2, 1.75, 0.06, 3.9, TEAL)
tb(s, 9.45, 1.95, 3.0, 0.35, "on the old sample", 15, False, GREY, italic=True)
tb(s, 9.45, 2.3, 3.0, 0.9, "+17 pp", 44, True, TEAL, font="Georgia")
tb(s, 9.45, 3.2, 3.0, 0.8, "AllParticles\n22.5 → 39.6%\n(fixed denominator)", 15, False, NAVY, line=1.2)
tb(s, 9.45, 4.25, 3.0, 1.2, "then it saturated:\nmore GNN did not\nmove the chains", 15, False, GREY, italic=True, line=1.25)
tb(s, MX, 5.9, 12.1, 0.45, "That saturation is exactly why this round moved upstream, to the pruning inputs.",
   20, False, NAVY, italic=True)
footer(s)

s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 4  ·  ROADMAP", "Four ideas, in order", "plus two engineering fixes")
items = [("v623", "Physics features"), ("v631", "Ranking + operating point"),
         ("v633/v640", "Remove terms, fix selection"), ("v648", "Competition context"),
         ("v651", "Line-graph attention")]
y = 1.9
for i, (v, t) in enumerate(items):
    rect(s, MX, y, 1.55, 0.62, TEAL if i < 2 else NAVY)
    tb(s, MX + 0.05, y + 0.14, 1.45, 0.35, v, 18, True, WHITE, align=PP_ALIGN.CENTER)
    tb(s, MX + 1.85, y + 0.1, 10.2, 0.45, t, 24, True, NAVY)
    y += 0.86
tb(s, MX, 6.25, 12.1, 0.35, "The next six slides explain the principles; measurements follow.", 15, False, GREY, italic=True)
footer(s)

# --- M1 ---
s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 4  ·  METHOD 1", "Physics features (v623)", "use the quantities a physicist would cut on")
rect(s, MX, 1.65, 6.4, 3.9, WHITE, line=ICE)
tb(s, MX + 0.25, 1.78, 5.9, 0.3, "two tracks from a common vertex", 16, True, NAVY)
rect(s, 1.0, 4.7, 0.2, 0.2, NAVY, shape=MSO_SHAPE.OVAL)
a = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(1.1), Inches(4.8), Inches(3.6), Inches(2.6))
b = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(1.1), Inches(4.8), Inches(5.2), Inches(3.2))
for ln in (a, b): ln.line.color.rgb = NAVY; ln.line.width = Pt(2.4)
rect(s, 1.85, 3.95, 0.7, 0.35, RGBColor(0xD8, 0xDE, 0xEA))
tb(s, 2.05, 3.98, 0.4, 0.28, "θ", 15, True, NAVY)
tb(s, MX + 0.25, 5.15, 5.9, 0.35, "dR · dz · DOCA · |Δz| · collinearity", 17, False, GREY)
rect(s, 7.25, 1.65, 5.45, 3.9, LIGHT); rect(s, 7.25, 1.65, 0.06, 3.9, TEAL)
bullets(s, 7.5, 1.95, 5.0, 3.4, ["Physical basis, so cuts are meaningful", "Computed locally, not inherited", "Result: best AUC, no chain gain"], BODY, NAVY, 14)
note(s, "Before this change the pruning score used raw inherited features. Here we recompute five geometric quantities per track pair",
        "and feed them to the same MLP.", 6.4)
footer(s)

# --- M2 ---
s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 4  ·  METHOD 2", "Pairwise ranking (v631)", "optimise the ordering, not the absolute score")
pic(s, P_RANK, 0.8, 1.75, w=5.6)
rect(s, 6.9, 1.65, 5.8, 2.4, LIGHT); rect(s, 6.9, 1.65, 0.06, 2.4, TEAL)
bullets(s, 7.15, 1.9, 5.35, 2.0, ["0.1% of edges are true", "BCE never orders candidates", "Ordering is what a threshold uses"], 22, NAVY, 10)
formula(s, 6.9, 4.2, 5.8, 1.1, "L = softplus( -(s_true - s_fake)/τ )")
tb(s, 6.9, 5.45, 5.8, 0.4, "hard negatives share a track with the positive", 15, False, GREY, italic=True)
note(s, "The loss compares a true pair against a fake pair that shares one track, so it learns to resolve genuine competition",
        "instead of the easy background.", 6.1)
footer(s)

# --- M3 ---
s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 4  ·  METHOD 3", "The operating point (v631)", "the threshold decides which chains enter")
pic(s, P_THR, 0.8, 1.75, w=5.6)
rect(s, 6.9, 1.65, 5.8, 2.9, LIGHT); rect(s, 6.9, 1.65, 0.06, 2.9, TEAL)
bullets(s, 7.15, 1.9, 5.35, 2.5, ["Lower threshold keeps more true pairs", "It also admits more background", "So this is a trade, not a free gain"], 22, NAVY, 10)
formula(s, 6.9, 4.7, 5.8, 1.1, "kept = { e : s_e > thr }")
note(s, "The threshold is not a detail: it decides how many truth chains reach the reconstruction at all.", 6.05)
footer(s)

# --- M4 ---
s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 4  ·  METHOD 4", "Competition and exclusivity (v648)", "a track has only one true partner")
rect(s, MX, 1.65, 6.3, 3.9, WHITE, line=ICE)
cx, cy = MX + 1.15, 3.55
rect(s, cx, cy, 0.22, 0.22, NAVY, shape=MSO_SHAPE.OVAL)
tb(s, cx - 0.2, cy + 0.28, 0.9, 0.28, "track", 15, False, GREY)
for x, y, col, lab in [(3.3, 2.1, TEAL, "0.91"), (4.4, 2.9, "#9AA3B2", "0.88"),
                       (4.5, 4.1, "#9AA3B2", "0.55"), (3.4, 4.85, "#9AA3B2", "0.31")]:
    ln = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(cx + 0.22), Inches(cy + 0.11), Inches(x + 0.09), Inches(y + 0.09))
    ln.line.color.rgb = _rgb(col); ln.line.width = Pt(1.8)
    rect(s, x, y, 0.18, 0.18, col, shape=MSO_SHAPE.OVAL)
    tb(s, x + 0.24, y - 0.03, 0.8, 0.26, lab, 15, False, GREY)
rect(s, 7.1, 1.65, 5.6, 3.9, LIGHT); rect(s, 7.1, 1.65, 0.06, 3.9, TEAL)
bullets(s, 7.35, 1.95, 5.15, 2.6, ["Per-edge scoring is blind to rivals", "Exclusivity is an assignment problem", "New features: max, mean, rank"], 22, NAVY, 10)
note(s, "Each edge is described by its own score plus summary statistics of the competing edges of the same track, so exclusivity",
        "becomes visible to a per-edge model.", 6.4)
formula(s, 7.1, 4.75, 5.6, 0.75, "f_e = [ s_e , max s_e' , rank ]", 17)
footer(s)

# --- M5a: attention in plain words ---
s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 4  ·  METHOD 5 (1/2)", "Attention, in plain words", "each candidate pair asks the others: did you take my track?")
rect(s, MX, 1.65, 12.1, 2.5, WHITE, line=ICE)
xs = [MX + 1.0, MX + 3.3, MX + 5.6, MX + 7.9, MX + 10.2]
y0 = 3.05
for i, x in enumerate(xs):
    rect(s, x - 0.28, y0 - 0.28, 0.56, 0.56, TEAL, shape=MSO_SHAPE.OVAL)
    tb(s, x - 0.25, y0 - 0.2, 0.5, 0.4, f"e{i+1}", 17, True, WHITE, align=PP_ALIGN.CENTER)
for i in range(len(xs) - 1):
    ln = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(xs[i] + 0.3), Inches(y0), Inches(xs[i + 1] - 0.32), Inches(y0))
    ln.line.color.rgb = RGBColor(0x8E, 0xC9, 0xC0); ln.line.width = Pt(2)
tb(s, MX + 0.2, 3.55, 11.8, 0.4, "the five candidate track pairs of one event", 15, False, GREY, italic=True)
bullets(s, MX, 4.45, 12.1, 2.2, ["A track can only belong to one true pair", "So each pair must know what the others claim", "Attention is exactly that exchange"], BODY, NAVY, 14)
note(s, "Attention runs on the line graph: its nodes are candidate track pairs, not tracks. Each pair aggregates information from",
        "the other pairs of the same event.", 6.4)
footer(s)

# --- M5b: the mechanism + geometry prior ---
s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 4  ·  METHOD 5 (2/2)", "The mechanism, and the geometric prior", "why the bias is the active ingredient")
formula(s, MX, 1.65, 6.2, 2.35,
        "q_i = W_q h_i      k_j = W_k h_j\n"
        "α_ij = softmax( q_i·k_j / √d\n"
        "                + b_ij )\n"
        "h'_i = Σ_j α_ij v_j", 18)
rect(s, 7.0, 1.65, 5.7, 2.35, LIGHT); rect(s, 7.0, 1.65, 0.06, 2.35, TEAL)
bullets(s, 7.25, 1.9, 5.25, 2.0, ["q, k, v: what I ask, what I offer, what I pass", "α: how much attention", "b: a learned map of geometry"], 19, NAVY, 8)
tb(s, MX, 4.15, 6.2, 0.7, "Without geometry, attention has\nnothing physical to attend to.", 18, True, RED, line=1.15)
pic(s, P_BIAS, 7.0, 4.15, w=5.7)
tb(s, MX, 4.8, 6.1, 1.6, "Ablation: the same layer, with and without b_ij.\nOnly the version with geometry moves the reconstruction.", 17, False, NAVY, line=1.2)
note(s, "q, k and v are learned projections of the pair representation; softmax decides how much each neighbour contributes.", 6.45)
footer(s)

# --- geometric prior explainer ---
s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 4  ·  BACKGROUND", "What a geometric prior is, and why it is a bias", None)
bullets(s, MX, 1.75, 12.1, 2.6,
        ["Attention alone compares learned vectors — it has no notion of distance",
         "A prior injects known physics into the attention weights",
         "Here: dR, dz, DOCA are mapped to a scalar b_ij and added before softmax"],
        BODY, NAVY, 16)
formula(s, MX, 4.35, 7.2, 1.0, "score_ij = q_i·k_j/√d  +  b_ij", 20)
rect(s, 8.1, 4.35, 4.6, 1.55, LIGHT); rect(s, 8.1, 4.35, 0.06, 1.55, TEAL)
tb(s, 8.3, 4.48, 4.2, 1.3, "Zero-initialised output: the layer starts as identity,\nso any change is learned, not injected.", 14, False, NAVY, line=1.25)
footer(s)

# --- ablation evidence ---
s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 4  ·  EVIDENCE", "How we know a change is real", "one variable, same base, same threshold")
pic(s, P_NEG, 0.8, 1.8, w=6.4)
rect(s, 7.5, 1.7, 5.2, 3.6, LIGHT); rect(s, 7.5, 1.7, 0.06, 3.6, TEAL)
bullets(s, 7.75, 1.95, 4.8, 3.1, ["Same checkpoint, same data", "One variable per run", "Controls decide: attention alone, node-pool mask"], 20, NAVY, 10)
tb(s, MX, 5.55, 12.1, 0.8, "A control that lands at zero is the strongest evidence that the extra ingredient is doing the work.", 18, False, GREY, italic=True)
footer(s)

# --- measurements ---
s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 4  ·  MEASUREMENTS", "14.5% → 28.0% AllParticles", "denominator = all truth chains (13 255)")
rows = [["Version", "Change", "AllPart. [%]", "Perfect [%]", "AUC", "PV [%]"],
        ["v601", "baseline", "14.5", "9.5", "0.972", "8.61"],
        ["v623", "physics features", "14.3", "9.6", "0.974", "8.61"],
        ["v631", "ranking + threshold", "19.8", "13.3", "0.935", "8.63"],
        ["v633", "remove terms", "21.8", "14.5", "0.924", "8.68"],
        ["v643", "DOCA fix", "26.0", "17.8", "0.924", "8.66"],
        ["v648", "competition", "27.0", "18.1", "0.932", "8.68"],
        ["v651", "attention + bias", "28.0", "19.0", "0.933", "8.64"]]
t = s.shapes.add_table(len(rows), 6, Inches(MX), Inches(1.7), Inches(12.1), Inches(3.5)).table
for i, w in enumerate([1.35, 4.2, 2.15, 2.0, 1.2, 1.2]):
    t.columns[i].width = Inches(w)
for r, row in enumerate(rows):
    for c, val in enumerate(row):
        cell = t.cell(r, c); cell.text = val; cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY if r == 0 else (WHITE if r % 2 else LIGHT)
        for p in cell.text_frame.paragraphs:
            p.alignment = PP_ALIGN.LEFT if c == 1 else PP_ALIGN.CENTER
            for run in p.runs:
                run.font.size = Pt(15 if r else 16); run.font.bold = (r == 0) or (c == 0)
                run.font.color.rgb = WHITE if r == 0 else NAVY; run.font.name = "Calibri"
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
tb(s, MX, 5.45, 12.1, 0.4, "PV association does not move — these changes target reconstruction. Numbers are fixed-denominator.", 15, False, GREY, italic=True)
footer(s)

s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 4  ·  MEASUREMENTS", "Two curves, opposite directions", None)
pic(s, P_LADDER, 0.7, 1.45, w=12.0)
bullets(s, MX, 5.4, 12.1, 1.4, ["AllParticles 14.5 → 28.0%, PerfectReco 9.5 → 19.0%", "AUC falls: the gain is not ranking"], BODY, NAVY, 10)
footer(s)

s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 4  ·  MEASUREMENTS", "What each step bought", "in percentage points")
pic(s, P_DELTA, 1.3, 1.6, w=10.6)
tb(s, MX, 5.35, 12.1, 0.5, "v623 is the only step that costs reconstruction — kept for its ranking, then superseded.", 16, False, GREY, italic=True)
footer(s)

s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 4  ·  BREAKDOWN", "The gain is not uniform in multiplicity", "single-B events are the physics case that matters")
pic(s, "report_figs/reco_fixed/F2_by_nB.png", 0.7, 1.55, w=11.9)
note(s, "Exact reconstruction per event class. Single-B events are the relevant case for triggering and for most analyses; "
        "multi-B events remain harder, and three-B and above are rare enough that we do not optimise for them.", 5.8)
footer(s)

s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "SUMMARY", "Where we are", None)
cards = [("PV association", "13–14% → 8.6% (minIP 10.15%)"),
         ("Reconstruction", "14.5% → 28.0% AllParticles"),
         ("Bugs fixed", "leak sample + DOCA feature"),
         ("Next", "the output stage")]
for i, (h, v) in enumerate(cards):
    xx = 0.8 + (i % 2) * 6.0; yy = 1.85 + (i // 2) * 2.25
    rect(s, xx, yy, 5.7, 2.0, LIGHT); rect(s, xx, yy, 0.07, 2.0, TEAL)
    tb(s, xx + 0.3, yy + 0.2, 5.1, 0.35, h, 20, True, NAVY, font="Georgia")
    tb(s, xx + 0.3, yy + 0.75, 5.1, 1.0, v, 22, False, NAVY)
footer(s)

s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "BACKUP", "Metric definitions", None)
defs = [("AllParticles", "all truth tracks of a chain in one object"),
        ("PerfectReco", "the track set matches the truth exactly"),
        ("PV misassociation", "tracks assigned to the wrong PV"),
        ("edge AUC", "ranking quality of the pruning score"),
        ("LCAG", "generation class of a track pair")]
y = 1.8
for k, v in defs:
    tb(s, MX, y, 3.0, 0.35, k, 18, True, NAVY)
    tb(s, MX + 3.2, y, 9.0, 0.4, v, 18, False, GREY)
    y += 0.75
tb(s, MX, 5.85, 12.1, 0.8, "Two normalisations appear: paper convention (surviving chains) and fixed denominator (all chains).\nThey are not interchangeable.", 15, False, GREY, italic=True)
footer(s)

prs.save(OUT)
print("saved", OUT, os.path.getsize(OUT))
