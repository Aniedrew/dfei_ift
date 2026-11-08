"""Deck v3 — adds principle slides: each new method is explained, not only measured."""
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
LIGHT = RGBColor(0xF3, 0xF6, 0xFB); WHITE = RGBColor(0xFF, 0xFF, 0xFF)
RED = RGBColor(0xC0, 0x39, 0x2B)
CODEBG = RGBColor(0xEE, 0xF1, 0xF7)
C = dict(p="#1E2761", s="#00A896", ice="#CADCFC", grey="#5A6270", red="#C0392B")

plt.rcParams.update({
    "axes.facecolor": "none", "figure.facecolor": "none", "axes.edgecolor": "#9AA3B2",
    "axes.labelcolor": C["p"], "text.color": C["p"], "xtick.color": C["p"], "ytick.color": C["p"],
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.18, "grid.color": "#9AA3B2",
})

VERS = ["v601", "v623", "v631", "v633", "v640", "v643", "v648", "v651"]
RAW_ALL = [25.38, 23.51, 34.25, 40.22, 45.17, 49.55, 56.39, 60.86]
RAW_PERF = [16.76, 15.72, 23.04, 26.64, 29.55, 33.78, 37.84, 41.37]
AUC = [0.9716, 0.9737, 0.9352, 0.9242, None, 0.9236, 0.9324, 0.9327]
PRUNED = [43.05, 39.12, 42.26, 45.71, 49.97, 47.45, 52.12, 54.06]
CONTAM = [42.50, 46.57, 37.97, 32.46, 27.43, 26.51, 20.88, 17.98]
CLEAN = [14.45, 14.31, 19.77, 21.83, 22.60, 26.04, 27.00, 27.96]


def save(fig, name, w, h):
    p = f"{FIGS}/{name}_{int(w*200)}x{int(h*200)}.png"
    fig.savefig(p, dpi=200, transparent=True); plt.close(fig); return p


def fig_pv():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.6, 3.6), dpi=200, gridspec_kw={"width_ratios": [1.45, 1]})
    lab = ["v31", "v38", "v47", "v557", "v601", "v623", "v651"]
    allv = [13.80, 13.40, 13.07, 14.02, 8.61, 8.61, 8.64]
    a1.bar(lab, allv, color=[C["grey"]] * 4 + [C["s"]] * 3, width=0.62)
    a1.axhline(10.15, color=C["red"], ls="--", lw=1.6)
    a1.text(6.5, 10.5, "minIP 10.15", color=C["red"], ha="right", fontsize=9)
    a1.axhline(2.0, color=C["ice"], ls=":", lw=2)
    a1.text(-0.45, 2.4, "leaky line (v560): 2.0", color="#8B93A7", fontsize=8.5)
    for i, v in enumerate(allv): a1.text(i, v + 0.25, f"{v:.1f}", ha="center", fontsize=8.5)
    a1.set_ylim(0, 16.5); a1.set_ylabel("misassociation [%]"); a1.set_title("All tracks", fontsize=11, loc="left")
    a2.bar(["v601", "v651", "minIP"], [16.50, 14.86, 19.99], color=[C["s"], C["s"], C["red"]], width=0.55)
    for i, v in enumerate([16.50, 14.86, 19.99]): a2.text(i, v + 0.3, f"{v:.1f}", ha="center", fontsize=9)
    a2.set_ylim(0, 23); a2.set_title("Signal population", fontsize=11, loc="left")
    fig.tight_layout(); return save(fig, "pv2", 11.6, 3.6)


def fig_decomp():
    fig, ax = plt.subplots(figsize=(11.2, 3.4), dpi=200)
    x = np.arange(len(VERS))
    ax.bar(x, PRUNED, color="#C9D0DD", label="pruned away (lost at pruning)")
    ax.bar(x, CONTAM, bottom=PRUNED, color=C["s"], label="survives, merged with background")
    ax.bar(x, CLEAN, bottom=np.array(PRUNED) + np.array(CONTAM), color=C["p"], label="exact reconstruction")
    for i in range(len(VERS)):
        ax.text(i, PRUNED[i] / 2, f"{PRUNED[i]:.0f}", ha="center", fontsize=8, color="#4A5568")
        ax.text(i, PRUNED[i] + CONTAM[i] / 2, f"{CONTAM[i]:.0f}", ha="center", fontsize=8, color="white")
        ax.text(i, PRUNED[i] + CONTAM[i] + CLEAN[i] / 2, f"{CLEAN[i]:.0f}", ha="center", fontsize=8, color="white")
    ax.set_xticks(x); ax.set_xticklabels(VERS); ax.set_ylim(0, 100)
    ax.set_ylabel("% of all truth chains")
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout(); return save(fig, "decomp", 11.2, 3.4)


def fig_scatter():
    fig, ax = plt.subplots(figsize=(6.0, 3.5), dpi=200)
    a = [(AUC[i], RAW_ALL[i], VERS[i]) for i in range(len(VERS)) if AUC[i]]
    ax.scatter([t[0] for t in a], [t[1] for t in a], s=55, color=C["s"], zorder=3)
    for x, y, v in a: ax.annotate(v, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8.5)
    ax.set_xlabel("edge AUC  (ranking quality)"); ax.set_ylabel("AllParticles [%]")
    ax.set_xlim(0.915, 0.985); ax.set_ylim(18, 66)
    ax.set_title("Better ranking does not mean better chains", fontsize=11, loc="left")
    fig.tight_layout(); return save(fig, "scatter", 6.0, 3.5)


def fig_ladder():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.6, 3.4), dpi=200)
    a1.plot(VERS, RAW_ALL, "o-", color=C["p"], lw=2.2, ms=5, label="AllParticles")
    a1.plot(VERS, RAW_PERF, "s-", color=C["s"], lw=2.2, ms=5, label="PerfectReco")
    for i in (0, len(VERS) - 1):
        a1.annotate(f"{RAW_ALL[i]:.1f}", (i, RAW_ALL[i]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.5)
        a1.annotate(f"{RAW_PERF[i]:.1f}", (i, RAW_PERF[i]), textcoords="offset points", xytext=(0, -13), ha="center", fontsize=8.5)
    a1.set_ylim(10, 70); a1.set_ylabel("[%]"); a1.legend(frameon=False, fontsize=9, loc="lower right")
    a1.set_title("Reconstruction rises …", fontsize=11, loc="left")
    idx = [t for t in range(len(VERS)) if AUC[t]]
    a2.plot([VERS[i] for i in idx], [AUC[i] for i in idx], "o-", color=C["grey"], lw=2.2, ms=5)
    a2.set_ylim(0.915, 0.985); a2.set_ylabel("edge AUC"); a2.set_title("… while ranking quality falls", fontsize=11, loc="left")
    fig.tight_layout(); return save(fig, "ladder2", 11.6, 3.4)


def fig_delta():
    steps = ["v623\nfeatures", "v631\noperating pt", "v633\nremove", "v640\nselection",
             "v643\nDOCA fix", "v648\ncontext", "v651\nattention"]
    da = [RAW_ALL[i + 1] - RAW_ALL[i] for i in range(7)]
    dp = [RAW_PERF[i + 1] - RAW_PERF[i] for i in range(7)]
    fig, ax = plt.subplots(figsize=(11.2, 3.3), dpi=200)
    y = np.arange(len(steps)); h = 0.36
    ax.barh(y + h / 2, da, h, color=C["p"], label="Δ AllParticles")
    ax.barh(y - h / 2, dp, h, color=C["s"], label="Δ PerfectReco")
    for i, (u, v) in enumerate(zip(da, dp)):
        ax.text(u + (0.25 if u >= 0 else -0.25), i + h / 2, f"{u:+.1f}", va="center", ha="left" if u >= 0 else "right", fontsize=8.5)
        ax.text(v + (0.25 if v >= 0 else -0.25), i - h / 2, f"{v:+.1f}", va="center", ha="left" if v >= 0 else "right", fontsize=8.5)
    ax.set_yticks(y); ax.set_yticklabels(steps); ax.invert_yaxis(); ax.axvline(0, color="#9AA3B2", lw=1)
    ax.set_xlabel("change vs the previous version [pp]"); ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout(); return save(fig, "delta", 11.2, 3.3)


def fig_neg():
    names = ["attention alone\n(v637)", "mask on node pooling\n(v805)", "geom. candidates\ntight (v1003)",
             "geom. candidates\nboth (v1005)", "slot decode\n(v803)"]
    vals = [0.0, -0.82, -3.20, -7.39, -23.92]
    fig, ax = plt.subplots(figsize=(7.2, 3.1), dpi=200)
    ax.barh(names, vals, color=[C["grey"]] + [C["red"]] * 4, height=0.55)
    for i, v in enumerate(vals):
        ax.text(v - 0.5 if v < 0 else v + 0.3, i, f"{v:+.1f}", va="center", ha="right" if v < 0 else "left", fontsize=9)
    ax.axvline(0, color="#9AA3B2", lw=1); ax.invert_yaxis(); ax.set_xlim(-28, 3)
    ax.set_xlabel("Δ frontier residual (0 = no change)")
    fig.tight_layout(); return save(fig, "neg", 7.2, 3.1)


def fig_doca():
    fig, ax = plt.subplots(figsize=(4.8, 3.1), dpi=200)
    x = [0, 1]; wd = 0.32
    a = [40.22, 49.55]; p = [26.64, 33.78]
    ax.bar([i - wd / 2 for i in x], a, wd, color=C["p"], label="AllParticles")
    ax.bar([i + wd / 2 for i in x], p, wd, color=C["s"], label="PerfectReco")
    for i in range(2):
        ax.text(i - wd / 2, a[i] + 0.8, f"{a[i]:.1f}", ha="center", fontsize=9)
        ax.text(i + wd / 2, p[i] + 0.8, f"{p[i]:.1f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(["v633\nbuggy DOCA", "v643\ncorrected"])
    ax.set_ylim(0, 58); ax.set_ylabel("[%]"); ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout(); return save(fig, "doca", 4.8, 3.1)


# ---------- principle figures ----------
def fig_rank_loss():
    fig, ax = plt.subplots(figsize=(5.6, 2.9), dpi=200)
    x = np.linspace(0, 1, 300)
    neg = 0.9 * np.exp(-((x - 0.42) ** 2) / 0.020)
    pos = 0.9 * np.exp(-((x - 0.76) ** 2) / 0.016)
    ax.plot(x, neg, color="#9AA3B2", lw=1.8); ax.fill_between(x, neg, color="#9AA3B2", alpha=0.22)
    ax.plot(x, pos, color=C["s"], lw=1.8); ax.fill_between(x, pos, color=C["s"], alpha=0.22)
    ax.annotate("", xy=(0.76, 0.62), xytext=(0.42, 0.62),
                arrowprops=dict(arrowstyle="<->", color=C["p"], lw=1.6))
    ax.text(0.59, 0.66, "margin", ha="center", fontsize=8.5, color=C["p"])
    ax.scatter([0.42], [0.22], color="#9AA3B2", zorder=3)
    ax.scatter([0.76], [0.24], color=C["s"], zorder=3)
    ax.text(0.37, 0.08, "same track,\nfake partner", fontsize=7.5, color="#6B7280", ha="right")
    ax.text(0.80, 0.08, "same track,\ntrue partner", fontsize=7.5, color=C["s"])
    ax.set_yticks([]); ax.set_xlim(0, 1); ax.set_xlabel("pruning score")
    fig.tight_layout(); return save(fig, "rank_loss", 5.6, 2.9)


def fig_threshold():
    fig, ax = plt.subplots(figsize=(5.6, 2.9), dpi=200)
    x = np.linspace(0, 1, 400)
    bg = 1.0 * np.exp(-((x - 0.38) ** 2) / 0.016)
    sig = 0.55 * np.exp(-((x - 0.80) ** 2) / 0.018)
    ax.plot(x, bg, color="#9AA3B2", lw=1.8); ax.fill_between(x, bg, color="#9AA3B2", alpha=0.20)
    ax.plot(x, sig, color=C["s"], lw=1.8); ax.fill_between(x, sig, color=C["s"], alpha=0.22)
    ax.axvspan(0.90, 1.0, color=C["s"], alpha=0.10)
    ax.axvspan(0.90, 0.97, color=C["red"], alpha=0.10)
    ax.axvline(0.97, color=C["red"], ls=":", lw=1.3); ax.axvline(0.90, color=C["red"], ls="--", lw=1.4)
    ax.text(0.965, 0.9, "thr 0.97", rotation=90, va="top", ha="right", fontsize=8, color=C["red"])
    ax.text(0.895, 0.9, "thr 0.90", rotation=90, va="top", ha="right", fontsize=8, color=C["red"])
    ax.annotate("admitted by 0.90,\nnot by 0.97", xy=(0.935, 0.12), xytext=(0.66, 0.55),
                fontsize=7.5, color="#4A5568",
                arrowprops=dict(arrowstyle="->", color="#9AA3B2", lw=1))
    ax.set_yticks([]); ax.set_xlim(0, 1); ax.set_xlabel("pruning score")
    fig.tight_layout(); return save(fig, "threshold", 5.6, 2.9)


P_PV, P_DECOMP, P_SCAT = fig_pv(), fig_decomp(), fig_scatter()
P_LADDER, P_DELTA, P_NEG, P_DOCA = fig_ladder(), fig_delta(), fig_neg(), fig_doca()
P_RANK, P_THR = fig_rank_loss(), fig_threshold()

prs = Presentation(); prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
W, H, MX = 13.333, 7.5, 0.55


def _rgb(v):
    if isinstance(v, str):
        v = v.lstrip("#"); return RGBColor(int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16))
    return v


def tb(s, x, y, w, h, text, size=14, bold=False, color=NAVY, align=PP_ALIGN.LEFT,
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


def rect(s, x, y, w, h, fill, line=None, shape=MSO_SHAPE.RECTANGLE, radius=None, lw=1.2):
    sh = s.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb = _rgb(fill)
    if line is None: sh.line.fill.background()
    else:
        sh.line.color.rgb = _rgb(line); sh.line.width = Pt(lw)
    sh.shadow.inherit = False
    if radius is not None and shape == MSO_SHAPE.ROUNDED_RECTANGLE: sh.adjustments[0] = radius
    return sh


def bullets(s, x, y, w, h, items, size=12, color=NAVY, gap=6):
    box = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame; tf.word_wrap = True
    for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, m, 0)
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(gap)
        r = p.add_run(); r.text = "—  " + it
        r.font.size = Pt(size); r.font.color.rgb = color; r.font.name = "Calibri"
    return box


def formula(s, x, y, w, h, lines, size=11.5):
    rect(s, x, y, w, h, CODEBG)
    tb(s, x + 0.16, y + 0.12, w - 0.3, h - 0.2, lines, size, False, NAVY, font="Consolas", line=1.3)


def header(s, kicker, title, sub=None):
    rect(s, 0, 0, W, 0.09, TEAL)
    tb(s, MX, 0.3, 11.5, 0.28, kicker, 11, True, TEAL)
    tb(s, MX, 0.56, 12.3, 0.6, title, 25, True, NAVY, font="Georgia")
    if sub: tb(s, MX, 1.13, 12.3, 0.34, sub, 12.5, False, GREY, italic=True)


def footer(s):
    tb(s, MX, H - 0.36, 7.0, 0.22, "DFEI · bug fixes and pruning optimization", 8.5, False, GREY)
    tb(s, W - 1.0, H - 0.36, 0.5, 0.22, str(len(prs.slides)), 8.5, True, GREY, align=PP_ALIGN.RIGHT)


def pic(s, path, x, y, w=None, h=None):
    if w: s.shapes.add_picture(path, Inches(x), Inches(y), width=Inches(w))
    else: s.shapes.add_picture(path, Inches(x), Inches(y), height=Inches(h))


def cover():
    s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, NAVY); rect(s, 0, 0, 0.14, H, TEAL)
    tb(s, 0.95, 1.8, 11.6, 0.35, "DFEI GROUP MEETING  ·  2026-10-23", 11.5, True, TEAL)
    tb(s, 0.95, 2.2, 11.6, 1.7, "DFEI: Bug Fixes and\nPruning Optimization", 40, True, WHITE, font="Georgia", line=1.12)
    tb(s, 0.95, 3.95, 11.0, 0.45, "From a corrected simulation sample to a stronger pruning stage", 15.5, False, ICE, italic=True)
    rect(s, 0.95, 4.65, 1.5, 0.04, TEAL)
    tb(s, 0.95, 4.85, 8.0, 0.3, "Qingxiang Guo  ·  UCAS", 13.5, False, WHITE)
    tb(s, 0.95, 6.6, 11.0, 0.28, "All numbers on the corrected sample unless stated otherwise", 9.5, False, ICE)


def toc():
    s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, LIGHT)
    header(s, "CONTENTS", "Four questions, four answers")
    rows = [("1", "What did the leak fix cost?", "PV association fell behind minIP; a PV-aware head brought it back"),
            ("2", "Where does the reconstruction loss sit?", "two failure modes — and only one is fixable downstream"),
            ("3", "What was wrong with DOCA?", "a silent feature corruption worth ~7 pp of perfect reconstruction"),
            ("4", "How far did pruning optimization get?", "four methods explained, then measured one change at a time")]
    y = 1.72
    for i, (n, t, d) in enumerate(rows):
        rect(s, MX, y, 12.25, 1.06, WHITE); rect(s, MX, y, 0.06, 1.06, TEAL if i % 2 == 0 else NAVY)
        tb(s, MX + 0.24, y + 0.15, 0.7, 0.6, n, 25, True, TEAL if i % 2 == 0 else NAVY, font="Georgia")
        tb(s, MX + 1.0, y + 0.12, 10.9, 0.32, t, 16, True, NAVY)
        tb(s, MX + 1.0, y + 0.53, 11.0, 0.32, d, 11.5, False, GREY, italic=True)
        y += 1.2
    footer(s)


def divider(num, title, sub, items):
    s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, NAVY)
    tb(s, 0.9, 1.5, 3.0, 0.35, f"PART {num}", 12.5, True, TEAL)
    tb(s, 0.9, 1.9, 11.4, 1.0, title, 33, True, WHITE, font="Georgia")
    tb(s, 0.9, 3.0, 11.0, 0.4, sub, 14.5, False, ICE, italic=True)
    rect(s, 0.9, 3.6, 1.4, 0.04, TEAL)
    bullets(s, 0.9, 3.9, 11.0, 2.3, items, 13, WHITE, 9)


cover(); toc()

# ================= PART 1 =================
divider(1, "What the leak fix cost", "One metric moved the wrong way",
        ["The corrected sample removes truth-level information from the inputs",
         "PV association is the metric that lost the most",
         "A PV-aware pruning head brought it back below the minIP baseline"])

s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 1  ·  CONTEXT", "Where pruning sits in DFEI", "Everything downstream inherits whatever pruning keeps")
boxes = [("Track graph\n(~150 tracks)", 0.75, GREY), ("Node pruning\n(background out)", 3.15, TEAL),
         ("Edge pruning\n(false pairs out)", 5.55, TEAL), ("LCAG classification\n(4 classes per pair)", 7.95, NAVY),
         ("Chain assembly\n(connected components)", 10.35, NAVY)]
for txt, x, col in boxes:
    rect(s, x, 2.1, 2.2, 1.15, WHITE, line=col, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.12)
    tb(s, x + 0.12, 2.33, 1.96, 0.75, txt, 11, True, col, align=PP_ALIGN.CENTER)
for x in (2.98, 5.38, 7.78, 10.18):
    c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x), Inches(2.67), Inches(x + 0.16), Inches(2.67))
    c.line.color.rgb = GREY; c.line.width = Pt(1.6)
rect(s, 2.55, 3.55, 4.05, 0.6, RGBColor(0xE6, 0xF4, 0xF1))
tb(s, 2.66, 3.66, 3.9, 0.4, "our work: these two stages", 11, True, TEAL)
rect(s, 9.45, 3.55, 3.05, 0.6, RGBColor(0xEE, 0xEE, 0xF6))
tb(s, 9.56, 3.66, 2.9, 0.4, "next round", 11, True, NAVY)
bullets(s, MX, 4.6, 12.2, 1.7, [
    "Pruning decides which truth chains even enter the reconstruction — a chain lost here can never be recovered",
    "PV association shares the same pruning representation, which is why one change can move both"], 12.5, NAVY, 8)
footer(s)

s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 1  ·  PV ASSOCIATION", "Old models fall behind minIP; retraining plus a PV head recovers it",
       "Misassociation rate, lower is better")
pic(s, P_PV, 0.45, 1.6, w=12.4)
rect(s, MX, 5.5, 12.25, 1.2, LIGHT); rect(s, MX, 5.5, 0.05, 1.2, TEAL)
bullets(s, 0.78, 5.63, 11.9, 1.0, [
    "Models trained on the leaky sample: 13.1–14.0% on the corrected sample — every one worse than minIP (10.15%)",
    "Retrained with a PV-aware pruning head: 8.6% (all tracks), 14.9% (signal) — below minIP, far from the 2.0% the leak suggested"],
    10.5, NAVY, 4)
footer(s)

s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 1  ·  THE FIX", "What the PV-aware pruning head does", "An auxiliary target on the same representation")
rect(s, MX, 1.65, 6.35, 3.55, WHITE, line=ICE)
tb(s, MX + 0.2, 1.76, 5.9, 0.28, "The head predicts a PV for every surviving track", 11.5, True, NAVY, font="Georgia")
for i, yy in enumerate([2.3, 2.75, 3.2]):
    rect(s, MX + 0.3, yy, 0.16, 0.16, NAVY, shape=MSO_SHAPE.OVAL)
    tb(s, MX + 0.55, yy - 0.05, 0.7, 0.25, f"PV {i+1}", 9, False, GREY)
trk = [(3.4, 2.36, TEAL, "t1"), (3.9, 2.62, TEAL, "t2"), (4.5, 3.05, TEAL, "t3"),
       (5.0, 2.45, "#9AA3B2", "t4"), (5.4, 3.3, "#9AA3B2", "t5")]
for x, y, col, lab in trk:
    rect(s, x, y, 0.14, 0.14, col, shape=MSO_SHAPE.OVAL)
    tb(s, x - 0.02, y + 0.16, 0.4, 0.2, lab, 8, False, GREY)
for x, y, *_ in trk[:3]:
    ln = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x + 0.14), Inches(y + 0.07), Inches(MX + 0.46), Inches(2.38))
    ln.line.color.rgb = TEAL; ln.line.width = Pt(1.1); ln.line.dash_style = 4
for x, y, *_ in trk[3:]:
    ln = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x + 0.14), Inches(y + 0.07), Inches(MX + 0.46), Inches(2.83))
    ln.line.color.rgb = rgb_grey = RGBColor(0x9A, 0xA3, 0xB2); ln.line.width = Pt(1.1); ln.line.dash_style = 4
tb(s, MX + 0.3, 4.55, 5.8, 0.5, "Assignment is soft: a track contributes to the PV loss with a weight,\nso ambiguous tracks do not dominate.", 9.5, False, GREY, italic=True)
rect(s, 7.15, 1.65, 5.65, 3.55, LIGHT); rect(s, 7.15, 1.65, 0.05, 3.55, TEAL)
tb(s, 7.4, 1.8, 5.1, 0.3, "Why it changes pruning", 13.5, True, NAVY, font="Georgia")
bullets(s, 7.4, 2.25, 5.15, 2.8, [
    "PV assignment is a property of the whole event, not of one track",
    "Learning it forces the shared representation to encode detector geometry",
    "Those same features make track–track edges easier to judge",
    "Effect: 13–14% → 8.6%, with no loss in reconstruction"], 10.5, NAVY, 6)
footer(s)

# ================= PART 2 =================
divider(2, "Where the loss actually sits", "Two failure modes, one lever",
        ["A truth chain either disappears at pruning or survives and is merged with background",
         "Assembly never fails on its own: every surviving chain is found",
         "So the pruning stage sets both terms"])

s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 2  ·  DIAGNOSIS", "Each step trades one failure mode for the other",
       "All 13 255 truth chains of the corrected sample, fixed denominator")
pic(s, P_DECOMP, 0.55, 1.55, w=12.2)
tb(s, MX, 5.4, 12.2, 0.28, "Grey = lost at pruning.  Teal = found but merged with background.  Navy = exact reconstruction.", 10, False, GREY)
rect(s, MX, 5.78, 12.25, 0.95, LIGHT); rect(s, MX, 5.78, 0.05, 0.95, TEAL)
tb(s, 0.78, 5.9, 11.9, 0.75,
   "The exchange is visible: v631 and v633 buy exact chains by dropping more of them; v643 and v648 shrink the contaminated band.\n"
   "No version shrinks both at once — that is the frontier we are trying to break.", 10.2, False, NAVY)
footer(s)

s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 2  ·  A WARNING FROM THE DATA", "Ranking quality and reconstruction quality come apart",
       "Every version with both measurements on the corrected sample")
pic(s, P_SCAT, 0.6, 1.7, w=6.1)
rect(s, 7.15, 1.7, 5.65, 3.5, LIGHT); rect(s, 7.15, 1.7, 0.05, 3.5, TEAL)
tb(s, 7.4, 1.85, 5.1, 0.3, "Why we report four benchmarks", 13.5, True, NAVY, font="Georgia")
bullets(s, 7.4, 2.3, 5.15, 2.7, [
    "v601 has the best edge AUC and one of the worst reconstruction rates",
    "v651 has a lower AUC and the best reconstruction",
    "A ranking number cannot tell us whether a change helped the physics",
    "So AUC is a diagnostic, never the acceptance criterion"], 10.5, NAVY, 7)
footer(s)

# ================= PART 3 =================
divider(3, "The DOCA bug", "A feature that was silently wrong",
        ["Distance of closest approach was inherited from an upstream quantity",
         "That quantity is not the geometric DOCA",
         "The pruning MLP learned from a biased input for months"])

s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 3  ·  THE BUG", "What DOCA is, and what we were feeding", "No crash, no NaN — just a wrong number")
rect(s, MX, 1.6, 6.15, 3.9, WHITE, line=ICE)
tb(s, MX + 0.2, 1.7, 5.7, 0.26, "Geometric quantity (what the name promises)", 11, True, NAVY, font="Georgia")
rect(s, 1.05, 4.55, 0.18, 0.18, NAVY, shape=MSO_SHAPE.OVAL)
tb(s, 1.28, 4.5, 0.9, 0.24, "PV", 9, False, GREY)
l1 = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(1.14), Inches(4.64), Inches(3.95), Inches(2.55))
l2 = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(1.14), Inches(4.64), Inches(5.15), Inches(3.35))
for ln in (l1, l2): ln.line.color.rgb = NAVY; ln.line.width = Pt(2.2)
rect(s, 3.95, 2.45, 0.16, 0.16, TEAL, shape=MSO_SHAPE.OVAL)
rect(s, 5.15, 3.25, 0.16, 0.16, TEAL, shape=MSO_SHAPE.OVAL)
tb(s, 4.1, 2.2, 1.3, 0.24, "track 1", 9, False, GREY)
tb(s, 5.3, 3.3, 1.3, 0.24, "track 2", 9, False, GREY)
ln = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(4.02), Inches(2.62), Inches(4.98), Inches(3.62))
ln.line.color.rgb = RED; ln.line.width = Pt(1.8); ln.line.dash_style = 4
tb(s, 3.3, 3.35, 2.8, 0.3, "DOCA: shortest distance between the two lines", 9, False, RED, italic=True)
tb(s, MX + 0.2, 5.05, 5.8, 0.35, "dR = sqrt(Δη² + Δφ²)   ·   collinearity = cos θ", 9.5, False, GREY)
rect(s, 6.9, 1.6, 5.9, 3.9, LIGHT); rect(s, 6.9, 1.6, 0.05, 3.9, RED)
tb(s, 7.15, 1.75, 5.4, 0.28, "What the network actually received", 13.5, True, NAVY, font="Georgia")
bullets(s, 7.15, 2.2, 5.4, 3.2, [
    "log DOCA was read from an upstream reconstructed quantity",
    "Its definition differs from the geometric DOCA on the left",
    "So the MLP learned a mapping from a biased feature",
    "Nothing crashed: the score stayed plausible, which is why it survived",
    "Fix: recompute dR, dz, DOCA, |Δz| locally from track states"], 10.2, NAVY, 6)
footer(s)

s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 3  ·  THE EFFECT", "Correcting DOCA: +9.3 pp AllParticles, +7.1 pp PerfectReco",
       "Same sample, same threshold, one variable changed")
pic(s, P_DOCA, 0.8, 1.85, w=5.6)
rect(s, 6.75, 1.85, 6.05, 3.55, LIGHT); rect(s, 6.75, 1.85, 0.05, 3.55, TEAL)
tb(s, 7.0, 2.0, 5.5, 0.3, "Read carefully", 13.5, True, NAVY, font="Georgia")
bullets(s, 7.0, 2.45, 5.55, 2.8, [
    "Edge AUC is flat (0.924 → 0.924): the fix restores feature correctness, not ranking power",
    "The gain appears in assembled chains — the quantity we care about",
    "Cleanest single-variable improvement in the whole line"], 10.5, NAVY, 8)
footer(s)

# ================= PART 4 =================
divider(4, "Pruning optimization", "Four methods, explained before they are measured",
        ["Physics-derived features · pairwise ranking · competition context · line-graph attention",
         "Each one is a controlled change on top of the previous best",
         "Benchmarks: AllParticles, PerfectReco, edge AUC, PV association"])

# --- roadmap ---
s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 4  ·  ROADMAP", "Four ideas, in the order we tried them", "Plus two engineering fixes that are not new methods")
items = [("v623", "Physics features", "make the input basis physical"),
         ("v631", "Pairwise ranking + operating point", "what the score is optimised for"),
         ("v633 / v640", "Remove + select", "engineering: drop bad terms, fix selection"),
         ("v648", "Competition context", "break per-edge independence"),
         ("v651", "Line-graph attention", "let edges exchange information")]
y = 1.75
for i, (v, t, d) in enumerate(items):
    rect(s, MX, y, 0.95, 0.52, TEAL if i < 2 else NAVY)
    tb(s, MX + 0.05, y + 0.13, 0.85, 0.3, v, 12, True, WHITE, align=PP_ALIGN.CENTER)
    tb(s, MX + 1.15, y + 0.02, 5.0, 0.28, t, 14, True, NAVY)
    tb(s, MX + 1.15, y + 0.3, 10.9, 0.25, d, 10.5, False, GREY, italic=True)
    y += 0.78
tb(s, MX, 5.7, 12.2, 0.35, "Model selection (v640) is the other engineering fix: the total loss mixes auxiliary terms, so we select the checkpoint by pruning AP instead.", 9.8, False, GREY, italic=True)
tb(s, MX, 6.05, 12.2, 0.35, "The next five slides explain the principles; the measurements follow on the ladder.", 10.2, False, GREY, italic=True)
footer(s)

# --- M1: physics features ---
s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 4  ·  METHOD 1", "Physics-derived features (v623)", "Replace inherited numbers with quantities a physicist would cut on")
rect(s, MX, 1.6, 6.5, 3.9, WHITE, line=ICE)
tb(s, MX + 0.2, 1.7, 6.0, 0.26, "Two tracks from a common vertex", 11, True, NAVY, font="Georgia")
rect(s, 1.0, 4.45, 0.16, 0.16, NAVY, shape=MSO_SHAPE.OVAL); tb(s, 0.75, 4.65, 0.7, 0.22, "PV", 9, False, GREY)
a = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(1.08), Inches(4.53), Inches(3.6), Inches(2.7))
b = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(1.08), Inches(4.53), Inches(5.2), Inches(3.1))
for ln in (a, b): ln.line.color.rgb = NAVY; ln.line.width = Pt(2)
tb(s, 3.3, 2.35, 1.2, 0.24, "track 1", 9, False, GREY)
tb(s, 5.0, 3.18, 1.2, 0.24, "track 2", 9, False, GREY)
# angle arc
rect(s, 2.0, 3.85, 0.55, 0.18, RGBColor(0xD5, 0xDB, 0xE8))
tb(s, 2.15, 3.88, 0.4, 0.2, "θ", 9, True, NAVY)
tb(s, MX + 0.2, 5.05, 6.1, 0.35, "dR = sqrt(Δη²+Δφ²)  ·  |Δz|  ·  DOCA  ·  collinearity = cos θ", 9.2, False, GREY)
rect(s, 7.3, 1.6, 5.5, 3.9, LIGHT); rect(s, 7.3, 1.6, 0.05, 3.9, TEAL)
tb(s, 7.55, 1.75, 5.0, 0.28, "Why these five", 13.5, True, NAVY, font="Georgia")
bullets(s, 7.55, 2.2, 5.0, 3.2, [
    "They are the quantities a physicist cuts on, so the MLP can learn cuts in a meaningful basis",
    "They are computed from track states, so they do not inherit upstream conventions",
    "Zero-initialised adapters keep the old behaviour as the starting point",
    "Result: best edge AUC of the line (0.974), but no chain gain yet"], 10.2, NAVY, 6)
footer(s)

# --- M2: ranking loss ---
s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 4  ·  METHOD 2", "Pairwise ranking (v631)", "Optimise the ordering, not the absolute score")
pic(s, P_RANK, 0.7, 1.7, w=5.9)
rect(s, 6.85, 1.65, 5.95, 3.0, LIGHT); rect(s, 6.85, 1.65, 0.05, 3.0, TEAL)
tb(s, 7.1, 1.8, 5.4, 0.28, "The problem with plain BCE", 13, True, NAVY, font="Georgia")
bullets(s, 7.1, 2.2, 5.45, 2.3, [
    "Only ~0.1% of candidate edges are true, so pos_weight ≈ 700",
    "The gradient then only pushes true edges up — the ordering among candidates is never optimised",
    "At a low threshold, what matters is precisely that ordering"], 10.2, NAVY, 6)
formula(s, 0.7, 4.85, 5.9, 0.8, "L = mean over (true, fake) pairs\n    softplus( -( s_true - s_fake ) / τ )")
rect(s, 6.85, 4.85, 5.95, 0.8, WHITE, line=ICE)
tb(s, 7.05, 4.95, 5.6, 0.6, "Hard negatives share a track with the positive:\n the loss compares candidates that genuinely compete.", 9.8, False, GREY, italic=True)
footer(s)

# --- M3: operating point ---
s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 4  ·  METHOD 3", "The operating point (v631)", "The threshold decides which chains even enter the reconstruction")
pic(s, P_THR, 0.7, 1.7, w=5.9)
rect(s, 6.85, 1.65, 5.95, 3.0, LIGHT); rect(s, 6.85, 1.65, 0.05, 3.0, TEAL)
tb(s, 7.1, 1.8, 5.4, 0.28, "Why 0.90 and not 0.97", 13, True, NAVY, font="Georgia")
bullets(s, 7.1, 2.2, 5.45, 2.3, [
    "Lowering the threshold admits more true pairs (good: fewer chains destroyed)",
    "It also admits more background pairs (bad: more contamination)",
    "There is no free lunch — the gain is a different point on the same curve",
    "That is why the ranking loss must come with it, to keep the background side in check"], 9.8, NAVY, 5)
formula(s, 0.7, 4.85, 5.9, 0.8, "kept edges = { e : s_e > thr }\nretention ↑  →  chain recall ↑, contamination ↑")
rect(s, 6.85, 4.85, 5.95, 0.8, WHITE, line=ICE)
tb(s, 7.05, 4.95, 5.6, 0.6, "Measured: AllParticles +10.8 pp, PerfectReco +7.3 pp\n— the largest single step in the line.", 9.8, False, GREY, italic=True)
footer(s)

# --- M4: competition ---
s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 4  ·  METHOD 4", "Competition and exclusivity (v648)", "A track has only one true partner — so rivals are evidence")
rect(s, MX, 1.6, 6.3, 3.9, WHITE, line=ICE)
tb(s, MX + 0.2, 1.7, 5.9, 0.26, "One track, four candidate partners", 11, True, NAVY, font="Georgia")
cx, cy = MX + 1.1, 3.6
rect(s, cx, cy, 0.2, 0.2, NAVY, shape=MSO_SHAPE.OVAL); tb(s, cx - 0.15, cy + 0.24, 0.9, 0.22, "track", 9, False, GREY)
for x, y, col, lab in [(3.0, 2.25, TEAL, "true  s=0.91"), (4.2, 2.95, "#9AA3B2", "fake  s=0.88"),
                       (4.35, 4.05, "#9AA3B2", "fake  s=0.55"), (3.1, 4.75, "#9AA3B2", "fake  s=0.31")]:
    ln = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(cx + 0.2), Inches(cy + 0.1), Inches(x + 0.08), Inches(y + 0.08))
    ln.line.color.rgb = _rgb(col); ln.line.width = Pt(1.6)
    rect(s, x, y, 0.16, 0.16, col, shape=MSO_SHAPE.OVAL); tb(s, x + 0.2, y - 0.02, 1.5, 0.22, lab, 8.2, False, GREY)
tb(s, MX + 0.25, 5.05, 5.9, 0.35, "The fake at 0.88 beats most edges in the event — per-edge scoring cannot tell.", 9.2, False, GREY, italic=True)
rect(s, 7.1, 1.6, 5.7, 3.9, LIGHT); rect(s, 7.1, 1.6, 0.05, 3.9, TEAL)
tb(s, 7.35, 1.75, 5.2, 0.28, "What the new features add", 13, True, NAVY, font="Georgia")
formula(s, 7.35, 2.15, 5.2, 1.05, "f_e = [ s_e ,\n        max_{e'≠e} s_e' ,  mean_{e'≠e} s_e' ,\n        rank(s_e) among its track's edges ]")
bullets(s, 7.35, 3.35, 5.2, 2.0, [
    "Per-edge classification is blind to the competitor set",
    "Exclusivity makes the decision a small assignment problem",
    "Result: first block that improves ranking and chains together"], 9.8, NAVY, 5)
footer(s)

# --- M5: attention ---
s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 4  ·  METHOD 5", "Line-graph attention with a geometric prior (v651)",
       "What attention is, and why it needs the prior")
rect(s, MX, 1.55, 6.55, 3.55, WHITE, line=ICE)
tb(s, MX + 0.2, 1.65, 6.1, 0.26, "Nodes of the line graph = candidate edges", 11, True, NAVY, font="Georgia")
xs = [MX + 0.75, MX + 1.85, MX + 2.95, MX + 4.05, MX + 5.15]
y0 = 3.35
for i in range(len(xs) - 1):
    ln = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(xs[i]), Inches(y0), Inches(xs[i + 1]), Inches(y0))
    ln.line.color.rgb = NAVY; ln.line.width = Pt(1.6)
for i, x in enumerate(xs):
    rect(s, x - 0.1, y0 - 0.1, 0.2, 0.2, TEAL, shape=MSO_SHAPE.OVAL)
    tb(s, x - 0.22, y0 + 0.18, 0.5, 0.2, f"e{i+1}", 8.2, False, GREY, align=PP_ALIGN.CENTER)
# attention arcs
for i, j in [(0, 2), (1, 3), (2, 4)]:
    ln = s.shapes.add_connector(MSO_CONNECTOR.CURVE, Inches(xs[i]), Inches(y0 - 0.12), Inches(xs[j]), Inches(y0 - 0.12))
    ln.line.color.rgb = RGBColor(0x8E, 0xC9, 0xC0); ln.line.width = Pt(1.2)
tb(s, MX + 0.25, 2.55, 6.1, 0.3, "each edge attends to the others in the same event", 9.2, False, GREY, italic=True)
tb(s, MX + 0.25, 4.2, 6.15, 0.7, "The edge's own score is not enough: whether e1 is real depends on whether\ne2 already took its track. Attention lets the edges negotiate.", 9.2, False, GREY)
rect(s, 7.25, 1.55, 5.55, 3.55, LIGHT); rect(s, 7.25, 1.55, 0.05, 3.55, TEAL)
tb(s, 7.5, 1.7, 5.05, 0.28, "The mechanism, in three lines", 13, True, NAVY, font="Georgia")
formula(s, 7.5, 2.1, 5.05, 1.55,
        "q_i = W_q h_i ,  k_j = W_k h_j\n"
        "α_ij = softmax_j( q_i·k_j / √d\n"
        "                + b_ij )\n"
        "h'_i = Σ_j α_ij v_j")
bullets(s, 7.5, 3.75, 5.05, 1.3, [
    "b_ij is a learned map of the pair's geometry (dR, dz, DOCA)",
    "Output projection is zero-initialised: the layer starts as identity"], 9.5, NAVY, 4)
tb(s, MX, 5.25, 12.2, 0.55, "Ablation: remove b_ij and the layer becomes neutral (v637). The attention itself is not the gain — the geometry prior is.",
   10.2, False, RED, italic=True)
footer(s)

# --- ladder table ---
s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 4  ·  MEASUREMENTS", "From 25% to 61% AllParticles", "Paper-convention numbers; read the trend, not the absolute level")
rows = [["Version", "Change", "AllParticles [%]", "PerfectReco [%]", "edge AUC", "PV misassoc. [%]"],
        ["v601", "baseline on the corrected sample", "25.4", "16.8", "0.972", "8.61"],
        ["v623", "physics-derived inputs (M1)", "23.5", "15.7", "0.974", "8.61"],
        ["v631", "ranking + operating point (M2, M3)", "34.3", "23.0", "0.935", "8.63"],
        ["v633", "drop the two negative inputs", "40.2", "26.6", "0.924", "8.68"],
        ["v643", "corrected DOCA geometry", "49.6", "33.8", "0.924", "8.66"],
        ["v648", "competition context (M4)", "56.4", "37.8", "0.932", "8.68"],
        ["v651", "line-graph attention (M5)", "60.9", "41.4", "0.933", "8.64"]]
t = s.shapes.add_table(len(rows), 6, Inches(MX), Inches(1.62), Inches(12.25), Inches(3.2)).table
for i, w in enumerate([1.15, 4.65, 2.0, 1.95, 1.35, 1.25]):
    t.columns[i].width = Inches(w)
for r, row in enumerate(rows):
    for c, val in enumerate(row):
        cell = t.cell(r, c); cell.text = val; cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY if r == 0 else (WHITE if r % 2 else LIGHT)
        for p in cell.text_frame.paragraphs:
            p.alignment = PP_ALIGN.LEFT if c == 1 else PP_ALIGN.CENTER
            for run in p.runs:
                run.font.size = Pt(10.6 if r else 11); run.font.bold = (r == 0) or (c == 0)
                run.font.color.rgb = WHITE if r == 0 else NAVY; run.font.name = "Calibri"
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
tb(s, MX, 5.05, 12.2, 0.3, "PV association is the one column that does not move — these changes were made for reconstruction.", 10, False, GREY, italic=True)
footer(s)

s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 4  ·  MEASUREMENTS", "Two curves that do not run in parallel", None)
pic(s, P_LADDER, 0.55, 1.3, w=12.2)
rect(s, MX, 5.3, 12.25, 1.35, LIGHT); rect(s, MX, 5.3, 0.05, 1.35, TEAL)
bullets(s, 0.8, 5.43, 11.9, 1.1, [
    "AllParticles 25.4 → 60.9%, PerfectReco 16.8 → 41.4%",
    "edge AUC 0.972 → 0.933: the gain does not come from a better ranking",
    "It comes from correct inputs and from the operating point"], 10.6, NAVY, 5)
footer(s)

s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 4  ·  MEASUREMENTS", "The per-step budget", "Change relative to the previous version, in percentage points")
pic(s, P_DELTA, 0.55, 1.55, w=12.2)
tb(s, MX, 5.45, 12.2, 0.4, "v623 is the only step that costs reconstruction — kept because it gave the best edge AUC and became the base for v631.",
   10, False, GREY, italic=True)
footer(s)

s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 4  ·  NEGATIVE RESULTS", "What we tried and discarded", "Δ relative to the frontier; negative means worse than doing nothing")
pic(s, P_NEG, 0.6, 1.65, w=7.7)
rect(s, 8.55, 1.65, 4.25, 3.3, LIGHT); rect(s, 8.55, 1.65, 0.05, 3.3, RED)
tb(s, 8.8, 1.8, 3.8, 0.28, "Also discarded", 13, True, NAVY, font="Georgia")
bullets(s, 8.8, 2.2, 3.85, 2.5, [
    "Bigger capacity: no gain",
    "Auxiliary heads (mass, structure, source): no transfer",
    "Loose chain rewards: more chains kept, more contamination"], 9.8, NAVY, 6)
tb(s, MX, 5.3, 12.2, 0.35, "Reported so that nobody repeats them — each cost at least one controlled run.", 10, False, GREY, italic=True)
footer(s)

# summary + backup
s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, NAVY)
tb(s, 0.9, 0.7, 11.0, 0.35, "SUMMARY", 12, True, TEAL)
tb(s, 0.9, 1.08, 11.6, 0.55, "Where we are", 29, True, WHITE, font="Georgia")
cards = [("PV association", "13–14% on the old models → 8.6% with the PV-aware head; minIP 10.15%"),
         ("Reconstruction", "AllParticles 25.4 → 60.9%; PerfectReco 16.8 → 41.4% in six steps"),
         ("Two bugs fixed", "truth leak in the sample and the DOCA feature — both silent, both measurable"),
         ("Next", "the output stage: edge classification plus connected components is the remaining ceiling")]
for i, (h, v) in enumerate(cards):
    xx = 0.9 + (i % 2) * 5.95; yy = 1.95 + (i // 2) * 2.05
    rect(s, xx, yy, 5.65, 1.8, RGBColor(0x2A, 0x34, 0x75))
    tb(s, xx + 0.25, yy + 0.16, 5.0, 0.28, h, 13.5, True, TEAL, font="Georgia")
    tb(s, xx + 0.25, yy + 0.55, 5.1, 1.1, v, 11.5, False, WHITE, line=1.15)
tb(s, 0.9, 6.5, 11.5, 0.3, "Thank you — questions welcome", 11.5, False, ICE, italic=True)

s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "BACKUP", "Metric definitions", "So the numbers on the previous slides are unambiguous")
defs = [("AllParticles", "a reconstructed object contains all truth tracks of a chain (extra tracks allowed)"),
        ("PerfectReco", "the reconstructed track set equals the truth track set exactly"),
        ("PV misassociation", "fraction of tracks assigned to a PV different from the true one; HGNN vs minIP"),
        ("edge AUC", "ranking quality of the pruning score over all candidate edges"),
        ("LCAG", "lowest-common-ancestor generation class of a track pair; we quote class-1 accuracy")]
y = 1.8
for k, v in defs:
    tb(s, MX, y, 2.6, 0.3, k, 11.8, True, NAVY)
    tb(s, MX + 2.75, y, 9.6, 0.5, v, 10.8, False, GREY)
    y += 0.74
tb(s, MX, 5.95, 12.2, 0.7,
   "Two normalisations appear in this deck: the paper convention (denominator = truth chains surviving pruning) on the measurement slides,\n"
   "and a fixed denominator (all truth chains) on the diagnosis slide. The two must not be compared directly.",
   9.6, False, GREY, italic=True)
footer(s)

prs.save(OUT)
print("saved", OUT, os.path.getsize(OUT))
