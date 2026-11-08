"""Friday report deck v2 — much more visual: every slide carries a figure or a diagram."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
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
C = dict(p="#1E2761", s="#00A896", ice="#CADCFC", grey="#5A6270", red="#C0392B", light="#E8EDF6")

plt.rcParams.update({
    "axes.facecolor": "none", "figure.facecolor": "none",
    "axes.edgecolor": "#9AA3B2", "axes.labelcolor": C["p"],
    "text.color": C["p"], "xtick.color": C["p"], "ytick.color": C["p"],
    "font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.18, "grid.color": "#9AA3B2",
})

VERS = ["v601", "v623", "v631", "v633", "v640", "v643", "v648", "v651"]
RAW_ALL = [25.38, 23.51, 34.25, 40.22, 45.17, 49.55, 56.39, 60.86]
RAW_PERF = [16.76, 15.72, 23.04, 26.64, 29.55, 33.78, 37.84, 41.37]
AUC = [0.9716, 0.9737, 0.9352, 0.9242, None, 0.9236, 0.9324, 0.9327]
# fixed-denominator accounting (all truth chains = 13255)
PRUNED = [43.05, 39.12, 42.26, 45.71, 49.97, 47.45, 52.12, 54.06]
CONTAM = [42.50, 46.57, 37.97, 32.46, 27.43, 26.51, 20.88, 17.98]
CLEAN = [14.45, 14.31, 19.77, 21.83, 22.60, 26.04, 27.00, 27.96]


def save(fig, name, w, h):
    p = f"{FIGS}/{name}_{int(w*200)}x{int(h*200)}.png"
    fig.savefig(p, dpi=200, transparent=True); plt.close(fig); return p


def fig_pv():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.6, 3.7), dpi=200,
                                 gridspec_kw={"width_ratios": [1.45, 1]})
    lab = ["v31", "v38", "v47", "v557", "v601", "v623", "v651"]
    allv = [13.80, 13.40, 13.07, 14.02, 8.61, 8.61, 8.64]
    col = [C["grey"]] * 4 + [C["s"]] * 3
    a1.bar(lab, allv, color=col, width=0.62)
    a1.axhline(10.15, color=C["red"], ls="--", lw=1.6)
    a1.text(6.5, 10.5, "minIP 10.15", color=C["red"], ha="right", fontsize=9)
    a1.axhline(2.0, color=C["ice"], ls=":", lw=2)
    a1.text(-0.45, 2.4, "leaky line (v560): 2.0", color="#8B93A7", fontsize=8.5)
    for i, v in enumerate(allv):
        a1.text(i, v + 0.25, f"{v:.1f}", ha="center", fontsize=8.5)
    a1.set_ylim(0, 16.5); a1.set_ylabel("misassociation [%]")
    a1.set_title("All tracks", fontsize=11, loc="left")
    a2.bar(["v601", "v651", "minIP"], [16.50, 14.86, 19.99],
           color=[C["s"], C["s"], C["red"]], width=0.55)
    for i, v in enumerate([16.50, 14.86, 19.99]):
        a2.text(i, v + 0.3, f"{v:.1f}", ha="center", fontsize=9)
    a2.set_ylim(0, 23); a2.set_title("Signal population", fontsize=11, loc="left")
    fig.tight_layout(); return save(fig, "pv2", 11.6, 3.7)


def fig_decomp():
    fig, ax = plt.subplots(figsize=(11.2, 3.5), dpi=200)
    x = np.arange(len(VERS))
    ax.bar(x, PRUNED, color="#C9D0DD", label="pruned away (lost at pruning)")
    ax.bar(x, CONTAM, bottom=PRUNED, color=C["s"], label="survives, merged with background")
    ax.bar(x, CLEAN, bottom=np.array(PRUNED) + np.array(CONTAM), color=C["p"], label="exact reconstruction")
    for i in range(len(VERS)):
        ax.text(i, PRUNED[i] / 2, f"{PRUNED[i]:.0f}", ha="center", fontsize=8, color="#4A5568")
        ax.text(i, PRUNED[i] + CONTAM[i] / 2, f"{CONTAM[i]:.0f}", ha="center", fontsize=8, color="white")
        ax.text(i, PRUNED[i] + CONTAM[i] + CLEAN[i] / 2, f"{CLEAN[i]:.0f}", ha="center",
                fontsize=8, color="white")
    ax.set_xticks(x); ax.set_xticklabels(VERS)
    ax.set_ylim(0, 100); ax.set_ylabel("% of all truth chains")
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout(); return save(fig, "decomp", 11.2, 3.5)


def fig_scatter():
    fig, ax = plt.subplots(figsize=(6.2, 3.6), dpi=200)
    a = [(AUC[i], RAW_ALL[i], VERS[i]) for i in range(len(VERS)) if AUC[i]]
    ax.scatter([t[0] for t in a], [t[1] for t in a], s=55, color=C["s"], zorder=3)
    for x, y, v in a:
        ax.annotate(v, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8.5)
    ax.set_xlabel("edge AUC  (ranking quality)"); ax.set_ylabel("AllParticles [%]")
    ax.set_xlim(0.915, 0.985); ax.set_ylim(18, 66)
    ax.set_title("Better ranking does not mean better chains", fontsize=11, loc="left")
    fig.tight_layout(); return save(fig, "scatter", 6.2, 3.6)


def fig_ladder():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.6, 3.5), dpi=200)
    a1.plot(VERS, RAW_ALL, "o-", color=C["p"], lw=2.2, ms=5, label="AllParticles")
    a1.plot(VERS, RAW_PERF, "s-", color=C["s"], lw=2.2, ms=5, label="PerfectReco")
    for i in (0, len(VERS) - 1):
        a1.annotate(f"{RAW_ALL[i]:.1f}", (i, RAW_ALL[i]), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8.5)
        a1.annotate(f"{RAW_PERF[i]:.1f}", (i, RAW_PERF[i]), textcoords="offset points",
                    xytext=(0, -13), ha="center", fontsize=8.5)
    a1.set_ylim(10, 70); a1.set_ylabel("[%]"); a1.legend(frameon=False, fontsize=9, loc="lower right")
    a1.set_title("Reconstruction rises …", fontsize=11, loc="left")
    ax2 = [t for t in range(len(VERS)) if AUC[t]]
    a2.plot([VERS[i] for i in ax2], [AUC[i] for i in ax2], "o-", color=C["grey"], lw=2.2, ms=5)
    a2.set_ylim(0.915, 0.985); a2.set_ylabel("edge AUC")
    a2.set_title("… while ranking quality falls", fontsize=11, loc="left")
    fig.tight_layout(); return save(fig, "ladder2", 11.6, 3.5)


def fig_delta():
    steps = ["v623\nfeatures", "v631\noperating pt", "v633\nremove", "v640\nselection",
             "v643\nDOCA fix", "v648\ncontext", "v651\nattention"]
    da = [RAW_ALL[1] - RAW_ALL[0], RAW_ALL[2] - RAW_ALL[1], RAW_ALL[3] - RAW_ALL[2],
          RAW_ALL[4] - RAW_ALL[3], RAW_ALL[5] - RAW_ALL[4], RAW_ALL[6] - RAW_ALL[5],
          RAW_ALL[7] - RAW_ALL[6]]
    dp = [RAW_PERF[1] - RAW_PERF[0], RAW_PERF[2] - RAW_PERF[1], RAW_PERF[3] - RAW_PERF[2],
          RAW_PERF[4] - RAW_PERF[3], RAW_PERF[5] - RAW_PERF[4], RAW_PERF[6] - RAW_PERF[5],
          RAW_PERF[7] - RAW_PERF[6]]
    fig, ax = plt.subplots(figsize=(11.2, 3.5), dpi=200)
    y = np.arange(len(steps)); h = 0.36
    ax.barh(y + h / 2, da, h, color=C["p"], label="Δ AllParticles")
    ax.barh(y - h / 2, dp, h, color=C["s"], label="Δ PerfectReco")
    for i, (u, v) in enumerate(zip(da, dp)):
        ax.text(u + (0.25 if u >= 0 else -0.25), i + h / 2, f"{u:+.1f}", va="center",
                ha="left" if u >= 0 else "right", fontsize=8.5)
        ax.text(v + (0.25 if v >= 0 else -0.25), i - h / 2, f"{v:+.1f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=8.5)
    ax.set_yticks(y); ax.set_yticklabels(steps); ax.invert_yaxis()
    ax.axvline(0, color="#9AA3B2", lw=1)
    ax.set_xlabel("change vs the previous version [pp]")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout(); return save(fig, "delta", 11.2, 3.5)


def fig_neg():
    names = ["attention alone\n(v637)", "mask on node pooling\n(v805)", "geometry candidates\ntight (v1003)",
             "geometry candidates\nboth (v1005)", "slot decode\n(v803)"]
    vals = [0.0, -0.82, -3.20, -7.39, -23.92]
    fig, ax = plt.subplots(figsize=(7.4, 3.2), dpi=200)
    cols = [C["grey"]] + [C["red"]] * 4
    ax.barh(names, vals, color=cols, height=0.55)
    for i, v in enumerate(vals):
        ax.text(v - 0.5 if v < 0 else v + 0.3, i, f"{v:+.1f}", va="center",
                ha="right" if v < 0 else "left", fontsize=9)
    ax.axvline(0, color="#9AA3B2", lw=1); ax.invert_yaxis()
    ax.set_xlim(-28, 3); ax.set_xlabel("Δ frontier residual (0 = no change)")
    fig.tight_layout(); return save(fig, "neg", 7.4, 3.2)


def fig_doca():
    fig, ax = plt.subplots(figsize=(5.0, 3.3), dpi=200)
    x = [0, 1]; wd = 0.32
    a = [40.22, 49.55]; p = [26.64, 33.78]
    ax.bar([i - wd / 2 for i in x], a, wd, color=C["p"], label="AllParticles")
    ax.bar([i + wd / 2 for i in x], p, wd, color=C["s"], label="PerfectReco")
    for i in range(2):
        ax.text(i - wd / 2, a[i] + 0.8, f"{a[i]:.1f}", ha="center", fontsize=9)
        ax.text(i + wd / 2, p[i] + 0.8, f"{p[i]:.1f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(["v633\nbuggy DOCA", "v643\ncorrected"])
    ax.set_ylim(0, 58); ax.set_ylabel("[%]"); ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout(); return save(fig, "doca", 5.0, 3.3)


P_PV, P_DECOMP, P_SCAT = fig_pv(), fig_decomp(), fig_scatter()
P_LADDER, P_DELTA, P_NEG, P_DOCA = fig_ladder(), fig_delta(), fig_neg(), fig_doca()

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


def rect(s, x, y, w, h, fill, line=None, shape=MSO_SHAPE.RECTANGLE, radius=None):
    sh = s.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb = _rgb(fill)
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = _rgb(line); sh.line.width = Pt(1.2)
    sh.shadow.inherit = False
    if radius is not None and shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        sh.adjustments[0] = radius
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


def header(s, kicker, title, sub=None):
    rect(s, 0, 0, W, 0.09, TEAL)
    tb(s, MX, 0.3, 11.5, 0.28, kicker, 11, True, TEAL)
    tb(s, MX, 0.56, 12.3, 0.6, title, 25, True, NAVY, font="Georgia")
    if sub:
        tb(s, MX, 1.13, 12.3, 0.34, sub, 12.5, False, GREY, italic=True)


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
            ("2", "Where does the reconstruction loss actually sit?", "two failure modes — and only one of them is fixable downstream"),
            ("3", "What was wrong with DOCA?", "a silent feature corruption that cost ~7 pp of perfect reconstruction"),
            ("4", "How far did pruning optimization get?", "seven versions, four benchmarks, one change at a time")]
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

# ---------- Part 1 ----------
divider(1, "What the leak fix cost", "One metric moved the wrong way",
        ["The corrected sample removes truth-level information from the inputs",
         "PV association is the metric that lost the most",
         "A PV-aware pruning head brought it back below the minIP baseline"])

# pipeline diagram
s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 1  ·  CONTEXT", "Where pruning sits in DFEI", "Everything downstream inherits whatever pruning keeps")
boxes = [("Track graph\n(~150 tracks)", 0.75, GREY),
         ("Node pruning\n(background out)", 3.15, TEAL),
         ("Edge pruning\n(false pairs out)", 5.55, TEAL),
         ("LCAG classification\n(4 classes per pair)", 7.95, NAVY),
         ("Chain assembly\n(connected components)", 10.35, NAVY)]
for txt, x, col in boxes:
    b = rect(s, x, 2.15, 2.2, 1.15, WHITE, line=col, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.12)
    tb(s, x + 0.12, 2.38, 1.96, 0.75, txt, 11, True, col, align=PP_ALIGN.CENTER)
for x in (2.98, 5.38, 7.78, 10.18):
    c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x), Inches(2.72), Inches(x + 0.16), Inches(2.72))
    c.line.color.rgb = GREY; c.line.width = Pt(1.6)
rect(s, 2.55, 3.6, 4.05, 0.62, RGBColor(0xE6, 0xF4, 0xF1))
tb(s, 2.66, 3.72, 3.9, 0.4, "our work: these two stages", 11, True, TEAL)
rect(s, 9.45, 3.6, 3.05, 0.62, RGBColor(0xEE, 0xEE, 0xF6))
tb(s, 9.56, 3.72, 2.9, 0.4, "next round", 11, True, NAVY)
bullets(s, MX, 4.65, 12.2, 1.7, [
    "Pruning decides which truth chains even enter the reconstruction — a chain lost here can never be recovered",
    "PV association shares the same pruning representation, which is why one change can move both"], 12.5, NAVY, 8)
footer(s)

# PV figure
s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 1  ·  PV ASSOCIATION", "Old models fall behind minIP; retraining plus a PV head recovers it",
       "Misassociation rate, lower is better")
pic(s, P_PV, 0.45, 1.62, w=12.4)
rect(s, MX, 5.62, 12.25, 1.15, LIGHT); rect(s, MX, 5.62, 0.05, 1.15, TEAL)
bullets(s, 0.78, 5.75, 11.9, 0.95, [
    "Models trained on the leaky sample: 13.1–14.0% on the corrected sample — every one of them worse than minIP (10.15%)",
    "Retrained together with a PV-aware pruning head: 8.6% (all tracks), 14.9% (signal) — below minIP again, but far from the 2.0% the leak suggested"],
    10.5, NAVY, 4)
footer(s)

# PV head mechanism
s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 1  ·  THE FIX", "What the PV-aware pruning head changed", "Three additions to the pruning stage; the reconstruction target is untouched")
cards = [("1  Inputs", "Track–PV consistency enters the\npruning features: impact parameter,\nz-association, collinearity"),
         ("2  Auxiliary target", "A head predicts the PV of every\nsurviving track, trained jointly\nwith the pruning score"),
         ("3  Effect", "PV misassociation 13–14 → 8.6%\nBelow minIP (10.15%)\nNo loss in reconstruction efficiency")]
for i, (h, v) in enumerate(cards):
    x = MX + i * 4.12
    rect(s, x, 1.75, 3.9, 2.35, LIGHT); rect(s, x, 1.75, 3.9, 0.06, [NAVY, TEAL, TEAL][i])
    tb(s, x + 0.24, 1.92, 3.4, 0.3, h, 14.5, True, NAVY, font="Georgia")
    tb(s, x + 0.24, 2.32, 3.45, 1.6, v, 11, False, GREY, line=1.25)
# mini sketch: track -> PV assignment
rect(s, MX, 4.35, 6.0, 2.1, WHITE, line=ICE)
tb(s, MX + 0.2, 4.45, 5.6, 0.28, "What the head sees", 11.5, True, NAVY, font="Georgia")
for i, (yy, lb) in enumerate([(4.95, "PV 1"), (5.35, "PV 2"), (5.75, "PV 3")]):
    rect(s, MX + 0.25, yy, 0.16, 0.16, NAVY, shape=MSO_SHAPE.OVAL)
    tb(s, MX + 0.48, yy - 0.05, 0.6, 0.25, lb, 9, False, GREY)
for i, yy in enumerate([5.0, 5.4, 5.65, 5.2, 5.55]):
    rect(s, MX + 3.0 + i * 0.32, yy, 0.12, 0.12, TEAL, shape=MSO_SHAPE.OVAL)
tb(s, MX + 0.25, 6.02, 5.5, 0.3, "each surviving track is softly assigned to one PV", 9.5, False, GREY, italic=True)
rect(s, 6.85, 4.35, 5.95, 2.1, LIGHT); rect(s, 6.85, 4.35, 0.05, 2.1, TEAL)
bullets(s, 7.1, 4.5, 5.5, 1.8, [
    "PV information was previously invisible to the pruning score",
    "Adding it changes which tracks survive, not how chains are assembled",
    "Caveat: retraining and the head were introduced together; this sample does not separate them"], 10.5, NAVY, 6)
footer(s)

# ---------- Part 2 ----------
divider(2, "Where the loss actually sits", "Two failure modes, one lever",
        ["A truth chain either disappears at pruning or survives and is merged with background",
         "Assembly never fails on its own: every surviving chain is found",
         "So the pruning stage sets both terms"])

s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 2  ·  DIAGNOSIS", "Each step trades one failure mode for the other",
       "All 13 255 truth chains of the corrected sample, fixed denominator")
pic(s, P_DECOMP, 0.55, 1.62, w=12.2)
tb(s, MX, 5.55, 12.2, 0.3, "Grey = lost at pruning.  Teal = found but merged with background.  Navy = exact reconstruction.", 10, False, GREY)
rect(s, MX, 5.95, 12.25, 0.9, LIGHT); rect(s, MX, 5.95, 0.05, 0.9, TEAL)
tb(s, 0.78, 6.05, 11.9, 0.7,
   "Note the exchange: v631 and v633 buy exact chains by dropping more of them; v643 and v648 shrink the contaminated band.\n"
   "No version so far shrinks both at once — that is the frontier we are trying to break.", 10.5, False, NAVY)
footer(s)

s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 2  ·  A WARNING FROM THE DATA", "Ranking quality and reconstruction quality come apart",
       "Every version with both measurements on the corrected sample")
pic(s, P_SCAT, 0.65, 1.75, w=6.3)
rect(s, 7.35, 1.75, 5.45, 3.55, LIGHT); rect(s, 7.35, 1.75, 0.05, 3.55, TEAL)
tb(s, 7.6, 1.9, 5.0, 0.3, "Why this matters for the next slide", 14, True, NAVY, font="Georgia")
bullets(s, 7.6, 2.35, 5.05, 2.8, [
    "v601 has the best edge AUC (0.972) and one of the worst reconstruction rates",
    "v651 has a lower AUC and the best reconstruction",
    "A single ranking number cannot tell us whether a change helped the physics",
    "So we report four benchmarks, and treat AUC as a diagnostic"], 10.8, NAVY, 7)
footer(s)

# ---------- Part 3 ----------
divider(3, "The DOCA bug", "A feature that was silently wrong",
        ["Distance of closest approach was inherited from an upstream quantity",
         "That quantity is not the geometric DOCA",
         "The pruning MLP learned from a biased input for months"])

s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 3  ·  THE BUG", "What was wrong", "No crash, no NaN — just a wrong number")
# sketch on the left
rect(s, MX, 1.7, 6.0, 3.7, WHITE, line=ICE)
tb(s, MX + 0.2, 1.8, 5.6, 0.28, "What DOCA should be", 11.5, True, NAVY, font="Georgia")
rect(s, 1.15, 4.35, 0.18, 0.18, NAVY, shape=MSO_SHAPE.OVAL)
tb(s, 1.38, 4.3, 0.9, 0.25, "PV", 9.5, False, GREY)
l1 = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(1.24), Inches(4.44), Inches(4.05), Inches(2.6))
l2 = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(1.24), Inches(4.44), Inches(5.1), Inches(3.5))
for ln in (l1, l2):
    ln.line.color.rgb = NAVY; ln.line.width = Pt(2)
rect(s, 4.05, 2.5, 0.16, 0.16, TEAL, shape=MSO_SHAPE.OVAL)
rect(s, 5.1, 3.4, 0.16, 0.16, TEAL, shape=MSO_SHAPE.OVAL)
tb(s, 4.25, 2.25, 1.4, 0.25, "decay vertex", 9.5, False, GREY)
tb(s, 2.3, 3.95, 2.6, 0.3, "closest approach of the two tracks", 9.5, False, GREY, italic=True)
# text right
rect(s, 6.85, 1.7, 5.95, 3.7, LIGHT); rect(s, 6.85, 1.7, 0.05, 3.7, RED)
tb(s, 7.1, 1.85, 5.4, 0.3, "What we actually fed the network", 14, True, NAVY, font="Georgia")
bullets(s, 7.1, 2.3, 5.45, 3.0, [
    "log DOCA was read from an upstream reconstructed quantity",
    "Its definition differs from the geometric DOCA drawn on the left",
    "The MLP therefore learned a mapping from a biased feature",
    "No error surfaced — the score stayed plausible"], 10.8, NAVY, 7)
tb(s, MX, 5.6, 12.2, 0.35, "Fix: recompute dR, dz, DOCA and |Δz| locally from the track states; one feature replaced, architecture unchanged.",
   11, False, GREY, italic=True)
footer(s)

s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 3  ·  THE EFFECT", "Correcting DOCA: +9.3 pp AllParticles, +7.1 pp PerfectReco",
       "Same sample, same threshold, one variable changed")
pic(s, P_DOCA, 0.75, 1.85, w=5.9)
rect(s, 6.95, 1.85, 5.85, 3.55, LIGHT); rect(s, 6.95, 1.85, 0.05, 3.55, TEAL)
tb(s, 7.2, 2.0, 5.3, 0.3, "Read carefully", 14, True, NAVY, font="Georgia")
bullets(s, 7.2, 2.45, 5.35, 2.8, [
    "Edge AUC is flat (0.924 → 0.924): the fix restores feature correctness, not ranking power",
    "The gain appears in assembled chains — exactly the quantity we care about",
    "This is the cleanest single-variable improvement in the whole line"], 10.8, NAVY, 8)
footer(s)

# ---------- Part 4 ----------
divider(4, "Pruning optimization", "Seven versions, four benchmarks, one change at a time",
        ["Physics inputs → operating point → remove what fails → context → attention",
         "Benchmarks: AllParticles, PerfectReco, edge AUC, PV association",
         "Every step is a controlled change on top of the previous best"])

s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 4  ·  THE LADDER", "From 25% to 61% AllParticles", "Paper-convention numbers; read the trend, not the absolute level")
rows = [["Version", "Change", "AllParticles [%]", "PerfectReco [%]", "edge AUC", "PV misassoc. [%]"],
        ["v601", "baseline on the corrected sample", "25.4", "16.8", "0.972", "8.61"],
        ["v623", "physics-derived pruning inputs", "23.5", "15.7", "0.974", "8.61"],
        ["v631", "threshold + pairwise ranking", "34.3", "23.0", "0.935", "8.63"],
        ["v633", "drop the two negative inputs", "40.2", "26.6", "0.924", "8.68"],
        ["v643", "corrected DOCA geometry", "49.6", "33.8", "0.924", "8.66"],
        ["v648", "competition / exclusivity context", "56.4", "37.8", "0.932", "8.68"],
        ["v651", "line-graph attention + geometry bias", "60.9", "41.4", "0.933", "8.64"]]
t = s.shapes.add_table(len(rows), 6, Inches(MX), Inches(1.72), Inches(12.25), Inches(3.3)).table
for i, w in enumerate([1.15, 4.55, 2.0, 1.95, 1.35, 1.25]):
    t.columns[i].width = Inches(w)
for r, row in enumerate(rows):
    for c, val in enumerate(row):
        cell = t.cell(r, c); cell.text = val; cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY if r == 0 else (WHITE if r % 2 else LIGHT)
        for p in cell.text_frame.paragraphs:
            p.alignment = PP_ALIGN.LEFT if c == 1 else PP_ALIGN.CENTER
            for run in p.runs:
                run.font.size = Pt(10.8 if r else 11.2); run.font.bold = (r == 0) or (c == 0)
                run.font.color.rgb = WHITE if r == 0 else NAVY; run.font.name = "Calibri"
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
tb(s, MX, 5.28, 12.2, 0.35, "PV association is the one column that does not move — the pruning changes were made for reconstruction, not for PV.",
   10.2, False, GREY, italic=True)
footer(s)

s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 4  ·  THE LADDER", "Two curves that do not run in parallel", None)
pic(s, P_LADDER, 0.55, 1.35, w=12.2)
rect(s, MX, 5.35, 12.25, 1.35, LIGHT); rect(s, MX, 5.35, 0.05, 1.35, TEAL)
bullets(s, 0.8, 5.48, 11.9, 1.15, [
    "AllParticles 25.4 → 60.9%, PerfectReco 16.8 → 41.4%",
    "edge AUC 0.972 → 0.933: the reconstruction gain does not come from a better ranking",
    "It comes from correct inputs and from moving the operating point"], 10.8, NAVY, 5)
footer(s)

s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 4  ·  WHAT EACH CHANGE BOUGHT", "The per-step budget", "Change relative to the previous version, in percentage points")
pic(s, P_DELTA, 0.55, 1.62, w=12.2)
tb(s, MX, 5.55, 12.2, 0.4, "v623 is the only step that costs reconstruction — it was kept because it gave the best edge AUC and became the base for v631.",
   10.2, False, GREY, italic=True)
footer(s)

# step mechanism slides
def step_slide(kicker, title, sub, sketch, res_items, extra=None):
    s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
    header(s, kicker, title, sub)
    rect(s, MX, 1.65, 6.55, 4.25, WHITE, line=ICE)
    sketch(s)
    rect(s, 7.35, 1.65, 5.45, 4.25, LIGHT); rect(s, 7.35, 1.65, 0.05, 4.25, TEAL)
    tb(s, 7.6, 1.8, 5.0, 0.3, "What it bought", 14, True, NAVY, font="Georgia")
    bullets(s, 7.6, 2.25, 5.05, 3.5, res_items, 10.8, NAVY, 8)
    if extra:
        tb(s, MX, 6.05, 12.2, 0.4, extra, 10.2, False, GREY, italic=True)
    footer(s)


def sk_features(s):
    tb(s, MX + 0.2, 1.78, 6.1, 0.28, "Derived per-pair quantities", 11.5, True, NAVY, font="Georgia")
    for i, (lb, col) in enumerate([("dR", TEAL), ("dz", TEAL), ("DOCA", NAVY), ("|Δz|", NAVY), ("collinearity", GREY)]):
        rect(s, MX + 0.25, 2.22 + i * 0.62, 0.09, 0.4, col)
        tb(s, MX + 0.5, 2.25 + i * 0.62, 2.6, 0.3, lb, 11, False, NAVY)
        rect(s, MX + 2.9, 2.35 + i * 0.62, 3.3, 0.16, RGBColor(0xDD, 0xE3, 0xEE))
        rect(s, MX + 2.9, 2.35 + i * 0.62, 1.2 + i * 0.42, 0.16, col)
    tb(s, MX + 0.25, 5.35, 6.0, 0.4, "Zero-initialised adapters: the old behaviour is the starting point,\nso the change is isolated.", 9.8, False, GREY, italic=True)


def sk_threshold(s):
    tb(s, MX + 0.2, 1.78, 6.1, 0.28, "Edge score distribution (schematic)", 11.5, True, NAVY, font="Georgia")
    x = np.linspace(0, 1, 200)
    bg = 120 * np.exp(-((x - 0.35) ** 2) / 0.012)
    sig = 60 * np.exp(-((x - 0.78) ** 2) / 0.02)
    fig, ax = plt.subplots(figsize=(3.5, 2.0), dpi=200)
    ax.plot(x, bg, color="#9AA3B2", lw=1.8, label="background pairs")
    ax.fill_between(x, bg, color="#9AA3B2", alpha=0.25)
    ax.plot(x, sig, color=C["s"], lw=1.8, label="true pairs")
    ax.fill_between(x, sig, color=C["s"], alpha=0.25)
    ax.axvline(0.90, color=C["red"], ls="--", lw=1.4)
    ax.axvline(0.97, color=C["red"], ls=":", lw=1.2)
    ax.text(0.895, 100, "0.90", color=C["red"], fontsize=8.5, ha="right")
    ax.text(0.975, 100, "0.97", color=C["red"], fontsize=8.5, ha="left")
    ax.set_yticks([]); ax.set_xlim(0, 1); ax.set_xlabel("pruning score")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    p = f"{FIGS}/sk_thr_{int(3.5*200)}x{int(2.0*200)}.png"; fig.tight_layout(); fig.savefig(p, dpi=200, transparent=True); plt.close(fig)
    s.shapes.add_picture(p, Inches(MX + 1.1), Inches(2.15), width=Inches(4.3))
    tb(s, MX + 0.25, 5.0, 6.1, 0.7, "Lower threshold: more true pairs survive, but more background pairs enter.\nThe pairwise ranking term is what keeps the background side under control.",
       9.8, False, GREY, italic=True)


def sk_competition(s):
    tb(s, MX + 0.2, 1.78, 6.1, 0.28, "Exclusivity: how many rivals does a track have?", 11.5, True, NAVY, font="Georgia")
    cx, cy = MX + 1.35, 3.65
    rect(s, cx, cy, 0.2, 0.2, NAVY, shape=MSO_SHAPE.OVAL)
    tb(s, cx - 0.1, 3.92, 1.4, 0.25, "track", 9, False, GREY)
    pts = [(MX + 3.4, 2.35, TEAL, "true partner"), (MX + 4.2, 3.1, "#9AA3B2", "rival"),
           (MX + 4.5, 4.2, "#9AA3B2", "rival"), (MX + 3.2, 4.9, "#9AA3B2", "rival")]
    for x, y, col, lab in pts:
        ln = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(cx + 0.18), Inches(cy + 0.1),
                                    Inches(x + 0.08), Inches(y + 0.08))
        ln.line.color.rgb = _rgb(col); ln.line.width = Pt(1.6)
        rect(s, x, y, 0.16, 0.16, col, shape=MSO_SHAPE.OVAL)
        tb(s, x + 0.22, y - 0.04, 1.5, 0.24, lab, 8.5, False, GREY)
    tb(s, MX + 0.25, 5.25, 6.0, 0.6, "The old score looked at each pair in isolation.\nThe new block also counts the competitors each track has.",
       9.8, False, GREY, italic=True)


def sk_attention(s):
    tb(s, MX + 0.2, 1.78, 6.1, 0.28, "Line-graph attention, biased by the geometry", 11.5, True, NAVY, font="Georgia")
    xs = [MX + 0.8, MX + 2.0, MX + 3.2, MX + 4.4, MX + 5.6]
    y0 = 3.3
    for i in range(len(xs) - 1):
        ln = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(xs[i]), Inches(y0),
                                    Inches(xs[i + 1]), Inches(y0))
        ln.line.color.rgb = NAVY; ln.line.width = Pt(1.8)
    for x in xs:
        rect(s, x - 0.09, y0 - 0.09, 0.18, 0.18, TEAL, shape=MSO_SHAPE.OVAL)
    for i in range(len(xs) - 1):
        tb(s, (xs[i] + xs[i + 1]) / 2 - 0.35, y0 + 0.16, 0.7, 0.22, "e%d" % (i + 1), 8.5, False, GREY, align=PP_ALIGN.CENTER)
    tb(s, MX + 0.25, 3.9, 6.1, 0.3, "each edge attends to the other edges of the same event", 9.8, False, GREY, italic=True)
    rect(s, MX + 0.5, 4.45, 5.4, 0.75, RGBColor(0xE6, 0xF4, 0xF1))
    tb(s, MX + 0.68, 4.55, 5.1, 0.55, "attention bias  ←  the same geometry features (dR, dz, DOCA)\nWithout this bias the layer is neutral.",
       9.8, True, TEAL)


step_slide("PART 4  ·  STEP 1", "v623 — physics back into the inputs",
           "Derived quantities instead of inherited ones",
           sk_features,
           ["AllParticles 25.4 → 23.5% (a small loss)",
            "Edge AUC 0.972 → 0.974 — the best ranking we ever had",
            "Kept as the base for v631: the representation improved even though the chains did not"],
           "Lesson: better ranking alone does not become better chains. This is the first evidence of the decoupling.")

step_slide("PART 4  ·  STEP 2", "v631 — move the operating point, rank pairs",
           "Threshold 0.97 → 0.90, re-balanced classes, pairwise ranking loss",
           sk_threshold,
           ["AllParticles 23.5 → 34.3% (+10.8 pp)",
            "PerfectReco 15.7 → 23.0% (+7.3 pp)",
            "Edge AUC 0.974 → 0.935 — lower, and yet the chains are better",
            "The largest single jump in the line"])

step_slide("PART 4  ·  STEP 3", "v633 / v640 — remove what does not work",
           "Two ablations and a model-selection fix",
           lambda s: (tb(s, MX + 0.2, 1.78, 6.1, 0.28, "Two terms removed, one selection rule fixed", 11.5, True, NAVY, font="Georgia"),
                      bullets(s, MX + 0.3, 2.3, 6.0, 2.2, ["Event-count head: −19% on its own", "PV-overlap injection: −2.6%"], 11.5, NAVY, 8),
                      rect(s, MX + 0.3, 3.35, 5.9, 0.02, ICE),
                      tb(s, MX + 0.2, 3.5, 6.1, 0.28, "Then: select the checkpoint by pruning AP, not by total loss", 10.5, True, NAVY),
                      tb(s, MX + 0.3, 3.9, 5.8, 0.6, "Same model, different checkpoint — and it alone is worth about +1 pp.", 10, False, GREY, italic=True)),
           ["v633: AllParticles 34.3 → 40.2%, PerfectReco 23.0 → 26.6%",
            "v640: selection change, ≈ +1 pp on top",
            "Removing a bad term was worth more than adding two good ones"])

step_slide("PART 4  ·  STEP 4", "v648 — let each edge see its competitors",
           "Competition and exclusivity enter the pruning score",
           sk_competition,
           ["AllParticles 49.6 → 56.4%",
            "PerfectReco 33.8 → 37.8%",
            "Hard-pool edge AP 0.668 — the best of the whole line",
            "First block that helps ranking and reconstruction at the same time"])

step_slide("PART 4  ·  STEP 5", "v651 — attention with a geometric prior",
           "Line-graph attention, biased by the geometry features",
           sk_attention,
           ["AllParticles 56.4 → 60.9%",
            "PerfectReco 37.8 → 41.4%",
            "LCAG true-edge classification 74.7 → 75.4%, the best we have",
            "PV association unchanged at 8.6%"])

s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
header(s, "PART 4  ·  NEGATIVE RESULTS", "What we tried and discarded", "Δ relative to the frontier; negative means worse than doing nothing")
pic(s, P_NEG, 0.6, 1.7, w=7.9)
rect(s, 8.75, 1.7, 4.05, 3.4, LIGHT); rect(s, 8.75, 1.7, 0.05, 3.4, RED)
tb(s, 9.0, 1.85, 3.6, 0.3, "Also discarded", 13.5, True, NAVY, font="Georgia")
bullets(s, 9.0, 2.3, 3.65, 2.6, [
    "Bigger capacity (wider layers, more heads): no gain",
    "Auxiliary heads (mass, structure, source): no transfer",
    "Loose chain rewards: more chains kept, more contamination"], 10.2, NAVY, 6)
tb(s, MX, 5.5, 12.2, 0.35, "Reported so that nobody repeats them — each cost at least one controlled run.", 10.2, False, GREY, italic=True)
footer(s)

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
y = 1.85
for k, v in defs:
    tb(s, MX, y, 2.6, 0.3, k, 12, True, NAVY)
    tb(s, MX + 2.75, y, 9.6, 0.5, v, 11, False, GREY)
    y += 0.78
tb(s, MX, 6.15, 12.2, 0.6,
   "Two normalisations appear in this deck. Slides 5–6, 12–17 use the paper convention (denominator = truth chains that survive pruning).\nSlide 8 uses a fixed denominator (all truth chains). The two must not be compared directly.",
   9.8, False, GREY, italic=True)
footer(s)

prs.save(OUT)
print("saved", OUT, os.path.getsize(OUT))
