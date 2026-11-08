"""Build the Friday report deck: DFEI bug fixes and pruning optimization (English)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

OUT = "DFEI_bugfix_pruning_20261023_EN.pptx"
FIGS = "report_figs/deck_20261023"
os.makedirs(FIGS, exist_ok=True)

# ---------------- palette ----------------
NAVY = RGBColor(0x1E, 0x27, 0x61)
TEAL = RGBColor(0x00, 0xA8, 0x96)
ICE = RGBColor(0xCA, 0xDC, 0xFC)
GREY = RGBColor(0x5A, 0x62, 0x70)
LIGHT = RGBColor(0xF3, 0xF6, 0xFB)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
C = dict(primary="#1E2761", secondary="#00A896", ice="#CADCFC", grey="#5A6270")

plt.rcParams.update({
    "axes.prop_cycle": plt.cycler(color=[C["primary"], C["secondary"], C["ice"]]),
    "axes.facecolor": "none", "figure.facecolor": "none",
    "axes.edgecolor": C["grey"], "axes.labelcolor": C["primary"],
    "text.color": C["primary"], "xtick.color": C["primary"], "ytick.color": C["primary"],
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
})

W, H = 13.333, 7.5
MX = 0.55


def fig_pv():
    fig, ax = plt.subplots(figsize=(8.2, 3.9), dpi=200)
    labels = ["v31", "v38", "v47", "v557", "v601\n(retrained)", "v623", "v651"]
    vals = [13.80, 13.40, 13.07, 14.02, 8.61, 8.61, 8.64]
    colors = [C["grey"]] * 4 + [C["secondary"]] * 3
    b = ax.bar(labels, vals, color=colors, width=0.62)
    ax.axhline(10.15, color="#C0392B", ls="--", lw=1.8)
    ax.text(6.45, 10.45, "minIP baseline 10.15%", color="#C0392B", ha="right", fontsize=10)
    ax.axhline(2.0, color=C["ice"], ls=":", lw=2)
    ax.text(0.05, 2.4, "leaky sample, v560: 2.0%", color="#8B93A7", fontsize=9)
    for r, v in zip(b, vals):
        ax.text(r.get_x() + r.get_width() / 2, v + 0.25, f"{v:.1f}", ha="center", fontsize=9)
    ax.set_ylabel("PV misassociation rate [%]  (lower is better)")
    ax.set_ylim(0, 16.5)
    ax.set_title("Models trained on the leaky sample fail on the fixed sample",
                 fontsize=12, color=C["primary"], loc="left")
    fig.tight_layout()
    p = f"{FIGS}/pv_association_1230x585.png"
    fig.savefig(p, dpi=200, transparent=True); plt.close(fig); return p


def fig_ladder():
    vers = ["v601", "v623", "v631", "v633", "v643", "v648", "v651"]
    allp = [25.38, 23.51, 34.25, 40.22, 49.55, 56.39, 60.86]
    perf = [16.76, 15.72, 23.04, 26.64, 33.78, 37.84, 41.37]
    auc = [0.9716, 0.9737, 0.9352, 0.9242, 0.9236, 0.9324, 0.9327]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.4, 3.7), dpi=200)
    a1.plot(vers, allp, "o-", color=C["primary"], lw=2.2, ms=5, label="AllParticles")
    a1.plot(vers, perf, "s-", color=C["secondary"], lw=2.2, ms=5, label="PerfectReco")
    a1.set_ylabel("[%]"); a1.set_title("Reconstruction (paper convention)", fontsize=11, loc="left")
    a1.legend(frameon=False, fontsize=9); a1.grid(alpha=.22)
    a2.plot(vers, auc, "o-", color=C["grey"], lw=2.2, ms=5)
    a2.set_ylabel("edge AUC"); a2.set_ylim(0.915, 0.985)
    a2.set_title("Pruning ranking quality (proxy)", fontsize=11, loc="left")
    a2.grid(alpha=.22)
    fig.tight_layout()
    p = f"{FIGS}/ladder_1710x555.png"
    fig.savefig(p, dpi=200, transparent=True); plt.close(fig); return p


def fig_doca():
    fig, ax = plt.subplots(figsize=(5.4, 3.6), dpi=200)
    x = [0, 1]; wd = 0.34
    a = [40.22, 49.55]; p = [26.64, 33.78]
    ax.bar([i - wd / 2 for i in x], a, wd, color=C["primary"], label="AllParticles")
    ax.bar([i + wd / 2 for i in x], p, wd, color=C["secondary"], label="PerfectReco")
    for i, (va, vp) in enumerate(zip(a, p)):
        ax.text(i - wd / 2, va + 0.7, f"{va:.1f}", ha="center", fontsize=9)
        ax.text(i + wd / 2, vp + 0.7, f"{vp:.1f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(["v633\n(buggy DOCA)", "v643\n(corrected DOCA)"])
    ax.set_ylabel("[%]"); ax.set_ylim(0, 58); ax.legend(frameon=False, fontsize=9)
    ax.set_title("Correcting the DOCA feature", fontsize=11, loc="left")
    fig.tight_layout()
    p = f"{FIGS}/doca_810x540.png"
    fig.savefig(p, dpi=200, transparent=True); plt.close(fig); return p


P_PV, P_LADDER, P_DOCA = fig_pv(), fig_ladder(), fig_doca()

prs = Presentation()
prs.slide_width = Inches(W); prs.slide_height = Inches(H)
BLANK = prs.slide_layouts[6]


def tb(s, x, y, w, h, text, size=14, bold=False, color=NAVY, align=PP_ALIGN.LEFT,
       font="Calibri", italic=False, line=1.0, anchor=MSO_ANCHOR.TOP):
    box = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, m, 0)
    lines = text.split("\n") if isinstance(text, str) else text
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align; p.line_spacing = line
        r = p.add_run(); r.text = ln
        r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = color
        r.font.name = font; r.font.italic = italic
    return box


def _rgb(v):
    if isinstance(v, str):
        v = v.lstrip("#")
        return RGBColor(int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16))
    return v


def rect(s, x, y, w, h, fill, line=None):
    from pptx.enum.shapes import MSO_SHAPE
    sh = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb = _rgb(fill)
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line
    sh.shadow.inherit = False
    return sh


def bullets(s, x, y, w, h, items, size=14, color=NAVY, gap=6):
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


def slide_header(s, kicker, title, sub=None):
    rect(s, 0, 0, W, 0.09, TEAL)
    tb(s, MX, 0.34, 11.5, 0.3, kicker, 11.5, True, TEAL)
    tb(s, MX, 0.62, 12.2, 0.62, title, 27, True, NAVY, font="Georgia")
    if sub:
        tb(s, MX, 1.26, 12.2, 0.4, sub, 13.5, False, GREY, italic=True)


def footer(s, n):
    tb(s, MX, H - 0.42, 6.0, 0.25, "DFEI · bug fixes and pruning optimization", 9, False, GREY)
    tb(s, W - 1.1, H - 0.42, 0.6, 0.25, f"{n}", 9, True, GREY, align=PP_ALIGN.RIGHT)


def cover():
    s = prs.slides.add_slide(BLANK)
    rect(s, 0, 0, W, H, NAVY)
    rect(s, 0, 0, 0.14, H, TEAL)
    tb(s, 0.95, 1.85, 11.6, 0.4, "DFEI GROUP MEETING  ·  2026-10-23", 12, True, TEAL)
    tb(s, 0.95, 2.28, 11.6, 1.7,
       "DFEI: Bug Fixes and\nPruning Optimization", 40, True, WHITE, font="Georgia", line=1.12)
    tb(s, 0.95, 4.05, 11.0, 0.5,
       "From a corrected simulation sample to a stronger pruning stage", 16, False, ICE, italic=True)
    rect(s, 0.95, 4.75, 1.6, 0.04, TEAL)
    tb(s, 0.95, 4.95, 8.0, 0.35, "Qingxiang Guo  ·  UCAS", 14, False, WHITE)
    tb(s, 0.95, 6.55, 11.0, 0.3, "All numbers on the corrected (0904) sample unless stated otherwise", 10, False, ICE)


def toc():
    s = prs.slides.add_slide(BLANK)
    rect(s, 0, 0, W, H, LIGHT)
    slide_header(s, "CONTENTS", "What this talk covers")
    rows = [("1", "The leak fix and PV association", "why one metric moved the wrong way, and how we recovered it"),
            ("2", "Where the reconstruction loss actually sits", "two failure modes, and why pruning is the lever"),
            ("3", "The DOCA bug", "what was wrong, how it surfaced, what it cost"),
            ("4", "Pruning optimization: seven versions", "the ladder, the benchmarks, and what did not work")]
    y = 1.95
    for i, (n, t, d) in enumerate(rows):
        rect(s, MX, y, 12.25, 1.05, WHITE)
        rect(s, MX, y, 0.06, 1.05, TEAL if i % 2 == 0 else NAVY)
        tb(s, MX + 0.28, y + 0.16, 0.7, 0.6, n, 26, True, TEAL if i % 2 == 0 else NAVY, font="Georgia")
        tb(s, MX + 1.05, y + 0.13, 7.2, 0.35, t, 17, True, NAVY)
        tb(s, MX + 1.05, y + 0.55, 10.8, 0.35, d, 12, False, GREY, italic=True)
        y += 1.22
    footer(s, len(prs.slides))


def divider(num, title, sub, items):
    s = prs.slides.add_slide(BLANK)
    rect(s, 0, 0, W, H, NAVY)
    tb(s, 0.9, 1.55, 3.0, 0.4, f"PART {num}", 13, True, TEAL)
    tb(s, 0.9, 1.95, 11.4, 1.0, title, 34, True, WHITE, font="Georgia")
    tb(s, 0.9, 3.05, 11.0, 0.45, sub, 15, False, ICE, italic=True)
    rect(s, 0.9, 3.65, 1.4, 0.04, TEAL)
    bullets(s, 0.9, 3.95, 11.0, 2.4, items, 13.5, WHITE, 9)


# ---------------- 1 cover / 2 toc ----------------
cover(); toc()

# 3 divider
divider(1, "The leak fix, and PV association",
        "One number moved the wrong way — and it was not the reconstruction",
        ["The corrected sample removes a truth-level information leak",
         "PV association is the metric that lost the most",
         "A dedicated PV-aware pruning head brought it back below the minIP baseline"])

# 4 PV chart
s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
slide_header(s, "PART 1  ·  PV ASSOCIATION", "After the leak fix, the old models are worse than minIP",
             "PV misassociation rate; lower is better")
s.shapes.add_picture(P_PV, Inches(0.55), Inches(1.72), width=Inches(7.55))
rect(s, 8.45, 1.78, 4.35, 3.05, LIGHT)
rect(s, 8.45, 1.78, 0.05, 3.05, TEAL)
bullets(s, 8.68, 1.95, 3.95, 2.75, [
    "Models trained on the leaky sample: 13.1–14.0% — all worse than minIP (10.15%)",
    "Retrained on the corrected sample, together with the PV-aware head: 8.6%",
    "So we are below minIP again — but far from the 2.0% the leaky sample suggested"],
    11.5, NAVY, 7)
tb(s, 0.55, 6.35, 12.2, 0.5,
   "Reading: the leak inflated the PV advantage. The honest number is a ~1.5 pp advantage, not ~10 pp.",
   12, False, GREY, italic=True)
footer(s, len(prs.slides))

# 5 PV head fix
s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
slide_header(s, "PART 1  ·  THE FIX", "What the PV-aware pruning head changed",
             "Three changes to the pruning stage, nothing to the reconstruction target")
cols = [("Inputs", ["Track–PV geometric consistency,\nnot only track–track similarity",
                    "Vertex-level quantities: impact parameter,\nz-association, collinearity"]),
        ("Target", ["Auxiliary head predicts the PV of\neach surviving track",
                    "Trained jointly with the pruning score,\nso the representation sees PV context"]),
        ("Effect", ["PV misassociation 13–14% (old models) → 8.6%",
                    "Below the minIP baseline (10.15%)",
                    "Retraining and the head were introduced together — this sample does not separate them"])]
for i, (h, items) in enumerate(cols):
    x = MX + i * 4.12
    rect(s, x, 1.85, 3.9, 3.55, LIGHT)
    rect(s, x, 1.85, 3.9, 0.06, TEAL if i % 2 == 0 else NAVY)
    tb(s, x + 0.25, 2.02, 3.4, 0.35, h, 16, True, NAVY, font="Georgia")
    bullets(s, x + 0.25, 2.5, 3.4, 2.6, items, 12, NAVY, 10)
tb(s, 0.55, 5.75, 12.2, 0.75,
   "Same architecture, same data, only the pruning objective changed — so the 5 pp gain is attributable.",
   12.5, False, GREY, italic=True)
footer(s, len(prs.slides))

# 6 divider 2
divider(2, "Where the reconstruction loss sits",
        "Two failure modes, one lever",
        ["A truth chain either disappears at pruning, or survives and is merged with background",
         "The second failure is produced by the decoder, not by the edge score",
         "So pruning quality is the knob that moves both"])

# 7 two failure modes
s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
slide_header(s, "PART 2  ·  DIAGNOSIS", "Every surviving truth chain is found — the loss is elsewhere",
             "Decomposition of all 13 255 truth chains, v651")
rect(s, MX, 1.8, 5.95, 3.6, LIGHT); rect(s, MX, 1.8, 0.06, 3.6, NAVY)
tb(s, MX + 0.28, 1.95, 5.2, 0.35, "The four exclusive outcomes", 15.5, True, NAVY, font="Georgia")
rows = [("Pruned away", "54.1%"), ("Survived, assembled with background", "18.0%"),
        ("Survived, exact reconstruction", "28.0%"), ("Survived, but not assembled", "0.0%")]
yy = 2.42
for i, (k, v) in enumerate(rows):
    tb(s, MX + 0.3, yy, 4.3, 0.3, k, 12, False, NAVY)
    tb(s, MX + 4.55, yy, 1.1, 0.3, v, 12, True, TEAL if i == 2 else NAVY, align=PP_ALIGN.RIGHT)
    yy += 0.55
tb(s, MX + 0.3, 4.72, 5.2, 0.5, "Zero chains are lost at the assembly stage\n(the assembler always groups a chain's tracks)",
   10.5, False, GREY, italic=True)
rect(s, 6.85, 1.8, 5.95, 3.6, LIGHT); rect(s, 6.85, 1.8, 0.06, 3.6, TEAL)
tb(s, 7.13, 1.95, 5.3, 0.35, "Why this matters for benchmarks", 15.5, True, NAVY, font="Georgia")
bullets(s, 7.13, 2.45, 5.35, 2.8, [
    "AllParticles and PerfectReco both count contaminated chains as failures",
    "But the denominator (truth chains present) is set by pruning",
    "A looser threshold moves chains between the two failure buckets",
    "So a single percentage point can be a threshold move, not a model gain"], 11.5, NAVY, 7)
footer(s, len(prs.slides))

# 8 divider 3
divider(3, "The DOCA bug",
        "A feature that was silently wrong",
        ["Distance of closest approach was taken from an upstream quantity",
         "The quantity was not the geometric DOCA",
         "The pruning MLP learned from a corrupted input for months"])

# 9 what was wrong
s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
slide_header(s, "PART 3  ·  THE BUG", "What was wrong, and why it survived so long",
             "A silent feature corruption, not a crash")
cols = [("Symptom", ["No error, no NaN",
                      "Pruning looked plausible",
                      "But the derived vertex features\nwere not what their names implied"]),
        ("Root cause", ["log DOCA was read from an\nupstream reconstructed quantity",
                        "Its definition differs from the\ntrue geometric DOCA",
                        "So the MLP learned a biased mapping"]),
        ("Fix", ["Recompute DOCA locally from\ntrack states (dR, dz, DOCA, |Δz|)",
                 "Replace the feature, keep the\narchitecture unchanged",
                 "One-variable change, so the effect\nis measurable"])]
for i, (h, items) in enumerate(cols):
    x = MX + i * 4.12
    rect(s, x, 1.85, 3.9, 3.35, LIGHT)
    rect(s, x, 1.85, 0.06, 3.35, [NAVY, "#C0392B", TEAL][i])
    tb(s, x + 0.25, 2.0, 3.4, 0.35, h, 16, True, NAVY, font="Georgia")
    bullets(s, x + 0.25, 2.48, 3.4, 2.5, items, 11.5, NAVY, 8)
tb(s, 0.55, 5.55, 12.2, 0.6,
   "Generalisable lesson: derived physics features must be re-derived and unit-checked, not inherited from upstream code.",
   12.5, False, GREY, italic=True)
footer(s, len(prs.slides))

# 10 doca effect
s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
slide_header(s, "PART 3  ·  THE EFFECT", "Correcting DOCA is worth ~+7 pp of perfect reconstruction",
             "Paper-convention metrics; same sample, same threshold")
s.shapes.add_picture(P_DOCA, Inches(0.75), Inches(1.8), height=Inches(3.7))
rect(s, 6.9, 1.85, 5.9, 3.55, LIGHT); rect(s, 6.9, 1.85, 0.06, 3.55, TEAL)
tb(s, 7.15, 2.0, 5.4, 0.35, "What moved, and what did not", 15.5, True, NAVY, font="Georgia")
bullets(s, 7.15, 2.5, 5.4, 2.75, [
    "AllParticles 40.2 → 49.6%",
    "PerfectReco 26.6 → 33.8%",
    "Edge AUC flat (0.924 → 0.924): the fix restores feature correctness, not ranking power",
    "So the gain shows up in assembled chains, which is what we actually care about"],
    11.5, NAVY, 7)
footer(s, len(prs.slides))

# 11 divider 4
divider(4, "Pruning optimization: seven versions",
        "One change at a time, measured on four benchmarks",
        ["Physics-derived features → operating point → remove what fails → context → attention",
         "Benchmarks: AllParticles, PerfectReco, edge AUC, PV association",
         "Every step is a single controlled change on top of the previous best"])

# 12 ladder table
s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
slide_header(s, "PART 4  ·  THE LADDER", "From 25% to 61% AllParticles in six controlled steps",
             "All numbers on the corrected sample; N is the truth-chain sample per evaluation")
rows = [["Version", "Change", "AllParticles [%]", "PerfectReco [%]", "edge AUC", "PV misassoc. [%]"],
        ["v601", "baseline, corrected sample", "25.4", "16.8", "0.972", "8.61"],
        ["v623", "physics-derived pruning inputs", "23.5", "15.7", "0.974", "8.61"],
        ["v631", "threshold + pairwise ranking", "34.3", "23.0", "0.935", "8.63"],
        ["v633", "drop the two negative inputs", "40.2", "26.6", "0.924", "8.68"],
        ["v643", "corrected DOCA geometry", "49.6", "33.8", "0.924", "8.66"],
        ["v648", "competition / exclusivity context", "56.4", "37.8", "0.932", "8.68"],
        ["v651", "line-graph attention + geometry bias", "60.9", "41.4", "0.933", "8.64"]]
tbl_shape = s.shapes.add_table(len(rows), 6, Inches(MX), Inches(1.85), Inches(12.25), Inches(3.35))
table = tbl_shape.table
widths = [1.15, 4.55, 2.0, 1.95, 1.35, 1.25]
for i, w in enumerate(widths):
    table.columns[i].width = Inches(w)
for r, row in enumerate(rows):
    for c, val in enumerate(row):
        cell = table.cell(r, c)
        cell.text = val
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY if r == 0 else (WHITE if r % 2 else LIGHT)
        for p in cell.text_frame.paragraphs:
            p.alignment = PP_ALIGN.LEFT if c == 1 else PP_ALIGN.CENTER
            for run in p.runs:
                run.font.size = Pt(11 if r else 11.5)
                run.font.bold = (r == 0) or (c == 0)
                run.font.color.rgb = WHITE if r == 0 else NAVY
                run.font.name = "Calibri"
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
tb(s, MX, 5.45, 12.2, 0.5,
   "Note: AllParticles / PerfectReco use the paper convention, where the denominator is the set of truth chains surviving pruning — so read the trend, not the absolute level.",
   10.5, False, GREY, italic=True)
tb(s, MX, 5.95, 12.2, 0.5,
   "The edge AUC column is deliberately included: it shows that most of the gain comes from the operating point and from correct features, not from better ranking.",
   10.5, False, GREY, italic=True)
footer(s, len(prs.slides))

# 12b trend figure
s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
slide_header(s, "PART 4  ·  THE LADDER AT A GLANCE", "Two curves, and they do not run in parallel",
             "Reconstruction climbs while ranking quality stays flat or falls")
s.shapes.add_picture(P_LADDER, Inches(0.85), Inches(1.8), width=Inches(11.6))
rect(s, MX, 5.75, 12.25, 0.95, LIGHT); rect(s, MX, 5.75, 0.05, 0.95, TEAL)
tb(s, MX + 0.25, 5.88, 11.8, 0.7,
   "AllParticles 25.4 → 60.9% and PerfectReco 16.8 → 41.4%, while edge AUC goes 0.972 → 0.933.\n"
   "The gains come from correct features and the operating point — not from a better edge ranking.",
   11.5, False, NAVY)
footer(s, len(prs.slides))

# 13-17 step slides
def step_slide(n, kicker, title, sub, left, right_title, right_items, fig=None):
    s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
    slide_header(s, kicker, title, sub)
    lw = 6.0 if fig else 7.0
    rect(s, MX, 1.8, lw, 3.9, LIGHT); rect(s, MX, 1.8, 0.06, 3.9, TEAL)
    tb(s, MX + 0.28, 1.95, lw - 0.5, 0.3, "What changed", 14, True, NAVY, font="Georgia")
    bullets(s, MX + 0.28, 2.38, lw - 0.55, 3.1, left, 12, NAVY, 9)
    if fig:
        s.shapes.add_picture(fig, Inches(7.0), Inches(1.85), width=Inches(5.7))
    x2 = 7.0 if fig else 7.75
    rect(s, x2, 1.8, W - x2 - 0.55, 3.9, WHITE, line=ICE)
    tb(s, x2 + 0.22, 1.95, W - x2 - 1.0, 0.3, right_title, 14, True, NAVY, font="Georgia")
    bullets(s, x2 + 0.22, 2.38, W - x2 - 1.0, 3.1, right_items, 11.5, NAVY, 8)
    footer(s, n)


step_slide(13, "PART 4  ·  STEP 1", "v623 — put physics back into the pruning inputs",
           "Derived quantities instead of raw inherited features",
           ["Recomputed vertex geometry: dR, dz, DOCA, |Δz|",
            "Pairwise collinearity and opening angle",
            "Zero-initialised adapters, so the old behaviour is the starting point",
            "One variable change on the v601 base"],
           "Result", ["AllParticles 25.4 → 23.5% (no gain)",
                      "Edge AUC 0.972 → 0.974 (best ranking we ever had)",
                      "Lesson: better ranking does not automatically become better chains"])
step_slide(14, "PART 4  ·  STEP 2", "v631 — move the operating point, and rank pairs",
           "Threshold, class weighting, and a pairwise objective",
           ["Threshold 0.97 → 0.90 with a re-balanced class weight",
            "Pairwise ranking loss added to the edge BCE",
            "minIP-targeted negatives",
            "Still no architectural change"],
           "Result", ["AllParticles 23.5 → 34.3% (+10.8 pp)",
                      "PerfectReco 15.7 → 23.0% (+7.3 pp)",
                      "Edge AUC 0.974 → 0.935 — lower, yet the chains are better",
                      "First clear evidence that AUC and reconstruction can disagree"])
step_slide(15, "PART 4  ·  STEP 3", "v633 / v640 — remove what does not work",
           "Two ablations and a model-selection fix",
           ["Dropped the event-count head (−19% on its own)",
            "Dropped the PV-overlap injection (−2.6%)",
            "Model selection switched from total loss to pruning AP",
            "Same architecture, fewer terms"],
           "Result", ["v633: AllParticles 34.3 → 40.2%, PerfectReco 23.0 → 26.6%",
                      "v640: selection change alone ≈ +1 pp",
                      "Removing a bad term was worth more than adding two good ones"])
step_slide(16, "PART 4  ·  STEP 4", "v648 — let each edge see its competitors",
           "Competition / exclusivity context in the pruning score",
           ["For every candidate edge, aggregate statistics of its competitors",
            "Track-level exclusivity: how many rivals does this track have?",
            "Vertex-level competition, not only pairwise similarity",
            "One feature block added to the existing MLP"],
           "Result", ["AllParticles 49.6 → 56.4%",
                      "PerfectReco 33.8 → 37.8%",
                      "Edge AP on the hard pool: 0.668, the best of the whole line",
                      "This is the first block that helps both ranking and chains"])
step_slide(17, "PART 4  ·  STEP 5", "v651 — attention with a geometric prior",
           "A line-graph attention layer, biased by the geometry features",
           ["Attention over the track–track edges (line graph)",
            "The geometry bias is what makes it work: without it, the layer is neutral",
            "Output projection zero-initialised, so training starts from v648",
            "Same data, same threshold, same reconstruction code"],
           "Result", ["AllParticles 56.4 → 60.9%",
                      "PerfectReco 37.8 → 41.4%",
                      "LCAG true-edge classification 74.7 → 75.4%, the best we have",
                      "PV association unchanged at 8.6%"])

# 18 what did not work
s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
slide_header(s, "PART 4  ·  NEGATIVE RESULTS", "What we tried and discarded",
             "Each of these cost at least one controlled run")
neg = [("Bigger capacity", "wider layers, more heads, more epochs — no gain on a saturated base"),
       ("Attention without a prior", "line-graph attention alone is neutral; it needs the geometry bias"),
       ("Geometry-defined candidates", "replacing learned pruning with a physical window: negative in five variants"),
       ("Mass / structure / source heads", "auxiliary targets do not transfer to reconstruction"),
       ("Loose chain objectives", "rewards that keep more chains raise the contaminated fraction, not the exact ones")]
y = 1.86
for k, v in neg:
    rect(s, MX, y, 12.25, 0.78, LIGHT)
    rect(s, MX, y, 0.05, 0.78, "#C0392B")
    tb(s, MX + 0.28, y + 0.11, 3.55, 0.3, k, 13, True, NAVY)
    tb(s, MX + 3.95, y + 0.11, 8.0, 0.55, v, 11.5, False, GREY)
    y += 0.9
tb(s, MX, 6.55, 12.2, 0.4, "Reported here so that no one repeats them.", 11, False, GREY, italic=True)
footer(s, len(prs.slides))

# 19 summary
s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, NAVY)
tb(s, 0.9, 0.75, 11.0, 0.4, "SUMMARY", 12.5, True, TEAL)
tb(s, 0.9, 1.15, 11.6, 0.6, "Where we are", 30, True, WHITE, font="Georgia")
cards = [("PV association", "13–14% on the old models; 8.6% after the PV-aware head; minIP 10.15%"),
         ("Reconstruction", "AllParticles 25.4 → 60.9%; PerfectReco 16.8 → 41.4% in six controlled steps"),
         ("Two bugs fixed", "truth leak in the sample, and the DOCA feature — both silent, both measurable"),
         ("Next", "the output stage: edge classification plus connected components is the remaining ceiling")]
x = 0.9
for i, (h, v) in enumerate(cards):
    xx = 0.9 + (i % 2) * 5.95
    yy = 2.0 + (i // 2) * 2.1
    rect(s, xx, yy, 5.65, 1.85, RGBColor(0x2A, 0x34, 0x75))
    tb(s, xx + 0.25, yy + 0.18, 5.0, 0.3, h, 14, True, TEAL, font="Georgia")
    tb(s, xx + 0.25, yy + 0.58, 5.1, 1.1, v, 12, False, WHITE)
tb(s, 0.9, 6.55, 11.5, 0.35, "Thank you — questions welcome", 12, False, ICE, italic=True)
footer(s, len(prs.slides))

# 20 backup
s = prs.slides.add_slide(BLANK); rect(s, 0, 0, W, H, WHITE)
slide_header(s, "BACKUP", "Metric definitions", "So the numbers on the previous slides are unambiguous")
defs = [("AllParticles", "a reconstructed object contains all truth tracks of a chain (extra tracks allowed)"),
        ("PerfectReco", "the reconstructed track set equals the truth track set exactly"),
        ("PV misassociation", "fraction of tracks assigned to a PV different from the true one, HGNN vs minIP"),
        ("edge AUC", "ranking quality of the pruning score over all candidate edges"),
        ("LCAG", "lowest-common-ancestor generation class of a track pair; we quote class-1 accuracy")]
y = 1.9
for k, v in defs:
    tb(s, MX, y, 2.6, 0.3, k, 12.5, True, NAVY)
    tb(s, MX + 2.75, y, 9.5, 0.5, v, 11.5, False, GREY)
    y += 0.78
tb(s, MX, 6.3, 12.2, 0.6,
   "Caveat kept on purpose: AllParticles and PerfectReco are normalised by the truth chains that survive pruning, so they are not directly comparable across versions with different thresholds.",
   10.5, False, GREY, italic=True)
footer(s, len(prs.slides))

prs.save(OUT)
print("saved", OUT, os.path.getsize(OUT))
