#!/usr/bin/env python3
"""DFEI group meeting, 2026-10-12 - the talk.

"Which optimisations actually worked - and why": eleven ideas tested on one
ruler, the mechanism behind the winners, and the pattern that separates them
from the losers.

Style primitives are the ones from make_dfei_meeting_20260915.py (chip,
title + rule, bullets, figures, italic note, footer).  Only the date and the
figure directory changed; every number comes from the talk brief.
"""
import os

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

BASE = '/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn'
FIG = BASE + '/meeting_figs_20261012'
OUT = FIG + '/DFEI_principles_20261012_EN.pptx'

BLUE = RGBColor(0x1F, 0x4E, 0x79)
DARK = RGBColor(0x2B, 0x2B, 0x2B)
GRAY = RGBColor(0x77, 0x77, 0x77)
PURPLE = RGBColor(0x8E, 0x44, 0xAD)
TEAL = RGBColor(0x2C, 0x7A, 0x7B)
PINK = RGBColor(0xC2, 0x18, 0x5B)
GOLD = RGBColor(0xB8, 0x86, 0x0B)
GREEN = RGBColor(0x2E, 0x8B, 0x57)
LIGHTBG = RGBColor(0xF7, 0xF7, 0xF7)
ROWALT = RGBColor(0xF2, 0xF6, 0xFA)
FONT = 'Calibri'
BODY = 24
NOTE = 21
TITLE = 30

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]

LEFT, LW = 0.55, 6.05          # bullet column
RX, RW = 6.85, 6.05            # right column
N = [0]


def footer(s):
    tb = s.shapes.add_textbox(Inches(0.55), Inches(7.12), Inches(12.30), Inches(0.30))
    r = tb.text_frame.paragraphs[0].add_run()
    r.text = 'Qingxiang Guo · UCAS · DFEI group meeting · 2026-10-12          ' + str(N[0])
    r.font.size = Pt(13); r.font.color.rgb = GRAY; r.font.name = FONT


def slide(chip, chip_color, title=None):
    N[0] += 1
    s = prs.slides.add_slide(BLANK)
    if title is None:                       # title page
        footer(s)
        return s
    if chip:
        cw = 0.30 * len(str(chip)) + 0.34
        b = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.42), Inches(0.16),
                               Inches(cw), Inches(0.50))
        b.fill.solid(); b.fill.fore_color.rgb = chip_color
        b.line.fill.background()
        tf = b.text_frame
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
        r = p.add_run(); r.text = str(chip)
        r.font.size = Pt(17); r.font.bold = True
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF); r.font.name = FONT
        left = 0.42 + cw + 0.20
    else:
        left = 0.55
    tb = s.shapes.add_textbox(Inches(left), Inches(0.12), Inches(13.333 - left - 0.30), Inches(0.92))
    tf = tb.text_frame; tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    avail = 13.333 - left - 0.30
    size = 24
    for cand in (TITLE, 27, 24):
        if len(title) * 0.0082 * cand <= avail:
            size = cand
            break
    r = tf.paragraphs[0].add_run(); r.text = title
    r.font.size = Pt(size); r.font.bold = True; r.font.color.rgb = BLUE; r.font.name = FONT
    ln = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.55), Inches(1.10), Inches(12.30), Pt(2.6))
    ln.fill.solid(); ln.fill.fore_color.rgb = BLUE; ln.line.fill.background()
    footer(s)
    return s


def bullets(s, items, left, top, width, height, size=BODY, space=8, lsp=None, color=None):
    tb = s.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame; tf.word_wrap = True
    for i, it in enumerate(items):
        txt, lvl = it if isinstance(it, tuple) else (it, 0)
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(space)
        if lsp:
            p.line_spacing = lsp
        r = p.add_run()
        r.text = ('      ' * lvl) + ('•  ' if lvl == 0 else '–  ') + txt
        r.font.size = Pt(size - 2 * lvl)
        r.font.color.rgb = color or DARK
        r.font.name = FONT
    return tb


def note(s, txt, top=6.62, size=NOTE, left=0.55, width=12.30):
    top = min(top, 6.62)                    # keep clear of the footer at 7.12 in
    tb = s.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(0.46))
    tf = tb.text_frame; tf.word_wrap = True
    r = tf.paragraphs[0].add_run(); r.text = txt
    r.font.size = Pt(size); r.font.color.rgb = GRAY; r.font.name = FONT
    r.font.italic = True
    return tb


def pic(s, path, left, top, max_w, max_h, center_in=None, center_v=None):
    if not os.path.exists(path):
        print('[warn] missing figure: ' + path)
        return None
    from PIL import Image
    iw, ih = Image.open(path).size
    ar = iw / float(ih)
    if max_w / max_h > ar:
        hh, ww = max_h, max_h * ar
    else:
        ww, hh = max_w, max_w / ar
    if center_in is not None:
        left = center_in[0] + max(0.0, (center_in[1] - ww) / 2.0)
    if center_v is not None:
        top = center_v[0] + max(0.0, (center_v[1] - hh) / 2.0)
    return s.shapes.add_picture(path, Inches(left), Inches(top), width=Inches(ww), height=Inches(hh))


def card(s, header, items, left, top, width, height, color, size=14):
    b = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top),
                           Inches(width), Inches(height))
    b.fill.solid(); b.fill.fore_color.rgb = LIGHTBG
    b.line.color.rgb = color; b.line.width = Pt(1.25)
    hb = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top),
                            Inches(width), Inches(0.52))
    hb.fill.solid(); hb.fill.fore_color.rgb = color; hb.line.fill.background()
    tf = hb.text_frame; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = Inches(0.15); tf.margin_top = tf.margin_bottom = 0
    r = tf.paragraphs[0].add_run(); r.text = header
    r.font.size = Pt(16); r.font.bold = True
    r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF); r.font.name = FONT
    tb = s.shapes.add_textbox(Inches(left + 0.16), Inches(top + 0.60),
                              Inches(width - 0.32), Inches(height - 0.68))
    tf = tb.text_frame; tf.word_wrap = True
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(3)
        r = p.add_run(); r.text = '•  ' + it
        r.font.size = Pt(size); r.font.color.rgb = DARK; r.font.name = FONT
    return b


def table(s, data, left, top, width, height, colw=None, size=16):
    rows, cols = len(data), len(data[0])
    shp = s.shapes.add_table(rows, cols, Inches(left), Inches(top),
                             Inches(width), Inches(height))
    tbl = shp.table
    if colw:
        for j, w in enumerate(colw):
            tbl.columns[j].width = Inches(w)
    for i, row in enumerate(data):
        for j, val in enumerate(row):
            c = tbl.cell(i, j)
            c.text = str(val)
            c.margin_left = c.margin_right = Inches(0.06)
            c.margin_top = c.margin_bottom = Inches(0.02)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = c.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER if j > 0 else PP_ALIGN.LEFT
            r = p.runs[0]
            r.font.size = Pt(size); r.font.name = FONT
            r.font.bold = (i == 0)
            r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF) if i == 0 else DARK
            c.fill.solid()
            c.fill.fore_color.rgb = BLUE if i == 0 else (ROWALT if i % 2 else RGBColor(0xFF, 0xFF, 0xFF))
    return shp


# ============================================================ 1 title
s = slide(None, None)
tb = s.shapes.add_textbox(Inches(0.85), Inches(2.05), Inches(11.6), Inches(1.6))
tf = tb.text_frame; tf.word_wrap = True
r = tf.paragraphs[0].add_run(); r.text = 'Which optimisations actually worked - and why'
r.font.size = Pt(40); r.font.bold = True; r.font.color.rgb = BLUE; r.font.name = FONT
tb = s.shapes.add_textbox(Inches(0.85), Inches(3.55), Inches(11.6), Inches(0.7))
r = tb.text_frame.paragraphs[0].add_run()
r.text = 'DFEI · 11 ideas tested on the same ruler'
r.font.size = Pt(BODY); r.font.color.rgb = DARK; r.font.name = FONT
tb = s.shapes.add_textbox(Inches(0.85), Inches(4.30), Inches(11.6), Inches(0.6))
r = tb.text_frame.paragraphs[0].add_run()
r.text = 'Qingxiang Guo · University of Chinese Academy of Sciences · 12 October 2026'
r.font.size = Pt(22); r.font.color.rgb = GRAY; r.font.name = FONT

# ============================================================ 2 outline
s = slide('OUTLINE', BLUE, 'Outline')
items = [('PART 0', 'the scoreboard - 11 ideas, and which ones moved', TEAL),
         ('PART 1', 'first, the ruler - why the criterion comes before the model', BLUE),
         ('PART 2', 'the loss - kill the weakest link of every chain', PINK),
         ('PART 3', 'attention over the neighbourhood', GOLD),
         ('PART 4', 'competition context - what trees need, GNNs compute', PURPLE),
         ('PART 5-6', 'the pattern, and where the physics is still empty', GREEN),
         ('SUMMARY', 'what we know, and one methodological lesson', BLUE)]
y = 1.42
for tag, txt, col in items:
    b = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.75), Inches(y),
                           Inches(1.95), Inches(0.58))
    b.fill.solid(); b.fill.fore_color.rgb = col; b.line.fill.background()
    tf = b.text_frame; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = tag
    r.font.size = Pt(18); r.font.bold = True
    r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF); r.font.name = FONT
    tb = s.shapes.add_textbox(Inches(3.00), Inches(y + 0.02), Inches(9.6), Inches(0.58))
    tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    r = tf.paragraphs[0].add_run(); r.text = txt
    r.font.size = Pt(BODY); r.font.color.rgb = col; r.font.name = FONT
    y += 0.74
note(s, 'Every verdict rests on a controlled probe run on the same pool - the numbers are '
        'the ones in the brief.', top=6.45)

# ============================================================ 3 PART 0 scoreboard
s = slide('PART 0', TEAL, 'The scoreboard: 11 ideas, tested on one ruler')
bullets(s, [
    'The ruler: 18 versions, 500-event test set, hard-pool AP and chain survival.',
    'Eleven distinct ideas were tried; the ones that moved the number all changed '
    'what is optimised or what the model can see.',
    'The ones that failed only changed how the same quantity is computed.',
], LEFT, 1.28, LW, 4.9, space=12)
card(s, 'EFFECTIVE', [
    'chain-level loss (weakest-link)',
    'attention over the neighbourhood',
    'competition-context feature (derived_comp)',
    '4 GNN layers',
    'learning rate 3e-4',
    'seed / determinism fixes',
], RX, 1.28, RW, 2.42, GREEN)
card(s, 'INEFFECTIVE', [
    'line-graph edge-edge attention',
    'geometric pair-bias',
    'top-k · sparsemax',
    'per-edge Sinkhorn matching',
    'vertex geometry · hand-crafted ctx',
    'more input features (for the GNN)',
    'more capacity / recipes / metrics',
], RX, 3.92, RW, 2.62, PINK)
note(s, 'Six ideas moved the needle; five families did not - this talk is the evidence for each '
        'verdict, and the mechanism behind it.', top=6.68)

# ============================================================ 4 PART 1 ruler
s = slide('PART 1', BLUE, 'First the ruler: two orderings that disagree')
bullets(s, [
    'Two rulers on the same 18 versions: hard-pool AP and chain survival.',
    'v632 is last on AP (0.6081) but 4th on chain survival (49.9 %).',
    'v623 is first on AP (0.6534) but only 5th on chain survival (47.1 %).',
    'v648 ranks #3 on AP (0.6524) but #17 on chain survival (38.7 %).',
    'v642 / v641 rank #9 / #10 on AP but hold the top two chain survivals (61.6 % / 59.2 %).',
    'The criterion was re-cut twice: All_fix, then threshold-free AP, then threshold-dependent '
    'chain survival.',
], LEFT, 1.26, LW, 5.45, space=7)
pic(s, FIG + '/F1_dumbbell_ap_vs_chain.png', RX, 1.26, RW, 5.20,
    center_in=(RX, RW), center_v=(1.26, 5.20))
note(s, 'Pick the criterion before the model: the metric decides which optimisation looks good.',
     top=6.85)

# ============================================================ 5 PART 1 why it diverges
s = slide('PART 1', BLUE, 'Why the two rulers diverge')
bullets(s, [
    'The physical criterion is an AND over a chain: it survives only if every link does.',
    'So chain survival falls as p^n - a 20 % per-link loss still costs ~36 % of 2-link chains.',
    'Short chains (n = 2) are the most sensitive; the 18 arms collapse onto the p^n curve.',
    'Edge-level AP is dominated by the many easy background edges, so it hides long-chain damage.',
], LEFT, 1.26, LW, 5.35, space=9)
pic(s, FIG + '/F3_and_semantics.png', RX, 1.30, RW, 2.42)
pic(s, FIG + '/F2_chain_surv_bar.png', RX, 3.92, RW, 2.55)
note(s, 'Same model, two rulers - the one that matches the physics is the one to optimise.',
     top=6.85)

# ============================================================ 6 PART 2 principle
s = slide('PART 2', PINK, 'Idea 1: kill the weakest link (the loss)')
bullets(s, [
    'Per-edge BCE optimises the mean; the target is "no link of the chain may fail".',
    'The chain loss writes the weakest link of every truth chain into the objective.',
    'Softmin: surv(gamma) = -ln(mean(exp(-gamma·p))) / gamma, differentiable everywhere.',
    'At gamma -> 1 it is the per-edge average; at large gamma it tends to min(p).',
    'One hyper-parameter interpolates smoothly between "average" and "weakest link".',
], LEFT, 1.26, LW, 5.30, space=9)
pic(s, FIG + '/F4_softmin_gamma.png', RX, 1.26, RW, 5.20,
    center_in=(RX, RW), center_v=(1.26, 5.20))
note(s, 'The idea is not a new loss name - it is a new quantity being optimised.', top=6.80)

# ============================================================ 7 PART 2 evidence
s = slide('PART 2', PINK, 'The evidence: chain survival up on every seed')
bullets(s, [
    'Multi-B subset, 3 seeds, 696 events / 281 truth chains / 5248 test edges.',
    'Chain survival at the edge-precision-90% point: 24.2 % (w = 0) -> 32.6 % (softmin w = 1.0).',
    'All three seeds improve: 24.6 / 23.6 / 24.5 -> 33.5 / 28.5 / 35.8.',
    'The edge-level rulers move the same way: AP 0.8393 -> 0.8533, AUC 0.8650 -> 0.8732.',
    'So the chain loss is not a trade - it lifts the edge and the chain rulers together.',
], LEFT, 1.30, LW, 5.35, space=9)
pic(s, FIG + '/F5_chainloss_probe.png', RX, 1.35, RW, 5.00,
    center_in=(RX, RW), center_v=(1.35, 5.00))
note(s, 'Chain survival +8.4 pp on every seed, with edge AP and AUC also up.', top=6.80)

# ============================================================ 8 PART 2 non-linearity
s = slide('PART 2', PINK, 'Why non-linearity matters')
bullets(s, [
    'The product / geometric mean (≈ per-edge BCE in log space) only reaches 28.0 %.',
    'Softmin reaches 32.6 %, and softmin w = 0.3 sits in between at 30.1 %.',
    'So the gain comes from penalising the weakest link, not from "using a product".',
    'Main-model check: v641 / v642 (v633 + chain_recall) hold 59.2 % and 61.6 % chain survival.',
    'Their AP is only #10 / #9 (0.6380 / 0.6389), while the AP champion v623 sits at 47.1 %.',
], LEFT, 1.30, LW, 5.35, space=9)
pic(s, FIG + '/F5_chainloss_probe.png', RX, 1.35, RW, 5.00,
    center_in=(RX, RW), center_v=(1.35, 5.00))
note(s, 'A non-linear penalty on the weakest link is what works - the product form is not.',
     top=6.80)

# ============================================================ 9 PART 3 attention
s = slide('PART 3', GOLD, 'Idea 2: attention over the neighbourhood')
bullets(s, [
    'Each edge should see its neighbours - message passing over the graph, not only its own features.',
    'Attention on 30-dim node+geo+ctx: AP 0.8460 ± 0.0118, AUC 0.8745 ± 0.0094.',
    'The same attention on all 47 raw features: AP 0.8450 ± 0.0172 - raw input adds nothing.',
    'A plain MLP on the same 47 features: AP 0.8015 ± 0.0149, AUC 0.8480 ± 0.0047.',
    'The MLP has the smallest spread, so the ~4.5 AP-point gap is far outside the noise.',
], LEFT, 1.30, LW, 5.35, space=9)
pic(s, FIG + '/F6_attn_vs_mlp.png', RX, 1.35, RW, 5.00,
    center_in=(RX, RW), center_v=(1.35, 5.00))
note(s, 'Neighbourhood attention is the second real lever; extra raw features are not.', top=6.80)

# ============================================================ 10 PART 3 boundary
s = slide('PART 3', GOLD, 'And the boundary: fancy attention buys nothing')
bullets(s, [
    'Attention itself works; the elaborate attention variants do not.',
    'Top-k with k = 4 / 8 / 16 / 32 all land in 0.79 - 0.80 - no separation at all.',
    'Sparsemax is worse than softmax (0.7847).',
    'Edge-level Sinkhorn scores 0.72 - 0.75, far below pairwise ≈ 0.81; combining adds +0.001 - 0.002.',
    'Geometry pair-bias and vertex geometry sit on the same plateau as the feature stack.',
], LEFT, 1.30, LW, 5.35, space=9)
pic(s, FIG + '/F7_negatives.png', RX, 1.35, RW, 5.00,
    center_in=(RX, RW), center_v=(1.35, 5.00))
note(s, 'Changing how attention is computed bought nothing.', top=6.80)

# ============================================================ 11 PART 4 competition context
s = slide('PART 4', PURPLE, 'Idea 3: competition context')
bullets(s, [
    'A true edge differs from a cross-chain fake not only in geometry, but in how many competitors it has.',
    'Idea: add the affinity distribution of the other candidate edges of the same track (derived_comp).',
    'GBDT probe on multi-B: node alone 0.6916, and 0.8391 once context is included.',
    'The competition context (degree - how many partners an edge has) is one of the top signals.',
], LEFT, 1.30, LW, 5.35, space=10)
pic(s, FIG + '/F8_features.png', RX, 1.30, RW, 5.00,
    center_in=(RX, RW), center_v=(1.30, 5.00))
note(s, 'Give the model the competition, and the ambiguous cross-chain edges separate.', top=6.80)

# ============================================================ 12 PART 4 trees vs GNNs
s = slide('PART 4', PURPLE, 'Why trees and GNNs disagree')
bullets(s, [
    'Trees must be fed geometry explicitly: trdist (+0.073) and flight (+0.048) carry the signal.',
    'Attention computes those relations itself - that is why raw features add nothing to the GNN.',
    'Geometry adds +0.113 AP on multi-B but only +0.032 on single-B: 3.5x more valuable when '
    'chains compete.',
    'On the full set it is less again: node 0.6786, +geo 0.7327.',
    'Same information, different mechanism - it matters who computes it.',
], LEFT, 1.30, LW, 5.35, space=9)
pic(s, FIG + '/F8_features.png', RX, 1.30, RW, 5.00,
    center_in=(RX, RW), center_v=(1.30, 5.00))
note(s, 'The gap between GBDT and GNN is a difference in mechanism, not in information.', top=6.80)

# ============================================================ 13 PART 5 the pattern
s = slide('PART 5', GREEN, 'The pattern behind the scoreboard')
bullets(s, [
    'Every change that worked altered WHAT is optimised or WHAT the model can see.',
    'Every change that failed only altered HOW the same thing is computed.',
    'Loss: optimise the chain, not the edge.',
    'Features: add the competition context, not more raw channels.',
    'Neighbourhood: let edges see neighbours - not a cleverer attention kernel.',
], LEFT, 1.30, LW, 6.05, space=9)
card(s, 'CHANGED THE OBJECTIVE / THE VIEW', [
    'weakest-link chain loss',
    'message passing over neighbours',
    'competition-context features',
    'deeper stack (4 layers), lr = 3e-4',
], RX, 1.30, RW, 2.55, GREEN)
card(s, 'CHANGED THE MACHINERY ONLY', [
    'attention sparsifiers (top-k, sparsemax)',
    'edge matching (Sinkhorn)',
    'geometric pair-bias, vertex geometry',
    'more capacity, more raw features',
], RX, 4.00, RW, 2.55, PINK)

# ============================================================ 14 PART 6 stratification
s = slide('PART 6', TEAL, 'Where the physics is still empty')
bullets(s, [
    'Bench: model AP gain over the traditional geometric baseline (base2), split by truth chains '
    'per event (n_B).',
    'n_B = 1: positive for every version (+0.027 to +0.048).',
    'n_B = 2: still positive (+0.045 to +0.092).',
    'n_B = 3: negative for every version (-0.035, -0.081, -0.120).',
    'And chain survival for n_B = 3 is exactly 0 for all versions.',
], LEFT, 1.30, LW, 5.35, space=9)
table(s, [
    ['version', 'n_B = 1', 'n_B = 2', 'n_B = 3'],
    ['v623', '+0.027', '+0.092', '-0.120'],
    ['v648', '+0.048', '+0.077', '-0.081'],
    ['v643', '+0.042', '+0.045', '-0.035'],
], RX, 1.55, RW, 1.95, colw=[2.05, 1.33, 1.33, 1.34], size=18)
note(s, 'The dense, multi-chain regime (n_B >= 3) is the next target: the gain is still negative there.',
     top=6.80)

# ============================================================ 15 PART 6 running now
s = slide('PART 6', TEAL, 'Running now')
bullets(s, [
    'In the bench: v653 (softmin weakest link), v654 (chain_recall w = 5), v655 (chain_recall w = 15).',
    'v651 (geometry bias + capacity), v652 (derived_vertex), v645 / v647 (pair_sym).',
    'Next: choose the base model by chain survival, not by AP.',
    'And add a chain-level metric to validation, so the checkpoint is picked on the same ruler.',
], LEFT, 1.30, LW, 5.35, space=11)
arms = [('v653', 'softmin weakest link', GREEN),
        ('v654', 'chain_recall w = 5', PINK),
        ('v655', 'chain_recall w = 15', PINK),
        ('v651', 'geometry bias + capacity', GOLD),
        ('v652', 'derived_vertex', PURPLE),
        ('v645 / v647', 'pair_sym', TEAL)]
y = 1.42
for tag, txt, col in arms:
    b = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(RX), Inches(y),
                           Inches(2.25), Inches(0.58))
    b.fill.solid(); b.fill.fore_color.rgb = col; b.line.fill.background()
    tf = b.text_frame; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = tag
    r.font.size = Pt(16); r.font.bold = True
    r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF); r.font.name = FONT
    tb = s.shapes.add_textbox(Inches(RX + 2.40), Inches(y), Inches(3.65), Inches(0.58))
    tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    r = tf.paragraphs[0].add_run(); r.text = txt
    r.font.size = Pt(15); r.font.color.rgb = col; r.font.name = FONT
    y += 0.72
note(s, 'Selection and checkpointing move to the ruler that matches the physics.', top=6.80)

# ============================================================ 16 summary
s = slide('SUMMARY', GREEN, 'Summary and the lesson')
bullets(s, [
    'Worked - chain-level (weakest-link) loss: chain survival 24.2 -> 32.6 %, 3/3 seeds.',
    'Worked - neighbourhood attention: AP 0.8460 against 0.8015 for a plain MLP.',
    'Worked - competition context: multi-B GBDT AP 0.6916 -> 0.8391.',
    'Did not - sparsifiers, Sinkhorn, geometry pair-bias, vertex geometry, more capacity.',
    'The pattern: winners change the objective or the view; losers change the machinery.',
    'Lesson: pick the ruler before the model, and assert invariants so a silent mis-framing is caught.',
], LEFT, 1.30, 12.30, 4.9, space=13)
note(s, 'Everything here is on the 500-event test set or the multi-B probe pool.', top=6.60)

prs.save(OUT)
print('[ok] saved: ' + OUT)
print('     slides: ' + str(len(prs.slides._sldIdLst)))
