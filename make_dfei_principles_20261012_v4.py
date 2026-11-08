#!/usr/bin/env python3
"""DFEI group meeting, 2026-10-12, v4 - the talk.

"Which optimisations actually worked - and why": only the optimisations that
*moved the number* are kept, and each one is explained from first principles.

v4 = v3 with every issue from the review fixed, and a new PART 0 "Setup":

  E1  the two chain-survival readings are named apart and never conflated:
        benchmark -> "chain survival at the 0.9 score threshold"     (fixed 0.9)
        probe     -> "chain survival at the probe's 90%-edge-precision threshold"
      and the slides state explicitly that the two thresholds differ.
  E2  the head-count claim now matches the table (8 heads 0.8047 is inside
      the 4-head run-to-run spread, +/-0.0118) - numbers unchanged.
  E3  the body text now uses v633 -> v642 (p2 0.668 -> 0.803), matching F3.
  E4  the MLP spread is now labelled by metric: AUC +/-0.0047 and AP
      +/-0.0149, exactly as archived in F6.
  E5  the feature-ceiling and learning-rate numbers come from the archived
      re-runs under report_figs/ (principles_provenance_features.json and
      principles_provenance_lr.json), read at build time.  The feature re-run
      reproduced the v3 numbers exactly; the learning-rate re-run differs, so
      the deck now quotes the re-run values (3e-4 best, ~0.809 +/- 0.012).

  PART 0 is a SINGLE setup page (the objects, the two pools, the version codes);
  the ruler becomes the opening of PART 1.  Every other term is defined on the
  slide where it is first used (p_n on Idea 1a, softmin on Idea 1b, the probe
  working point on Idea 1c, attention / message passing / permutation
  invariance on Idea 2a, the competition features on Idea 3a, ...).

Run:
    export PATH=$HOME/miniconda3/envs/dfei/bin:$PATH
    python make_principles_figs_20261012_v4.py     # write the figures first
    python make_dfei_principles_20261012_v4.py
"""
import datetime as _dt
import json
import os
import re as _re
import shutil as _shutil
import zipfile as _zipfile

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

BASE = '/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn'
FIG = BASE + '/meeting_figs_20261012_v4'
OUT = FIG + '/DFEI_principles_20261012_v4_EN.pptx'

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
NOTE = 17
TITLE = 30

# ---- E5: read the archived feature-ceiling re-run ------------------------
_FC = None
try:
    with open(BASE + '/report_figs/principles_provenance_features.json') as _f:
        _FC = json.load(_f)
    print('[deck] feature ceiling read from principles_provenance_features.json')
except Exception as _e:                                            # noqa: BLE001
    print('[deck] WARN: no principles_provenance_features.json (%s) - using v3 values' % _e)

if _FC is not None:
    _g = _FC['multi_b']['groups']
    FC_NODE = _g['node']['ap']
    FC_GEO = _g['node+geo']['ap']
    FC_FULL = _g['node+geo+raw+der+vrt+ctx']['ap']
    _imp = {x['name']: x['importance'] for x in _FC['multi_b']['importance']}
    FC_IMP = [(k, _imp.get(k, 0.0)) for k in ('doca', 'trdist', 'flight', 'm')]
else:
    FC_NODE, FC_GEO, FC_FULL = 0.6916, 0.8043, 0.8391
    FC_IMP = [('doca', 0.166), ('trdist', 0.073), ('flight', 0.048), ('m', 0.022)]

# ---- E5: read the archived learning-rate sweep --------------------------
_LR = None
try:
    with open(BASE + '/report_figs/principles_provenance_lr.json') as _f:
        _LR = json.load(_f)
    print('[deck] learning-rate sweep read from principles_provenance_lr.json')
except Exception as _e:                                            # noqa: BLE001
    print('[deck] WARN: no principles_provenance_lr.json (%s) - using v3 values' % _e)

if _LR is not None:
    _lr = _LR['lr']
    LR_BEST = _lr['3e-4']
    LR_TEXT = [
        '%d seeds (test AP): %.4f +/- %.4f.' % (LR_BEST['n_seeds'], LR_BEST['mean'], LR_BEST['std']),
        'Best mean; 1e-4 %.4f, 1e-3 %.4f.' % (_lr['1e-4']['mean'], _lr['1e-3']['mean']),
    ]
    LR_CAVEAT = ('3e-4 has the best mean, but the gap to 1e-3 is inside the seed '
                 'spread - a tie-breaker, not a result. Source: '
                 'report_figs/principles_provenance_lr.json.')
else:
    LR_TEXT = ['10 seeds: 0.8124 +/- 0.002.',
               'Beats 1e-4 (0.8108) and 1e-3 (0.8018).']
    LR_CAVEAT = 'Small settings, but they are what makes every other comparison trustworthy.'

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


def note(s, txt, top=6.45, size=NOTE, left=0.55, width=12.30):
    top = min(top, 6.45)                    # keep clear of the footer at 7.12 in
    tb = s.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(0.64))
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


def eqpic(s, name, top, max_h=0.74, max_w=None):
    """Centre a mathtext formula image in the right-hand column."""
    max_w = max_w or RW
    return pic(s, FIG + '/' + name, RX, top, max_w, max_h, center_in=(RX, RW))


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
    r.font.size = Pt(15); r.font.bold = True
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
            c.margin_top = c.margin_bottom = Inches(0.01)
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
tb = s.shapes.add_textbox(Inches(0.85), Inches(1.95), Inches(11.6), Inches(1.6))
tf = tb.text_frame; tf.word_wrap = True
r = tf.paragraphs[0].add_run(); r.text = 'Which optimisations actually worked - and why'
r.font.size = Pt(40); r.font.bold = True; r.font.color.rgb = BLUE; r.font.name = FONT
tb = s.shapes.add_textbox(Inches(0.85), Inches(3.35), Inches(11.9), Inches(0.9))
tf = tb.text_frame; tf.word_wrap = True
r = tf.paragraphs[0].add_run()
r.text = 'the three ideas that moved the physics, explained from first principles'
r.font.size = Pt(BODY); r.font.color.rgb = DARK; r.font.name = FONT
tb = s.shapes.add_textbox(Inches(0.85), Inches(4.25), Inches(11.9), Inches(0.6))
r = tb.text_frame.paragraphs[0].add_run()
r.text = 'DFEI · chain-level loss · neighbourhood attention · competition context'
r.font.size = Pt(20); r.font.color.rgb = GRAY; r.font.name = FONT
tb = s.shapes.add_textbox(Inches(0.85), Inches(4.95), Inches(11.9), Inches(0.6))
r = tb.text_frame.paragraphs[0].add_run()
r.text = 'Qingxiang Guo · University of Chinese Academy of Sciences · 12 October 2026'
r.font.size = Pt(20); r.font.color.rgb = GRAY; r.font.name = FONT

# ============================================================ 2 outline
s = slide('OUTLINE', BLUE, 'Outline')
items = [('PART 0', 'setup - the objects, the two pools, the version codes', TEAL),
         ('PART 1', 'the loss - kill the weakest link of every truth chain', PINK),
         ('PART 2', 'attention over the neighbourhood', GOLD),
         ('PART 3', 'competition context, not just geometry', PURPLE),
         ('PART 4', 'training hygiene - depth, learning rate, determinism', GREEN),
         ('SUMMARY', 'three conclusions, and one methodological lesson', BLUE)]
y = 1.44
for tag, txt, col in items:
    b = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.75), Inches(y),
                           Inches(1.95), Inches(0.58))
    b.fill.solid(); b.fill.fore_color.rgb = col; b.line.fill.background()
    tf = b.text_frame; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = tag
    r.font.size = Pt(17); r.font.bold = True
    r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF); r.font.name = FONT
    tb = s.shapes.add_textbox(Inches(3.00), Inches(y + 0.02), Inches(9.6), Inches(0.58))
    tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    r = tf.paragraphs[0].add_run(); r.text = txt
    r.font.size = Pt(21); r.font.color.rgb = col; r.font.name = FONT
    y += 0.84
note(s, 'PART 0 is a single page; every later term is defined where it is first '
        'used. The ruler is chain survival at the 0.9 score threshold, not per-edge AP.',
     top=6.45)

# ============================================================ 3 PART 0 setup
# ONE setup page only: the global objects, the two pools and the version codes.
# It is the only page of shared prerequisites; every other term is defined on
# the page where it is first used (hard pool here, p_n on Idea 1a, the probe
# working point on Idea 1c, attention on Idea 2a, ...).
s = slide('PART 0', TEAL, 'Setup - the objects, the two pools, the version codes')
bullets(s, [
    'Candidate edge = any pair of tracks in one event (a complete graph); '
    'only ~0.13% of them are truth structural edges.',
    'Truth structural edge = a candidate edge that lies on a truth chain; '
    'truth chain = one connected component of them (~one B hadron).',
    'Hard pool = the truth structural edges + two fake families whose both ends '
    'are real: fake_intra (same chain, wrong depth) and fake_inter (two chains).',
    'PV = the pp collision point; SV = the secondary vertex where a B decays. '
    'Pruning = two heads at a FIXED threshold 0.9: node ("is this track '
    'signal?") and edge ("is this edge structural?").',
    'Reconstruction then glues the survivors into chains: a chain counts only '
    'if EVERY link survives - it is an AND.',
], LEFT, 1.26, LW, 5.10, size=15, space=8)
bullets(s, [
    'Two pools, NOT comparable: bench working point = the FIXED 0.9 score '
    'threshold (500-event hard pool 5964 edges; 3000-event hard pool 33166 '
    'edges); probe working point = the score threshold that gives 90% edge '
    'precision (696-event multi-B probe, 281 truth chains / 5248 test edges).',
    'v6xx = trained configurations of the SAME model, each a base plus one '
    'documented change, all re-scored on one benchmark: v601 production base; '
    'v623 + derived inputs (the hard-pool AP champion); v633 the "clean-stack" '
    'base; v643 = v633 + corrected geometry; v648 = v643 + competition '
    'context; v642 = v633 + chain_recall; v654 = v648 + chain_recall.',
    'Source: make_prune_variant_configs.py records in its docstring exactly '
    'what each version changed - nothing here is folklore.',
], RX, 1.26, RW, 5.10, size=15, space=11)
note(s, 'The only page of shared prerequisites. Every other term (p_n, softmin, '
        'attention, permutation invariance, competition, ...) is defined where '
        'it is first used.', top=6.62)

# ============================================================ 4 PART 1 ruler
s = slide('PART 1', PINK, 'The ruler: chain survival, not edge AP')
bullets(s, [
    'Physical criterion: a chain survives only if every link is recovered - an '
    'AND. chain survival = the fraction of truth chains whose every track AND '
    'every truth structural edge is above the working-point threshold.',
    'Two readings, always named apart: at the 0.9 score threshold (benchmark, '
    'FIXED) and at the probe\'s 90%-edge-precision threshold (probe, data-driven).',
    'AP = average precision (PR-curve area); AUC = area under the ROC curve; AP '
    'is the honest ranking metric at ~0.13% positives.',
    'F1: ranking the 18 versions by hard-pool AP and by chain survival are '
    'almost unrelated - e.g. v632 is last on AP (0.6081) yet 4th on survival (49.9%).',
    'So the ruler is chain survival at the 0.9 score threshold - every claim '
    'below is measured with it.',
], LEFT, 1.26, LW, 4.95, size=15, space=8)
pic(s, FIG + '/F1_dumbbell_ap_vs_chain.png', RX, 1.26, RW, 5.00,
    center_in=(RX, RW), center_v=(1.26, 5.00))
note(s, 'Pick the criterion before the model: the metric decides which optimisation looks good.',
     top=6.62)

# ============================================================ 8 PART 1 Idea 1a
s = slide('PART 1', PINK, 'Idea 1a - per-edge BCE optimises the wrong quantity')
bullets(s, [
    'BCE = binary cross-entropy; the per-edge loss trains each candidate edge as an independent yes / no.',
    'Chain survival is an AND: S = prod_e 1[y_e correct]; in expectation S ~ prod_e p_e.',
    'Per-edge BCE maximises each edge log-likelihood alone - it optimises the mean correctness.',
    'Bridge: training sees the full graph, but inference prunes at the 0.9 score threshold first - so the AND is decided after the loss was fit.',
    'per-link p_n = (n-edge chain survival)^(1/n). Measured (500-event bench, 0.9 score threshold): v633 -> v642, p2 0.668 -> 0.803 lifts 2-link survival 44.7% -> 64.5%.',
    'So a small per-link gain is amplified by the product - but BCE only sees the average.',
], LEFT, 1.24, LW, 5.05, size=16, space=6)
eqpic(s, 'EQ_survival.png', 1.22, max_h=0.66)
pic(s, FIG + '/F3_and_semantics.png', RX, 2.02, RW, 4.10,
    center_in=(RX, RW), center_v=(2.02, 4.10))
note(s, 'The objective must be the product of per-link success, not the average of per-edge likelihoods.',
     top=6.40)

# ============================================================ 9 PART 1 Idea 1b
s = slide('PART 1', PINK, 'Idea 1b - softmin: a differentiable weakest link')
bullets(s, [
    'softmin_gamma aggregates the per-link probabilities of one chain - a smooth minimum (formula on the right).',
    'Limits: gamma -> 0 gives mean(p); gamma -> infinity gives min(p) = the weakest link.',
    'Loss L_chain = -ln(softmin_gamma(p)), averaged over all truth chains of the event.',
    'scatter_add = a grouped sum in one tensor op (no Python loop); group key = event id x chain id.',
    'Reference (F4): p = [0.95, 0.50, 0.05], weakest link 0.05 vs mean 0.50.',
    'Implementation: clamp p to [1e-6, 1-1e-6], gamma = 10.',
], LEFT, 1.24, LW, 5.05, size=16, space=6)
eqpic(s, 'EQ_softmin.png', 1.20, max_h=0.60)
eqpic(s, 'EQ_chainloss.png', 1.92, max_h=0.50)
pic(s, FIG + '/F4_softmin_gamma.png', RX, 2.56, RW, 3.40,
    center_in=(RX, RW), center_v=(2.56, 3.40))
note(s, 'One hyper-parameter interpolates smoothly between "average" and "weakest link".',
     top=6.40)

# ============================================================ 10 PART 1 Idea 1c
s = slide('PART 1', PINK, 'Idea 1c - the fix has to be non-linear (probe)')
bullets(s, [
    'All numbers on this slide and the next come from the 696-event multi-B probe, NOT the 500-event benchmark; the probe working point = the score threshold that gives 90% edge precision.',
    'Product / geometric mean S_prod = (prod p)^(1/n); its log is mean(ln p) ~ per-edge BCE.',
    'Measured at the probe\'s 90%-edge-precision threshold: the product form reaches 28.0% chain survival there, barely above the 24.2% baseline.',
    'Softmin reaches 32.6% at the same threshold; the softer softmin w = 0.3 sits between at 30.1%.',
    'So what works is a non-linear penalty on the weakest link - not "switching to a product".',
], LEFT, 1.28, LW, 5.00, size=16, space=10)
pic(s, FIG + '/F5_chainloss_probe.png', RX, 1.45, RW, 4.75,
    center_in=(RX, RW), center_v=(1.45, 4.75))
note(s, 'The gain comes from pricing the weakest link, not from changing the loss name.', top=6.40)

# ============================================================ 11 PART 1 Idea 1d
s = slide('PART 1', PINK, 'Idea 1d - the evidence, seed by seed (probe)')
bullets(s, [
    'Multi-B probe: 696 events / 281 truth chains / 5248 test edges, 3 seeds (a separate pool from the benchmark).',
    'All three seeds improve: 24.6 / 23.6 / 24.5 -> 33.5 / 28.5 / 35.8 (mean 24.2 -> 32.6).',
    'Edge-level rulers rise too: AP 0.8393 -> 0.8533, AUC 0.8650 -> 0.8732 (the probe\'s own edge pool).',
    'Single-B also rises: 33.1 / 34.2 -> 44.0 / 38.1 (and 45.4 for w = 0.3).',
    'It is not a trade-off: the chain loss lifts the chain and the edge ruler together.',
], LEFT, 1.28, LW, 5.00, size=16, space=10)
pic(s, FIG + '/F5_chainloss_probe.png', RX, 1.45, RW, 4.75,
    center_in=(RX, RW), center_v=(1.45, 4.75))
note(s, 'Chain survival +8.4 pp on every seed, with edge AP and AUC also up.', top=6.40)

# ============================================================ 12 PART 1 Idea 1e
s = slide('PART 1', PINK, 'Idea 1e - the same idea is already inside the main model')
bullets(s, [
    'v641 / v642 add a chain_recall weakest-link loss on top of v633 (500-event benchmark, threshold 0.9).',
    'Chain survival: v642 61.6%, v641 59.2% - the AP champion v623 (best hard-pool AP) reaches only 47.1%.',
    'Base v633 = the feature-stack base we had been building on: 42.3% - a genuinely stronger weakest link.',
    'Not a re-ranking: survival uses the fixed 0.9 score threshold, so higher survival = more links above the same bar.',
    'Per-link p2: 0.803 / 0.808 vs 0.710 (v623) and 0.668 (v633).',
], LEFT, 1.28, LW, 5.00, size=16, space=10)
pic(s, FIG + '/F2_chain_surv_bar.png', RX, 1.30, RW, 4.85,
    center_in=(RX, RW), center_v=(1.30, 4.85))
note(s, 'Change the implementation, keep the idea - the physics follows either way.', top=6.55)

# ============================================================ 13 PART 1 Idea 1f
s = slide('PART 1', PINK, 'Idea 1f - main model, 3000 events: the chain loss reproduces at scale')
bullets(s, [
    '3000-event benchmark, hard pool 33166 edges; chain survival read at the 0.9 score threshold (FIXED) - far beyond the noise.',
    'v648 -> v654 = v648 + chain_recall (a min-pooling weakest-link loss): survival 35.8% -> 52.3% (+16.5 pp, +46% relative).',
    'The long chains gain the most: 3-edge chains 35.0% -> 52.8% (2-edge 36.9% -> 53.0%).',
    'Cost: hard-pool AP 0.6684 -> 0.6591 (only -0.009); per-link p2 0.608 -> 0.728.',
    'chain_recall = the chain-level loss that punishes the weakest link of each truth chain (the softmin form was defined in Idea 1b).',
    'Reproduces v641 / v642 on v633 (survival 59.2% / 61.6%) on a far larger pool - the mechanism, not the aggregator, is what matters.',
], LEFT, 1.24, LW, 5.05, size=16, space=8)
pic(s, FIG + '/F11_chainloss_main.png', RX, 1.35, RW, 4.85,
    center_in=(RX, RW), center_v=(1.35, 4.85))
note(s, 'The probe (24.2 -> 32.6%) stays the controlled small experiment; this 3000-event run is the main evidence. '
        'Source: report_figs/prune_loss_chains_v648_mb.csv / prune_loss_chains_v654_mb.csv.',
     top=6.48, size=17)

# ============================================================ 14 PART 2 Idea 2a
s = slide('PART 2', GOLD, 'Idea 2a - attention over the neighbourhood')
bullets(s, [
    'Baseline MLP scores each candidate edge alone: f(x_e), information only from that edge (the probe baseline).',
    'Attention aggregates the neighbours: alpha_ef = softmax_f((W_q h_e).(W_k h_f)/sqrt(d)).',
    'Update h_e <- sum_{f in N(e)} alpha_ef W_v h_f - one message-passing step.',
    'permutation-invariant = the output does not change if the neighbour set is reordered (the set is unordered).',
    'receptive field / hops: one layer sees one ring of neighbours; L layers see L hops.',
    'Bridge: attention can represent functions of the neighbour distribution; a tree sees only the scalars it is fed.',
], LEFT, 1.24, LW, 5.05, size=16, space=6)
eqpic(s, 'EQ_attention.png', 1.22, max_h=0.62)
pic(s, FIG + '/F9_attention_principle.png', RX, 1.96, RW, 4.00,
    center_in=(RX, RW), center_v=(1.96, 4.00))
note(s, 'Neighbourhood attention is the second real lever: what an edge may see is what changes.',
     top=6.42)

# ============================================================ 15 PART 2 Idea 2b
s = slide('PART 2', GOLD, 'Idea 2b - it helps most where chains compete')
bullets(s, [
    'Multi-B probe, 3-4 seeds. Attention (30-dim = node+geo+ctx) AP 0.8460 +/- 0.0118, AUC 0.8745 +/- 0.0094.',
    'Plain MLP on all 47 features (all six groups): AP 0.8015 +/- 0.0149, AUC 0.8480 +/- 0.0047.',
    'Gap: AP +0.044, AUC +0.027. The MLP\'s AUC run-to-run spread is only +/-0.0047, so the AUC gap is ~6x that spread (its AP spread is +/-0.0149, so the AP gap is ~3x).',
    'Bridge: this is why more raw channels do not help - the gain is from seeing the neighbourhood, not feature count.',
    'Attention keeps its lead at both input sizes: 30-dim AP 0.8460, 47-dim AP 0.8450.',
], LEFT, 1.28, LW, 5.00, size=17, space=10)
pic(s, FIG + '/F6_attn_vs_mlp.png', RX, 1.50, RW, 4.70,
    center_in=(RX, RW), center_v=(1.50, 4.70))
note(s, 'Attention is the lever; more raw channels are not.', top=6.55)

# ============================================================ 16 PART 2 Idea 2c
s = slide('PART 2', GOLD, 'Idea 2c - depth is part of the mechanism')
bullets(s, [
    'L layers -> a receptive field of L hops: 2 layers 0.7937, 4 layers 0.8018, 6 layers 0.7947 (probe, 3 seeds).',
    'Paired over 3 seeds, 4 vs 2 layers gives Delta = +0.008, positive on 3/3 seeds.',
    '6 layers falls back - too deep for this graph (over-smoothing / overfitting).',
    'Head count: 2 heads 0.7900, 4 heads 0.8018, 8 heads 0.8047.',
    '4 heads is our operating point: 8 heads scores 0.8047, inside the run-to-run spread of the 4-head result (+/-0.0118 over seeds), so we keep the smaller head count.',
    'So "attention + depth" is one recipe, not a set of independent knobs.',
], LEFT, 1.28, LW, 5.05, size=16, space=7)
table(s, [
    ['configuration', 'probe AP'],
    ['2 layers', '0.7937'],
    ['4 layers', '0.8018'],
    ['6 layers', '0.7947'],
    ['2 heads', '0.7900'],
    ['4 heads (ref.)', '0.8018'],
    ['8 heads', '0.8047'],
], RX, 1.55, RW, 3.30, colw=[3.65, 2.40], size=16)
note(s, 'Depth and head count are part of the same recipe - tuned together, not one at a time.',
     top=6.55)

# ============================================================ 17 PART 3 Idea 3a
s = slide('PART 3', PURPLE, 'Idea 3a - competition context, not just geometry')
bullets(s, [
    'A true edge and a cross-chain fake (same PV, different secondary vertex) are geometrically close.',
    'PV = pp collision point; secondary vertex (SV) = where the B decays; same PV + different SV = the hard fake.',
    'The signal is the competition: how many candidate edges share the track, and how spread their affinities are.',
    'Definitions: a = exp(-d3/50); r = a / max_f a_f; hi = # high-affinity partners; cnt = # partners.',
    'Symmetry is essential: an undirected edge feature must not change if (i, j) are swapped (we use min / max).',
    'Ordered pairs leak "which end is first" (max|Delta| = 0.99) - the 0702 bug: a 2026-07 dataset whose features '
    'silently depended on the storage order of the two ends.',
], LEFT, 1.24, LW, 5.05, size=15, space=6)
eqpic(s, 'EQ_competition.png', 1.24, max_h=0.58)
pic(s, FIG + '/F10_competition_context.png', RX, 2.02, RW, 3.95,
    center_in=(RX, RW), center_v=(2.02, 3.95))
note(s, 'Give the model the competition, and the ambiguous cross-chain edges separate.', top=6.42)

# ============================================================ 18 PART 3 Idea 3b
_s = lambda x: '%.4f' % x
s = slide('PART 3', PURPLE, 'Idea 3b - the evidence (GBDT feature-upper-bound probe)')
bullets(s, [
    'v648 = v643 + competition context: edge AP +0.015, chain survival 35.95 -> 38.67 (500-event, threshold 0.9).',
    'F8 is a GBDT feature-upper-bound probe (trees on raw features, NOT a model): AP ' + _s(FC_NODE) +
    ' (node) -> ' + _s(FC_FULL) + ' (all groups, multi-B).',
    'Multi-B permutation importance (drop in AP when a feature is shuffled): ' +
    ', '.join('%s +%.3f' % (n, v) for n, v in FC_IMP) + '.',
    'Abbreviations: doca = distance of closest approach; trdist = transverse distance; flight = flight distance; '
    'm = pair invariant mass.',
    'Competition / degree signals rank alongside the geometry ones; geometry adds 3.5x more on multi-B than single-B '
    '(+0.113 vs +0.032).',
    'That is exactly where multi-B is hard: many competing candidate chains.',
], LEFT, 1.24, LW, 5.05, size=14, space=6)
pic(s, FIG + '/F8_features.png', RX, 1.45, RW, 4.65,
    center_in=(RX, RW), center_v=(1.45, 4.65))
note(s, 'The competition signal is one of the strongest features on the multi-B pool. '
        'Source: report_figs/principles_provenance_features.json.', top=6.52)

# ============================================================ 19 PART 4 hygiene
s = slide('PART 4', GREEN, 'Training hygiene - three small things')
card(s, 'Depth 4 layers', [
    'Same recipe as Idea 2c.',
    'Receptive field = 4 hops.',
], 0.55, 1.45, 3.95, 2.10, GOLD)
card(s, 'Learning rate 3e-4', LR_TEXT,
     4.70, 1.45, 3.95, 2.10, TEAL)
card(s, 'Deterministic seeds', [
    'Fixed numpy / torch seeds.',
    'Two runs bit-identical: 5964 edges, 10 versions.',
], 8.85, 1.45, 3.95, 2.10, PURPLE)
bullets(s, [
    'Determinism is the precondition for same-pool A/B: identical edge counts and strata every run.',
    'Before the fix each evaluation drew a different 200-event sample (the early-stopping sample) - comparisons were noise.',
], LEFT, 3.85, 12.30, 2.2, size=20, space=10)
note(s, LR_CAVEAT, top=6.45)

# ============================================================ 20 summary
s = slide('SUMMARY', GREEN, 'Summary and the lesson')
card(s, 'Worked - chain loss', [
    '3000-event bench: survival 35.8 -> 52.3%.',
    'Cost only -0.009 AP.',
], 0.55, 1.42, 3.95, 2.25, PINK)
card(s, 'Worked - attention', [
    'Multi-B probe: AP 0.8460 vs MLP 0.8015.',
    'AUC 0.8745 vs 0.8480.',
], 4.70, 1.42, 3.95, 2.25, GOLD)
card(s, 'Worked - competition ctx', [
    'GBDT probe (multi-B): AP ' + _s(FC_NODE) + ' -> ' + _s(FC_FULL) + '.',
    'Geometry +0.113 on multi-B.',
], 8.85, 1.42, 3.95, 2.25, PURPLE)
bullets(s, [
    'Pattern: every winner changed WHAT is optimised or WHAT the model can see.',
    'Lesson: fix the ruler before the model, and assert invariants so a mis-framing cannot pass.',
], LEFT, 3.95, 12.30, 2.2, size=20, space=10)
note(s, 'Each card names its own pool: the 3000-event benchmark, the multi-B probe, the GBDT probe.', top=6.55)

# ============================================================ 21 what runs next
s = slide('NEXT', BLUE, 'What runs next')
bullets(s, [
    'v655 (chain_recall w = 15) and a v654 weight sweep are running - to see if the weight is the limiting factor.',
    'The three writings of the chain-level loss are compared head-to-head on one base (v648), ranked by chain survival.',
    'Selection and checkpointing move onto the chain-survival ruler (the fixed-threshold one).',
], LEFT, 1.45, 12.30, 3.0, size=22, space=14)
note(s, 'One idea, three implementations - the ruler decides which one ships.', top=6.55)

# ---- clean the file metadata: python-pptx leaves its own fingerprints
#      (author 'Steve Canny', comment 'generated using python-pptx', a 2013
#      template timestamp), which show up in the file properties.
_cp = prs.core_properties
_cp.author = 'Qingxiang Guo'
_cp.last_modified_by = 'Qingxiang Guo'
_cp.title = 'Which optimisations actually worked - and why'
_cp.subject = 'DFEI group meeting, 2026-10-12'
_cp.comments = ''
_cp.category = ''
_cp.keywords = 'DFEI, LHCb, GNN, chain-level loss, attention'
_cp.revision = 1
_NOW = _dt.datetime(2026, 10, 12, 11, 0, 0)
_cp.created = _NOW
_cp.modified = _NOW

prs.save(OUT)
print('[ok] saved: ' + OUT)

# docProps/app.xml is not reachable through python-pptx -> rewrite it in the package
_tmp = OUT + '.tmp'
with _zipfile.ZipFile(OUT) as _zin, _zipfile.ZipFile(_tmp, 'w', _zipfile.ZIP_DEFLATED) as _zout:
    for _it in _zin.infolist():
        _data = _zin.read(_it.filename)
        if _it.filename == 'docProps/app.xml':
            _x = _data.decode('utf-8')
            _x = _x.replace('Microsoft Macintosh PowerPoint', 'Microsoft Office PowerPoint')
            _x = _x.replace('On-screen Show (4:3)', 'On-screen Show (16:9)')
            _x = _re.sub(r'<Slides>\d+</Slides>',
                         '<Slides>%d</Slides>' % len(prs.slides._sldIdLst), _x)
            _data = _x.encode('utf-8')
        _zout.writestr(_it, _data)
_shutil.move(_tmp, OUT)
print('[ok] metadata cleaned')
print('     slides: ' + str(len(prs.slides._sldIdLst)))
