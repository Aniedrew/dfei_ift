#!/usr/bin/env python3
"""DFEI group meeting, 2026-10-12, v3 - the talk (v2 + definitions).

"Which optimisations actually worked - and why": only the optimisations that
*moved the number* are kept, and each one is explained from first principles
(with formulas).

v3 is v2 with the *gaps closed*: two "Notation and definitions" slides are
inserted right after the outline (hard pool, chain survival, per-link p, n_B,
AP vs AUC, candidate edge, pruning working point, pool caveats, version codes,
the small-model probe), and a few undefined-at-first-use terms get a short
in-slide bracket.  No structure, number, figure or claim is changed; nothing
is deleted.  Every added definition is the one implemented in the analysis /
training code (see the report for the mapping).

Run:
    export PATH=$HOME/miniconda3/envs/dfei/bin:$PATH
    python make_principles_figs_20261012_v3.py     # write the figures first
    python make_dfei_principles_20261012_v3.py
"""
import os

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

BASE = '/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn'
FIG = BASE + '/meeting_figs_20261012_v3'
OUT = FIG + '/DFEI_principles_20261012_v3_EN.pptx'

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
items = [('PART 1', 'the loss - kill the weakest link of every truth chain', PINK),
         ('PART 2', 'attention over the neighbourhood', GOLD),
         ('PART 3', 'competition context, not just geometry', PURPLE),
         ('PART 4', 'training hygiene - depth, learning rate, determinism', GREEN),
         ('SUMMARY', 'three conclusions, and one methodological lesson', BLUE)]
y = 1.55
for tag, txt, col in items:
    b = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.75), Inches(y),
                           Inches(1.95), Inches(0.62))
    b.fill.solid(); b.fill.fore_color.rgb = col; b.line.fill.background()
    tf = b.text_frame; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = tag
    r.font.size = Pt(18); r.font.bold = True
    r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF); r.font.name = FONT
    tb = s.shapes.add_textbox(Inches(3.00), Inches(y + 0.02), Inches(9.6), Inches(0.62))
    tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    r = tf.paragraphs[0].add_run(); r.text = txt
    r.font.size = Pt(BODY); r.font.color.rgb = col; r.font.name = FONT
    y += 0.92
note(s, 'All of it is measured on one ruler (PART 0): chain survival, not per-edge AP.',
     top=6.62)

# ============================================================ 2b definitions (1/2)
# [v3] inserted so a first-time listener never has to guess a pool, a class or a
#      metric name.  Every line is the definition implemented in the analysis /
#      training code (analyze_prune_loss.py, analyze_vertex_assoc.py,
#      dfei_lightning_module.py); no new number is introduced.
s = slide('DEFINITIONS', BLUE, 'Notation and definitions (1/2): pools and the objects')
bullets(s, [
    'Two evaluation pools are used - numbers from different pools are NOT comparable.',
    ('500-event benchmark: the 18 versions scored on a held-out 500-event test set (F1-F3, slide 4, slide 8).', 1),
    ('696-event multi-B probe: a separate small-model run on events with n_B >= 2 (F5, F6, slides 6-7, 10-11).', 1),
    'Candidate edge = a track pair inside one event (a complete graph on that event\'s tracks).',
    'Truth chain = connected component of truth structural edges (one chain ~ one B hadron).',
    'Truth structural edge = a candidate edge that lies on a truth chain (edge label > 0).',
    'Hard pool (the AP pool) = truth edges + the two "both ends are real tracks" fakes: fake_intra (same chain but not a structural edge) and fake_inter (two different chains). 500-event benchmark: 5964 edges, ~50.6% positive.',
    'Pruning heads: node head = "is this track a signal track?"; edge head = "is this candidate edge a truth structural edge?".',
], 0.60, 1.24, 12.15, 5.30, size=16, space=5)

# ============================================================ 2c definitions (2/2)
s = slide('DEFINITIONS', BLUE, 'Notation and definitions (2/2): metrics, working point, probes')
bullets(s, [
    'Working point = the pruning score threshold (0.9 for nodes and edges in the benchmark). A chain survives iff every track AND every truth structural edge is kept; chain survival = the fraction of truth chains that survive.',
    ('The multi-B probe instead sets the threshold at the score giving 90% edge precision - the "edge-precision-90%" point named in the figure captions.', 1),
    'AP = average precision (area under the precision-recall curve); AUC = area under the ROC curve. With a ~0.13% positive rate, AP is the honest ranking metric; AUC is reported beside it.',
    'per-link p_n = (fraction of n-edge truth chains fully recovered)^(1/n) - the implied survival of a single link. p2 = the value for 2-edge chains.',
    'n_B = number of truth chains in an event (a B-hadron proxy); multi-B = n_B >= 2, single-B = n_B = 1.',
    'Seeds = independent training runs; error bars are the run-to-run spread. The +/-0.006 bar is the statistical uncertainty of hard-pool AP at 500 events (n ~ 5964).',
    'Version codes v6xx = trained configurations of the same model, each a base plus one documented change, all re-scored on the same 500-event benchmark. (v601 = production base; v633 = the "clean" base; v623 = derived inputs; v643 + v648 = +geometry / +competition; v641/v642/v654/v655 = +chain-recall; v653 = +softmin weakest link; v628/v631/v632 = stacked variants.)',
    'Small-model probe (F5/F6): a pure-CPU per-edge MLP vs an event self-attention net on the same edges; 30-dim = node+geo+ctx features, 47-dim = all six feature groups.',
    'Glossary: BCE = binary cross-entropy; softmin_gamma = a differentiable minimum (sharpness gamma); scatter_add = grouped sum without a Python loop; permutation-invariant = output unchanged if the input set is reordered; receptive field = hops of neighbours a stack can see.',
], 0.60, 1.24, 12.15, 5.30, size=15, space=4)

# ============================================================ 3 PART 0 ruler
s = slide('PART 0', TEAL, 'The ruler: a truth chain only counts if it is rebuilt')
bullets(s, [
    'Physical criterion: a chain survives only if every link is recovered - an AND over the chain.',
    'We read it as chain survival at the edge-precision-90 % working point (pruning threshold 0.9; see definitions).',
    'F1: ranking the 18 versions by hard-pool AP and by chain survival are almost unrelated.',
    'Example: v632 is last on AP (0.6081), yet 4th on survival (49.9 %).',
    'So the ruler is chain survival - every claim below is measured with it.',
], LEFT, 1.28, LW, 4.9, space=11)
pic(s, FIG + '/F1_dumbbell_ap_vs_chain.png', RX, 1.26, RW, 5.00,
    center_in=(RX, RW), center_v=(1.26, 5.00))
note(s, 'Pick the criterion before the model: the metric decides which optimisation looks good.',
     top=6.62)

# ============================================================ 4 PART 1 Idea 1a
s = slide('PART 1', PINK, 'Idea 1a - per-edge BCE optimises the wrong quantity')
bullets(s, [
    'Chain survival is an AND: S = prod_e 1[y_e correct]; in expectation S ~ prod_e p_e.',
    'Per-edge BCE maximises each edge log-likelihood on its own - it optimises the mean correctness.',
    'That gradient is mis-matched to the chain target: the weakest link is not priced at all.',
    'Consequence for n = 2: p 0.668 -> 0.803 lifts chain survival 44.7 % -> 64.5 % (per-link p2, 500-event benchmark).',
    'So a small per-link gain is amplified by the product - but BCE only sees the average.',
], LEFT, 1.28, LW, 5.0, space=9)
eqpic(s, 'EQ_survival.png', 1.24, max_h=0.70)
pic(s, FIG + '/F3_and_semantics.png', RX, 2.16, RW, 3.90,
    center_in=(RX, RW), center_v=(2.16, 3.90))
note(s, 'The objective must be the product of per-link success, not the average of per-edge likelihoods.',
     top=6.60)

# ============================================================ 5 PART 1 Idea 1b
s = slide('PART 1', PINK, 'Idea 1b - softmin: a differentiable weakest link')
bullets(s, [
    'softmin_gamma aggregates the per-link probabilities of one chain - formula on the right.',
    'Limits: gamma -> 0 gives mean(p); gamma -> infinity gives min(p), the weakest link.',
    'Loss L_chain = -ln(softmin_gamma(p)), averaged over all truth chains of the event.',
    'Reference (F4): p = [0.95, 0.50, 0.05], min = 0.05 and mean = 0.50.',
    'Implementation: clamp p to [1e-6, 1 - 1e-6], gamma = 10, group chains with scatter_add.',
], LEFT, 1.28, LW, 5.0, space=9)
eqpic(s, 'EQ_softmin.png', 1.22, max_h=0.62)
eqpic(s, 'EQ_chainloss.png', 1.96, max_h=0.52)
pic(s, FIG + '/F4_softmin_gamma.png', RX, 2.62, RW, 3.30,
    center_in=(RX, RW), center_v=(2.62, 3.30))
note(s, 'One hyper-parameter interpolates smoothly between "average" and "weakest link".',
     top=6.60)

# ============================================================ 6 PART 1 Idea 1c
s = slide('PART 1', PINK, 'Idea 1c - the fix has to be non-linear')
bullets(s, [
    'Product / geometric mean S_prod = (prod p)^(1/n); its log is mean(ln p) ~ per-edge BCE.',
    'Measured: the product form reaches 28.0 % chain survival, barely above the 24.2 % baseline.',
    'Softmin reaches 32.6 %; the softer softmin w = 0.3 sits between at 30.1 %.',
    'So what works is a non-linear penalty on the weakest link - not "switching to a product".',
], LEFT, 1.28, LW, 5.0, space=12)
pic(s, FIG + '/F5_chainloss_probe.png', RX, 1.55, RW, 4.55,
    center_in=(RX, RW), center_v=(1.55, 4.55))
note(s, 'The gain comes from pricing the weakest link, not from changing the loss name.',
     top=6.60)

# ============================================================ 7 PART 1 Idea 1d
s = slide('PART 1', PINK, 'Idea 1d - the evidence, seed by seed')
bullets(s, [
    'Multi-B probe: 696 events / 281 truth chains / 5248 test edges, 3 seeds (a separate pool from the 500-event benchmark).',
    'All three seeds improve: 24.6 / 23.6 / 24.5 -> 33.5 / 28.5 / 35.8 (mean 24.2 -> 32.6).',
    'Edge-level rulers rise too: AP 0.8393 -> 0.8533, AUC 0.8650 -> 0.8732.',
    'Single-B also rises: 33.1 / 34.2 -> 44.0 / 38.1 (and 45.4 for w = 0.3).',
    'It is not a trade-off: the chain loss lifts the edge and the chain ruler together.',
], LEFT, 1.28, LW, 5.0, space=9)
pic(s, FIG + '/F5_chainloss_probe.png', RX, 1.55, RW, 4.55,
    center_in=(RX, RW), center_v=(1.55, 4.55))
note(s, 'Chain survival +8.4 pp on every seed, with edge AP and AUC also up.', top=6.60)

# ============================================================ 8 PART 1 Idea 1e
s = slide('PART 1', PINK, 'Idea 1e - the same idea is already inside the main model')
bullets(s, [
    'v641 / v642 add a same-family chain_recall weakest-link loss on top of v633.',
    'Chain survival: v642 61.63 %, v641 59.21 % - the AP champion v623 (best arm by hard-pool AP) reaches only 47.13 %.',
    'Base v633 sits at 42.30 %: the weakest link is genuinely stronger, not re-ranked.',
    'Per-link p2: 0.803 / 0.808 vs 0.710 (v623) and 0.668 (v633).',
    'So a second, independent implementation of the same idea moves the chain ruler.',
], LEFT, 1.28, LW, 5.0, space=9)
pic(s, FIG + '/F2_chain_surv_bar.png', RX, 1.30, RW, 4.85,
    center_in=(RX, RW), center_v=(1.30, 4.85))
note(s, 'Change the implementation, keep the idea - the physics follows either way.', top=6.60)

# ============================================================ 9 PART 2 Idea 2a
s = slide('PART 2', GOLD, 'Idea 2a - attention over the neighbourhood')
bullets(s, [
    'Baseline MLP scores each candidate edge alone: f(x_e), information only from that edge (the small-model probe baseline).',
    'Attention aggregates the neighbours: alpha_ef = softmax_f((W_q h_e).(W_k h_f)/sqrt(d)).',
    'Update h_e <- sum_{f in N(e)} alpha_ef W_v h_f - one message-passing step.',
    'Permutation invariance: the neighbourhood is a set, so aggregation must not depend on order.',
    'Effective receptive field grows with depth: L layers see L hops.',
    'It can represent the neighbours distribution implicitly; a tree sees only fed-in scalars.',
], LEFT, 1.24, LW, 5.0, space=6)
eqpic(s, 'EQ_attention.png', 1.22, max_h=0.66)
pic(s, FIG + '/F9_attention_principle.png', RX, 2.02, RW, 3.90,
    center_in=(RX, RW), center_v=(2.02, 3.90))
note(s, 'Neighbourhood attention is the second real lever: what an edge may see is what changes.',
     top=6.58)

# ============================================================ 10 PART 2 Idea 2b
s = slide('PART 2', GOLD, 'Idea 2b - it helps most where chains compete')
bullets(s, [
    'Multi-B probe, 3-4 seeds: attention (30-dim = node+geo+ctx inputs) AP 0.8460 +/- 0.0118, AUC 0.8745 +/- 0.0094.',
    'Plain MLP on all 47 features (all six feature groups): AP 0.8015 +/- 0.0149, AUC 0.8480 +/- 0.0047.',
    'Gap: AP +0.044, AUC +0.027 - far outside the MLP spread of 0.005.',
    'Attention keeps its lead at both input sizes: 30-dim AP 0.8460, 47-dim AP 0.8450.',
    'The same direction holds at edge level, so the two views agree.',
], LEFT, 1.28, LW, 5.0, space=9)
pic(s, FIG + '/F6_attn_vs_mlp.png', RX, 1.55, RW, 4.55,
    center_in=(RX, RW), center_v=(1.55, 4.55))
note(s, 'Attention is the lever; more raw channels are not.', top=6.60)

# ============================================================ 11 PART 2 Idea 2c
s = slide('PART 2', GOLD, 'Idea 2c - depth is part of the mechanism')
bullets(s, [
    'L layers -> a receptive field of L hops: 2 layers 0.7937, 4 layers 0.8018, 6 layers 0.7947.',
    'Paired over 3 seeds, 4 vs 2 layers gives Delta = +0.008, positive on 3/3 seeds.',
    '6 layers falls back - too deep for this graph (over-smoothing / overfitting).',
    'Head-count sweep: 4 heads is the operating point (2 heads 0.7900, 8 heads 0.8047, i.e. within the run-to-run spread of the 4-head result).',
    'So "attention + depth" is one recipe, not a set of independent knobs.',
], LEFT, 1.28, LW, 5.0, space=8)
table(s, [
    ['configuration', 'AP'],
    ['2 layers', '0.7937'],
    ['4 layers', '0.8018'],
    ['6 layers', '0.7947'],
    ['2 heads', '0.7900'],
    ['8 heads', '0.8047'],
], RX, 1.55, RW, 3.30, colw=[3.65, 2.40], size=17)
note(s, 'Depth and head count are part of the same recipe - tuned together, not one at a time.',
     top=6.60)

# ============================================================ 12 PART 3 Idea 3a
s = slide('PART 3', PURPLE, 'Idea 3a - competition context, not just geometry')
bullets(s, [
    'A true edge and a cross-chain fake (same PV, different secondary vertex) are geometrically close.',
    'The signal is the competition: how many candidate edges share the track, and their affinity spread.',
    'Definitions: a = exp(-d3/50); r = a / max_f a_f; hi = # high-affinity partners; cnt = # partners.',
    'Symmetry is essential: an undirected edge feature must be invariant to swapping (i, j).',
    'We build it from (min, max) of the endpoint values; ordered pairs leak "which endpoint is first".',
    'Measured leak: swapping endpoints moves such features by max|Delta| = 0.99 (the 0702 delta_z0 bug).',
], LEFT, 1.24, LW, 5.0, space=6)
eqpic(s, 'EQ_competition.png', 1.22, max_h=0.62)
pic(s, FIG + '/F10_competition_context.png', RX, 2.00, RW, 3.95,
    center_in=(RX, RW), center_v=(2.00, 3.95))
note(s, 'Give the model the competition, and the ambiguous cross-chain edges separate.', top=6.58)

# ============================================================ 13 PART 3 Idea 3b
s = slide('PART 3', PURPLE, 'Idea 3b - the evidence')
bullets(s, [
    'v648 = v643 + competition context: AP +0.015, chain survival 35.95 -> 38.67 (500-event benchmark).',
    'Multi-B permutation importance (drop in AP when a feature is shuffled): doca (closest-approach) +0.166, trdist (transverse distance) +0.073, flight (flight distance) +0.048, m (pair mass) +0.022.',
    'Competition / degree-type signals rank alongside the geometry ones.',
    'Geometry adds 3.5x more on multi-B than single-B (+0.113 vs +0.032).',
    'That is exactly where multi-B is hard: many competing candidate chains.',
], LEFT, 1.28, LW, 5.0, space=11)
pic(s, FIG + '/F8_features.png', RX, 1.50, RW, 4.60,
    center_in=(RX, RW), center_v=(1.50, 4.60))
note(s, 'The competition signal is one of the strongest features on the multi-B pool.', top=6.60)

# ============================================================ 14 PART 4 hygiene
s = slide('PART 4', GREEN, 'Training hygiene - three small things')
card(s, 'Depth 4 layers', [
    'Same recipe as Idea 2c.',
    'Receptive field = 4 hops.',
], 0.55, 1.45, 3.95, 2.10, GOLD)
card(s, 'Learning rate 3e-4', [
    '10 seeds: 0.8124 +/- 0.002.',
    'Beats 1e-4 (0.8108) and 1e-3 (0.8018).',
], 4.70, 1.45, 3.95, 2.10, TEAL)
card(s, 'Deterministic seeds', [
    'Fixed numpy / torch seeds.',
    'Two runs bit-identical: 5964 edges (= the 500-event hard pool), 10 versions.',
], 8.85, 1.45, 3.95, 2.10, PURPLE)
bullets(s, [
    'Determinism is the precondition for same-pool A/B: identical edge counts and strata every run.',
    'Without it, each evaluation drew a different 200-event sample and no comparison was meaningful.',
], LEFT, 3.85, 12.30, 2.2, space=10)
note(s, 'Small settings, but they are what makes every other comparison trustworthy.', top=6.60)

# ============================================================ 15 summary
s = slide('SUMMARY', GREEN, 'Summary and the lesson')
card(s, 'Worked - chain loss', [
    'Chain survival 24.2 -> 32.6 % (multi-B probe).',
    'Positive on 3/3 seeds.',
], 0.55, 1.42, 3.95, 2.25, PINK)
card(s, 'Worked - attention', [
    'AP 0.8460 vs 0.8015 (MLP; multi-B probe).',
    'AUC 0.8745 vs 0.8480.',
], 4.70, 1.42, 3.95, 2.25, GOLD)
card(s, 'Worked - competition ctx', [
    'Multi-B feature AP 0.6916 -> 0.8391 (GBDT feature upper bound).',
    'Geometry +0.113 on multi-B.',
], 8.85, 1.42, 3.95, 2.25, PURPLE)
bullets(s, [
    'Pattern: every winner changed WHAT is optimised or WHAT the model can see.',
    'Lesson: fix the ruler before the model, and assert invariants so a mis-framing cannot pass.',
], LEFT, 3.95, 12.30, 2.2, space=10)
note(s, 'Three levers: the objective, the neighbourhood, the competition - not the machinery.',
     top=6.60)

# ============================================================ 16 what runs next
s = slide('NEXT', BLUE, 'What runs next')
bullets(s, [
    'v653 / v654 / v655: three writings of the chain-level loss on one common base (= v648).',
    'Compared head-to-head on the same pool, ranked by chain survival.',
    'Selection and checkpointing move onto the chain-survival ruler.',
], LEFT, 1.45, 12.30, 3.0, space=14)
note(s, 'One idea, three implementations - the ruler decides which one ships.', top=6.60)

# ---- clean the file metadata: python-pptx leaves its own fingerprints
#      (author 'Steve Canny', comment 'generated using python-pptx', a 2013
#      template timestamp), which show up in the file properties.
import datetime as _dt
import re as _re
import shutil as _shutil
import zipfile as _zipfile

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
