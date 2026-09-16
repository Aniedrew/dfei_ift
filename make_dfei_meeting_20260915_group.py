#!/usr/bin/env python3
"""DFEI group meeting, 2026-09-15 — the SHORT version, for the group.

Same talk as make_dfei_meeting_20260915.py, but:

  * Part 0 (five slides of pipeline walkthrough) is gone.  All of it is compressed
    into ONE page: the whole pipeline in a single picture, with a line underneath
    for what we changed in each block.
  * Less text, bigger figures afterwards: three or four short bullets per slide,
    and the figures get the space that the prose used to take.
  * Plain wording for the noise: "±0.43 % due to random seed", not
    "the run-to-run spread of the protocol".
  * The numbers for what runs now are up to date (the pruning rebalance is the
    first line that beats v47).

The long deck is untouched: DFEI_meeting_20260915_EN.pptx.
"""
import os
import shutil

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

BASE = '/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn'
FIG = BASE + '/meeting_figs_20260915'
FS = FIG + '/legacy/slide_size'          # the 0902 figures, re-rendered at slide size
OUT = FIG + '/DFEI_meeting_20260915_group_EN.pptx'

BLUE = RGBColor(0x1F, 0x4E, 0x79)
DARK = RGBColor(0x2B, 0x2B, 0x2B)
GRAY = RGBColor(0x77, 0x77, 0x77)
PURPLE = RGBColor(0x8E, 0x44, 0xAD)
TEAL = RGBColor(0x2C, 0x7A, 0x7B)
PINK = RGBColor(0xC2, 0x18, 0x5B)
GOLD = RGBColor(0xB8, 0x86, 0x0B)
GREEN = RGBColor(0x2E, 0x8B, 0x57)
FONT = 'Calibri'
BODY = 24
NOTE = 21
TITLE = 30

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]

LEFT, LW = 0.55, 6.05          # bullet column
RX, RW = 6.85, 6.05            # figure column
BAND_T, BAND_B = 1.26, 6.52    # figure band
N = [0]


def footer(s):
    tb = s.shapes.add_textbox(Inches(0.55), Inches(7.12), Inches(12.30), Inches(0.30))
    r = tb.text_frame.paragraphs[0].add_run()
    r.text = 'Qingxiang Guo · UCAS · DFEI group meeting · 2026-09-15          ' + str(N[0])
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


def fig1(s, f):
    """One figure: the whole right band."""
    return pic(s, FIG + '/' + f, RX, BAND_T, RW, BAND_B - BAND_T,
               center_in=(RX, RW), center_v=(BAND_T, BAND_B - BAND_T))


def fig2(s, f1, f2):
    """Two figures stacked: each gets half the right band (~2.55 in tall, was 1.95)."""
    h = (BAND_B - BAND_T - 0.16) / 2.0
    pic(s, FIG + '/' + f1, RX, BAND_T, RW, h, center_in=(RX, RW))
    pic(s, FIG + '/' + f2, RX, BAND_T + h + 0.16, RW, h, center_in=(RX, RW))


def stripR(s, name, top=1.22):
    return pic(s, FIG + '/' + name, RX, top, RW, 1.30, center_in=(RX, RW))


def stripL(s, name, top=5.02):
    return pic(s, FIG + '/' + name, LEFT, top, LW, 1.30, center_in=(LEFT, LW))


# ============================================================ 1 title
s = slide(None, None)
tb = s.shapes.add_textbox(Inches(0.85), Inches(2.45), Inches(11.6), Inches(1.5))
tf = tb.text_frame; tf.word_wrap = True
r = tf.paragraphs[0].add_run(); r.text = 'Optimizing DFEI on CERN Monte Carlo'
r.font.size = Pt(44); r.font.bold = True; r.font.color.rgb = BLUE; r.font.name = FONT
tb = s.shapes.add_textbox(Inches(0.85), Inches(3.95), Inches(11.6), Inches(0.7))
r = tb.text_frame.paragraphs[0].add_run()
r.text = 'The whole pipeline in one page, the v31 → v47 line, and where it broke down'
r.font.size = Pt(BODY); r.font.color.rgb = DARK; r.font.name = FONT
tb = s.shapes.add_textbox(Inches(0.85), Inches(6.05), Inches(11.6), Inches(0.6))
r = tb.text_frame.paragraphs[0].add_run()
r.text = 'Qingxiang Guo · University of Chinese Academy of Sciences · 15 September 2026'
r.font.size = Pt(22); r.font.color.rgb = GRAY; r.font.name = FONT

# ============================================================ 2 data + overall effect
s = slide('DATA', BLUE, 'Data and the overall effect')
bullets(s, [
    'CERN official MC (DFEI_IFT_20260702), threshold 0.9, twenty test files.',
    'v31 → v47: PerfectReco 12.30 → 23.15 %, AllParticles 22.49 → 39.56 %.',
    '±0.43 % due to random seed — anything smaller than that is not a result.',
], LEFT, 1.40, LW, 3.40, space=14)
fig1(s, 'main_line.png')

# ============================================================ 3 outline
s = slide('OUTLINE', BLUE, 'Outline')
items = [('PIPELINE', 'the whole model in one page', TEAL),
         ('PART 1', 'differentiable pruning with annealing (v36)', BLUE),
         ('PART 2', 'rewarding the chain — hinge and chain-CE (v37, v38)', PINK),
         ('PART 3', 'supervising the representation with physics (v46 → v47)', GOLD),
         ('PART 4', 'what came after v47 — and the diagnosis', PURPLE),
         ('SUMMARY', 'what we know, and what runs now', GREEN)]
y = 1.55
for tag, txt, col in items:
    b = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.75), Inches(y), Inches(1.95), Inches(0.60))
    b.fill.solid(); b.fill.fore_color.rgb = col; b.line.fill.background()
    tf = b.text_frame; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = tag
    r.font.size = Pt(18); r.font.bold = True
    r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF); r.font.name = FONT
    tb = s.shapes.add_textbox(Inches(3.00), Inches(y + 0.04), Inches(9.6), Inches(0.60))
    tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    r = tf.paragraphs[0].add_run(); r.text = txt
    r.font.size = Pt(BODY); r.font.color.rgb = col; r.font.name = FONT
    y += 0.86
note(s, 'Each change is labelled with the pipeline block it acts on.', top=6.55)

# ============================================================ 4 the pipeline, one page
s = slide('PIPELINE', TEAL, 'The model in one page — and what we changed in it')
pic(s, FIG + '/p0_stages.png', 0.40, 1.22, 12.55, 2.40, center_in=(0.40, 12.55))

tb = s.shapes.add_textbox(Inches(0.55), Inches(3.66), Inches(12.30), Inches(0.40))
r = tb.text_frame.paragraphs[0].add_run()
r.text = 'What we changed, block by block:'
r.font.size = Pt(21); r.font.bold = True; r.font.color.rgb = BLUE; r.font.name = FONT

for col_left, subs, items in (
        (LEFT, 'stage 1 · pruning — decide what survives', [
            'GNN blocks — four label-free targets on the shared representation: '
            'source, mass, depth + RC, momentum.',
            'point / edge prune heads — annealing, plus the loss rebalanced '
            '(point 1 → 5, edge 33 → 3).']),
        (RX, 'stage 2 · reconstruction — build the chain', [
            'LCAG head — class-2 weight, hinge (v37), chain cross-entropy '
            'on truth-chain edges only (v38).',
            'chain assembly — untouched; the physics reaches it only through '
            'the representation.'])):
    tb = s.shapes.add_textbox(Inches(col_left), Inches(4.08), Inches(RW), Inches(0.36))
    r = tb.text_frame.paragraphs[0].add_run(); r.text = subs
    r.font.size = Pt(18); r.font.bold = True
    r.font.color.rgb = TEAL if col_left == RX else PURPLE
    r.font.name = FONT
    bullets(s, items, col_left, 4.44, RW, 1.70, size=19, space=7)
note(s, 'Best so far: the pruning rebalance — 42.01 % AllParticles / 24.34 % PerfectReco, '
        'above the v47 line.', top=6.45, size=20)

# ============================================================ PART 1
s = slide('PART 1', BLUE, 'Differentiable pruning with annealing')
bullets(s, [
    'Problem: training sees the full graph, inference prunes first — two different graphs.',
    'Fix: a soft mask, w_eff = w · σ((w − cut)/τ), with τ annealed 1.0 → 0.1.',
    'Training only: inference still cuts at w ≥ 0.9.',
    'v31 → v36: 12.30 → 18.77 (PerfectReco), 22.49 → 35.17 (AllParticles).',
], LEFT, 1.40, LW, 3.40, space=12)
stripR(s, 'w_v36.png', 1.22)
pic(s, FIG + '/p1_anneal.png', RX, 2.70, RW, 3.85, center_in=(RX, RW), center_v=(2.70, 3.85))

# ============================================================ PART 2
s = slide('PART 2', PINK, 'Why chains die (1/2) — one wrong edge is enough')
bullets(s, [
    'At inference, an edge called background is pruned.',
    'One missing edge and the chain can no longer be assembled — it is gone.',
    'So what matters is the accuracy on the edges that carry a chain.',
], LEFT, 1.40, LW, 3.40, space=14)
fig2(s, 'chain_before.png', 'chain_after.png')
stripL(s, 'w_edge.png')

s = slide('PART 2', PINK, 'Why chains die (2/2) — the class imbalance')
bullets(s, [
    'Background is 99.9 % of all edges; each structural class is about 0.04 %.',
    'The LCAG loss is a global cross-entropy, so almost all of its gradient says “background”.',
    'Per-class accuracy collapses on the rare classes — worst on class 2, the sisters.',
], LEFT, 1.40, LW, 3.40, space=14)
fig2(s, 'legacy/slide_size/chain_lca_imbalance_dist.png',
     'legacy/slide_size/chain_lca_imbalance_acc.png')
stripL(s, 'w_class.png')

s = slide('PART 2', PINK, 'Reward 1 (v37) — the hinge')
bullets(s, [
    'Confidence = max(p0…p3) from the LCAG head.',
    'Penalty = max(0, 0.3 − confidence), paid only on truth-chain edges.',
    'v36 → v37: 18.77 → 19.50 — and class 2 got worse (47.7 → 44.2), we over-corrected.',
], LEFT, 1.40, LW, 3.40, space=14)
fig2(s, 'd_hinge.png', 'legacy/slide_size/chain_lca_hinge_curve.png')
stripL(s, 'w_hinge.png')

s = slide('PART 2', PINK, 'Reward 2 (v38) — chain cross-entropy')
bullets(s, [
    'CE = −log p_true — it pays for being the right class, not merely for being confident.',
    'Paid only on truth-chain edges (classes 1/2/3), so its gradient is not diluted.',
    'v37 → v38: 19.50 → 21.10; class 1 67.8 → 76.8, class 2 41.3 → 47.9.',
], LEFT, 1.40, LW, 3.40, space=14)
stripR(s, 'w_ce.png', 1.22)
pic(s, FIG + '/d_cewhere.png', RX, 2.70, RW, 3.85, center_in=(RX, RW), center_v=(2.70, 3.85))

s = slide('PART 2', PINK, 'The strategy works')
bullets(s, [
    'v31 → v38: PerfectReco 12.30 → 21.10, AllParticles 22.49 → 37.59.',
    'The classes that moved are the ones that decide whether a chain survives.',
    'Still a training-only change.',
], RX, 1.40, RW, 3.30, space=14)
pic(s, FIG + '/p1_v31_v38.png', LEFT, BAND_T, 6.05, BAND_B - BAND_T,
    center_in=(LEFT, 6.05), center_v=(BAND_T, BAND_B - BAND_T))
stripR(s, 'w_evid.png', 5.10)

# ============================================================ PART 3
s = slide('PART 3', GOLD, 'First, a probe — the physics is not there')
bullets(s, [
    'Freeze the backbone, fit one linear layer, try to read a physical quantity out.',
    'Before the mass head: ππ mass R² = 0.003, momentum R² ≈ 0.',
    'After: edge mass R² = 0.930 — physics has to be supervised in.',
], LEFT, 1.40, LW, 3.40, space=14)
fig2(s, 'd_probe.png', 'legacy/slide_size/probe_r2.png')
stripL(s, 'w_probe.png')

s = slide('PART 3', GOLD, 'The head zoo — nine targets on one backbone')
bullets(s, [
    'Five are the original DFEI machinery: LCAG, node prune, edge prune, PV association, chain scorer.',
    'Four are ours, all label-free: source (v36), mass (v46/47), struct (v48/52), momentum (v48/51).',
], LEFT, 1.30, 12.30, 1.60, space=10)
pic(s, FIG + '/m_zoo.png', 0.40, 3.05, 12.55, 2.55, center_in=(0.40, 12.55), center_v=(3.05, 2.55))
note(s, 'Adding a head never changes the backbone — it only adds a supervised target.',
     top=5.85, left=0.55, width=5.90)
stripR(s, 'w_zoo.png', 5.72)

s = slide('PART 3', GOLD, 'The mass head — the one that pays')
bullets(s, [
    'Target: the invariant mass of the track pair on every tt edge, from MC truth.',
    'v46 regressed the raw mass: 21.28 — inside the noise. v47 adds log10 and masking.',
    'v38 → v47: 21.10 → 23.15 (PerfectReco), 37.59 → 39.56 (AllParticles).',
    'The same recipe on source, struct and momentum is a wash.',
], LEFT, 1.40, LW, 3.40, space=12)
fig2(s, 's_mass.png', 'd_mass.png')
stripL(s, 'w_mass.png')

s = slide('PART 3', GOLD, 'The struct head — tree position without labels')
bullets(s, [
    'Depth: how far a node sits from the chain root, in BFS layers.',
    'Rumor Centrality: which node is the source of the chain.',
    'Both come from the truth tree — no manual labels.',
    'Best single addition on the weak base; a wash on the main line.',
], LEFT, 1.40, LW, 3.40, space=10)
fig2(s, 'depth_calc.png', 'rc_calc.png')
stripL(s, 'w_struct.png')

# ============================================================ PART 4
s = slide('PART 4', PURPLE, 'What came after v47')
bullets(s, [
    'Five directions in parallel: capacity, attention, restructuring the event, '
    'retraining the recipe, another dataset.',
    'The verdict is the same everywhere: the network is not the bottleneck.',
], LEFT, 1.30, 12.30, 1.90, space=10)
pic(s, FIG + '/m_directions.png', 0.40, 3.20, 12.55, 2.45, center_in=(0.40, 12.55), center_v=(3.20, 2.45))
note(s, 'Adding capacity, attention or training time moves nothing — and combining the few things '
        'that do work moves nothing either.', top=5.85)

s = slide('PART 4', PURPLE, 'The diagnosis — and what runs now')
bullets(s, [
    'Oracle test: freeze v47, rewrite the pruning decisions with MC truth → 99.75 % '
    'AllParticles against 39.56 %.',
    'So the score is decided at the pruning step, not by the network.',
    'Two causes: the point head carried 1/33 of the loss weight, and the step size was 10× too small.',
    'What runs now: rebalanced pruning losses at lr 3e-4.',
], LEFT, 1.40, LW, 3.40, space=11)
fig1(s, 'oracle_ladder.png')

# ============================================================ summary
s = slide('SUMMARY', GREEN, 'Summary and next steps')
bullets(s, ['What we now know',
            ('v31 → v47 is real: PerfectReco 12.30 → 23.15 %, AllParticles 22.49 → 39.56 %', 1),
            ('nothing added to the network after v47 beats it — not capacity, not attention, not longer training', 1),
            ('the bottleneck is the pruning classification, not the reconstruction', 1)],
       LEFT, 1.26, 12.30, 2.15, space=6, lsp=0.95)
bullets(s, ['What runs now',
            ('rebalanced pruning losses at lr 3e-4 — 42.01 % AllParticles / 24.34 % PerfectReco, '
             'the first line above v47', 1),
            ('next: point-pruning precision, hard negatives, chain-level losses', 1)],
       LEFT, 3.60, 12.30, 1.70, space=6, lsp=0.95)
bullets(s, ['Open questions',
            ('is same-source (class-2) clustering the real problem DFEI should solve?', 1),
            ('what time budget would HLT2 leave for a lightweight GNN per event?', 1)],
       LEFT, 5.35, 12.30, 1.30, space=6, lsp=0.95)

prs.save(OUT)
print('[ok] saved: ' + OUT)
print('     slides: ' + str(len(prs.slides._sldIdLst)))

VDIR = FIG + '/versions'
os.makedirs(VDIR, exist_ok=True)
n = 1
while os.path.exists('%s/talk_v%02d_group_DFEI_meeting_20260915_EN.pptx' % (VDIR, n)):
    n += 1
for f in (os.path.abspath(__file__), BASE + '/meeting_20260915_group_script.md'):
    if os.path.exists(f):
        shutil.copy(f, '%s/talk_v%02d_group_%s' % (VDIR, n, os.path.basename(f)))
shutil.copy(OUT, '%s/talk_v%02d_group_DFEI_meeting_20260915_EN.pptx' % (VDIR, n))
print('[ok] archived as talk_v%02d_group in %s' % (n, VDIR))
