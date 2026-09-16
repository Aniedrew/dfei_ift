#!/usr/bin/env python3
"""DFEI group meeting, 2026-09-15 — the talk.

This is the 0902 deck brought up to date:
  * every number is the fixed-denominator number (N = 17 561 truth B candidates);
  * the figures that were drawn with PowerPoint shapes are proper figures now;
  * a new Part 0 walks through the pipeline step by step, and every later change
    carries a small "where in the pipeline" strip;
  * Part 4/5 of the old deck became a short summary of the five directions we
    tried after v47 — no version-by-version detail.

Layout: title + rule, bullets on the left, strip and figures on the right,
footer at 7.12 in.
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
OUT = FIG + '/DFEI_meeting_20260915_EN.pptx'

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
RX, RW = 6.85, 6.05            # right column
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
    # one title line only: centre it in the band above the rule
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


def strip(s, name, left=RX, top=1.22, w=RW):
    return pic(s, FIG + '/' + name, left, top, w, 1.30, center_in=(left, w))


# ============================================================ 1 title
s = slide(None, None)
tb = s.shapes.add_textbox(Inches(0.85), Inches(2.35), Inches(11.6), Inches(1.5))
tf = tb.text_frame; tf.word_wrap = True
r = tf.paragraphs[0].add_run(); r.text = 'Optimizing DFEI on CERN Monte Carlo'
r.font.size = Pt(44); r.font.bold = True; r.font.color.rgb = BLUE; r.font.name = FONT
tb = s.shapes.add_textbox(Inches(0.85), Inches(3.85), Inches(11.6), Inches(0.7))
r = tb.text_frame.paragraphs[0].add_run()
r.text = 'How the model works, the v31 → v47 line, and what came after'
r.font.size = Pt(BODY); r.font.color.rgb = DARK; r.font.name = FONT
tb = s.shapes.add_textbox(Inches(0.85), Inches(6.05), Inches(11.6), Inches(0.6))
r = tb.text_frame.paragraphs[0].add_run()
r.text = 'Qingxiang Guo · University of Chinese Academy of Sciences · 15 September 2026'
r.font.size = Pt(22); r.font.color.rgb = GRAY; r.font.name = FONT

# ============================================================ 2 data + overall effect
s = slide('DATA', BLUE, 'Data and the overall effect')
bullets(s, [
    'CERN official MC production (DFEI_IFT_20260702), threshold 0.9, 20 test files.',
    'Over the line v31 → v47: PerfectReco 12.30 → 23.15 % and AllParticles 22.49 → 39.56 %.',
    'Both are fractions of the 17 561 truth B candidates in the sample — the two metrics are '
    'defined in Part 0.',
    'The run-to-run spread of the protocol is ±0.43 pp, so anything below ~0.5 pp is not a result.',
], LEFT, 1.30, LW, 5.20, space=11)
pic(s, FIG + '/main_line.png', RX, 1.30, RW, 5.20, center_in=(RX, RW), center_v=(1.30, 5.20))

# ============================================================ 3 outline
s = slide('OUTLINE', BLUE, 'Outline')
items = [('PART 0', 'how the model works — the pipeline, step by step', TEAL),
         ('PART 1', 'differentiable pruning with annealing (v36)', BLUE),
         ('PART 2', 'rewarding the chain — hinge and chain-CE (v37, v38)', PINK),
         ('PART 3', 'supervising the representation with physics (v46 → v47)', GOLD),
         ('PART 4', 'what came after v47 — five directions, and the diagnosis', PURPLE),
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
note(s, 'In Parts 1–3 every slide shows a strip with the pipeline block the change acts on highlighted.', top=6.55)

# ============================================================ PART 0
s = slide('PART 0', TEAL, 'How the model works')
bullets(s, [
    'One event → one graph: nodes are tracks and PVs, edges are track pairs (tt).',
    'Every edge carries a truth class: 0 = background, 1/2/3 = structural.',
    'Stage 1 prunes the graph with learned scores; stage 2 rebuilds the chain from what is left.',
    'We score whole chains, not single edges.',
], LEFT, 1.26, 12.30, 1.90, space=6)
pic(s, FIG + '/p0_stages.png', 0.40, 3.40, 12.55, 2.60, center_in=(0.40, 12.55), center_v=(3.40, 2.60))
note(s, 'The rest of the talk is about where in this picture each change acts.', top=6.20)

s = slide('PART 0', TEAL, 'From one event to a graph')
bullets(s, [
    'Only tracks become nodes — the B and the J/ψ do not survive to the tracker.',
    'Every track pair becomes an edge; node and edge features carry kinematics, impact '
    'parameters and detector information.',
    'The truth chain from MC marks which tracks and which pairs belong to the B decay.',
    'It is a handful of tracks inside a very dense background graph.',
], LEFT, 1.30, LW, 5.10, space=12)
pic(s, FIG + '/p0_event.png', RX, 1.55, RW, 3.05, center_in=(RX, RW))
strip(s, 'w_ingraph.png', RX, 5.05)

# ------------------------------------------------- the four classes, defined once
s = slide('PART 0', TEAL, 'The four edge classes')
bullets(s, [
    'The head returns four numbers per pair — p0…p3, a softmax over the classes — '
    'and the largest one wins.',
    'class 0 — background: the two tracks share no common ancestor in this event; '
    '99.9 % of all edges.',
    'class 1 — parent–child · class 2 — sisters (same mother) · '
    'class 3 — grandparent–grandchild.',
    'class 2 carries the chain: a sister pair marks one decay vertex, and chain assembly '
    'is built from them.',
], LEFT, 1.30, LW, 5.20, space=9)
pic(s, FIG + '/p0_class.png', RX, 1.45, RW, 3.30, center_in=(RX, RW))
strip(s, 'w_classdef.png', RX, 5.00)

s = slide('PART 0', TEAL, 'Stage 1 — pruning: decide what survives')
bullets(s, [
    'Two binary heads do the pruning: a point head scores every node, an edge head '
    'scores every edge.',
    'Each score is a learned weight w in [0, 1] — does this track, or this pair, belong '
    'to a B decay?',
    'At inference a hard cut is applied: only nodes and edges with w ≥ 0.9 survive.',
    'The four-class LCAG classification is NOT part of pruning — it runs on the edges '
    'that survive, in the next step.',
], LEFT, 1.30, LW, 4.60, space=10)
pic(s, FIG + '/p0_prune.png', RX, 1.26, RW, 3.85, center_in=(RX, RW))
strip(s, 'w_stage1.png', RX, 5.25)

s = slide('PART 0', TEAL, 'Stage 2 — reconstruction: build the chain')
bullets(s, [
    'On the surviving edges, the LCAG head classifies every pair: 0 = background, '
    '1/2/3 = structural.',
    'Chain assembly joins tracks while the LCA stays consistent; a class-0 edge cannot '
    'glue two tracks.',
    'The heads read one shared 16-d representation; every message carries the learned '
    'weight w.',
    'The candidates are compared with the truth chain — that is where the metrics come from.',
], LEFT, 1.30, LW, 5.20, space=8)
pic(s, FIG + '/p0_recon.png', RX, 1.26, RW, 3.85, center_in=(RX, RW))
strip(s, 'w_stage2.png', RX, 5.25)

s = slide('PART 0', TEAL, 'How we score it')

bullets(s, [
    'AllParticles: all the truth particles are in one recovered chain — and nothing else is.',
    'PerfectReco: the content matches AND the connections match — the tree is rebuilt exactly.',
    'So PerfectReco is the stricter flag. N = 17 561 truth B candidates at threshold 0.9.',
    'Four identical runs spread over ±0.43 pp — that is the noise floor.',
], LEFT, 1.30, LW, 5.20, space=9)
pic(s, FIG + '/p0_metric.png', RX, 1.35, RW, 3.45, center_in=(RX, RW))
strip(s, 'w_score.png', RX, 5.00)

# ============================================================ PART 1
s = slide('PART 1', BLUE, 'Differentiable pruning with annealing')
bullets(s, [
    'Problem: training sees the full graph, but inference prunes with a hard cut first.',
    'Fix: multiply the message weights by a soft mask, w_eff = w · σ((w − cut)/τ).',
    'τ is annealed 1.0 → 0.1; the cut tightens 0.5 → 0.7 → 0.85, and inference keeps w ≥ 0.9.',
    'Evidence: v31 → v36 gives 12.30 → 18.77 (PerfectReco) and 22.49 → 35.17 (AllParticles) — '
    'the largest step before the mass head.',
], LEFT, 1.30, LW, 5.10, space=9)
strip(s, 'w_v36.png')
pic(s, FIG + '/p1_anneal.png', RX, 2.62, RW, 3.95, center_in=(RX, RW), center_v=(2.62, 3.95))
note(s, 'A training-time change: inference keeps the same hard cut.', top=6.60)

# ============================================================ PART 2
s = slide('PART 2', PINK, 'Why chains die (1/2) — one wrong edge is enough')
bullets(s, [
    'At inference, a chain edge that the classifier calls background (class 0) is pruned.',
    'If a single structural edge of a chain is missing, the chain can no longer be assembled.',
    'So the accuracy that matters is the accuracy on the edges that carry a chain — and those '
    'are the rarest ones.',
], LEFT, 1.30, LW, 4.60, space=10)
pic(s, FIG + '/chain_before.png', RX, 1.26, RW, 2.00, center_in=(RX, RW))
pic(s, FIG + '/chain_after.png', RX, 3.38, RW, 2.00, center_in=(RX, RW))
strip(s, 'w_edge.png', RX, 5.42)

s = slide('PART 2', PINK, 'Why chains die (2/2) — the class imbalance')
bullets(s, [
    'class 0 (background) is 99.9 % of all edges; each structural class is about 0.04 %.',
    'The LCAG loss is a global cross-entropy, so almost all of its gradient says “background”.',
    'Per-class accuracy collapses on the rare classes — worst on class 2, the sisters.',
    'So the structural classes must be supervised directly: that is what the rewards do.',
], LEFT, 1.30, LW, 5.20, space=9)
pic(s, FS + '/chain_lca_imbalance_dist.png', RX, 1.26, RW, 1.95, center_in=(RX, RW))
pic(s, FS + '/chain_lca_imbalance_acc.png', RX, 3.32, RW, 1.95, center_in=(RX, RW))
strip(s, 'w_class.png', RX, 5.42)

s = slide('PART 2', PINK, 'Reward 1 (v37) — the hinge')
bullets(s, [
    'Confidence comes from the LCAG head: confidence = max(p0…p3).',
    'Penalty = max(0, margin − confidence), margin 0.3 — nothing is paid above that.',
    'It is paid only on truth-chain edges, known from MC truth during training.',
    'Effect: 18.77 → 19.50, but class-2 accuracy went DOWN (47.7 → 44.2) — weight 3.0 over-corrected.',
], LEFT, 1.30, LW, 5.20, space=12)
pic(s, FIG + '/d_hinge.png', RX, 1.26, RW, 1.95, center_in=(RX, RW))
pic(s, FS + '/chain_lca_hinge_curve.png', RX, 3.32, RW, 2.05, center_in=(RX, RW))
strip(s, 'w_hinge.png', RX, 5.42)

s = slide('PART 2', PINK, 'Reward 2 (v38) — chain cross-entropy')
bullets(s, [
    'CE = −log p_true: it pays for being the right class, not merely for being confident.',
    'It is paid only on truth-chain edges (classes 1/2/3), so its gradient is undiluted.',
    'v38 also brings the class-2 weight back to 2.0 and tightens the cut to 0.85.',
    'Effect: 19.50 → 21.10; class-1 accuracy 67.8 → 76.8 and class-2 41.3 → 47.9.',
], LEFT, 1.30, LW, 5.20, space=12)
pic(s, FIG + '/d_cewhere.png', RX, 2.62, RW, 3.95, center_in=(RX, RW), center_v=(2.62, 3.95))
strip(s, 'w_ce.png')

s = slide('PART 2', PINK, 'The strategy works')
bullets(s, [
    'Over the whole line (v31 → v38): PerfectReco 12.30 → 21.10 and AllParticles 22.49 → 37.59.',
    'Per-class accuracy: class-1 67.8 → 76.8, class-2 41.3 → 47.9 — the classes that decide '
    'whether a chain survives.',
    'It stays a training-only change: inference keeps the same hard cut at 0.9.',
], RX, 2.62, 6.35, 3.90, space=11)
strip(s, 'w_evid.png', RX, 1.22)
pic(s, FIG + '/p1_v31_v38.png', LEFT, 1.28, 5.60, 5.30, center_in=(LEFT, 5.60), center_v=(1.28, 5.30))

# ============================================================ PART 3
s = slide('PART 3', GOLD, 'First, a probe — the physics is not there')
bullets(s, [
    'Method: freeze the backbone, fit one linear layer, read a physical quantity out.',
    'If the quantity is not encoded the probe fails: R² ≈ 0.',
    'Before any physics head: ππ mass R² = 0.003 and momentum R² ≈ 0 — the physics is lost.',
    'After the mass head: edge mass R² = 0.930 — so physics is not free, it must be supervised in.',
], LEFT, 1.30, LW, 5.20, space=12)
pic(s, FIG + '/d_probe.png', RX, 1.26, RW, 2.20, center_in=(RX, RW))
pic(s, FS + '/probe_r2.png', RX, 3.55, RW, 2.20, center_in=(RX, RW))
strip(s, 'w_probe.png', RX, 5.42)

s = slide('PART 3', GOLD, 'The head zoo — nine targets on one backbone')
bullets(s, [
    'Five heads are the original DFEI machinery: LCAG, node prune, edge prune, PV association, '
    'chain scorer.',
    'Four are ours, and they are all label-free from MC truth: source (v36), mass (v46/47), '
    'struct (v48/52), momentum (v48/51).',
], LEFT, 1.28, 12.30, 1.30, space=6)
pic(s, FIG + '/m_zoo.png', 0.40, 3.05, 12.55, 2.55, center_in=(0.40, 12.55), center_v=(3.05, 2.55))
note(s, 'Adding a head never changes the backbone — it only adds a supervised target on top of the '
        'same representation.', top=5.75, left=0.55, width=6.05)
strip(s, 'w_zoo.png', RX, 5.70)

s = slide('PART 3', GOLD, 'The mass head — the one that pays')
bullets(s, [
    'Target: the invariant mass of the track pair on every tt edge, from MC truth.',
    'v46 regressed the raw mass: 21.28 — inside the noise.',
    'v47 adds log10 normalisation and masks the sentinel edges.',
    'Evidence: v38 → v47 gives 21.10 → 23.15 and 37.59 → 39.56 (AllParticles).',
    'The same recipe on source, struct and momentum is a wash.',
], LEFT, 1.30, LW, 5.10, space=9)
pic(s, FIG + '/s_mass.png', RX, 1.26, RW, 1.90, center_in=(RX, RW))
pic(s, FIG + '/d_mass.png', RX, 3.28, RW, 2.35, center_in=(RX, RW))
strip(s, 'w_mass.png', RX, 5.70)

s = slide('PART 3', GOLD, 'The struct head — tree position without labels')
bullets(s, [
    'Depth: how far a node sits from the chain root, in BFS layers.',
    'Rumor Centrality: which node is the source of the chain (Shah & Zaman).',
    'Both come from the truth tree, so no manual labels are needed.',
    'On the weak ablation base this was the best single addition; on the main line it is a wash.',
    'Stacking three heads at full weight (v48) is also a wash: 21.40 / 37.36 against 21.10 / 37.59.',
], LEFT, 1.30, LW, 5.20, space=10)
pic(s, FIG + '/depth_calc.png', RX, 1.26, RW, 2.10, center_in=(RX, RW))
pic(s, FIG + '/rc_calc.png', RX, 3.42, RW, 2.20, center_in=(RX, RW))
strip(s, 'w_struct.png', RX, 5.42)

# ============================================================ PART 4
s = slide('PART 4', PURPLE, 'What came after v47')
bullets(s, [
    'Once the mass head landed, we pushed five directions in parallel: capacity, '
    'context/attention, restructuring the event, retraining the recipe, and another dataset.',
    'The verdict is the same everywhere: the network is not the bottleneck.',
], LEFT, 1.28, 12.30, 1.35, space=6)
pic(s, FIG + '/m_directions.png', 0.40, 3.30, 12.55, 2.30, center_in=(0.40, 12.55), center_v=(3.30, 2.30))
note(s, 'Same pattern in all five: adding capacity, attention or training time moves nothing, '
        'and combining the few things that do work also moves nothing.', top=5.75)

s = slide('PART 4', PURPLE, 'The diagnosis — and what runs now')
bullets(s, [
    'The oracle test: freeze v47 and rewrite the pruning decisions with MC truth → the same '
    'pipeline reaches 99.75 % AllParticles, against 39.56 % baseline.',
    'So the score is decided at the pruning step, not by the network.',
    'Two causes: the point head carried 1/33 of the loss weight, and the step size was 10× too small.',
    'What runs now: rebalanced pruning losses, focal loss, a chain-level recall term.',
], RX, 1.60, 6.35, 4.85, space=8)
pic(s, FIG + '/oracle_ladder.png', LEFT, 1.26, 5.60, 5.32, center_in=(LEFT, 5.60), center_v=(1.26, 5.32))

# ============================================================ summary
s = slide('SUMMARY', GREEN, 'Summary and next steps')
bullets(s, ['What we now know',
            ('the main line v31 → v47 is real: PerfectReco 12.30 → 23.15 %, '
             'AllParticles 22.49 → 39.56 %', 1),
            ('nothing added after v47 beats it — not capacity, not attention, not longer training', 1),
            ('the reconstruction is not the bottleneck; the pruning classification is', 1)],
       LEFT, 1.26, 12.30, 2.15, space=6, lsp=0.95)
bullets(s, ['Next',
            ('finish the pruning-reweighting arms and re-test the old verdicts at lr 3e-4', 1),
            ('if the point head responds, attack precision: hard negatives and chain-level losses', 1),
            ('continue the public line to see whether the levers transfer', 1)],
       LEFT, 3.42, 12.30, 1.70, space=6, lsp=0.95)
bullets(s, ['Open questions',
            ('is same-source (class-2) clustering the real problem DFEI should solve?', 1),
            ('what time budget would HLT2 leave for a lightweight GNN per event?', 1)],
       LEFT, 5.42, 12.30, 1.30, space=6, lsp=0.95)

prs.save(OUT)
print('[ok] saved: ' + OUT)
print('     slides: ' + str(len(prs.slides._sldIdLst)))

VDIR = FIG + '/versions'
os.makedirs(VDIR, exist_ok=True)
n = 1
while os.path.exists('%s/talk_v%02d_DFEI_meeting_20260915_EN.pptx' % (VDIR, n)):
    n += 1
for f in (os.path.abspath(__file__), BASE + '/make_dfei_figs_20260915.py',
          BASE + '/make_meeting_maps_dot.py', BASE + '/make_meeting_20260915_figs2.py',
          BASE + '/make_meeting_20260915_legacy.py',
          BASE + '/meeting_20260915_talk_script.md'):
    shutil.copy(f, '%s/talk_v%02d_%s' % (VDIR, n, os.path.basename(f)))
shutil.copy(OUT, '%s/talk_v%02d_DFEI_meeting_20260915_EN.pptx' % (VDIR, n))
print('[ok] archived as talk_v%02d in %s' % (n, VDIR))
