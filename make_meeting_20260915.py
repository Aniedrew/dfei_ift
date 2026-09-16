#!/usr/bin/env python3
"""DFEI group meeting deck — 2026-09-15.   English.

Structure: title · contents · the complete map · one chapter per subject.
Every map/schematic is drawn with graphviz `dot` (make_meeting_maps_dot.py);
every before/after chart and every per-attempt card comes from matplotlib
(make_meeting_20260915_figs2.py, make_attempt_figs.py).

Each chapter opens with a head page (its slice of the map + the chapter result)
and then gives every single attempt its own page: how it works, and what it gave.

Body text (bullets, captions) is at least 24 pt.

Build:  /lzufs/home/guoqingxiang/miniconda3/envs/dfei/bin/python make_meeting_20260915.py
Every build is archived under meeting_figs_20260915/versions/ together with the sources.
"""
import os
import shutil

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

BASE = '/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn'
FIG = BASE + '/meeting_figs_20260915'
OLD = BASE + '/meeting_figs'
OUT = FIG + '/DFEI_progress_20260915_EN.pptx'

BLUE = RGBColor(0x1F, 0x4E, 0x79)
DARK = RGBColor(0x33, 0x33, 0x33)
GRAY = RGBColor(0x77, 0x77, 0x77)
PURPLE = RGBColor(0x7A, 0x44, 0xAA)
PLUM = RGBColor(0x9B, 0x59, 0xB6)
TEAL = RGBColor(0x0B, 0x72, 0x85)
PINK = RGBColor(0xC2, 0x18, 0x5B)
GOLD = RGBColor(0xB8, 0x86, 0x0B)
GREEN = RGBColor(0x2E, 0x7D, 0x32)
CAPC = RGBColor(0x8E, 0x44, 0xAD)
FONT = 'Calibri'
BODY = 24            # every piece of body text is at least this

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


_N = [0]


def new_slide(idx, chip=None, title=None, sub=None, chip_color=BLUE):
    _N[0] += 1
    idx = _N[0]
    s = prs.slides.add_slide(BLANK)
    if title:
        if chip:
            cw = 0.42 * len(str(chip)) + 0.30
            b = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.42), Inches(0.16),
                                   Inches(cw), Inches(0.48))
            b.fill.solid(); b.fill.fore_color.rgb = chip_color
            b.line.fill.background()
            tf = b.text_frame
            tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
            tf.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
            r = p.add_run(); r.text = str(chip)
            r.font.size = Pt(18 if len(str(chip)) > 3 else 22)
            r.font.bold = True; r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF); r.font.name = FONT
            left = 0.42 + cw + 0.18
        else:
            left = 0.5
        tb = s.shapes.add_textbox(Inches(left), Inches(0.10), Inches(13.333 - left - 0.30), Inches(0.95))
        tf = tb.text_frame; tf.word_wrap = True
        r = tf.paragraphs[0].add_run(); r.text = title
        r.font.size = Pt(30 if len(title) <= 40 else 27)
        r.font.bold = True; r.font.color.rgb = BLUE; r.font.name = FONT
        if sub:
            p2 = tf.add_paragraph()
            r2 = p2.add_run(); r2.text = sub
            r2.font.size = Pt(BODY); r2.font.color.rgb = GRAY; r2.font.name = FONT
        ln = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(1.10), Inches(12.35), Pt(2.6))
        ln.fill.solid(); ln.fill.fore_color.rgb = BLUE; ln.line.fill.background()
    tb = s.shapes.add_textbox(Inches(0.5), Inches(7.12), Inches(12.35), Inches(0.30))
    r = tb.text_frame.paragraphs[0].add_run()
    r.text = 'Qingxiang Guo · UCAS · DFEI group meeting · 2026-09-15          ' + str(idx)
    r.font.size = Pt(14); r.font.color.rgb = GRAY; r.font.name = FONT
    return s


def bullets(s, items, left, top, width, height, size=BODY, color=None, space=6):
    tb = s.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame; tf.word_wrap = True
    for i, it in enumerate(items):
        txt, lvl = it if isinstance(it, tuple) else (it, 0)
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(space)
        r = p.add_run()
        r.text = ('      ' * lvl) + ('•  ' if lvl == 0 else '–  ') + txt
        r.font.size = Pt(size)
        r.font.color.rgb = color or DARK; r.font.name = FONT
    return tb


def note(s, text, top=6.50, color=GRAY, size=BODY, left=0.5, width=12.35):
    tb = s.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(0.5))
    tf = tb.text_frame; tf.word_wrap = True
    r = tf.paragraphs[0].add_run(); r.text = text
    r.font.size = Pt(size); r.font.color.rgb = color; r.font.name = FONT
    r.font.italic = True
    return tb


def pic_fit(s, path, left, top, max_w, max_h, center_in=None, center_v=None):
    if not os.path.exists(path):
        print('[warn] missing figure:', path)
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


def head(idx, part, title, sub, fig, chip_color, caption=None, points=(), width=4.40):
    """Chapter page: the map on the left, the chapter summary on the right."""
    s = new_slide(idx, part, title, sub, chip_color=chip_color)
    bulge = 0.40 + width + 0.35
    pic_fit(s, FIG + '/' + fig, 0.40, 1.26, width, 5.05, center_v=(1.26, 5.05))
    bullets(s, list(points), bulge, 1.28, 12.95 - bulge, 4.90, size=BODY, space=8)
    if caption:
        note(s, caption, top=6.50, size=BODY)
    return s


def headfig(idx, part, title, sub, mapfig, chart, chip_color, caption=None):
    """Chapter page: the map on the left, the chapter result on the right."""
    s = new_slide(idx, part, title, sub, chip_color=chip_color)
    pic_fit(s, FIG + '/' + mapfig, 0.40, 1.26, 4.40, 5.05, center_v=(1.26, 5.05))
    pic_fit(s, FIG + '/' + chart, 5.05, 1.24, 7.65, 4.85, center_in=(5.05, 7.65),
            center_v=(1.24, 4.85))
    if caption:
        note(s, caption, top=6.38, size=BODY)
    return s


def split(idx, part, title, sub, points, figs, chip_color, note_txt=None, fig_w=4.30):
    """Text on the left, evidence figures on the right."""
    s = new_slide(idx, part, title, sub, chip_color=chip_color)
    tw = 12.25 - fig_w
    bullets(s, list(points), 0.50, 1.24, tw, 5.10, size=BODY, space=7)
    slot = 5.10 / len(figs)
    for k, f in enumerate(figs):
        pic_fit(s, f, 12.90 - fig_w, 1.26 + k * slot, fig_w, slot - 0.10)
    if note_txt:
        note(s, note_txt, top=6.50, size=BODY)
    return s


def attempt(idx, chip, chip_color, ver, title, sub, how, figkey, takeaway):
    """One page per attempt: what it is on the left, the card on the right."""
    s = new_slide(idx, chip, '%s — %s' % (ver, title), sub, chip_color=chip_color)
    bullets(s, list(how), 0.50, 1.24, 6.10, 5.10, size=BODY, space=9)
    pic_fit(s, FIG + '/att_%s.png' % figkey, 6.80, 1.22, 6.10, 5.05,
            center_in=(6.80, 6.10), center_v=(1.22, 5.05))
    note(s, takeaway, top=6.42, size=BODY)
    return s


# =========================================================== 1 title
s = new_slide(1)
tb = s.shapes.add_textbox(Inches(0.85), Inches(2.35), Inches(11.8), Inches(1.5))
tf = tb.text_frame; tf.word_wrap = True
r = tf.paragraphs[0].add_run(); r.text = 'Optimizing DFEI on CERN Monte Carlo'
r.font.size = Pt(44); r.font.bold = True; r.font.color.rgb = BLUE; r.font.name = FONT
tb = s.shapes.add_textbox(Inches(0.85), Inches(3.85), Inches(11.8), Inches(0.6))
tf = tb.text_frame
r = tf.paragraphs[0].add_run(); r.text = 'Version map, every attempt, and the diagnosis'
r.font.size = Pt(BODY); r.font.color.rgb = DARK; r.font.name = FONT
tb = s.shapes.add_textbox(Inches(0.85), Inches(6.05), Inches(11.8), Inches(0.6))
tf = tb.text_frame
r = tf.paragraphs[0].add_run()
r.text = 'Qingxiang Guo · University of Chinese Academy of Sciences · 15 September 2026'
r.font.size = Pt(BODY); r.font.color.rgb = GRAY; r.font.name = FONT

# =========================================================== 2 contents
s = new_slide(2, 'CONTENTS', 'Contents')
items = [
    ('MAP', 'the complete version map', BLUE),
    ('PART 1', 'v31 → v38 — the recipe that worked', BLUE),
    ('PART 2', 'v38 → v47 — physics into the representation', GOLD),
    ('PART 3', 'the five branches after v47 — every attempt', CAPC),
    ('PART 4', 'the diagnosis — what limits the score', TEAL),
    ('PART 5', 'what runs now, and what comes next', GREEN),
]
y = 1.36
for tag, t, col in items:
    b = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.75), Inches(y), Inches(1.75), Inches(0.56))
    b.fill.solid(); b.fill.fore_color.rgb = col; b.line.fill.background()
    tf = b.text_frame; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = tag
    r.font.size = Pt(18); r.font.bold = True; r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF); r.font.name = FONT
    tb = s.shapes.add_textbox(Inches(2.75), Inches(y + 0.02), Inches(10.0), Inches(0.56))
    tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    r = tf.paragraphs[0].add_run(); r.text = t
    r.font.size = Pt(BODY); r.font.color.rgb = col; r.font.name = FONT
    y += 0.86
note(s, 'PerfectReco = the fraction of truth B candidates whose full chain is recovered.', top=6.68, size=BODY)

# =========================================================== 3 the complete map
s = new_slide(3, 'MAP', 'The complete version map')
pic_fit(s, FIG + '/m_master.png', 0.30, 1.24, 11.90, 5.10, center_in=(0.30, 11.95),
        center_v=(1.24, 5.10))
pic_fit(s, FIG + '/cbar.png', 12.45, 1.75, 0.80, 3.40)
note(s, 'Colour = PerfectReco · red dashed = closed branch · grey = not evaluated', top=6.52, size=BODY)

# =========================================================== PART 1
s = head(4, 'PART 1', 'v31 → v38 — the recipe that worked',
         'Four training-time changes; evaluation untouched',
         'm_p1.png', BLUE,
         caption='This is the part we would keep as it is today.',
         points=['The main line: v31 → v32/v35 → v36 → v37 → v38.',
                 'Every change acts during training only; inference keeps a hard cut of 0.9.',
                 'Evidence: PerfectReco 12.30 → 21.10, AllParticles 22.49 → 37.59.'])

s = split(5, 'PART 1', 'v31 → v38 — what each step bought',
          'Four changes, one step at a time',
          ['v31 — the reference point: fix a silent class-weight bug in the LCAG loss.',
           'v32/v35 — plain continuation (lr 1e-3 → 1e-4, 150 epochs): the scores do not move.',
           'v36 — soft mask + source head: +6.47 pp PerfectReco.',
           'v37 — class-2 weight 3.0 + hinge, cut 0.7: +0.73 pp only, and class-2 got worse.',
           'v38 — class-2 back to 2.0 + chain-CE, cut 0.85: +1.60 pp.'],
          [FIG + '/p1_v31_v38.png'], BLUE, fig_w=6.60)

s = split(6, 'PART 1', 'v36 — differentiable pruning with annealing',
          'Align training with the pruned graph at inference',
          ['Problem: training sees the full graph, but inference prunes with a hard cut first.',
           'Fix: multiply the message weights by a soft mask w_eff = w · σ((w − cut)/τ).',
           'τ is annealed 1.0 → 0.1: large τ keeps gradients alive, small τ approaches the hard cut.',
           'The cut tightens over versions: 0.5 (v36) → 0.7 (v37) → 0.85 (v38); inference at 0.9.',
           'Evidence: PerfectReco 12.30 → 18.77 (+6.5 pp) — the largest step before the mass head.'],
          [FIG + '/m_mech.png', OLD + '/pruning_sigmoid_tau.png'], BLUE, fig_w=5.30)

s = split(7, 'PART 1', 'Why chains die — class-0 dilution',
          'The global cross-entropy is ~99.9% “background”',
          ['class 0 (background) is 99.9% of all edges; each structural class is about 0.04%.',
           'The LCAG loss is a global cross-entropy, so almost all of its gradient says “background”.',
           'Per-class accuracy collapses exactly on the rare structural classes — worst on class 2.',
           'At inference one misclassified structural edge is pruned and the whole chain is lost.',
           'So the structural classes must be supervised directly. That is what the rewards do.'],
          [OLD + '/chain_lca_imbalance_dist.png', OLD + '/chain_lca_imbalance_acc.png'],
          BLUE, fig_w=5.30)

s = split(8, 'PART 1', 'v37 — the first reward: a hinge',
          'Class-2 weight 3.0 · hinge margin 0.3 · cut 0.7',
          ['Class-2 weighting: multiply the sister-edge loss term by 3.0.',
           'Hinge — a training-only confidence bonus. Confidence = max(p0…p3) from the LCAG head.',
           'Penalty = max(0, margin − confidence) with margin 0.3: nothing is paid above that.',
           'It is paid ONLY on truth-chain edges, known from MC truth during training.',
           'Evidence: 18.77 → 19.50 — and class-2 went DOWN, 47.7 → 44.2, so 3.0 over-corrected.'],
          [OLD + '/chain_lca_hinge_mechanism.png', OLD + '/chain_lca_hinge_curve.png'],
          BLUE, fig_w=5.30)

s = split(9, 'PART 1', 'v38 — the second reward: chain-CE',
          'Class-2 back to 2.0 · chain-CE · cut 0.85',
          ['Class-2 weighting is brought back to 2.0, undoing the over-correction of v37.',
           'Chain-CE: a 4-class cross-entropy on chain edges, added on top of the hinge.',
           'CE = −log p_true: it pays for being the RIGHT class, not merely for being confident.',
           'Paid only on truth-chain edges (classes 1/2/3), so its gradient is undiluted.',
           'Evidence: the two rewards together move class-1 67.8 → 76.8 and class-2 41.3 → 47.9.'],
          [OLD + '/chain_lca_ce_where.png', FIG + '/d_rewards.png'], BLUE, fig_w=5.40)

# =========================================================== PART 2
s = head(10, 'PART 2', 'v38 → v47 — physics into the representation',
         'Heads on one backbone; the architecture is unchanged',
         'm_p2.png', GOLD,
         caption='Before any head the mass probe gave R² = 0.003.',
         points=['Purpose: make the shared representation carry physics.',
                 'Principle: small regression heads on the same backbone.',
                 'Evidence: only the mass head pays — 21.10 → 23.15.'])

s = split(11, 'PART 2', 'First, a probe: the physics is not there',
          'What the backbone does not encode',
          ['Method: freeze the backbone, fit one linear layer, read a physical quantity out.',
           'If the quantity is not encoded, the probe fails: R² ≈ 0.',
           'Before any head: ππ mass R² = 0.003 and momentum R² ≈ 0 — the physics is lost.',
           'After the mass head (masshead2): edge mass R² = 0.930.',
           'So the reconstruction objective does not teach physics for free. It has to be supervised in.'],
          [OLD + '/probe_method.png', OLD + '/probe_r2.png'], GOLD, fig_w=5.30)

s = new_slide(12, 'PART 2', 'The head zoo — nine targets on one backbone',
              'Every head reads the same representation',
              chip_color=GOLD)
pic_fit(s, FIG + '/m_zoo.png', 0.40, 1.22, 12.5, 2.35, center_in=(0.45, 12.45))
bullets(s, ['Five heads are the original DFEI machinery: LCAG, node prune, edge prune, PV '
            'association, chain scorer.',
            'Our four: source (v36), mass (v46/47), struct (v48/52), momentum (v48/51) — all '
            'label-free from MC truth.',
            'All nine read the same 16-dim representation; the heads are parallel, every arrow '
            'starting at the backbone.'],
        0.50, 3.70, 12.35, 2.85, size=BODY)

s = split(13, 'PART 2', 'v46 / v47 — the mass head',
          'The one head that pays',
          ['Target: the invariant mass of the track pair on each tt edge, taken from MC truth.',
           'v46 (masshead1) regressed the raw mass: 21.28 — inside the noise.',
           'v47 (masshead2) adds log10 normalisation and masks the sentinel edges.',
           'Evidence: 21.10 → 23.15 PerfectReco, 37.59 → 39.56 AllParticles — still unbeaten.',
           'The identical recipe on source, struct and momentum is a wash.'],
          [FIG + '/s_mass.png', FIG + '/d_mass.png'], GOLD, fig_w=5.30)

# =========================================================== PART 3
s = head(19, 'PART 3', 'The five branches after v47',
         'What we tried once the main line stopped',
         'm_branches.png', CAPC,
         caption='None of the five branches reaches the main line.',
         points=['Six groups of attempts, all starting from v31, v38 or v47.',
                 'From here on every single attempt gets its own page.',
                 'Two branches were closed outright, three produced one useful fact each.'])

# ---- one page per attempt, in order ---------------------------------------
A_CAP = [
 ('PART 3a', CAPC, 'v48', 'three heads at once',
  'base v38 · three heads together, 130 epochs',
  ['All three heads read the same 16-d node and edge representation.',
   'Their losses sum to 0.877 against 0.559 for the main reconstruction task.',
   'So on this base the auxiliary terms — not the reconstruction — drive the update.'],
  'cap_v48', 'Result: 37.36 / 21.40 against v38 at 37.59 / 21.10 — a wash.'),
 ('PART 3a', CAPC, 'v50 / v51 / v52', 'one head at a time',
  'base v47 · a 5-file probe, one added head per run',
  ['With stacking ruled out, each head is added on its own to see whether it still pays.',
   'The probe is deliberately small — 20 training files, 5-file evaluation — so it is compared '
   'only against its own control.',
   'v50 is the control (mass head only); v51 adds momentum, v52 adds the struct head.'],
  'cap_probe', 'Both added heads move AllParticles: +2.4 and +3.1 respectively.'),
 ('PART 3a', CAPC, 'v53', 'widen the latent space, from scratch',
  'from scratch · tracks 32-d, tt edges 24-d',
  ['The idea: 16 dimensions are shared by nine heads, and a track carries 12–14 physical '
   'degrees of freedom, so the space may be the bottleneck.',
   'The encoder, the GNN blocks and the decoder are all widened; everything else is unchanged.',
   'Trained from scratch because the widened layers have no counterpart in v38.'],
  'cap_v53', '26.78 / 15.84 after 150 epochs — far below v38, and it had converged.'),
 ('PART 3a', CAPC, 'v512', 'the same widening, inherited',
  'base v38 · same widening, weights carried over',
  ['Same architecture change as v53, but the compatible weights are inherited from v38.',
   'The point is to separate “width” from “from scratch”: if v53 failed only because it started '
   'from zero, inheriting should fix it.',
   '51 conflicting layers and the source head are re-initialised; the rest is carried over.'],
  'cap_v512', '24.03 / 13.34 — worse than from scratch: the inherited trunk is disturbed.'),
 ('PART 3a', CAPC, 'v513 / v514', 'widen the GN hidden layer',
  'base v38 · GNN hidden width 128 → 256, 20 epochs',
  ['The lightest capacity change available: only the four GNN update networks get wider.',
   'The latent stays 16-d, so the encoder, the decoder and every head interface are untouched.',
   'v513 hit CUDA OOM at batch 8 mid-run; v514 continued it at batch 6.'],
  'cap_v514', '25.44 / 15.35 — the widened GNN never recovers the v38 level.'),
 ]

A_ATT = [
 # ---- attention
 ('PART 3b', PINK, 'v510', 'track-level self-attention',
  'base v38 · attention after the last GN block',
  ['Motivation: class-0 dilution means an edge is only judged from its two endpoints. '
   'Attention lets a track see the other tracks of its own event.',
   'Implementation: multi-head self-attention after the last GN block, residual + LayerNorm, '
   'masked so that events never mix.',
   '16-dimensional, 4 heads, the rest of the stack untouched.'],
  'att_v510', '38.01 / 21.75 — the first sign that event context is worth something.'),
 ('PART 3b', PINK, 'v511', 'add the edge-feature bias',
  'base v510 · tt edge features as attention bias',
  ['Pure content attention ignores the track-pair features the GNN already computes.',
   'Fix: an MLP maps the tt edge feature to one logit per head, scattered into the attention '
   'scores — pairs without an edge keep the content-only score.',
   'This is the ParT-style P-MHA construction, and the only difference from v510 is this bias.'],
  'att_v511', '38.39 / 22.11 — +1.64 pp against v38; class-1 recall 78.27%.'),
 ('PART 3b', PINK, 'v515 / v516', 'the same module on v47',
  'base v47 · attention on the best model, +60 epochs',
  ['v511 was validated on v38; the natural test is to put the same module on the current best, v47.',
   'Only the attention layers are new — the whole trunk, including the mass head, is inherited.',
   'v516 then continues v515 for 60 more epochs to separate “not enough budget” from “conflict”.'],
  'att_v515', '22.95 and 22.72 against v47 at 23.15 — the two gains do not add up.'),
 ('PART 3b', PINK, 'v517 / v519', 'generation: attention + mass',
  'base v38 · attention + mass trained together',
  ['Post-hoc grafting failed, so the two new capabilities are introduced together instead, '
   'from a base that is not yet set in its ways.',
   'Generation method: start from v38 and let attention and the mass head train jointly for the '
   'full budget, with early stopping disabled.',
   'This is the same path that had worked for the mass head (v46 → v47).'],
  'att_v519', '38.12 / 22.00 — better than v38, still below v47.'),
 ('PART 3b', PINK, 'v518 / v520', 'generation: the struct head',
  'base v47 · struct head, run to 179 epochs',
  ['The struct head had never been given a long budget; the earlier attempt (v48) was cut short.',
   'v518 starts from v47 with the struct head added; v520 continues it to the full 179 epochs.',
   'If structure supervision is useful, a long run should show it.'],
  'att_v520', '22.52 → 22.69 over 48 more epochs; v47 stays at 23.15.'),
 ]

A_RES = [
 # ---- restructure
 ('PART 3c', PLUM, 'v39', 'split the event by truth PV',
  'base v38 · cluster the tracks, then reconstruct',
  ['The event graph is highly connected: 91–139 nodes per event, and chains interfere across '
   'primary vertices.',
   'The idea: split the tracks into PV clusters first, reconstruct each cluster on its own, '
   'then merge the results.',
   'A hard truth split is the cheapest way to test whether the idea can work at all.'],
  'res_v39', 'Stopped itself after 15 epochs: the split only exists at inference.'),
 ('PART 3c', PLUM, 'v40', 'make the split trainable',
  'base v38 · a learned cluster head, Gumbel',
  ['Fixing the previous failure: the split becomes a learned MLP head, supervised by truth PV, '
   'with a Gumbel-annealed assignment so real subgraphs are cut during training.',
   'This creates a mismatch: training runs on subgraphs, validation still ran on the full graph.',
   'The resumption also started with the annealing temperature already at its final value.'],
  'res_v40', 'Validation 35.9 → 84.9 → 114 → 155: the run diverged.'),
 ('PART 3c', PLUM, 'v41', 'align train and validation',
  'base v38 · same idea, problems addressed',
  ['Validation is now split the same way as training, so the two see the same graph structure.',
   'A curriculum moves the split from truth to the learned head over 30 epochs, and the number '
   'of subgraphs per step is capped to keep the run affordable.',
   'Early stopping is reset at the start, since the validation metric has changed meaning.'],
  'res_v41', 'Correct code, but 3.2 h/epoch and a curve that never fell.'),
 ('PART 3c', PLUM, 'v42', 'the same question at 50 files',
  'base v38 · the same recipe at 50 files',
  ['The previous run could not answer the question because it was too slow, so the same code is '
   'run on a quarter of the data.',
   'This version does converge — the curriculum and the validation alignment work as intended.',
   'It is the clean test of whether training on subgraphs helps.'],
  'res_v42', 'It does not: −2.98 pp PerfectReco, class-1 76.8% → 56.4%.'),
 ]

A_RET = [
 # ---- retrain
 ('PART 3d', PURPLE, 'v500 → v509', 'the ablation protocol',
  'one change per run from a common base',
  ['Each optimisation is added on its own to the same base (v500) and kept only if AllParticles '
   'does not drop.',
   'This isolates each ingredient, but it also forbids the greedy chain from keeping anything '
   'that only works in combination.',
   'A tenth run (v509) adds nothing at all, purely to measure the drift of the protocol.'],
  'ret_proto', 'Eight single-change runs, and the spread is mostly inside the noise.'),
 ('PART 3d', PURPLE, 'v501', 'B2 differentiable pruning',
  'base v500 · re-enable the soft mask, cut 0.85',
  ['B2 is the soft mask that aligns training with the hard cut used at inference.',
   'On this base it is re-enabled at the same cut the base itself was built with.',
   'So the run re-applies a mechanism that is already implicit in the starting weights.'],
  'ret_v501', '32.70 / 17.72 — no gain, as expected for a redundant change.'),
 ('PART 3d', PURPLE, 'v502', 'class-2 loss weight 2.0',
  'base v500 · sister-edge weight 2.0 in the LCAG loss',
  ['Class-2 (sister) edges are the rarest structural class and the glue of a chain.',
   'The change multiplies their contribution to the main classification loss by two.',
   'The base had already been rebalanced, which is what this run tests.'],
  'ret_v502', '33.27 / 18.01 — class-2 recall +0.3 pp: marginal.'),
 ('PART 3d', PURPLE, 'v503', 'chain hinge',
  'base v500 · penalty = max(0, 0.3 − confidence)',
  ['The hinge pays a penalty only while a chain edge is less than 0.3 confident.',
   'On a freshly trained model this pushes the rare classes towards confidence.',
   'But this base is already trained, and its chain edges are all well above 0.3.'],
  'ret_v503', '32.78 / 17.74 — the penalty is identically zero, so no gradient flows.'),
 ('PART 3d', PURPLE, 'v504', 'chain-CE on truth-chain edges',
  'base v500 · 4-class CE on the chain edges',
  ['Where the hinge only asked for confidence, chain-CE asks for the correct class: '
   'CE = −log p_true.',
   'It is paid only on truth-chain edges, so background edges cannot dilute its gradient.',
   'This directly attacks the class-0 dilution that the global loss suffers from.'],
  'ret_v504', '33.69 / 19.13 — +216 events: the only robust winner in the chain.'),
 ('PART 3d', PURPLE, 'v505', 'source head',
  'base v504 · predict the chain root',
  ['The source head regresses which node is the root of the chain, using Rumor Centrality '
   'computed from MC truth.',
   'It forces the node representation to encode where a chain starts.',
   'Added on top of the current best model in the chain, not on the original base.'],
  'ret_v505', '33.58 / 19.13 — class-1 +4.35 pp, class-2 −2.76 pp: they cancel.'),
 ('PART 3d', PURPLE, 'v506', 'mass head',
  'base v504 · regress log10 m_ππ on every tt edge',
  ['This is the head that gave the largest single gain in the campaign, but here it is added to '
   'the weak base rather than to v38.',
   'It supervises the invariant mass of each track pair, so sister relations become visible.',
   'The comparison is against v504, not against v47.'],
  'ret_v506', '33.20 / 19.23 — class-2 +4.08, but class-1 −0.74 drags AllParticles down.'),
 ('PART 3d', PURPLE, 'v507', 'struct head',
  'base v504 · regress BFS depth + RC, weight 0.3',
  ['Two targets per node: the depth in the truth tree (BFS layers) and the normalised '
   'Rumor Centrality.',
   'The low loss weight is deliberate — v48 had shown that stacking heads at full weight '
   'competes with the main task.',
   'It is the last accepted step of the greedy chain.'],
  'ret_v507', '34.02 / 19.62 — the best single addition in the chain (+0.68 AllParticles).'),
 ('PART 3d', PURPLE, 'v509', 'the empty control',
  'base v500 · nothing added at all, 20 epochs',
  ['Every other run in the chain starts from v500 and trains for 20 epochs.',
   'This one does exactly the same and changes nothing, so it measures how much the score moves '
   'on its own.',
   'Without it, none of the comparisons above can be read at the 0.1 pp level.'],
  'ret_v509', '33.27 / 17.98 against the base at 33.34 / 17.90 — zero drift.'),
 ('PART 3d', PURPLE, 'v530', 'B2 + source kept as a group',
  'base v500 · cumulative chain, no roll-back',
  ['The greedy chain drops any single change that does not pay immediately, which can cut '
   'synergies: B2 and the source head were both rejected on their own.',
   'The cumulative chain instead adds B2 and the source head as a group and keeps them whatever '
   'happens, exactly as the original v31 → v47 path did.',
   'This tests whether greedy roll-back was the reason the chain stalled.'],
  'ret_v530', '32.58 / 17.71 — no gain, so greedy roll-back is not the explanation.'),
 ('PART 3d', PURPLE, 'v549', 'mass + momentum on the best model',
  'base v507 · the two remaining heads stacked',
  ['v507 is the strongest model the ablation chain produced, so the question is whether it can '
   'be improved by more supervision.',
   'The mass and momentum heads are added together at reduced weight.',
   'This is the “does the sum beat the parts” test the single-change protocol cannot answer.'],
  'ret_v549', '33.92 / 19.47 against v507 at 34.02 / 19.62 — no gain.'),
 ('PART 3d', PURPLE, 'v540 / v545–548', 'context-aware pruning',
  'base v38 / v511 · a context-aware pruning score',
  ['The oracle said the pruning heads are the bottleneck, so this replaces the pruning score with '
   'a context-aware one.',
   'Each track and edge score is refined by aggregating its neighbourhood — a SAGPool/CRF-style '
   'update attached to the last GNN block.',
   'The first four attempts were dead on arrival: the residual layer and the gate were both '
   'zero-initialised, so the head could never leave its identity.'],
  'ret_v540', 'Four variants at 21.4–22.0 against 22.11 for attention: no gain.'),
 ]

A_PUB = [
 # ---- public
 ('PART 3e', GREEN, 'v27', 'start from the simple stack',
  'public sample · no B2, no chain losses, no heads',
  ['The public sample has no PID, and its class-2/class-3 edge counts differ from CERN by a '
   'factor of four to fourteen.',
   'This configuration is the plain stack: original reconstruction losses only, and a gentle '
   'learning rate.',
   'It is the reference the public line is built on.'],
  'pub_v27', '51.66 / 22.76 — the best public result, from the simplest stack.'),
 ('PART 3e', GREEN, 'v45', 'transfer the whole CERN stack',
  'public sample · the complete v38 recipe',
  ['The obvious move: take everything that works on CERN and train it on the public data.',
   'That includes B2 with cut 0.85, the class-2 weight, chain-CE and the source head.',
   'The job was interrupted at epoch 66, and the epoch-51 weights are the best it produced.'],
  'pub_v45', '19.70 / 8.06 against v27 at 51.66 / 22.76 — negative transfer.'),
 ('PART 3e', GREEN, 'v49', 'continue the transferred model',
  'base v45 · resumed to epoch 88',
  ['v45 was interrupted rather than finished, so it is resumed with the class-weight key fixed.',
   'The learning rate is left at 1e-3 and training runs on to epoch 88.',
   'This settles whether v45 was simply under-trained.'],
  'pub_v49', '14.15 / 5.95 — worse, so the recipe is the problem, not the budget.'),
 ('PART 3e', GREEN, 'v60 / v61', 'rebuild layer by layer instead',
  'base v27 · add one CERN ingredient at a time',
  ['If the whole stack transfers badly, the fix is to follow the CERN order instead: add one '
   'ingredient, check, then add the next.',
   'P1 adds B2 (cut 0.5) and the source head to v27.',
   'P2 (v61) then adds the class-2 weight and the hinge; that run is in flight.'],
  'pub_v60', '54.52 / 23.43 — above v27: rebuilding works where copying did not.'),
]


def chapter(part, title, sub, mapfig, chart, colour, caption, attempts):
    headfig(0, part, title, sub, mapfig, chart, colour, caption=caption)
    for a in attempts:
        chip, col, ver, ttl, sb, how, figkey, take = a
        attempt(0, chip, col, ver, ttl, sb, how, figkey, take)


chapter('PART 3a', 'Capacity is not what limits the score', 'Δ AllParticles against v38',
        'm_cap.png', 'd_capacity.png', CAPC,
        'Adding width or dimension costs 8–14 pp of AllParticles.', A_CAP)
chapter('PART 3b', 'Context helps only a plastic base',
        'Δ PerfectReco against the base it is grafted onto',
        'm_attn.png', 'd_attention.png', PINK,
        'Same module, opposite outcome — the base matters more than the module.', A_ATT)
chapter('PART 3c', 'Restructuring the event instead of the model',
        'Subgraph training vs the full-graph pass',
        'm_res.png', 'd_restructure.png', PLUM,
        'Training on subgraphs makes the full-graph forward pass worse.', A_RES)
chapter('PART 3d', 'Retraining the recipe', 'Δ AllParticles against the v500 base',
        'm_ret.png', 'd_retrain.png', PURPLE,
        'Only chain-CE clears the noise floor; the rest are flat.', A_RET)
chapter('PART 3e', 'The same question on another dataset', 'PerfectReco on the public sample',
        'm_pub.png', 'd_public.png', GREEN,
        'Public scores use N = 12 774 and are not comparable with the CERN numbers.', A_PUB)

# =========================================================== PART 4
s = head(0, 'PART 4', 'The diagnosis — what limits the score',
         'Three measurements that changed our reading',
         'm_diag.png', TEAL,
         caption='This is why we now work on the pruning heads, not on the network.',
         points=['With ideal pruning the same pipeline already reaches 99.8% AllParticles.',
                 'Nothing added after v47 beats it, and more compute does not help.',
                 'The point head carried 1/33 of the loss weight; the fine-tuning step was 10× too small.'])

s = new_slide(0, 'PART 4', 'First, the oracle test: what it is',
              'Rewrite the pruning decisions with the truth', chip_color=TEAL)
bullets(s, ['Question: how much of the score is decided before the network even runs?',
            'Method: take v47 frozen — never retrained — and replace the point and edge pruning '
            'decisions with MC truth.',
            'Only the input changes, so any difference is attributable to pruning.'],
        0.50, 1.24, 12.35, 1.9, size=BODY)
pic_fit(s, FIG + '/s_oracle.png', 0.40, 3.28, 12.5, 2.72, center_in=(0.45, 12.45))
note(s, 'Six variants, plus a reverse control that adds fake tracks. '
        'All% = P(survive the point prune) × P(recovered).', top=6.12, size=BODY)

s = split(0, 'PART 4', 'Then, the oracle result',
          'What each pruning axis is worth',
          ['Baseline (v47 at thr 0.9): AllParticles 39.56, PerfectReco 23.15.',
           'Fix point recall only — put back the truth tracks that were wrongly deleted: 54.09.',
           'Fix point precision only — remove the false positives, recall untouched: 54.68.',
           'Fix both point axes: 71.99. Fix only the edge decisions: 71.10. Fix everything: 99.75.',
           'Reverse control — add fake tracks instead: it collapses to 16.91, so the effect is real '
           'and signed.'],
          [FIG + '/oracle_ladder.png'], TEAL, fig_w=6.10)

s = split(0, 'PART 4', 'Second: the plateau',
          'The test: 16 runs after v47, and how to read them',
          ['The test: after v47, 16 more runs — capacity, attention, long generations, recombination.',
           'Only one even ties it. More compute does not help: v520 ran 179 epochs and ended 0.46 pp '
           'below, with a flat curve.',
           'More model does not help: the widening attempts cost 7–13 pp of AllParticles.',
           'Noise floor, measured: four functionally identical runs spread over ±0.43 pp / ±75 events '
           '— so anything under ~0.5 pp is not a result.'],
          [FIG + '/plateau.png'], TEAL, fig_w=5.60)

s = split(0, 'PART 4', 'Third: single changes vs combinations',
          'The test: take what works alone, then try every combination',
          ['The four that clear the noise floor on their own: chain-CE, edge-loss weight 33 → 3, '
           'attention with the edge bias, and learning rate 3e-4.',
           'Every combination of them failed — chain-CE + struct, mass + momentum, B2 + source, '
           'attention + mass, and the four winners stacked.',
           'Combining confirmed winners has never produced a gain here: a plateau, not bad ideas.',
           'Read together with the oracle, this says the headroom is in the pruning heads.'],
          [FIG + '/works_apart.png'], TEAL, fig_w=5.60)

s = split(0, 'PART 4', 'Fourth: the two concrete causes',
          'The tests: a learning-rate ladder, and an audit of the loss',
          ['The lr ladder: the same base and the same 20 epochs at 3e-5, 1e-4 and 3e-4.',
           'At 3e-5 nothing moves; at 3e-4 the score gains 5.9 pp and ties the best model of the '
           'campaign — so the model was never frozen, the step was just too small.',
           'The loss audit: the objective weights the edge head 33× and the point head 1×, and the '
           'config keys written to rebalance it were never read by the code.',
           'Both causes point at the same place — the point-pruning head.'],
          [FIG + '/lr_discovery.png', FIG + '/loss_weights.png'], TEAL, fig_w=5.00)

# =========================================================== PART 5
s = split(0, 'PART 5', 'What is running now',
          'The pruning-loss weights are config switches',
          ['v552 — point-loss weight 1 → 5: the point head is under-weighted.',
           'v553 — edge-loss weight 33 → 3: 22.84, already above v38 — the edge head dominated the objective.',
           'v554 — focal loss γ = 2 on both pruning heads, to push the hard examples (precision).',
           'v555 — a chain-level min-pooling recall loss: a chain dies at its weakest link.',
           'v556 — all confirmed levers stacked on v551 at lr 3e-4; target: break 40% AllParticles.',
           'v557–v561 — high-lr controls and a re-test of the B2 verdict.'],
          [FIG + '/running_arms.png'], GREEN, fig_w=5.40)

s = new_slide(0, 'SUMMARY', 'Summary and next steps', chip_color=GREEN)
bullets(s, ['What we now know',
            ('the main line v31 → v47 is real: PerfectReco 12.3 → 23.2%, AllParticles 22.5 → 39.6%', 1),
            ('nothing added after v47 beats it — not capacity, not attention, not longer training', 1),
            ('the reconstruction is not the bottleneck; the pruning classification is', 1),
            ('the point head is worth ~15 pp each way and carried 1/33 of the loss weight', 1)],
        0.55, 1.28, 12.2, 3.0, size=BODY)
bullets(s, ['Next',
            ('finish the pruning-reweighting arms and re-test the old verdicts at lr 3e-4', 1),
            ('if the point head responds, attack precision: hard negatives and chain-level losses', 1),
            ('continue the public line to see whether the levers transfer', 1)],
        0.55, 4.42, 12.2, 1.9, size=BODY)

# =========================================================== save + archive
prs.save(OUT)
print('[ok] saved: ' + OUT)
print('     slides: ' + str(len(prs.slides._sldIdLst)))

VDIR = FIG + '/versions'
os.makedirs(VDIR, exist_ok=True)
n = 1
while os.path.exists('%s/v%02d_DFEI_progress_20260915_EN.pptx' % (VDIR, n)):
    n += 1
for f in (os.path.abspath(__file__), BASE + '/make_meeting_maps_dot.py',
          BASE + '/make_meeting_20260915_figs2.py', BASE + '/make_attempt_figs.py'):
    tag = os.path.basename(f).replace('make_', '').replace('.py', '')
    shutil.copy(f, '%s/v%02d_%s.py' % (VDIR, n, tag))
shutil.copy(OUT, '%s/v%02d_DFEI_progress_20260915_EN.pptx' % (VDIR, n))
print('[ok] archived as v%02d in %s' % (n, VDIR))
