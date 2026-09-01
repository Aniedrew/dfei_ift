#!/usr/bin/env python3
"""DFEI group meeting deck v4 (English, formal): theme-based, larger fonts, full v38-line coverage.

v4 changes:
- body font 24pt / 22pt (larger), titles 38-48pt
- bug-fix details removed (one line only)
- Part 1 fully covers the v38-line: differentiable pruning with annealing, class-2 weighting,
  in-chain LCA consistency (hinge -> CE), all with the annealing/cut evolution
"""
import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

BASE = '/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn'
FIG = BASE + '/meeting_figs'
OUT = FIG + '/DFEI_progress_20260902_EN.pptx'
os.makedirs(FIG, exist_ok=True)

BLUE = RGBColor(0x1F, 0x4E, 0x79)
DARK = RGBColor(0x33, 0x33, 0x33)
GRAY = RGBColor(0x66, 0x66, 0x66)
LIGHT = RGBColor(0xEA, 0xF2, 0xF8)
RED = RGBColor(0xC0, 0x39, 0x2B)
GREEN = RGBColor(0x2C, 0xA0, 0x2C)
FONT = 'Calibri'
BODY = 24          # 正文
BODY_SM = 22       # 次要
TITLE = 38
MAIN_TITLE = 48

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


def add_title_bar(slide, text, sub=None):
    tb = slide.shapes.add_textbox(Inches(0.55), Inches(0.18), Inches(12.2), Inches(1.0))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = text
    r.font.size = Pt(TITLE)
    r.font.bold = True
    r.font.color.rgb = BLUE
    r.font.name = FONT
    if sub:
        p2 = tf.add_paragraph()
        r2 = p2.add_run()
        r2.text = sub
        r2.font.size = Pt(17)
        r2.font.color.rgb = GRAY
        r2.font.name = FONT
    ln = slide.shapes.add_shape(1, Inches(0.6), Inches(1.3), Inches(12.1), Pt(3))
    ln.fill.solid()
    ln.fill.fore_color.rgb = BLUE
    ln.line.fill.background()


def add_bullets(slide, items, left, top, width, height, size=BODY, color=None):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(10)
        if isinstance(it, tuple):
            txt, lvl = it
        else:
            txt, lvl = it, 0
        r = p.add_run()
        r.text = ('    ' * lvl) + ('•  ' if lvl == 0 else '–  ') + txt
        r.font.size = Pt(size - 2 * lvl)
        r.font.color.rgb = color or DARK
        r.font.name = FONT
    return tb


def add_table(slide, data, left, top, width, height, col_widths=None, font_size=20, header_fill=BLUE):
    rows, cols = len(data), len(data[0])
    tbl = slide.shapes.add_table(rows, cols, left, top, width, height).table
    if col_widths:
        for j, w in enumerate(col_widths):
            tbl.columns[j].width = w
    for i in range(rows):
        for j in range(cols):
            cell = tbl.cell(i, j)
            cell.text = str(data[i][j])
            cell.margin_left = Inches(0.06)
            cell.margin_right = Inches(0.06)
            cell.margin_top = Inches(0.02)
            cell.margin_bottom = Inches(0.02)
            for p in cell.text_frame.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(font_size)
                    r.font.name = FONT
                    r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF) if i == 0 else DARK
                p.alignment = PP_ALIGN.CENTER if i == 0 else PP_ALIGN.LEFT
            if i == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = header_fill
            elif i % 2 == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = LIGHT
    return tbl


def add_pic(slide, path, left, top, width=None, height=None):
    kw = {}
    if width:
        kw['width'] = width
    if height:
        kw['height'] = height
    if os.path.exists(path):
        slide.shapes.add_picture(path, left, top, **kw)
        return True
    print('[warn] missing fig:', path)
    return False


def _rgb(c):
    """Accept RGBColor or '#RRGGBB' string."""
    if isinstance(c, RGBColor):
        return c
    h = c.lstrip('#')
    return RGBColor(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def add_flow_box(slide, x, y, w, h, text, fill=LIGHT, line=BLUE, size=10, bold=False):
    """Native rounded-rectangle box for the flow chart (adjustable in PowerPoint)."""
    fill, line = _rgb(fill), _rgb(line)
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb = fill
    sh.line.color.rgb = line; sh.line.width = Pt(1.6)
    tf = sh.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.03)
    tf.margin_top = tf.margin_bottom = Inches(0.01)
    for i, ln in enumerate(text.split('\n')):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run(); r.text = ln
        r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = DARK; r.font.name = FONT
    return sh


def add_flow_arrow(slide, shape='right', x=0.0, y=0.0, w=0.3, h=0.3, color=BLUE):
    """Native block arrow (right/down/left/up), adjustable in PowerPoint."""
    m = {'right': MSO_SHAPE.RIGHT_ARROW, 'down': MSO_SHAPE.DOWN_ARROW,
         'left': MSO_SHAPE.LEFT_ARROW, 'up': MSO_SHAPE.UP_ARROW}
    sh = slide.shapes.add_shape(m[shape], Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb = color
    sh.line.fill.background()
    return sh


def add_flow_text(slide, x, y, w, h, text, size=9, color=GRAY, italic=False):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = text
    r.font.size = Pt(size); r.font.color.rgb = color
    r.font.name = FONT; r.font.italic = italic
    return tb


def add_footer(slide, idx):
    tb = slide.shapes.add_textbox(Inches(0.55), Inches(7.12), Inches(12.2), Inches(0.3))
    tf = tb.text_frame
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = 'Qingxiang Guo · UCAS · DFEI group meeting      ' + str(idx)
    r.font.size = Pt(10)
    r.font.color.rgb = GRAY
    r.font.name = FONT


def new_slide(idx):
    s = prs.slides.add_slide(BLANK)
    add_footer(s, idx)
    return s


# ============ S1 Title ============
s = new_slide(1)
tb = s.shapes.add_textbox(Inches(0.8), Inches(2.0), Inches(11.7), Inches(2.0))
tf = tb.text_frame; tf.word_wrap = True
r = tf.paragraphs[0].add_run()
r.text = 'Optimizing DFEI on CERN Monte Carlo'
r.font.size = Pt(MAIN_TITLE); r.font.bold = True; r.font.color.rgb = BLUE; r.font.name = FONT
tb = s.shapes.add_textbox(Inches(0.8), Inches(4.2), Inches(11.7), Inches(1.2))
tf = tb.text_frame; tf.word_wrap = True
r = tf.paragraphs[0].add_run()
r.text = 'A progress report on the optimization attempts, organized by theme'
r.font.size = Pt(22); r.font.color.rgb = DARK; r.font.name = FONT
tb = s.shapes.add_textbox(Inches(0.8), Inches(5.6), Inches(11.7), Inches(0.9))
tf = tb.text_frame
r = tf.paragraphs[0].add_run()
r.text = 'Qingxiang Guo · University of Chinese Academy of Sciences · 2026-09-02'
r.font.size = Pt(16); r.font.color.rgb = GRAY; r.font.name = FONT

# ============ S2 Data + overall effect ============
s = new_slide(2)
add_title_bar(s, 'Data and Overall Effect', 'CERN MC: /eos/lhcb/user/y/yukaiz/DFEI_IFT_20260702 · threshold 0.9 · 20 test files')
add_bullets(s, [
    'All work uses the CERN official Monte Carlo production (DFEI_IFT_20260702)',
    'Since fixing a silent class-weight bug, the line has raised',
    ('PerfectReco 23.9% → 32.7%   (+8.8pp)', 1),
    ('AllParticles 43.4% → 55.9%   (+12.5pp)', 1),
], Inches(0.7), Inches(1.6), Inches(6.0), Inches(4.6), size=BODY_SM)
add_pic(s, FIG + '/progress_line_v31_v47.png', Inches(7.0), Inches(2.3), width=Inches(5.9))

# ============ S3 Outline ============
s = new_slide(3)
add_title_bar(s, 'Outline', 'Organized by theme, not by chronology')
add_bullets(s, [
    'Part 1 — Differentiable Pruning with Annealing',
    ('training–inference alignment: soft mask w_eff = w·σ((w−cut)/τ), τ annealed', 1),
    'Part 2 — Rewarding the Chain',
    ('why chains die · the rewards (hinge, chain-CE) · evidence it works', 1),
    'Part 3 — Supervising Representations with Physics',
    ('linear probes first · the head zoo · depth & RC', 1),
    'Part 4 — Controlled Failures and Their Lessons',
    'Part 5 — Ongoing Attempts',
], Inches(0.9), Inches(1.7), Inches(11.5), Inches(4.8), size=BODY)

# ============ S4 Part 1: differentiable pruning ============
s = new_slide(4)
add_title_bar(s, 'Part 1 — Differentiable Pruning with Annealing', 'Align training with the pruned graphs seen at inference')
add_bullets(s, [
    'Problem: training sees the full graph; inference prunes with hard thresholds first',
    'Fix: soft mask w_eff = w · σ((w − cut)/τ), temperature τ annealed 1.0 → 0.1',
    ('large τ → smooth mask, stable gradients; small τ → approximates the hard threshold', 1),
    'Cut aligned to inference and tightened over versions: 0.5 (v36) → 0.7 (v37) → 0.85 (v38) → 0.9 (inference)',
], Inches(0.8), Inches(1.35), Inches(11.8), Inches(2.6), size=BODY_SM)
add_pic(s, FIG + '/pruning_sigmoid_tau.png', Inches(0.7), Inches(3.95), width=Inches(6.3))
# ---- native flow chart (adjustable shapes) ----
add_flow_box(s, 7.2, 4.0, 1.5, 0.85, 'GNN block\n(node/edge\nreprs)', size=9)
add_flow_arrow(s, 'right', 8.76, 4.22, 0.32, 0.4)
add_flow_box(s, 9.12, 4.0, 1.5, 0.85, 'Weight MLP\n(sigmoid head)', size=9)
add_flow_arrow(s, 'right', 10.68, 4.22, 0.32, 0.4)
add_flow_box(s, 11.05, 4.0, 1.0, 0.85, 'w ∈ [0,1]', size=10, bold=True)
add_flow_arrow(s, 'down', 11.42, 4.88, 0.3, 0.42)
add_flow_box(s, 9.95, 5.32, 3.1, 1.05, 'Soft mask\nw_eff = w·σ((w−cut)/τ)\ncut fixed per run (0.85)\nτ annealed: 1.0 → 0.1',
             fill='#FDEBD0', line=RED, size=9, bold=True)
add_flow_text(s, 7.2, 5.42, 2.7, 0.8, 'within a run:\ncut fixed, τ anneals\n→ same cut, sharper enforcement', size=8, color=RED)
add_flow_arrow(s, 'right', 9.9, 5.72, 0.5, 0.25, color=RED)
add_flow_arrow(s, 'left', 9.4, 5.9, 0.55, 0.25, color=GREEN)
add_flow_box(s, 6.9, 6.0, 2.4, 0.8, 'weighted message passing\n(back into the GN blocks)', fill=LIGHT, line=GREEN, size=9)
add_flow_arrow(s, 'down', 11.6, 6.4, 0.3, 0.4, color=BLUE)
add_flow_box(s, 10.3, 6.75, 2.7, 0.4, 'pruning loss vs truth (BCE)\n(mask makes it differentiable)', fill=LIGHT, line=BLUE, size=8)
add_flow_text(s, 7.2, 6.9, 2.6, 0.35, 'inference (no mask): keep if w ≥ 0.9', size=9, color=GRAY, italic=True)

# ============ S5 Part 2: why chains die (1/2): class imbalance ============
s = new_slide(5)
add_title_bar(s, 'Part 2 — Why Chains Die (1/2): The Class Imbalance', 'Structural edges are ~0.1% and the hardest to classify')
add_bullets(s, [
    'class 0 (background) is 99.9% of edges; each structural class is ~0.04%',
    'The GNN is weakest exactly on the rarest classes — especially class 2 (sister)',
    'Per-class accuracy: worst exactly on the rare structural classes',
], Inches(0.7), Inches(1.7), Inches(6.0), Inches(4.6), size=BODY_SM)
add_pic(s, FIG + '/chain_lca_imbalance_dist.png', Inches(7.0), Inches(1.5), width=Inches(5.9))
add_pic(s, FIG + '/chain_lca_imbalance_acc.png', Inches(7.0), Inches(4.2), width=Inches(5.9))
add_flow_text(s, 0.8, 6.85, 11.8, 0.4,
              'Implication: without extra supervision, the classifier barely learns classes 1/2/3 — the rewards exist to fix this',
              size=12, color=GRAY, italic=True)

# ============ S6 Part 2: why chains die (2/2): one misclassified edge ============
s = new_slide(6)
add_title_bar(s, 'Part 2 — Why Chains Die (2/2): One Misclassified Edge', 'Chain survival needs every structural edge correct')
add_bullets(s, [
    'At inference, an edge the classifier calls class 0 is pruned',
    'If ANY structural edge of a chain is misclassified, the whole chain dies',
    'The rewards (next slides) supervise truth-chain edge classes directly → chains survive',
], Inches(0.7), Inches(1.7), Inches(6.0), Inches(4.6), size=BODY_SM)
add_pic(s, FIG + '/chain_lca_before.png', Inches(7.0), Inches(1.5), width=Inches(5.9))
add_pic(s, FIG + '/chain_lca_after.png', Inches(7.0), Inches(4.15), width=Inches(5.9))
add_flow_text(s, 0.8, 6.85, 11.8, 0.4,
              'Implication: chain survival depends on every structural edge being classified correctly',
              size=12, color=GRAY, italic=True)

# ============ S7 Part 2: root cause — class-0 dilution ============
s = new_slide(7)
add_title_bar(s, 'Part 2 — The Root Cause: Class-0 Dilution', 'Why the standard classifier fails on chain structure')
add_bullets(s, [
    'class 0 (background) is 99.9% of edges; structural classes 1/2/3 together are ~0.1%',
    'The main LCAG loss is a global cross-entropy over every edge → its gradient is ~99.9% "predict background"',
    'Consequence: rare structural edges are undertrained → misclassified → chains break',
    'The fix: reward the chain — class-2 weighting (3.0 → 2.0), then hinge (v37), then chain-CE (v38)',
], Inches(0.7), Inches(1.5), Inches(6.0), Inches(5.2), size=BODY_SM)
add_pic(s, FIG + '/chain_lca_dilution.png', Inches(7.0), Inches(2.6), width=Inches(5.9))
add_flow_text(s, 0.8, 6.85, 11.8, 0.4,
              'Implication: with ~1000 background edges per structural edge, the global CE alone cannot teach the model classes 1/2/3',
              size=12, color=GRAY, italic=True)

# ============ S8 Part 2: Reward 1 (v37): hinge ============
s = new_slide(8)
add_title_bar(s, 'Reward 1 (v37): Hinge — Confidence Bonus', 'A training-only reward that pays out when chain edges are confidently classified')
add_bullets(s, [
    'It is a loss term that behaves like a reward: be confident on a chain edge → pay nothing',
    'Confidence comes from the LCAG head: per-edge 4-class softmax (p0…p3), confidence = max(pk)',
    'Penalty = max(0, margin − confidence), margin 0.3 → zero once confidence ≥ 0.3',
    'Paid ONLY on truth-chain edges — the true chains are known from MC truth during training',
    'Role: chain edges give the global CE almost no gradient → hinge keeps them confident; CE (next) fixes "which class"',
], Inches(0.7), Inches(1.5), Inches(6.0), Inches(5.3), size=BODY_SM)
add_pic(s, FIG + '/chain_lca_hinge_mechanism.png', Inches(7.0), Inches(1.5), width=Inches(5.9))
add_pic(s, FIG + '/chain_lca_hinge_curve.png', Inches(7.0), Inches(4.15), width=Inches(5.9))
add_flow_text(s, 0.8, 6.9, 11.8, 0.4,
              'Implication: hinge is a training-only confidence bonus on chain edges — confidence = max(pk), not the pruning weight w',
              size=12, color=GRAY, italic=True)

# ============ S9 Part 2: Reward 2 (v38): chain cross-entropy ============
s = new_slide(9)
add_title_bar(s, 'Reward 2 (v38): Chain-CE — Correctness Bonus', 'A training-only reward for being the RIGHT class, on chain edges')
add_bullets(s, [
    'CE = −log(p_true): p_true is the probability the model assigns to the edge’s TRUE class',
    'Right and confident (p_true → 1) → penalty ≈ 0; unsure or wrong → penalty large',
    'Chain-CE (v38): added ON TOP of the hinge, paid on truth-chain edges (classes 1/2/3) → direct, undiluted gradient',
    'v37 → v38 (weight 2.0 + chain-CE + cut 0.85): PerfectReco 27.3% → 29.3%',
    'Net effect of Part 2 (v31 → v38): 23.9% → 29.3%',
], Inches(0.7), Inches(1.5), Inches(6.0), Inches(5.3), size=BODY_SM)
add_pic(s, FIG + '/chain_lca_ce_curve.png', Inches(7.0), Inches(1.5), width=Inches(5.9))
add_pic(s, FIG + '/chain_lca_ce_where.png', Inches(7.0), Inches(4.15), width=Inches(5.9))
add_flow_text(s, 0.8, 6.9, 11.8, 0.4,
              'Implication: same function as the global CE — the reward is WHERE it is paid (chain edges only)',
              size=12, color=GRAY, italic=True)

# ============ S10 Part 2: the strategy works ============
s = new_slide(10)
add_title_bar(s, 'Part 2 — The Strategy Works', 'Rewards raise PerfectReco and per-class accuracy')
add_bullets(s, [
    'Reward line builds on Part 1 (pruning, v36: 26.3%)',
    ('hinge (v37): 26.3% → 27.3%', 1),
    ('chain-CE (v38): 27.3% → 29.3%', 1),
    'Per-class accuracy (v31 → v38): class1 67.8 → 76.8; class2 41.3 → 47.9',
    'Same evaluation: CERN MC, thr 0.9, 20 test files',
], Inches(0.7), Inches(1.6), Inches(6.0), Inches(4.6), size=BODY_SM)
add_pic(s, FIG + '/reward_results.png', Inches(7.0), Inches(2.4), width=Inches(5.9))
add_flow_text(s, 0.8, 6.85, 11.8, 0.4,
              'Implication: the rewards directly improve the structural classes the global CE ignores — chains survive more often',
              size=12, color=GRAY, italic=True)

# ============ S11 Part 3: linear probes (physics is invisible) ============
s = new_slide(11)
add_title_bar(s, 'Part 3 — First, Check: Physics is Invisible', 'Linear probes show what the backbone does NOT encode')
add_bullets(s, [
    'Probe: freeze the backbone, fit one linear layer, try to read a physical quantity from the representation (top right)',
    'If the quantity is not in the representation, the probe fails: R² ≈ 0',
    'Before any physics head: mass R² = 0.003, momentum R² ≈ 0 — physics is lost',
    'So we added heads that supervise physics directly (head zoo, next)',
    'After mass supervision (masshead2): edge mass R² = 0.930; PerfectReco 29.3 → 32.7, class2 44.7 → 51.1',
], Inches(0.7), Inches(1.5), Inches(6.0), Inches(5.3), size=BODY_SM)
add_pic(s, FIG + '/probe_method.png', Inches(7.0), Inches(1.5), width=Inches(5.9))
add_pic(s, FIG + '/probe_r2.png', Inches(7.0), Inches(4.15), width=Inches(5.9))
add_flow_text(s, 0.8, 6.9, 11.8, 0.4,
              'Implication: the backbone does not encode physics by itself — it must be supervised into the representation',
              size=12, color=GRAY, italic=True)

# ============ S12 Part 3: head zoo (native shapes) ============
s = new_slide(12)
add_title_bar(s, 'Part 3 — The Head Zoo', 'One backbone, many supervised targets — new heads in green')
add_bullets(s, [
    'Original heads (blue): the reconstruction machinery — classify, prune, assign',
    'New heads (green): supervise physical / structural quantities directly',
    ('source (v36) — chain root · mass — log10 m_ππ · struct — depth + RC · mom — momentum', 1),
    'Same backbone, different targets: supervision writes the physics into the representation',
], Inches(0.7), Inches(1.5), Inches(6.0), Inches(5.0), size=BODY_SM)
add_flow_box(s, 6.95, 1.7, 1.35, 4.7, 'shared GNN\nbackbone\n\nnode + edge\nrepresentations', size=9, bold=True)


def _hz(x, y, name, target, green):
    add_flow_box(s, x, y, 1.55, 1.25, name + '\n' + target,
                 fill='#EAFAF1' if green else LIGHT,
                 line=GREEN if green else BLUE, size=8, bold=True)


_hz(8.65, 5.3, 'LCAG', 'edge class 0-3', False)
_hz(10.4, 5.3, 'node prune', 'keep/remove node', False)
_hz(12.15, 5.3, 'edge prune', 'keep/remove edge', False)
_hz(8.65, 3.55, 'source', 'chain root (RC)', True)
_hz(10.4, 3.55, 'mass', 'log10 m_ππ', True)
_hz(12.15, 3.55, 'struct', 'depth + RC', True)
_hz(8.65, 1.8, 'mom', 'normalized p', True)
_hz(10.4, 1.8, 'PV asso', 'PV assignment', False)
_hz(12.15, 1.8, 'chain scorer', 'chain conf. (planned)', False)
for yc in (5.925, 4.175, 2.425):
    add_flow_arrow(s, 'right', 8.32, yc - 0.13, 0.26, 0.26)
    add_flow_arrow(s, 'right', 10.22, yc - 0.13, 0.12, 0.26)
    add_flow_arrow(s, 'right', 11.97, yc - 0.13, 0.12, 0.26)
add_flow_text(s, 8.6, 6.55, 2.2, 0.3, 'green = new physics heads', size=9, color=DARK)
add_flow_text(s, 11.0, 6.55, 2.2, 0.3, 'blue = original heads', size=9, color=DARK)

# ============ S13 Part 3: struct head — depth ============
s = new_slide(13)
add_title_bar(s, 'Part 3 — Structure Head: Depth', 'How the "where in the tree" target is computed')
add_bullets(s, [
    'Target: how far a node sits from the chain root (the B candidate)',
    'Computed by BFS on the truth chain: root d=0 → children d=1 → grandchildren d=2 (right)',
    'The struct head regresses this depth from the node representation',
    'Effect (5-file ablation): +struct → All 51.4 → 54.5, Perfect 29.7 → 30.5; class3 60.8 → 63.2',
], Inches(0.7), Inches(1.5), Inches(6.0), Inches(4.8), size=BODY_SM)
add_pic(s, FIG + '/depth_calc.png', Inches(7.0), Inches(2.2), width=Inches(5.9))
add_flow_text(s, 0.8, 6.85, 11.8, 0.4,
              'Implication: depth labels the node’s role in the chain — supervising it forces the representation to encode tree position',
              size=12, color=GRAY, italic=True)

# ============ S14 Part 3: struct head — rumor centrality ============
s = new_slide(14)
add_title_bar(s, 'Part 3 — Structure Head: Rumor Centrality', 'How the "is this the root?" score is computed')
add_bullets(s, [
    'RC answers: which node is the source of the chain? (Shah & Zaman)',
    'For each candidate root v: root the tree there, measure subtree sizes τ(u)',
    'log R(v) = −Σ log τ(u); the root maximizes it (right: B wins)',
    'Struct head: regress the normalized RC value; source head (v36): predict the argmax root',
], Inches(0.7), Inches(1.5), Inches(6.0), Inches(4.8), size=BODY_SM)
add_pic(s, FIG + '/rc_calc.png', Inches(7.0), Inches(2.2), width=Inches(5.9))
add_flow_text(s, 0.8, 6.85, 11.8, 0.4,
              'Implication: RC gives a physics-motivated "root-ness" target — no manual labels needed',
              size=12, color=GRAY, italic=True)

# ============ S15 Part 4: controlled failures ============
s = new_slide(15)
add_title_bar(s, 'Part 4 — Controlled Failures and Their Lessons', 'Two failed lines, two lessons')
add_bullets(s, [
    'PV subgraph training (v39-42): training on per-PV subgraphs while inference runs on the full graph',
    ('full-graph ability degraded: class1 76.8% → 56.4%; line closed', 1),
    'Combined mass + struct + mom (v48): aux losses (0.877) exceed the main task (0.559)',
    ('reconstruction dropped 5pp although LCAG did not — backbone pulled toward auxiliary tasks', 1),
    ('resolution: lower aux weights — mom 0.2, struct 0.3', 1),
    'Lessons: train/infer graph mismatch is fatal; gradient balance must be explicit',
], Inches(0.7), Inches(1.5), Inches(6.0), Inches(5.4), size=BODY_SM)
add_pic(s, FIG + '/v48_failure.png', Inches(7.0), Inches(2.2), width=Inches(5.9))
add_flow_text(s, 0.8, 6.85, 11.8, 0.4,
              'Implication: the v48 failure shows why the head zoo needs explicit gradient balance — and why we ablate before stacking',
              size=12, color=GRAY, italic=True)

# ============ S16 Part 5: ongoing ============
s = new_slide(16)
add_title_bar(s, 'Part 5 — Ongoing Attempts')
add_bullets(s, [
    'Wider latent space (v53): tracks nodes 32-dim, tt edges 24-dim — 16-dim sits at the physical-DOF lower bound',
    ('from-scratch training interrupted at ep74/150, not converged; resume planned', 1),
    'Public-data verification (v45/v49): no PID, class2/3 counts differ 4-14×; resume best val 33.4 @ep73',
    'Learned deterministic annealing: inference-side PV clustering with learned affinity (no subgraph training); core module CPU-verified',
    'Chain scoring for trigger assistance: criteria AUC 0.90 / 0.78 (realistic); training ready, needs GPU',
], Inches(0.8), Inches(1.4), Inches(11.8), Inches(5.6), size=BODY_SM)

# ============ S17 next + questions ============
s = new_slide(17)
add_title_bar(s, 'Next Steps and Open Questions')
add_bullets(s, [
    'Resume the wider-latent training to 150 ep; layer struct + mom at reduced weights',
    'Run queued evaluations; train the chain scorer; plug GNN affinity into learned DA',
    'Open questions:',
    ('trigger assistance vs full-event reconstruction — which metric would the collaboration trust?', 1),
    ('is same-source (class2) clustering the real problem DFEI should solve?', 1),
    ('HLT2 / Upgrade-II time budget for a lightweight GNN per event?', 1),
], Inches(0.8), Inches(1.4), Inches(11.8), Inches(5.4), size=BODY_SM)

prs.save(OUT)
print('[ok] 已保存: ' + OUT)
print('     共 ' + str(len(prs.slides._sldIdLst)) + ' 页')
