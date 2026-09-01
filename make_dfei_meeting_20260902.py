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
V47 = BASE + '/LHCb_logs/DFEI/version_47/plots_inclusive_00342442__v38_masshead2/edges'
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
    'Part 1 — Supervision and training–inference alignment',
    ('differentiable pruning with annealing · class-2 weighting · in-chain LCA consistency', 1),
    'Part 2 — Supervising representations with physics (source / mass / struct / mom heads)',
    'Part 3 — Controlled failures and their lessons',
    'Part 4 — Ongoing attempts (wider latent space, and others)',
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

# ============ S5 The problem: class-0 dilution ============
s = new_slide(5)
add_title_bar(s, 'Part 1 — The Problem: Class-0 Dilution', 'Why the standard classifier fails on chain structure')
add_bullets(s, [
    'class 0 (background) is 99.9% of edges; structural classes 1/2/3 together are ~0.1%',
    'The main LCAG loss is a global cross-entropy over every edge → its gradient is ~99.9% "predict background"',
    'Consequence: rare structural edges are undertrained → misclassified → chains break',
    'Fixes in Part 1: class-2 weighting (3.0 → 2.0), then hinge (v37), then chain-CE (v38)',
], Inches(0.7), Inches(1.5), Inches(6.0), Inches(5.2), size=BODY_SM)
add_pic(s, FIG + '/chain_lca_dilution.png', Inches(7.0), Inches(2.6), width=Inches(5.9))
add_flow_text(s, 0.8, 6.85, 11.8, 0.4,
              'Implication: with ~1000 background edges per structural edge, the global CE alone cannot teach the model classes 1/2/3',
              size=12, color=GRAY, italic=True)

# ============ S6 Reward 1 (v37): hinge ============
s = new_slide(6)
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

# ============ S7 Reward 2 (v38): chain cross-entropy ============
s = new_slide(7)
add_title_bar(s, 'Reward 2 (v38): Chain-CE — Correctness Bonus', 'A training-only reward for being the RIGHT class, on chain edges')
add_bullets(s, [
    'CE = −log(p_true): p_true is the probability the model assigns to the edge’s TRUE class',
    'Right and confident (p_true → 1) → penalty ≈ 0; unsure or wrong → penalty large',
    'Chain-CE (v38): added ON TOP of the hinge, paid on truth-chain edges (classes 1/2/3) → direct, undiluted gradient',
    'v37 → v38 (weight 2.0 + chain-CE + cut 0.85): PerfectReco 27.3% → 29.3%',
    'Net effect of Part 1 (v31 → v38): 23.9% → 29.3%',
], Inches(0.7), Inches(1.5), Inches(6.0), Inches(5.3), size=BODY_SM)
add_pic(s, FIG + '/chain_lca_ce_curve.png', Inches(7.0), Inches(1.5), width=Inches(5.9))
add_pic(s, FIG + '/chain_lca_ce_where.png', Inches(7.0), Inches(4.15), width=Inches(5.9))
add_flow_text(s, 0.8, 6.9, 11.8, 0.4,
              'Implication: same function as the global CE — the reward is WHERE it is paid (chain edges only)',
              size=12, color=GRAY, italic=True)

# ============ S8 Why chains die (1/2): class imbalance ============
s = new_slide(8)
add_title_bar(s, 'Why Chains Die (1/2): The Class Imbalance', 'Structural edges are ~0.1% and the hardest to classify')
add_bullets(s, [
    'class 0 (background) is 99.9% of edges; each structural class is ~0.04%',
    'The GNN is weakest exactly on the rarest classes — especially class 2 (sister)',
    'Per-class accuracy: worst exactly on the rare structural classes',
], Inches(0.7), Inches(1.7), Inches(6.0), Inches(4.6), size=BODY_SM)
add_pic(s, FIG + '/chain_lca_imbalance_dist.png', Inches(7.0), Inches(1.5), width=Inches(5.9))
add_pic(s, FIG + '/chain_lca_imbalance_acc.png', Inches(7.0), Inches(4.2), width=Inches(5.9))
add_flow_text(s, 0.8, 6.85, 11.8, 0.4,
              'Implication: without extra supervision, the classifier barely learns classes 1/2/3 — the chain losses exist to fix this',
              size=12, color=GRAY, italic=True)

# ============ S9 Why chains die (2/2): one misclassified edge ============
s = new_slide(9)
add_title_bar(s, 'Why Chains Die (2/2): One Misclassified Edge', 'Chain-CE keeps every structural edge correct')
add_bullets(s, [
    'At inference, an edge the classifier calls class 0 is pruned',
    'If ANY structural edge of a chain is misclassified, the whole chain dies',
    'Chain-CE (v38) supervises truth-chain edge classes directly → chains survive',
], Inches(0.7), Inches(1.7), Inches(6.0), Inches(4.6), size=BODY_SM)
add_pic(s, FIG + '/chain_lca_before.png', Inches(7.0), Inches(1.5), width=Inches(5.9))
add_pic(s, FIG + '/chain_lca_after.png', Inches(7.0), Inches(4.15), width=Inches(5.9))
add_flow_text(s, 0.8, 6.85, 11.8, 0.4,
              'Implication: chain survival depends on every structural edge being classified correctly — that is what the hinge + chain-CE provide',
              size=12, color=GRAY, italic=True)

# ============ S10 Part 2: physics supervision concept ============
s = new_slide(10)
add_title_bar(s, 'Part 2 — Supervising Representations with Physics', 'One common idea behind several heads')
add_bullets(s, [
    'Idea: physical / structural quantities are data-intrinsic; supervise the representation with them, keeping the model end-to-end',
    'Heads sharing this theme:',
    ('source head (v36) — Rumor-Centrality root; early instance of the same idea', 1),
    ('mass head — edge-level log10(m_ππ)', 1),
    ('structure head — node depth + RC value', 1),
    ('momentum head — node normalized momentum', 1),
    'Verification: linear probes on the frozen backbone (Ridge regression)',
], Inches(0.8), Inches(1.4), Inches(11.8), Inches(5.4), size=BODY_SM)

# ============ S11 mass head ============
s = new_slide(11)
add_title_bar(s, 'The Mass Head: Main Result', 'Edge-level regression of log10(m_ππ)')
add_bullets(s, [
    'Same-mother track pairs sit near resonance masses → the edge representation must encode sister relations',
    'Setup: SmoothL1 on log10(m_MeV), sentinel edges masked, weight 1.0',
    'v38 → masshead2 (20 test files):',
    ('PerfectReco 29.3% → 32.7%  (+3.4pp)', 1),
    ('AllParticles 52.1% → 55.9%  (+3.8pp)', 1),
    ('LCAG class2 44.7% → 51.1%  (+6.4pp)', 1),
], Inches(0.8), Inches(1.3), Inches(6.6), Inches(5.6), size=BODY_SM)
add_pic(s, V47 + '/NN_edges_2_roc.png', Inches(7.6), Inches(1.7), width=Inches(5.2))

# ============ S12 struct + mom ============
s = new_slide(12)
add_title_bar(s, 'Structure and Momentum Heads', 'Node-level supervision of tree position and momentum')
add_table(s, [
    ['Head', 'Target', 'Ablation (5-file eval)', 'Notes'],
    ['baseline', 'mass only', 'All 51.4 / Perfect 29.7', ''],
    ['+ struct', 'depth + RC', 'All 54.5 (+3.1) / Perfect 30.5 (+0.8)', 'best single head'],
    ['+ mom', 'normalized p', 'All 53.8 (+2.4) / Perfect 30.5 (+0.8)', 'fixes node R²≈0'],
], Inches(0.8), Inches(1.5), Inches(11.8), Inches(2.8), font_size=18)
add_bullets(s, [
    'Struct: depth (BFS to chain centroid) + RC value — node position in the tree; class3 60.8→63.2',
    'Mom: motivated by a probe — node representations were linearly unreadable for momentum (R² ≈ 0)',
    'Lesson: probe first, find the missing quantity, then supervise it',
], Inches(0.8), Inches(4.5), Inches(11.8), Inches(2.6), size=BODY_SM)

# ============ S13 probe ============
s = new_slide(13)
add_title_bar(s, 'Verification: Linear Probes on the Frozen Backbone', 'Ridge regression; R² of the physical quantity')
add_table(s, [
    ['Probe', 'v38', 'masshead2'],
    ['edge repr → log10(m_ππ)', 'R² = 0.003', 'R² = 0.930'],
    ['node repr → p (px/py/pz)', 'R² ≈ 0', 'R² ≈ 0'],
], Inches(0.8), Inches(1.5), Inches(8.5), Inches(1.8), font_size=20)
add_bullets(s, [
    'Mass supervision moves mass information into the edge representation — confirmed end-to-end',
    'Nodes remain unreadable for momentum → addressed by the momentum head (to be re-probed)',
], Inches(0.8), Inches(3.7), Inches(11.8), Inches(2.8), size=BODY_SM)

# ============ S14 controlled failures ============
s = new_slide(14)
add_title_bar(s, 'Part 3 — Controlled Failures and Their Lessons')
add_bullets(s, [
    'PV subgraph training (v39-42): training on per-PV subgraphs while inference runs on the full graph',
    ('full-graph ability degraded: class1 76.8% → 56.4%; line closed', 1),
    'Combined mass + struct + mom (v48): aux losses (0.877) exceed the main task (0.559)',
    ('reconstruction dropped 5pp although LCAG did not — backbone pulled toward auxiliary tasks', 1),
    ('resolution: lower aux weights — mom 0.2, struct 0.3', 1),
    'Lessons: train/infer graph mismatch is fatal; gradient balance between heads must be explicit',
], Inches(0.8), Inches(1.4), Inches(11.8), Inches(5.4), size=BODY_SM)

# ============ S15 ongoing ============
s = new_slide(15)
add_title_bar(s, 'Part 4 — Ongoing Attempts')
add_bullets(s, [
    'Wider latent space (v53): tracks nodes 32-dim, tt edges 24-dim — 16-dim sits at the physical-DOF lower bound',
    ('from-scratch training interrupted at ep74/150, not converged; resume planned', 1),
    'Public-data verification (v45/v49): no PID, class2/3 counts differ 4-14×; resume best val 33.4 @ep73',
    'Learned deterministic annealing: inference-side PV clustering with learned affinity (no subgraph training); core module CPU-verified',
    'Chain scoring for trigger assistance: criteria AUC 0.90 / 0.78 (realistic); training ready, needs GPU',
], Inches(0.8), Inches(1.4), Inches(11.8), Inches(5.6), size=BODY_SM)

# ============ S16 next + questions ============
s = new_slide(16)
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
