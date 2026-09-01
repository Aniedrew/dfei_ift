#!/usr/bin/env python3
"""DFEI group meeting deck v3 (English, formal): theme-based, simple structure.

- Title larger; no DFEI tutorial (audience knows the project)
- Opens with data (CERN MC) + overall result line chart
- Organized by theme, not by chronology (e.g. source/struct/mass heads share one theme)
- Last part: ongoing attempts (wider latent space, ...)
"""
import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

BASE = '/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn'
FIG = BASE + '/meeting_figs'
M11 = BASE + '/meeting_20260811_figs'
V47 = BASE + '/LHCb_logs/DFEI/version_47/plots_inclusive_00342442__v38_masshead2/edges'
OUT = FIG + '/DFEI_progress_20260902_EN.pptx'
os.makedirs(FIG, exist_ok=True)

BLUE = RGBColor(0x1F, 0x4E, 0x79)
DARK = RGBColor(0x33, 0x33, 0x33)
GRAY = RGBColor(0x66, 0x66, 0x66)
ACCENT = RGBColor(0x2C, 0xA0, 0x2C)
RED = RGBColor(0xC0, 0x39, 0x2B)
LIGHT = RGBColor(0xEA, 0xF2, 0xF8)
ORANGE = RGBColor(0xE6, 0x7E, 0x22)
FONT = 'Calibri'
BODY = 20
BODY_SM = 18
TITLE = 34

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
        r2.font.size = Pt(16)
        r2.font.color.rgb = GRAY
        r2.font.name = FONT
    ln = slide.shapes.add_shape(1, Inches(0.6), Inches(1.25), Inches(12.1), Pt(3))
    ln.fill.solid()
    ln.fill.fore_color.rgb = BLUE
    ln.line.fill.background()


def add_bullets(slide, items, left, top, width, height, size=BODY, color=None):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(8)
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


def add_table(slide, data, left, top, width, height, col_widths=None, font_size=18, header_fill=BLUE):
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
tb = s.shapes.add_textbox(Inches(0.8), Inches(2.0), Inches(11.7), Inches(1.8))
tf = tb.text_frame; tf.word_wrap = True
r = tf.paragraphs[0].add_run()
r.text = 'Optimizing DFEI on CERN Monte Carlo'
r.font.size = Pt(44); r.font.bold = True; r.font.color.rgb = BLUE; r.font.name = FONT
tb = s.shapes.add_textbox(Inches(0.8), Inches(4.0), Inches(11.7), Inches(1.2))
tf = tb.text_frame; tf.word_wrap = True
r = tf.paragraphs[0].add_run()
r.text = 'A progress report on the optimization attempts, organized by theme'
r.font.size = Pt(20); r.font.color.rgb = DARK; r.font.name = FONT
tb = s.shapes.add_textbox(Inches(0.8), Inches(5.5), Inches(11.7), Inches(0.9))
tf = tb.text_frame
r = tf.paragraphs[0].add_run()
r.text = 'Qingxiang Guo · University of Chinese Academy of Sciences · 2026-09-02'
r.font.size = Pt(15); r.font.color.rgb = GRAY; r.font.name = FONT

# ============ S2 Data + overall result ============
s = new_slide(2)
add_title_bar(s, 'Data and Overall Effect', 'CERN official MC: /eos/lhcb/user/y/yukaiz/DFEI_IFT_20260702 · threshold 0.9 · 20 test files')
add_bullets(s, [
    'All optimization work in this report is carried out on the CERN official Monte Carlo production (DFEI_IFT_20260702)',
    'Since fixing a silent class-weight bug, the optimization line has raised',
    ('PerfectReco 23.9% → 32.7%   (+8.8pp)', 1),
    ('AllParticles 43.4% → 55.9%   (+12.5pp)', 1),
], Inches(0.8), Inches(1.4), Inches(11.8), Inches(2.2), size=BODY_SM)
add_pic(s, FIG + '/progress_line_v31_v47.png', Inches(1.2), Inches(3.5), width=Inches(10.9))

# ============ S3 Outline ============
s = new_slide(3)
add_title_bar(s, 'Outline', 'Organized by theme, not by chronology')
add_bullets(s, [
    'Part 1 — Making the supervision signal actually effective',
    'Part 2 — Supervising the representations with physical / structural quantities',
    'Part 3 — Controlled failures and their lessons',
    'Part 4 — Ongoing attempts (wider latent space, and others)',
], Inches(0.9), Inches(1.7), Inches(11.5), Inches(4.8), size=BODY)

# ============ S4 Part 1: supervision effective ============
s = new_slide(4)
add_title_bar(s, 'Part 1 — Making the Supervision Signal Effective', 'Fix the silent failure, align training with inference')
add_bullets(s, [
    'Class-weight bug: config key LCA__weights (double underscore) never reached the loss → class1 ≈ 0%',
    ('fix → v31 baseline: PerfectReco 23.9%', 1),
    'Differentiable pruning: soft mask w·σ((w−cut)/τ), τ annealed 1.0→0.1, cut aligned to inference (0.5→0.85)',
    ('trains the model on the same pruned graphs that inference sees', 1),
    'Class-2 (same-mother) weighting, settled at 2.0',
    'In-chain LCA consistency: hinge + direct cross-entropy on truth-chain edges only',
    ('structural edges are ~0.1% of all edges — without direct CE, class0 dilutes them', 1),
    'Net: PerfectReco 23.9% → 29.3% (v31 → v38)',
], Inches(0.8), Inches(1.4), Inches(11.8), Inches(5.6), size=BODY_SM)

# ============ S5 Part 2: physics supervision concept ============
s = new_slide(5)
add_title_bar(s, 'Part 2 — Supervising Representations with Physics', 'A common idea behind several heads')
add_bullets(s, [
    'Idea: physical / structural quantities are data-intrinsic; supervise the representation with them, keep the model end-to-end',
    'Several heads share this theme:',
    ('source head (Rumor-Centrality root) — added early, same idea', 1),
    ('mass head — edge-level log10(m_ππ)', 1),
    ('structure head — node depth + RC value', 1),
    ('momentum head — node normalized momentum', 1),
    'Verification: linear probes (PhyIP-style) on the frozen backbone',
], Inches(0.8), Inches(1.4), Inches(11.8), Inches(5.4), size=BODY_SM)

# ============ S6 mass head ============
s = new_slide(6)
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

# ============ S7 struct + mom ============
s = new_slide(7)
add_title_bar(s, 'Structure and Momentum Heads', 'Node-level supervision of tree position and momentum')
add_table(s, [
    ['Head', 'Target', 'Ablation (5-file eval)', 'Notes'],
    ['baseline (mass only)', '—', 'All 51.4 / Perfect 29.7', ''],
    ['+ struct', 'depth + RC value', 'All 54.5 (+3.1) / Perfect 30.5 (+0.8)', 'best single head; class3 60.8→63.2'],
    ['+ mom', 'normalized p', 'All 53.8 (+2.4) / Perfect 30.5 (+0.8)', 'fixes node R²≈0 for momentum'],
], Inches(0.8), Inches(1.5), Inches(11.8), Inches(2.6), font_size=16)
add_bullets(s, [
    'Struct head: depth = BFS distance to chain centroid; RC = Rumor-Centrality value — node position in the tree',
    'Mom head motivated by a probe: node representations were linearly unreadable for momentum (R² ≈ 0)',
    'Lesson: probe first, find the missing quantity, then supervise it',
], Inches(0.8), Inches(4.3), Inches(11.8), Inches(2.6), size=BODY_SM)

# ============ S8 probe evidence ============
s = new_slide(8)
add_title_bar(s, 'Verification: Linear Probes on the Frozen Backbone', 'Ridge regression; R² of the physical quantity')
add_table(s, [
    ['Probe', 'v38', 'masshead2'],
    ['edge repr → log10(m_ππ)', 'R² = 0.003', 'R² = 0.930'],
    ['node repr → p (px/py/pz)', 'R² ≈ 0', 'R² ≈ 0'],
], Inches(0.8), Inches(1.5), Inches(8.5), Inches(1.8), font_size=18)
add_bullets(s, [
    'Mass supervision moves mass information into the edge representation — confirmed end-to-end',
    'Nodes remain unreadable for momentum → addressed by the momentum head (to be re-probed)',
], Inches(0.8), Inches(3.7), Inches(11.8), Inches(2.8), size=BODY_SM)

# ============ S9 controlled failures ============
s = new_slide(9)
add_title_bar(s, 'Part 3 — Controlled Failures and Their Lessons')
add_bullets(s, [
    'PV subgraph training (v39-42): train per-PV subgraphs → backbone sees only subgraphs, inference runs on the full graph',
    ('full-graph ability degraded: class1 76.8% → 56.4%; line closed', 1),
    'Combined mass + struct + mom (v48): auxiliary losses (0.877) exceed the main task (0.559) → reconstruction dropped 5pp',
    ('LCAG did not drop; the shared backbone was pulled toward the auxiliary tasks', 1),
    ('resolution: lower aux weights — mom 0.2, struct 0.3', 1),
    'Lessons: train/infer graph mismatch is fatal; gradient balance between heads must be explicit',
], Inches(0.8), Inches(1.4), Inches(11.8), Inches(5.4), size=BODY_SM)

# ============ S10 ongoing ============
s = new_slide(10)
add_title_bar(s, 'Part 4 — Ongoing Attempts')
add_bullets(s, [
    'Wider latent space (v53): tracks nodes 32-dim, tt edges 24-dim (16-dim sits at the physical-DOF lower bound)',
    ('from-scratch training interrupted at ep74/150 — not converged; resume planned', 1),
    'Public-data verification (v45/v49): published dataset has no PID, class2/3 counts differ 4-14×; resume best val 33.4 @ep73',
    'Learned deterministic annealing: inference-side PV clustering with a learned affinity (no subgraph training) — core module CPU-verified',
    'Chain scoring for trigger assistance: criteria AUC 0.90 / 0.78 (realistic); scorer training ready, needs GPU',
], Inches(0.8), Inches(1.4), Inches(11.8), Inches(5.6), size=BODY_SM)

# ============ S11 next + questions ============
s = new_slide(11)
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
