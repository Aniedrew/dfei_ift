#!/usr/bin/env python3
"""DFEI group meeting deck v2 (English, formal): full optimization journey with figures.

v2 changes vs v1:
- formal register (no filler words), body font >= 20pt
- v31->v38 optimization steps explained in full (audience is new to the project)
- figures embedded: training curves, ROC, event examples, B2 schematic, bug comparison
"""
import os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
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
BODY = 20          # 正文字号
BODY_SM = 18       # 次要文字
TITLE = 30

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


def add_title_bar(slide, text, sub=None):
    tb = slide.shapes.add_textbox(Inches(0.55), Inches(0.18), Inches(12.2), Inches(0.9))
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
        r2.font.size = Pt(15)
        r2.font.color.rgb = GRAY
        r2.font.name = FONT
    ln = slide.shapes.add_shape(1, Inches(0.6), Inches(1.12), Inches(12.1), Pt(3))
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
tb = s.shapes.add_textbox(Inches(0.8), Inches(1.7), Inches(11.7), Inches(1.6))
tf = tb.text_frame; tf.word_wrap = True
r = tf.paragraphs[0].add_run()
r.text = 'Optimizing DFEI: From a Silent Bug to Physics-Supervised Representations'
r.font.size = Pt(34); r.font.bold = True; r.font.color.rgb = BLUE; r.font.name = FONT
tb = s.shapes.add_textbox(Inches(0.8), Inches(3.6), Inches(11.7), Inches(1.2))
tf = tb.text_frame; tf.word_wrap = True
r = tf.paragraphs[0].add_run()
r.text = 'A full account of the optimization line, its rationale, results, and two controlled failures'
r.font.size = Pt(20); r.font.color.rgb = DARK; r.font.name = FONT
tb = s.shapes.add_textbox(Inches(0.8), Inches(5.5), Inches(11.7), Inches(0.9))
tf = tb.text_frame
r = tf.paragraphs[0].add_run()
r.text = 'Qingxiang Guo · University of Chinese Academy of Sciences · 2026-09-02'
r.font.size = Pt(15); r.font.color.rgb = GRAY; r.font.name = FONT

# ============ S2 Outline ============
s = new_slide(2)
add_title_bar(s, 'Outline')
add_bullets(s, [
    '1. Task definition, data, and evaluation metrics',
    '2. Starting point: the class-weight bug and the train–inference gap',
    '3. Optimization line v31 → v38: four verified steps',
    '4. Redirect: output-side physics supervision (mass / structure / momentum heads)',
    '5. Two controlled failures: combined heads, PV subgraph training',
    '6. Ongoing work: asymmetric latent dimensions, public-data verification',
    '7. New directions and open questions',
], Inches(0.9), Inches(1.6), Inches(11.5), Inches(5.4), size=BODY)

# ============ S3 Task ============
s = new_slide(3)
add_title_bar(s, 'Task: Full-Event Decay-Chain Reconstruction', 'DFEI — Deep Full Event Interpretation (García Pardiñas et al., arXiv:2304.08610)')
add_bullets(s, [
    'A pp collision produces ~150 charged tracks; a few originate from the B hadron of interest',
    'Goal: from tracks alone, classify, isolate, and hierarchically rebuild every heavy-hadron decay chain',
    'Four supervised sub-tasks:',
    ('LCAG edge classification (4 classes: background / parent-child / sister / grandparent)', 1),
    ('node pruning (signal vs background tracks) and edge pruning', 1),
    ('track-to-PV association', 1),
    ('offline: chain reconstruction from the classified edges (Rumor-Centrality roots)', 1),
], Inches(0.8), Inches(1.35), Inches(11.8), Inches(5.0), size=BODY_SM)

# ============ S4 Data & metrics ============
s = new_slide(4)
add_title_bar(s, 'Data and Evaluation Metrics')
add_bullets(s, [
    'Data: LHCb Run-3 (Upgrade-I) MC, ~150 tracks/event, normalized; 200 train / 20 val / 20 test files',
    'Public reference dataset (converted_LHCbcollision, arXiv:2304.08610) used for independent verification',
    'Metrics (threshold 0.9 on pruned graphs):',
    ('PerfectReco — fraction of events where every chain is fully recovered', 1),
    ('AllParticles — fraction of truth particles recovered across the event', 1),
    ('LCAG class accuracy per class (class2 = same-mother edges is the structural bottleneck)', 1),
], Inches(0.8), Inches(1.35), Inches(11.8), Inches(5.0), size=BODY_SM)

# ============ S5 Bug ============
s = new_slide(5)
add_title_bar(s, 'Starting Point: A Silent Bug Nullified the Signal', 'Inverse-frequency class weighting was configured but never reached the loss')
add_bullets(s, [
    'Config key LCA__weights (double underscore) vs LCA_weights in the code → weights silently ignored',
    'Consequence: class1 (parent–child) accuracy ≈ 0% — the model learned only the dominant background class',
    'Fix: single-underscore key; weights = N / (4·n_c), clamped',
    'After the fix (v31 baseline): PerfectReco 23.9%, AllParticles 43.4%',
], Inches(0.8), Inches(1.35), Inches(6.5), Inches(4.6), size=BODY_SM)
add_pic(s, M11 + '/fig09_class_acc_bug_vs_paper.png', Inches(7.5), Inches(1.7), width=Inches(5.2))

# ============ S6 Gap ============
s = new_slide(6)
add_title_bar(s, 'Second Issue: Train–Inference Gap', 'Training sees full graphs; inference prunes first')
add_bullets(s, [
    'Training: global optimization over the full graph',
    'Inference: hard-threshold pruning (conf > 0.9) → chains die at pruning',
    'Measured (CERN data): chain loss at edge-pruning dominates; overlap with node-pruning deaths is strong',
    'This gap motivates the first optimization (differentiable pruning)',
], Inches(0.8), Inches(1.35), Inches(6.5), Inches(4.6), size=BODY_SM)
add_pic(s, M11 + '/fig01_event_notfound.png', Inches(7.5), Inches(1.7), width=Inches(5.2))

# ============ S7 Optimization line overview ============
s = new_slide(7)
add_title_bar(s, 'Optimization Line v31 → v53: Overview', 'Every step is a controlled experiment')
add_table(s, [
    ['Version', 'Change', 'PerfectReco'],
    ['v31', 'bug-fixed baseline', '23.9%'],
    ['v36', 'differentiable pruning + source head', '26.3%'],
    ['v37', 'class2 weighting + in-chain LCA hinge', '27.3%'],
    ['v38', 'chain-LCA cross-entropy, cut 0.85', '29.3%'],
    ['v47', 'mass head (physics supervision)', '32.7%'],
    ['v53', 'asymmetric latent dims (32/24)', 'WIP (74/150 ep)'],
], Inches(0.8), Inches(1.5), Inches(6.8), Inches(4.0), font_size=17)
add_bullets(s, [
    'Two lines after v38: physics supervision (mass/struct/mom) and representation widening',
    'Two closed lines: PV subgraph training, naive head combination',
], Inches(0.8), Inches(5.7), Inches(11.8), Inches(1.5), size=BODY_SM)

# ============ S8 Differentiable pruning ============
s = new_slide(8)
add_title_bar(s, 'Optimization 1: Differentiable Pruning', 'Train on the graphs that inference actually sees')
add_bullets(s, [
    'Soft mask on node/edge weights: w_eff = w · σ((w − cut)/τ)',
    'Temperature τ annealed 1.0 → 0.1 during training',
    ('large τ → smooth mask and gradients; small τ → approximates the hard threshold', 1),
    'Pruning cut aligned to inference: 0.5 → 0.7 → 0.85',
    'Effect: the model learns to push background edges down and signal edges up explicitly',
], Inches(0.8), Inches(1.35), Inches(6.6), Inches(4.8), size=BODY_SM)
add_pic(s, M11 + '/fig12_eb2_schematic.png', Inches(7.5), Inches(1.6), width=Inches(5.2))

# ============ S9 class2 + chain LCA ============
s = new_slide(9)
add_title_bar(s, 'Optimization 2-3: Class-2 Weighting and In-Chain Consistency')
add_bullets(s, [
    'Class-2 (same-mother) edges are the structural bottleneck:',
    ('explicit weighting 3.0 → 2.0 (3.0 over-weighted, hurt class1)', 1),
    'In-chain LCA consistency losses — supervised only on truth-chain edges:',
    ('hinge: penalize low-confidence chain edges (margin 0.3)', 1),
    ('cross-entropy on chain-edge classes: structural edges are ~0.1% of all edges; direct CE prevents class0 dilution', 1),
    'Net: PerfectReco 26.3% → 29.3% (v36 → v38)',
], Inches(0.8), Inches(1.35), Inches(11.8), Inches(4.8), size=BODY_SM)
add_pic(s, M11 + '/fig02_training_loss.png', Inches(0.8), Inches(5.0), width=Inches(6.2))

# ============ S10 Source head ============
s = new_slide(10)
add_title_bar(s, 'Optimization 4: Source Head (Rumor-Centrality Training)')
add_bullets(s, [
    'Inference finds chain roots via Rumor Centrality; training never supervised roots',
    'Added a node head predicting "is this the chain root?" (BCE on the RC-argmax node of each truth chain)',
    'Limitation noted: RC-argmax is the centroid of the track graph, not necessarily the B itself — most tracks are final-state particles',
    ('this motivates the later structure supervision (depth + RC regression)', 1),
], Inches(0.8), Inches(1.35), Inches(11.8), Inches(4.2), size=BODY_SM)

# ============ S11 v31->v38 results ============
s = new_slide(11)
add_title_bar(s, 'Results v31 → v38', '20 test files, threshold 0.9')
add_table(s, [
    ['Model', 'PerfectReco', 'AllParticles', 'class1', 'class2'],
    ['v31 (baseline)', '23.9%', '43.4%', '67.8%', '41.3%'],
    ['v36', '26.3%', '49.2%', '64.5%', '47.7%'],
    ['v37', '27.3%', '50.6%', '69.1%', '44.2%'],
    ['v38', '29.3%', '52.1%', '76.8%', '47.9%'],
], Inches(0.8), Inches(1.5), Inches(9.5), Inches(2.8), font_size=18)
add_bullets(s, [
    'class2 gain at v36 is the structural driver; v37 rebalances toward class1',
], Inches(0.8), Inches(4.6), Inches(11.8), Inches(1.4), size=BODY)

# ============ S12 Redirect: physics supervision ============
s = new_slide(12)
add_title_bar(s, 'Redirect: Output-Side Physics Supervision', 'Keep end-to-end; force the latent space to carry physics')
add_bullets(s, [
    'Observation: physical quantities are computed analytically at trigger level — what is the GNN for? Answer: structure and context, encoded in its representation',
    'Approach: supervise representations with data-intrinsic physical targets',
    ('mass head (edge): log10(m_ππ) of each track pair — resonances = sister structure', 1),
    ('structure head (node): depth + Rumor-Centrality value — position in the tree', 1),
    ('momentum head (node): normalized track momentum', 1),
    'Verification: linear probes (PhyIP-style) on the frozen backbone',
], Inches(0.8), Inches(1.35), Inches(11.8), Inches(5.0), size=BODY_SM)

# ============ S13 Mass head ============
s = new_slide(13)
add_title_bar(s, 'Mass Head (v47): The Main Result', 'Edge-level regression of log10(m_ππ)')
add_bullets(s, [
    'Same-mother track pairs sit near resonance masses → the edge representation is forced to encode sister relations',
    'Setup: SmoothL1 on log10(m_MeV), sentinel edges masked, weight 1.0',
    'v38 → masshead2 (20 test files):',
    ('AllParticles 52.1% → 55.9%  (+3.8pp)', 1),
    ('PerfectReco 29.3% → 32.7%  (+3.4pp)', 1),
    ('LCAG class2 44.7% → 51.1%  (+6.4pp)', 1),
], Inches(0.8), Inches(1.3), Inches(6.6), Inches(5.6), size=BODY_SM)
add_pic(s, V47 + '/NN_edges_2_roc.png', Inches(7.6), Inches(1.7), width=Inches(5.2))

# ============ S14 Struct head ============
s = new_slide(14)
add_title_bar(s, 'Structure Head: Position in the Tree', 'Node-level depth + Rumor-Centrality regression')
add_bullets(s, [
    'Motivation: the source head uses one bit (root); depth and RC value describe the node position more fully',
    'Setup: depth = BFS distance to chain centroid; RC value; both regressed (SmoothL1)',
    'Ablation (from masshead2, +10 epochs, 5-file eval):',
    ('baseline: AllParticles 51.4 / Perfect 29.7', 1),
    ('+ struct:  AllParticles 54.5 (+3.1) / Perfect 30.5 (+0.8)  ← best single head', 1),
    'Also improves LCAG class3 (grandparent) 60.8% → 63.2%',
], Inches(0.8), Inches(1.3), Inches(11.8), Inches(5.6), size=BODY_SM)

# ============ S15 Mom head ============
s = new_slide(15)
add_title_bar(s, 'Momentum Head: Repairing a Silent Deficit', 'Node-level regression of normalized momentum')
add_bullets(s, [
    'Diagnosis: linear probes show node representations are unreadable for px/py/pz (R² ≈ 0)',
    ('graph normalization + ReLU in the blocks scramble momentum information', 1),
    'Setup: regress normalized momentum (px_n, py_n, pz_n) from the node representation (SmoothL1)',
    'Ablation (same protocol): +mom → AllParticles 53.8 (+2.4) / Perfect 30.5 (+0.8)',
    'General lesson: probe first, find the missing quantity, then supervise it',
], Inches(0.8), Inches(1.35), Inches(11.8), Inches(4.6), size=BODY_SM)

# ============ S16 Combined failure ============
s = new_slide(16)
add_title_bar(s, 'Controlled Failure: mass + struct + mom Together (v48)', 'Auxiliary-gradient competition, quantified')
add_table(s, [
    ['Loss component (ep114)', 'v47 masshead2', 'v48 combined'],
    ['LCA (main task)', '0.538', '0.559'],
    ['mass', '0.105', '0.045'],
    ['struct', '—', '0.307'],
    ['mom', '—', '0.525'],
    ['aux heads total', '0.105', '0.877'],
], Inches(0.8), Inches(1.4), Inches(6.2), Inches(3.2), font_size=16)
add_bullets(s, [
    'Auxiliary gradients (0.877) exceed the main task (0.559) → shared backbone pulled toward aux tasks',
    'LCAG accuracy did not drop; reconstruction dropped 5pp — the loss–reconstruction correlation is not monotonic',
    'Resolution: reduce aux weights (mom 0.2, struct 0.3) so supervision guides without dominating',
], Inches(0.8), Inches(4.8), Inches(11.8), Inches(2.4), size=BODY_SM)

# ============ S17 Probe evidence ============
s = new_slide(17)
add_title_bar(s, 'Verification: Linear Probes on the Frozen Backbone', 'Ridge regression; R² of the physics quantity')
add_table(s, [
    ['Probe', 'v38', 'masshead2'],
    ['edge repr → log10(m_ππ)', 'R² = 0.003', 'R² = 0.930'],
    ['node repr → p (px/py/pz)', 'R² ≈ 0', 'R² ≈ 0'],
], Inches(0.8), Inches(1.5), Inches(8.5), Inches(1.8), font_size=18)
add_bullets(s, [
    'Mass supervision moves mass information into the edge representation — confirmed end-to-end',
    'Nodes remain unreadable for momentum → the momentum head addresses exactly this (to be re-probed)',
], Inches(0.8), Inches(3.7), Inches(11.8), Inches(2.8), size=BODY_SM)

# ============ S18 Asym ============
s = new_slide(18)
add_title_bar(s, 'Asymmetric Latent Dimensions (v53)', 'Degrees-of-freedom analysis of the representations')
add_bullets(s, [
    'A 16-dim node representation must carry ~12-14 physical DOF (position, momentum, PID, mass) and serve 9 heads — at the lower bound',
    'Setup: tracks nodes 32-dim, tt edges 24-dim, others 16; from-scratch training (150 ep), v38 stack + mass head',
    'Status: interrupted at epoch 74/150 by the job limit — not converged (best val 39.9 vs ~35.6 in the v38 family)',
    ('preliminary eval at 74ep is not representative', 1),
    'Plan: resume to 150 epochs; then layer struct + mom with the reduced weights',
], Inches(0.8), Inches(1.3), Inches(11.8), Inches(5.4), size=BODY_SM)

# ============ S19 Public data ============
s = new_slide(19)
add_title_bar(s, 'Public-Data Verification (v45 / v49)', 'Independent check of the v38 stack on the published dataset')
add_bullets(s, [
    'Published dataset differs from CERN MC: no PID features; class2/class3 edge counts differ by 4-14×',
    'First run interrupted at epoch 66 (best ep51, val 33.2); resume to epoch 88/73 improved best val to 33.4',
    'ep73 evaluation queued (watchdog reserves any free GPU)',
    'Fair comparison to the paper requires adapting the stack to the PID-less dataset',
], Inches(0.8), Inches(1.35), Inches(11.8), Inches(4.4), size=BODY_SM)

# ============ S20 PV clustering (closed) ============
s = new_slide(20)
add_title_bar(s, 'Closed Line: PV Subgraph Training (v39-42)', 'Rationale, failure mode, and conclusion')
add_bullets(s, [
    'Rationale: split events into per-PV subgraphs (20-30 nodes) to reduce cross-chain interference',
    'Failure mode: the backbone trained only on subgraphs, while inference runs the GNN on the full graph first',
    ('full-graph forward ability degraded — class1 76.8% → 56.4%', 1),
    'Inference-only clustering also showed no gain (28.7% vs 29.3%)',
    'Conclusion: the train/infer graph mismatch is not fixable by val-side alignment; line closed',
    ('the clustering idea is pursued differently — see learned deterministic annealing (S22)', 1),
], Inches(0.8), Inches(1.3), Inches(11.8), Inches(5.4), size=BODY_SM)

# ============ S21 Results summary ============
s = new_slide(21)
add_title_bar(s, 'Results Summary', 'CERN MC, threshold 0.9; full 20-file eval unless noted')
add_table(s, [
    ['Model', 'PerfectReco', 'AllParticles', 'class2'],
    ['v31 baseline', '23.9%', '43.4%', '41.3%'],
    ['v38', '29.3%', '52.1%', '47.9%'],
    ['v47 masshead2 (best)', '32.7%', '55.9%', '51.1%'],
    ['v48 combined (rejected)', '29.0%', '50.6%', '50.4%'],
    ['v50-52 ablation (5-file)', '29.7/30.5/30.5', '51.4/53.8/54.5', '—'],
    ['v53 asym (74ep, WIP)', '22.8%', '38.5%', '—'],
], Inches(0.8), Inches(1.4), Inches(9.8), Inches(3.4), font_size=16)
add_bullets(s, [
    'masshead2 is the best model; physics supervision is reproducible',
    'Ablation rows use a small 5-file evaluation — directionally valid, not comparable to full-eval rows',
], Inches(0.8), Inches(5.0), Inches(11.8), Inches(1.8), size=BODY_SM)

# ============ S22 New directions ============
s = new_slide(22)
add_title_bar(s, 'New Directions', 'Learned deterministic annealing and chain scoring')
add_bullets(s, [
    'Learned DA (inference-side PV clustering): trained MLP provides track→PV affinity; deterministic annealing yields soft assignment with adaptive cluster count',
    ('no Gumbel sampling, no training-side subgraph split — avoids the v42 failure mode', 1),
    ('core DA module implemented and CPU-verified', 1),
    'Chain scoring for trigger-line assistance:',
    ('chain criteria AUC 0.90 (incl. random), 0.78 (pruned components only); best single feature 0.956', 1),
    ('scorer training script ready; requires GPU', 1),
], Inches(0.8), Inches(1.3), Inches(11.8), Inches(5.4), size=BODY_SM)

# ============ S23 Open questions ============
s = new_slide(23)
add_title_bar(s, 'Open Questions', 'The application form is the key decision')
add_bullets(s, [
    'Q1: trigger assistance (candidate scoring) vs full-event reconstruction — which metric would the collaboration trust?',
    'Q2: parent–child track relations are physically rare — is same-source (class2) clustering the real problem?',
    'Q3: invariant mass is analytic at trigger level — is the GNN\u2019s added value structure and context, not kinematics?',
    'Q4: can DFEI same-source clusters feed a transformer-based flavor tagger?',
    'Q5: realistic HLT2 / Upgrade-II time budget for a lightweight GNN per event?',
    'Q6: is physics supervision + linear-probe verification a direction to pursue? Better physical targets?',
], Inches(0.8), Inches(1.3), Inches(11.8), Inches(5.9), size=18)

# ============ S24 Next steps ============
s = new_slide(24)
add_title_bar(s, 'Next Steps')
add_bullets(s, [
    'Resume asymmetric training to 150 epochs; then add struct + mom at reduced weights',
    'Run queued evaluations (masshead2 best, public ep73) once GPUs are free',
    'Train the chain scorer; measure candidate-chain AUC on a full evaluation',
    'Plug GNN latent affinity into learned deterministic annealing',
    'Re-probe node representations after momentum-head training',
], Inches(0.8), Inches(1.4), Inches(11.8), Inches(4.8), size=BODY)

# ============ S25 Backup ============
s = new_slide(25)
add_title_bar(s, 'Backup: Setup')
add_bullets(s, [
    'Hardware: single GPU (10 GB cap), batch 8, ~40-60 min/epoch; long queues under heavy cluster load',
    'All changes reproducible from config files; version log maintained in the repository',
], Inches(0.8), Inches(1.4), Inches(11.8), Inches(2.4), size=BODY)

prs.save(OUT)
print('[ok] 已保存: ' + OUT)
print('     共 ' + str(len(prs.slides._sldIdLst)) + ' 页')
