#!/usr/bin/env python3
"""DFEI group meeting PPT (English): complete optimization journey, v31 → v53.

汇总自接手以来全部优化: bug 修复 -> 算法优化线 -> PV 分簇失败尝试 -> 输出侧物理监督
(mass/struct/mom) -> 消融定位 -> 不对称维度扩展 -> 公开数据验证 -> 新方向。
结果均为真实训练/评估数据 (2026-08-31 状态)。
"""
import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

BASE = '/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn'
FIG = BASE + '/meeting_figs'
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

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


def add_title_bar(slide, text, sub=None):
    tb = slide.shapes.add_textbox(Inches(0.55), Inches(0.22), Inches(12.2), Inches(0.8))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = text
    r.font.size = Pt(26)
    r.font.bold = True
    r.font.color.rgb = BLUE
    r.font.name = FONT
    if sub:
        p2 = tf.add_paragraph()
        r2 = p2.add_run()
        r2.text = sub
        r2.font.size = Pt(13)
        r2.font.color.rgb = GRAY
        r2.font.name = FONT
    ln = slide.shapes.add_shape(1, Inches(0.6), Inches(1.05), Inches(12.1), Pt(3))
    ln.fill.solid()
    ln.fill.fore_color.rgb = BLUE
    ln.line.fill.background()


def add_bullets(slide, items, left, top, width, height, size=15, color=None):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(6)
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


def add_table(slide, data, left, top, width, height, col_widths=None, font_size=12, header_fill=BLUE):
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


def add_footer(slide, idx):
    tb = slide.shapes.add_textbox(Inches(0.55), Inches(7.08), Inches(12.2), Inches(0.35))
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
tb = s.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.7), Inches(1.8))
tf = tb.text_frame; tf.word_wrap = True
r = tf.paragraphs[0].add_run()
r.text = 'Optimizing DFEI: The Full Two-Month Journey'
r.font.size = Pt(36); r.font.bold = True; r.font.color.rgb = BLUE; r.font.name = FONT
tb = s.shapes.add_textbox(Inches(0.8), Inches(3.5), Inches(11.7), Inches(1.4))
tf = tb.text_frame; tf.word_wrap = True
r = tf.paragraphs[0].add_run()
r.text = 'From a silent class-weight bug to physics-supervised representations — every attempt, explained'
r.font.size = Pt(18); r.font.color.rgb = DARK; r.font.name = FONT
tb = s.shapes.add_textbox(Inches(0.8), Inches(5.4), Inches(11.7), Inches(0.9))
tf = tb.text_frame
r = tf.paragraphs[0].add_run()
r.text = 'Qingxiang Guo · University of Chinese Academy of Sciences · 2026-09-02'
r.font.size = Pt(14); r.font.color.rgb = GRAY; r.font.name = FONT

# ============ S2 Overview / timeline ============
s = new_slide(2)
add_title_bar(s, 'Roadmap of Every Optimization Attempt', 'Color code: green = kept · red = abandoned · orange = in progress')
add_table(s, [
    ['Step', 'What', 'Result (PerfectReco @thr0.9)', 'Status'],
    ['v31', 'Class-weight bug fixed (baseline)', '23.9%', 'kept'],
    ['v36', 'B2 differentiable pruning + source head', '26.3%', 'kept'],
    ['v37', 'class2 weighting + chain-LCA hinge', '27.3%', 'kept'],
    ['v38', 'chain-LCA CE + b2 cut 0.85', '29.3%', 'kept'],
    ['v39-42', 'PV clustering (train on subgraphs)', '26.3% (degrades model)', 'closed'],
    ['v46-47', 'Mass head (physics supervision)', '32.7%', 'kept (best)'],
    ['v48', 'mass + struct + mom together', '29.0% (competition)', 'rejected'],
    ['v50-52', 'Single-head ablation (struct/mom)', 'struct +3.1, mom +2.4 (small-set)', 'guide next'],
    ['v53', 'Asymmetric latent dims (32/24)', 'trained 74/150 ep, not converged', 'resume'],
], Inches(0.8), Inches(1.4), Inches(11.8), Inches(4.6), font_size=12)
add_bullets(s, [
    'Also: public-data verification (v45/v49), chain-criteria AUC, learned-DA clustering (in dev)',
], Inches(0.8), Inches(6.2), Inches(11.8), Inches(0.8), size=13)

# ============ S3 Starting point & bug ============
s = new_slide(3)
add_title_bar(s, 'Starting Point: The Code Worked, But a Bug Silenced the Signal', 'v31 baseline')
add_bullets(s, [
    'DFEI = full-event GNN: classifies LCAG edges (4 classes), prunes nodes/edges, associates tracks to PVs, rebuilds all decay chains',
    'Bug found in inherited code: config key LCA__weights (double underscore) — the class weights never reached the loss',
    ('→ class1 (parent-child) accuracy stuck at ~0%: model learned only the dominant class0 background', 1),
    'Second gap: training on full graphs vs. hard-threshold pruning at inference (train-inference gap)',
    'Baseline after the fix: PerfectReco 23.9%, AllParticles 43.4%',
], Inches(0.8), Inches(1.4), Inches(11.8), Inches(5.0), size=16)

# ============ S4 v31→v38 optimization line ============
s = new_slide(4)
add_title_bar(s, 'Optimization Line v31 → v38 (each step verified)', 'PerfectReco 23.9% → 29.3%')
add_bullets(s, [
    'B2: differentiable pruning — soft mask w·σ((w−cut)/τ), τ annealed 1.0→0.1, cut aligned to inference (0.5→0.7→0.85)',
    ('trains the model on the same pruned graphs inference sees → removes the train-inference gap', 1),
    'class2 weighting: same-mother (sister) edges are the structural bottleneck; weight 3.0 → 2.0 after 3.0 hurt class1',
    'In-chain LCA consistency (truth-chain edges only): hinge (chains stay confident) + CE on chain-edge classes',
    ('structural edges are only ~0.1% of all edges — direct CE on them stops class0 from diluting the signal', 1),
    'Source head: train the backbone to predict the chain root (Rumor-Centrality argmax node, BCE)',
], Inches(0.8), Inches(1.4), Inches(11.8), Inches(5.6), size=15)

# ============ S5 PV clustering attempts (closed) ============
s = new_slide(5)
add_title_bar(s, 'PV Clustering (v39-42): Tried, Understood, Closed', 'Why: reduce cross-chain interference in high-multiplicity events')
add_bullets(s, [
    'Idea: split the event into per-PV subgraphs, rebuild chains independently, merge — fewer nodes per graph (20-30 vs 91-139)',
    'Training on subgraphs (trainable cluster head + Gumbel + curriculum) was the failure:',
    ('the backbone is only trained on subgraphs, but inference runs the GNN on the full graph first', 1),
    ('→ full-graph forward ability degraded: class1 76.8% → 56.4%', 1),
    'Inference-only clustering (v38 weights + pv_asso head) showed no gain (28.7% vs 29.3% baseline)',
    'Conclusion: train/infer graph mismatch is not fixable by val-side alignment → line closed',
], Inches(0.8), Inches(1.4), Inches(11.8), Inches(5.4), size=15)

# ============ S6 New direction: physics supervision ============
s = new_slide(6)
add_title_bar(s, 'New Direction: Output-Side Physics Supervision', 'Keep end-to-end; force the latent space to carry physics')
add_bullets(s, [
    'Instead of hand-crafting physics into the input, supervise the representations with physical targets',
    'Three auxiliary heads on top of the shared backbone:',
    ('mass head (edge-level): regress log10(m_ππ) of each track pair — resonance structure = sister information', 1),
    ('struct head (node-level): regress depth (BFS to chain centroid) + RC value — where the node sits in the tree', 1),
    ('mom head (node-level): regress normalized track momentum — nodes were linearly unreadable for p', 1),
    'Why this way: physical quantities are data-intrinsic (computed from tracks), no new labels needed',
    'Verify with linear probes (PhyIP-style): can a frozen backbone read physics from its own representations?',
], Inches(0.8), Inches(1.4), Inches(11.8), Inches(5.4), size=15)

# ============ S7 Mass head ============
s = new_slide(7)
add_title_bar(s, 'Mass Head (v46-47, masshead2): The Big Win', 'Edge-level log10(m_ππ) regression')
add_bullets(s, [
    'Motivation: edges connect two tracks; the pair mass under the pion hypothesis is cheap and physics-loaded',
    ('same-mother pairs sit near resonance masses → mass supervision forces the edge representation to encode sister relations', 1),
    'Setup: SmoothL1 on log10(m_MeV), mask the px≈py≈pz≈-1 sentinel edges; loss weight 1.0',
    'Result (20 test files, thr 0.9) — v38 → masshead2:',
    ('AllParticles 52.1% → 55.9%  (+3.8pp)', 1),
    ('PerfectReco 29.3% → 32.7%  (+3.4pp)', 1),
    ('LCAG class2 (sister) 44.7% → 51.1%  (+6.4pp)', 1),
    'Linear probe: edge repr → mass, R² = 0.003 → 0.930 (frozen backbone)',
], Inches(0.8), Inches(1.3), Inches(11.8), Inches(5.9), size=15)

# ============ S8 Struct head ============
s = new_slide(8)
add_title_bar(s, 'Structure Head: Where a Node Sits in the Tree', 'Node-level depth + RC regression')
add_bullets(s, [
    'Motivation: the source head only predicts the root (1 bit). The node position in the tree is richer: depth and Rumor Centrality',
    ('source-head roots are the track-graph centroid, not necessarily the B — most tracks are final-state particles', 1),
    'Setup: depth = BFS distance to the chain centroid; RC = rumor-centrality value; both regressed (SmoothL1)',
    'Ablation (from masshead2 best, 10 more epochs, small 5-file eval):',
    ('baseline (mass only):  AllParticles 51.4 / Perfect 29.7', 1),
    ('+ struct head:        AllParticles 54.5 (+3.1) / Perfect 30.5 (+0.8)  ← best single head', 1),
    'Struct head also improved LCAG class3 (grandparent) 60.8% → 63.2%',
], Inches(0.8), Inches(1.3), Inches(11.8), Inches(5.9), size=15)

# ============ S9 Mom head ============
s = new_slide(9)
add_title_bar(s, 'Momentum Head: Fixing a Silent Deficit', 'Node-level normalized p regression')
add_bullets(s, [
    'Diagnosis via linear probe: node representations are linearly unreadable for px/py/pz (R² ≈ 0)',
    ('graph_norm + ReLU in the blocks scramble the momentum information', 1),
    'Setup: regress normalized track momentum (px_n, py_n, pz_n) from the node representation (SmoothL1)',
    'Ablation result (same protocol as struct): +mom head → AllParticles 53.8 (+2.4) / Perfect 30.5 (+0.8)',
    'Lesson: a "silent" missing physical quantity can be found by probing and fixed by supervising',
], Inches(0.8), Inches(1.4), Inches(11.8), Inches(4.6), size=15)

# ============ S10 Combination failure + ablation ============
s = new_slide(10)
add_title_bar(s, 'Why Did mass+struct+mom Together Fail? (v48)', 'Gradient competition, quantified')
add_table(s, [
    ['Loss component (ep114)', 'v47 masshead2', 'v48 combined'],
    ['LCA (main task)', '0.538', '0.559'],
    ['mass', '0.105', '0.045'],
    ['struct', '—', '0.307'],
    ['mom', '—', '0.525'],
    ['aux heads total', '0.105', '0.877'],
], Inches(0.8), Inches(1.4), Inches(6.2), Inches(3.0), font_size=13)
add_bullets(s, [
    'In v48 the three auxiliary heads sum to 0.877 ≈ 1.6× the main LCA loss 0.559 → auxiliary gradients dominate',
    'LCAG accuracy did NOT drop (class1 even rose) — but reconstruction dropped 5pp: the shared representations were pulled toward the aux tasks at the expense of pruning/reconstruction',
    'Solution for the next version: lower aux weights (mom 0.2, struct 0.3) so supervision guides without dominating',
], Inches(0.8), Inches(4.7), Inches(11.8), Inches(2.6), size=14)

# ============ S11 PhyIP evidence ============
s = new_slide(11)
add_title_bar(s, 'Evidence: Physics Is Readable From the Representations', 'Linear probes on the frozen backbone (Ridge regression)')
add_table(s, [
    ['Probe', 'v38 (no physics supervision)', 'masshead2'],
    ['edge repr → log10(m_ππ)', 'R² = 0.003', 'R² = 0.930'],
    ['node repr → p (px/py/pz)', 'R² ≈ 0', 'R² ≈ 0'],
], Inches(0.8), Inches(1.5), Inches(11.8), Inches(1.6), font_size=14)
add_bullets(s, [
    'The mass supervision literally moves mass information into the edge representation — the end-to-end "physics in latent" claim holds',
    'Nodes are still unreadable for momenta → motivated the momentum head (S9); to be re-probed after mom-head training',
], Inches(0.8), Inches(3.5), Inches(11.8), Inches(2.6), size=15)

# ============ S12 Asymmetric latent expansion ============
s = new_slide(12)
add_title_bar(s, 'Asymmetric Latent Dimensions (v53)', 'Physics degrees-of-freedom analysis → widen tracks to 32, tt edges to 24')
add_bullets(s, [
    'Observation: node representation (16 dims) must carry ~12-14 physical DOF (position, momentum, PID, mass...) AND serve 9 heads',
    ('16 dims sits at the lower bound; edges need ~7-9 DOF but also compete with heads', 1),
    'Setup: tracks nodes 32-dim, tracks_tracks edges 24-dim, rest stays 16; from-scratch training, v38 stack + mass head',
    'Status: interrupted at epoch 74/150 (job limit); from-scratch not converged yet',
    ('best val_combined 39.9 vs v38-series ~35.6 → needs resume to 150 epochs before judging', 1),
    ('preliminary eval (74ep): All 38.5 / Perfect 22.8 — NOT representative', 1),
    'Plan: resume to 150 ep; then layer struct+mom on top with low weights (S10)',
], Inches(0.8), Inches(1.3), Inches(11.8), Inches(5.9), size=15)

# ============ S13 Public data verification ============
s = new_slide(13)
add_title_bar(s, 'Public-Data Verification (v45/v49)', 'Independent check of the v38 stack on the published dataset')
add_bullets(s, [
    'Published LHCb data (arXiv:2304.08610, converted_LHCbcollision): no PID features, class2/3 edge counts differ 4-14× from CERN MC',
    'v38 stack first run interrupted at ep66 (best ep51); resume ep66→88/73',
    ('resume best: val_combined 33.4 @ep73 (was 33.2 @ep51 pre-resume)', 1),
    'ep88/73 evaluation queued (watchdog to grab any free GPU)',
    'Caveat: the paper reports results on its own pipeline/data; our stack needs adaptation (PID-less) before a fair head-to-head',
], Inches(0.8), Inches(1.4), Inches(11.8), Inches(4.8), size=15)

# ============ S14 Results summary ============
s = new_slide(14)
add_title_bar(s, 'Results Summary (all CERN MC, thr 0.9)', 'Full 20-file eval unless noted')
add_table(s, [
    ['Model', 'AllParticles', 'PerfectReco', 'class2 acc', 'note'],
    ['v31 (bug-fixed baseline)', '43.4%', '23.9%', '~41%', ''],
    ['v38 (optimization line)', '52.1%', '29.3%', '~48%', ''],
    ['v47 masshead2 (best)', '55.9%', '32.7%', '51.1%', '+3.4pp Perfect'],
    ['v48 comb (rejected)', '50.6%', '29.0%', '50.4%', 'aux gradient competition'],
    ['v50-52 ablation (5-file)', '51.4/53.8/54.5', '29.7/30.5/30.5', '—', 'base/mom/struct'],
    ['v53 asym (74ep, WIP)', '38.5%', '22.8%', '—', 'not converged'],
], Inches(0.8), Inches(1.4), Inches(11.8), Inches(3.4), font_size=12)
add_bullets(s, [
    'masshead2 remains the best model — physics supervision works and is reproducible',
    'Ablation numbers use a small 5-file eval set, directionally valid, not comparable to the full-eval rows',
], Inches(0.8), Inches(5.0), Inches(11.8), Inches(1.8), size=13)

# ============ S15 Chain criteria AUC ============
s = new_slide(15)
add_title_bar(s, 'Trigger Perspective: Candidate-Chain Scoring', 'Full-event PerfectReco is the wrong metric for a trigger line')
add_bullets(s, [
    'Question: for HLT2-style selection what matters is candidate-chain classification, not full-event recovery',
    'Chain criteria from model outputs (masshead2, 20 events):',
    ('truth chains vs model-pruned components (realistic): AUC ≈ 0.78', 1),
    ('vs random combos (easy): AUC ≈ 0.90; strongest single feature struct_conf 0.956', 1),
    'Chain scorer (head 5) code is ready, training script written — needs GPU to train and to measure properly',
    'Timing: ~12 ms/event on GPU today; HLT2 would need 10-100× compression (distillation/pruning)',
], Inches(0.8), Inches(1.4), Inches(11.8), Inches(5.4), size=15)

# ============ S16 Learned DA ============
s = new_slide(16)
add_title_bar(s, 'New Direction: Learned Deterministic Annealing (PV)', 'Inference-side clustering with learnable affinity')
add_bullets(s, [
    'PV clustering was closed because training on subgraphs hurt the model — not because clustering is useless',
    'Learned DA (CMS-style): a trained MLP provides affinity (track→PV), deterministic annealing does soft assignment with adaptive cluster count',
    ('no Gumbel sampling, no training-side subgraph split — pure inference-side clustering', 1),
    'Core DA module implemented + CPU-verified (annealing schedule, adaptive splitting, merge post-processing)',
    'Next: plug GNN latent representations as the affinity source and evaluate on PV-finding / chain quality',
], Inches(0.8), Inches(1.4), Inches(11.8), Inches(4.8), size=15)

# ============ S17 Open questions ============
s = new_slide(17)
add_title_bar(s, 'Open Questions for the Group', 'Application form is the key decision')
add_bullets(s, [
    'Q1 — Trigger assistance (candidate scoring) vs full-event reconstruction: which metric would the collaboration trust?',
    'Q2 — Track-level parent-child relations are physically rare; is "same-source clustering" (class2) the real problem to solve?',
    'Q3 — Invariant mass is analytic in HLT2: what is the GNN\u2019s genuine added value — structure/context, not kinematics?',
    'Q4 — Could DFEI same-source clusters feed a transformer flavor tagger (same-side tagging inputs)?',
    'Q5 — Realistic HLT2 / Upgrade-II time budget for a lightweight GNN per event?',
    'Q6 — Is physics supervision (mass/depth/momentum + linear probes) a direction worth pursuing? Better physical targets?',
], Inches(0.8), Inches(1.3), Inches(11.8), Inches(6.0), size=14)

# ============ S18 Next steps ============
s = new_slide(18)
add_title_bar(s, 'Next Steps', '')
add_bullets(s, [
    'Resume asymmetric training to 150 ep; then layer struct + mom with low weights (0.3 / 0.2) on top',
    'Run the queued evaluations (masshead2 best, public ep73) as soon as GPUs free up',
    'Train the chain scorer; measure candidate-chain AUC on a full evaluation',
    'Plug GNN latent affinity into learned DA; compare PV clustering vs current association',
    'Re-probe node representations after mom-head training (did the R²≈0 deficit get fixed?)',
], Inches(0.8), Inches(1.4), Inches(11.8), Inches(4.6), size=15)

# ============ S19 Backup ============
s = new_slide(19)
add_title_bar(s, 'Backup: Data & Setup')
add_bullets(s, [
    'Data: LHCb Upgrade-I Run-3 MC, ~150 tracks/event, normalized, 200 train / 20 val / 20 test files',
    'Public dataset (converted_LHCbcollision) for independent verification',
    'Hardware: single GPU (10 GB cap), batch 8, ~40-60 min/epoch; long queues under heavy cluster load',
    'All changes reproducible from config files in the repo (version log maintained)',
], Inches(0.8), Inches(1.4), Inches(11.8), Inches(4.4), size=15)

prs.save(OUT)
print('[ok] 已保存: ' + OUT)
print('     共 ' + str(len(prs.slides._sldIdLst)) + ' 页')
