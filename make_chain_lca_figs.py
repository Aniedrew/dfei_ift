#!/usr/bin/env python3
"""Figures for the "in-chain LCA consistency" explanation (deck S5):
1) chain_lca_curves.png — 3 panels: class-0 dilution · hinge loss · cross-entropy
2) chain_lca_where.png  — which edges get which loss (truth-chain edges only)
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

FIG = '/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn/meeting_figs'
os.makedirs(FIG, exist_ok=True)
BLUE = '#1F4E79'
RED = '#C0392B'
ORANGE = '#E67E22'
GREEN = '#2CA02C'
GRAY = '#666666'

# ---------- Figure 1: three panels ----------
fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.1))

# (a) dilution
ax = axes[0]
vals = [99.9, 0.1]
labels = ['class 0\n(background)', 'structural\n(class 1/2/3)']
colors = [GRAY, GREEN]
bars = ax.bar(labels, vals, color=colors, width=0.5)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v * 1.15, f'{v}%',
            ha='center', fontsize=11, fontweight='bold', color='#222222')
ax.set_yscale('log'); ax.set_ylim(0.01, 200)
ax.set_ylabel('% of all edges', fontsize=10)
ax.set_title('The dilution problem', fontsize=12, fontweight='bold')
ax.text(0.5, 0.02, 'global CE gradient\n≈ 99.9% from background',
        transform=ax.transAxes, ha='center', fontsize=9, color=GRAY)

# (b) hinge
ax = axes[1]
conf = np.linspace(0, 1, 400)
margin = 0.3
hinge = np.maximum(0, margin - conf)
ax.plot(conf, hinge, lw=2.5, color=ORANGE)
ax.axvline(margin, color=GRAY, ls='--', lw=1.2)
ax.text(margin, 0.28, ' margin 0.3', color=GRAY, fontsize=9)
ax.fill_between(conf, 0, hinge, where=hinge > 0, color=ORANGE, alpha=0.15)
ax.set_xlabel('edge confidence', fontsize=11)
ax.set_ylabel('hinge loss', fontsize=11)
ax.set_title('Hinge (v37): "stay confident"\nloss = max(0, margin − conf)', fontsize=11, fontweight='bold')
ax.set_ylim(-0.02, 0.35)
ax.grid(alpha=0.3)

# (c) cross-entropy
ax = axes[2]
p = np.linspace(1e-3, 1, 400)
ce = -np.log(p)
ax.plot(p, ce, lw=2.5, color=BLUE)
ax.set_xlabel('p (probability of the TRUE class)', fontsize=11)
ax.set_ylabel('CE = −log(p)', fontsize=11)
ax.set_title('Cross-entropy (v38): "be the RIGHT class"\nCE → 0 if p → 1', fontsize=11, fontweight='bold')
ax.set_ylim(0, 5)
ax.grid(alpha=0.3)
ax.annotate('confident & correct\n(CE ≈ 0)', xy=(0.9, -np.log(0.9)), xytext=(0.45, 1.2),
            fontsize=9, color='#222222', arrowprops=dict(arrowstyle='->', color=GREEN))
ax.annotate('unsure / wrong\n(CE large)', xy=(0.05, -np.log(0.05)), xytext=(0.15, 3.8),
            fontsize=9, color='#222222', arrowprops=dict(arrowstyle='->', color=RED))

plt.tight_layout()
out1 = f'{FIG}/chain_lca_curves.png'
plt.savefig(out1, dpi=150, bbox_inches='tight')
plt.close()

# ---------- Figure 2: which edges get which loss ----------
fig, ax = plt.subplots(figsize=(11, 4.6))
ax.set_xlim(0, 110); ax.set_ylim(0, 50); ax.axis('off')

# decay tree: B -> J/psi -> mu+,mu- ; B -> K
nodes = {'B': (55, 42), 'J/psi': (30, 28), 'K': (80, 28), 'mu+': (18, 12), 'mu-': (44, 12)}
edges_tree = [('B', 'J/psi', 'parent-child'), ('B', 'K', 'parent-child'),
              ('J/psi', 'mu+', 'parent-child'), ('J/psi', 'mu-', 'parent-child'),
              ('mu+', 'mu-', 'sister (class 2)')]
bg = [(92, 8), (60, 8), (10, 45)]   # background tracks

# background edges (dashed gray) — only in global CE
for bnode in bg:
    for (x, y) in nodes.values():
        ax.plot([bnode[0], x], [bnode[1], y], color=GRAY, ls=':', lw=1.0, alpha=0.5)
# tree edges (solid green) — get hinge + chain-CE
for a, b, lab in edges_tree:
    (x1, y1), (x2, y2) = nodes[a], nodes[b]
    ax.plot([x1, x2], [y1, y2], color=GREEN, lw=2.2)
    if lab == 'sister (class 2)':
        ax.text((x1 + x2) / 2 + 3, (y1 + y2) / 2, lab, fontsize=9, color='#222222')
# nodes
for name, (x, y) in nodes.items():
    c = Circle((x, y), 3.2, fc='#EAEFF8', ec=BLUE, lw=1.6)
    ax.add_patch(c)
    ax.text(x, y + 4.5, name, ha='center', fontsize=10, color='#222222')
for bnode in bg:
    c = Circle(bnode, 2.6, fc='white', ec=GRAY, lw=1.2, ls=':')
    ax.add_patch(c)

# legend boxes
b1 = FancyBboxPatch((8, 3), 40, 6, boxstyle='round,pad=0.3', fc='white', ec=GREEN, lw=1.4)
ax.add_patch(b1)
ax.text(28, 6, 'solid green edges = on a TRUTH chain\n→ get hinge + chain-CE', ha='center', fontsize=9, color='#222222')
b2 = FancyBboxPatch((60, 3), 42, 6, boxstyle='round,pad=0.3', fc='white', ec=GRAY, lw=1.4)
ax.add_patch(b2)
ax.text(81, 6, 'dotted gray edges = background\n→ only in the global CE (no chain losses)', ha='center', fontsize=9, color='#222222')

ax.text(55, 48.5, 'In-chain consistency: losses applied ONLY to truth-chain edges (we know them from MC truth)',
        ha='center', fontsize=11, fontweight='bold', color='#222222')

plt.tight_layout()
out2 = f'{FIG}/chain_lca_where.png'
plt.savefig(out2, dpi=150, bbox_inches='tight')
plt.close()

# ---------- Figure 3: class imbalance vs per-class accuracy ----------
fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.4))

# (a) edge-class distribution (log scale)
ax = axes[0]
classes = ['class 0', 'class 1', 'class 2', 'class 3']
counts = [49814070, 21786, 19298, 1012]          # from a full eval (20 test files)
frac = np.array(counts) / sum(counts)
bars = ax.bar(classes, frac, color=[GRAY, BLUE, ORANGE, BLUE], width=0.55)
for b, f in zip(bars, frac):
    ax.text(b.get_x() + b.get_width() / 2, f * 1.4, f'{f * 100:.3f}%',
            ha='center', fontsize=9, fontweight='bold', color='#222222')
ax.set_yscale('log'); ax.set_ylim(1e-6, 3)
ax.set_ylabel('fraction of all edges', fontsize=10)
ax.set_title('Edge-class distribution (log scale)', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# (b) per-class accuracy
ax = axes[1]
acc = [98.08, 77.43, 49.17, 60.77]               # masshead2 baseline per-class accuracy
bars = ax.bar(classes, acc, color=[GRAY, BLUE, ORANGE, BLUE], width=0.55)
for b, v in zip(bars, acc):
    ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f'{v:.1f}%',
            ha='center', fontsize=10, fontweight='bold', color='#222222')
ax.set_ylim(0, 110)
ax.set_ylabel('accuracy (%)', fontsize=10)
ax.set_title('Per-class accuracy (best model)', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.annotate('class 2 = rare AND hardest\n→ structural bottleneck', xy=(2, 49.17),
            xytext=(0.6, 20), fontsize=10, color='#222222',
            arrowprops=dict(arrowstyle='->', color=ORANGE))

fig.suptitle('The GNN is weakest exactly where classes are rarest', fontsize=13, fontweight='bold', color='#222222')
plt.tight_layout(rect=[0, 0, 1, 0.92])
out3 = f'{FIG}/chain_lca_imbalance.png'
plt.savefig(out3, dpi=150, bbox_inches='tight')
plt.close()

# ---------- Figure 4: before / after tree ----------
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
nodes = {'B': (45, 38), 'J/psi': (24, 24), 'K': (68, 24), 'mu+': (12, 10), 'mu-': (36, 10)}
tree = [('B', 'J/psi', '1'), ('B', 'K', '1'), ('J/psi', 'mu+', '1'),
        ('J/psi', 'mu-', '1'), ('mu+', 'mu-', '2')]
bgpts = [(90, 8), (58, 2)]

for ax, (title, bad_edge) in zip(axes, [
        ('BEFORE — no chain-CE', ('J/psi', 'mu-')),
        ('AFTER — with chain-CE', None)]):
    ax.set_xlim(0, 105); ax.set_ylim(0, 52); ax.axis('off')
    # background edges (dotted gray)
    for bpt in bgpts:
        for (x, y) in nodes.values():
            ax.plot([bpt[0], x], [bpt[1], y], color=GRAY, ls=':', lw=0.9, alpha=0.45)
    # tree edges
    for a, b, lab in tree:
        (x1, y1), (x2, y2) = nodes[a], nodes[b]
        if (a, b) == bad_edge:
            ax.plot([x1, x2], [y1, y2], color=RED, lw=3.0, ls='--')
            ax.text((x1 + x2) / 2 + 2, (y1 + y2) / 2, 'misclassified as class 0\n→ pruned → chain dies',
                    fontsize=8, color='#222222', ha='left')
        else:
            ax.plot([x1, x2], [y1, y2], color=GREEN, lw=2.4)
            if lab == '2':
                ax.text((x1 + x2) / 2 + 2, (y1 + y2) / 2, f'class {lab}', fontsize=8, color='#222222')
            elif lab == '1':
                ax.text((x1 + x2) / 2 + 2, (y1 + y2) / 2 - 2.5, f'class {lab}', fontsize=8, color='#222222')
    # nodes
    for name, (x, y) in nodes.items():
        c = Circle((x, y), 3.0, fc='#EAEFF8', ec=BLUE, lw=1.5)
        ax.add_patch(c)
        ax.text(x, y + 4.3, name, ha='center', fontsize=9, color='#222222')
    for bpt in bgpts:
        c = Circle(bpt, 2.4, fc='white', ec=GRAY, lw=1.1, ls=':')
        ax.add_patch(c)
    # verdict
    if bad_edge:
        ax.text(50, 49.5, title, ha='center', fontsize=11, fontweight='bold', color='#222222')
        ax.text(50, 1.5, 'result: chain broken ✗', ha='center', fontsize=10, color='#222222')
    else:
        ax.text(50, 49.5, title, ha='center', fontsize=11, fontweight='bold', color='#222222')
        ax.text(50, 1.5, 'result: chain recovered ✓', ha='center', fontsize=10, color='#222222')

plt.tight_layout()
out4 = f'{FIG}/chain_lca_before_after.png'
plt.savefig(out4, dpi=150, bbox_inches='tight')
plt.close()

# ======================================================================
# Figure 5: the problem — class-0 dilution (single panel)
# ======================================================================
fig, ax = plt.subplots(figsize=(11, 4.8))
labels = ['class 0\n(background)', 'classes 1/2/3\n(structural)']
vals = [99.9, 0.1]
bars = ax.bar(labels, vals, color=[GRAY, GREEN], width=0.45)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v * 2.2, f'{v}%',
            ha='center', fontsize=15, fontweight='bold', color='#222222')
ax.set_yscale('log'); ax.set_ylim(0.01, 300); ax.set_xlim(-0.5, 2.3)
ax.set_ylabel('% of all edges (log scale)', fontsize=12)
ax.grid(axis='y', alpha=0.3)
ax.set_title('Edge-class distribution: ~1000 background edges per structural edge',
             fontsize=13, fontweight='bold')
ax.annotate('global CE gradient ≈ 99.9% from\n"predict class 0" → structural\nclasses are undertrained',
            xy=(1.0, 0.12), xytext=(1.45, 4),
            fontsize=11, color='#222222',
            arrowprops=dict(arrowstyle='->', color=RED, lw=1.8))
fig.suptitle('The Problem: Class-0 Dilution', fontsize=15, fontweight='bold', color='#222222')
plt.tight_layout(rect=[0, 0, 1, 0.94])
out5 = f'{FIG}/chain_lca_dilution.png'
plt.savefig(out5, dpi=150, bbox_inches='tight')
plt.close()


def _box(ax, x, y, w, h, text, fc='#EAEFF8', ec=BLUE, fs=10, bold=False):
    b = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.3', fc=fc, ec=ec, lw=1.6)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs,
            color='#222222', fontweight='bold' if bold else 'normal')


def _vline(ax, x, y1, y2, color=BLUE, lw=1.8):
    ax.add_patch(FancyArrowPatch((x, y1), (x, y2), arrowstyle='-|>',
                                 mutation_scale=16, color=color, lw=lw))


# ======================================================================
# Figure 6: hinge — where confidence comes from, where the loss goes
# ======================================================================
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={'width_ratios': [1.15, 1]})

# left: mechanism
ax = axes[0]
ax.set_xlim(0, 100); ax.set_ylim(0, 44); ax.axis('off')
_box(ax, 8, 34, 84, 6.5, 'LCAG head: per-edge 4-class softmax\n→ (p0, p1, p2, p3)', fs=10, bold=True)
_vline(ax, 50, 33, 29)
_box(ax, 8, 22.5, 84, 6.5, 'confidence = max(p0…p3)\nprobability of the class the model chose', fs=10)
_vline(ax, 50, 21.5, 17.5)
_box(ax, 8, 11, 84, 6.5, 'hinge loss = max(0, margin − confidence)\nmargin = 0.3 — extra term in the total loss', fc='#FDF2E9', ec=ORANGE, fs=10, bold=True)
_vline(ax, 50, 10, 6.5)
_box(ax, 8, 2, 84, 5.5, 'applied ONLY on truth-chain edges\n(known from MC truth — training only)', fc='#EAFAF1', ec=GREEN, fs=9.5)
ax.set_title('How the hinge works', fontsize=13, fontweight='bold')

# right: hinge curve
ax = axes[1]
conf = np.linspace(0, 1, 400)
margin = 0.3
hinge = np.maximum(0, margin - conf)
ax.plot(conf, hinge, lw=2.5, color=ORANGE)
ax.axvline(margin, color=GRAY, ls='--', lw=1.2)
ax.fill_between(conf, 0, hinge, where=hinge > 0, color=ORANGE, alpha=0.15)
ax.set_xlabel('edge confidence = max(pk)', fontsize=11)
ax.set_ylabel('hinge loss', fontsize=11)
ax.set_title('Loss vs confidence', fontsize=12, fontweight='bold')
ax.set_ylim(-0.02, 0.35)
ax.grid(alpha=0.3)
ax.text(margin + 0.02, 0.02, 'conf ≥ 0.3 → loss = 0', fontsize=10, color='#222222')
ax.text(0.02, 0.24, 'loses confidence →\npenalty grows linearly', fontsize=10, color='#222222')

fig.suptitle('Hinge (v37): keep chain edges CONFIDENT — whichever class', fontsize=14, fontweight='bold', color='#222222')
plt.tight_layout(rect=[0, 0, 1, 0.93])
out6 = f'{FIG}/chain_lca_hinge.png'
plt.savefig(out6, dpi=150, bbox_inches='tight')
plt.close()

# ======================================================================
# Figure 7: cross-entropy — what it means, where chain-CE is applied
# ======================================================================
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={'width_ratios': [1, 1.15]})

# left: CE curve
ax = axes[0]
p = np.linspace(1e-3, 1, 400)
ce = -np.log(p)
ax.plot(p, ce, lw=2.5, color=BLUE)
ax.set_xlabel('p_true (probability of the TRUE class)', fontsize=11)
ax.set_ylabel('CE = −log(p_true)', fontsize=11)
ax.set_title('Cross-entropy: be the RIGHT class', fontsize=12, fontweight='bold')
ax.set_ylim(0, 5)
ax.grid(alpha=0.3)
ax.annotate('confident & right\np_true → 1, CE ≈ 0', xy=(0.9, -np.log(0.9)), xytext=(0.40, 1.35),
            fontsize=10, color='#222222', arrowprops=dict(arrowstyle='->', color=GREEN))
ax.annotate('unsure / wrong\np_true small, CE large', xy=(0.05, -np.log(0.05)), xytext=(0.12, 3.6),
            fontsize=10, color='#222222', arrowprops=dict(arrowstyle='->', color=RED))

# right: where chain-CE is applied
ax = axes[1]
ax.set_xlim(0, 100); ax.set_ylim(0, 44); ax.axis('off')
_box(ax, 4, 32, 44, 8, 'global CE\n(ALL edges)', fc='white', ec=GRAY, fs=10, bold=True)
_box(ax, 52, 32, 44, 8, 'chain-CE\n(truth-chain edges, classes 1/2/3)', fc='#EAFAF1', ec=GREEN, fs=10, bold=True)
_vline(ax, 26, 30, 25); _vline(ax, 74, 30, 25)
ax.text(26, 17, 'gradient ≈ 99.9%\n"predict background"\n→ rare classes barely learn',
        ha='center', fontsize=10, color='#222222')
ax.text(74, 17, 'each structural edge gets a\ndirect gradient on its true class\n→ classes 1/2/3 learn',
        ha='center', fontsize=10, color='#222222')
ax.text(50, 3, 'same function −log(p_true) — the change is WHERE it is applied',
        ha='center', fontsize=10, style='italic', color='#222222')
ax.set_title('Where chain-CE is applied', fontsize=13, fontweight='bold')

fig.suptitle('Chain-CE (v38): same cross-entropy, applied only where the chain lives', fontsize=14, fontweight='bold', color='#222222')
plt.tight_layout(rect=[0, 0, 1, 0.93])
out7 = f'{FIG}/chain_lca_ce.png'
plt.savefig(out7, dpi=150, bbox_inches='tight')
plt.close()

print('[ok]', out1)
print('[ok]', out2)
print('[ok]', out3)
print('[ok]', out4)
print('[ok]', out5)
print('[ok]', out6)
print('[ok]', out7)
