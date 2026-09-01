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
            fontsize=9, color=GREEN, arrowprops=dict(arrowstyle='->', color=GREEN))
ax.annotate('unsure / wrong\n(CE large)', xy=(0.05, -np.log(0.05)), xytext=(0.15, 3.8),
            fontsize=9, color=RED, arrowprops=dict(arrowstyle='->', color=RED))

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
        ax.text((x1 + x2) / 2 + 3, (y1 + y2) / 2, lab, fontsize=9, color=GREEN)
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
        ha='center', fontsize=11, fontweight='bold', color=BLUE)

plt.tight_layout()
out2 = f'{FIG}/chain_lca_where.png'
plt.savefig(out2, dpi=150, bbox_inches='tight')
plt.close()

print('[ok]', out1)
print('[ok]', out2)
