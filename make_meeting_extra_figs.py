#!/usr/bin/env python3
"""Extra figures for deck parts 2-4:
1) reward_results.png    — the reward strategy works (PerfectReco + per-class accuracy)
2) head_zoo.png          — shared backbone -> 9 task heads schematic
3) struct_mom_tree.png   — what node depth and RC value mean (schematic)
4) probe_r2.png          — linear-probe R2 before/after mass supervision
5) v48_failure.png       — controlled failure: combined heads unbalance gradients
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


def save(fig, name):
    out = f'{FIG}/{name}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print('[ok]', out)


def _box(ax, x, y, w, h, text, fc='#EAEFF8', ec=BLUE, fs=10, bold=False):
    b = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.3', fc=fc, ec=ec, lw=1.6)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs,
            color='#222222', fontweight='bold' if bold else 'normal')


# ---------- 1) reward strategy works ----------
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))

ax = axes[0]
versions = ['v31\nbaseline', 'v36\npruning\n(Part 1)', 'v37\nhinge', 'v38\nchain-CE']
perfect = [23.9, 26.3, 27.3, 29.3]
bars = ax.bar(versions, perfect, color=[GRAY, '#9AA5B1', BLUE, GREEN], width=0.5)
for b, v in zip(bars, perfect):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.4, f'{v:.1f}%',
            ha='center', fontsize=11, fontweight='bold', color='#222222')
ax.set_ylim(0, 35)
ax.set_ylabel('PerfectReco (%)', fontsize=12)
ax.set_title('PerfectReco: what the rewards add', fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.text(2, 32.5, '+1.0pp', ha='center', fontsize=11, color='#222222')
ax.text(3, 32.5, '+2.0pp', ha='center', fontsize=11, color='#222222')

ax = axes[1]
classes = ['class 1', 'class 2']
v31 = [67.8, 41.3]
v38 = [76.8, 47.9]
x = np.arange(2); w = 0.35
b1 = ax.bar(x - w / 2, v31, w, color=GRAY, label='v31 (baseline)')
b2 = ax.bar(x + w / 2, v38, w, color=BLUE, label='v38 (chain-CE)')
for bars_ in (b1, b2):
    for b in bars_:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2,
                f'{b.get_height():.1f}', ha='center', fontsize=11, fontweight='bold', color='#222222')
ax.set_xticks(x); ax.set_xticklabels(classes)
ax.set_ylabel('LCAG accuracy (%)', fontsize=12)
ax.set_ylim(0, 92)
ax.set_title('Per-class accuracy: before vs after rewards', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

fig.suptitle('The reward strategy works: chains survive', fontsize=15, fontweight='bold', color='#222222')
plt.tight_layout(rect=[0, 0, 1, 0.92])
save(fig, 'reward_results')

# ---------- 2) head zoo ----------
fig, ax = plt.subplots(figsize=(11, 4.6))
ax.set_xlim(0, 100); ax.set_ylim(0, 50); ax.axis('off')
_box(ax, 2, 6, 24, 38, 'shared GNN\nbackbone\n\nnode + edge\nrepresentations', fs=10, bold=True)
heads = [
    ('LCAG', 'edge class 0/1/2/3'),
    ('node prune', 'keep / remove node'),
    ('edge prune', 'keep / remove edge'),
    ('source', 'chain root (RC)'),
    ('mass', 'log10 m_ππ'),
    ('struct', 'depth + RC value'),
    ('mom', 'normalized p'),
    ('PV asso', 'PV assignment'),
    ('chain scorer', 'chain confidence'),
]
cols = [30, 55, 80]
rows = [36, 23, 10]
for i, (name, target) in enumerate(heads):
    c = cols[i % 3]; r = rows[i // 3]
    _box(ax, c, r, 20, 10, f'{name}\n{target}', fs=8.5, bold=True)
    if i % 3 == 0:  # one arrow per row from the backbone
        ax.add_patch(FancyArrowPatch((26, r + 5), (30, r + 5), arrowstyle='-|>',
                                     mutation_scale=14, color=BLUE, lw=1.5))
fig.suptitle('The head zoo: one backbone, many supervised targets', fontsize=14, fontweight='bold', color='#222222')
plt.tight_layout(rect=[0, 0, 1, 0.93])
save(fig, 'head_zoo')

# ---------- 3) struct/mom concept: depth and RC on a tree ----------
fig, ax = plt.subplots(figsize=(11, 4.6))
ax.set_xlim(0, 110); ax.set_ylim(0, 52); ax.axis('off')
nodes = {'B': (55, 40), 'J/psi': (28, 26), 'K': (82, 26), 'mu+': (16, 12), 'mu-': (44, 12)}
tree = [('B', 'J/psi'), ('B', 'K'), ('J/psi', 'mu+'), ('J/psi', 'mu-')]
for a, b in tree:
    (x1, y1), (x2, y2) = nodes[a], nodes[b]
    ax.plot([x1, x2], [y1, y2], color=GREEN, lw=2.2)
depth = {'B': 0, 'J/psi': 1, 'K': 1, 'mu+': 2, 'mu-': 2}
rc = {'B': 0.52, 'J/psi': 0.21, 'K': 0.14, 'mu+': 0.07, 'mu-': 0.06}
for name, (x, y) in nodes.items():
    c = Circle((x, y), 3.2, fc='#EAEFF8', ec=BLUE, lw=1.6)
    ax.add_patch(c)
    ax.text(x, y + 5, f'{name}  (depth {depth[name]}, RC {rc[name]})', ha='center', fontsize=9, color='#222222')
ax.text(55, 3.5, 'depth = distance from the chain centroid (0 at the B) · RC = rumor-centrality of the root',
        ha='center', fontsize=10, color='#222222')
fig.suptitle('Structure head: node position in the tree (depth + RC value)', fontsize=14, fontweight='bold', color='#222222')
plt.tight_layout(rect=[0, 0, 1, 0.93])
save(fig, 'struct_mom_tree')

# ---------- 4) probe R2 ----------
fig, ax = plt.subplots(figsize=(11, 4.6))
probes = ['edge repr →\nlog10(m_ππ)', 'node repr →\np (px,py,pz)']
before = [0.003, 0.0]
after = [0.930, 0.0]
x = np.arange(2); w = 0.35
b1 = ax.bar(x - w / 2, before, w, color=GRAY, label='v38 (before mass head)')
b2 = ax.bar(x + w / 2, after, w, color=BLUE, label='masshead2')
for b, v in zip(b1, before):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f'{v:.3f}' if v > 0 else '≈0',
            ha='center', fontsize=11, fontweight='bold', color='#222222')
for b, v in zip(b2, after):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f'{v:.3f}' if v > 0 else '≈0',
            ha='center', fontsize=11, fontweight='bold', color='#222222')
ax.set_xticks(x); ax.set_xticklabels(probes)
ax.set_ylabel('linear-probe R²', fontsize=12)
ax.set_ylim(0, 1.08)
ax.set_title('Readability of physics quantities in the frozen backbone', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
fig.suptitle('Mass supervision wrote mass into the edge representation', fontsize=14, fontweight='bold', color='#222222')
plt.tight_layout(rect=[0, 0, 1, 0.93])
save(fig, 'probe_r2')

# ---------- 5) controlled failure: v48 gradient imbalance ----------
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))

ax = axes[0]
labels = ['main\n(LCAG)', 'aux\n(mass+struct+mom)']
losses = [0.559, 0.877]
bars = ax.bar(labels, losses, color=[BLUE, ORANGE], width=0.5)
for b, v in zip(bars, losses):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f'{v:.3f}',
            ha='center', fontsize=12, fontweight='bold', color='#222222')
ax.set_ylim(0, 1.05)
ax.set_ylabel('loss magnitude (v48)', fontsize=12)
ax.set_title('Aux losses outweigh the main task', fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

ax = axes[1]
vers = ['v47\n(mass only)', 'v48\n(combined)']
allp = [55.9, 50.6]
bars = ax.bar(vers, allp, color=[BLUE, ORANGE], width=0.5)
for b, v in zip(bars, allp):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f'{v:.1f}%',
            ha='center', fontsize=12, fontweight='bold', color='#222222')
ax.set_ylim(0, 62)
ax.set_ylabel('AllParticles (%)', fontsize=12)
ax.set_title('Reconstruction drops 5pp although LCAG did not', fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.text(1, 57, '−5.3pp', ha='center', fontsize=11, color='#222222')

fig.suptitle('Controlled failure: combined heads unbalance the gradients', fontsize=14, fontweight='bold', color='#222222')
plt.tight_layout(rect=[0, 0, 1, 0.92])
save(fig, 'v48_failure')
