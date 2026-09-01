#!/usr/bin/env python3
"""Figures for deck parts 2-4:
1) reward_results.png  — the reward strategy works (PerfectReco + per-class accuracy)
2) probe_method.png    — how a linear probe is calculated (methodology)
3) probe_r2.png        — linear-probe R2 before/after mass supervision
4) depth_calc.png      — how the struct-head depth target is computed (BFS)
5) rc_calc.png         — how Rumor Centrality is computed (subtree sizes)
6) v48_failure.png     — controlled failure: combined heads unbalance gradients
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


def _arrow(ax, x1, y1, x2, y2, color=BLUE, lw=1.8):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                                 mutation_scale=16, color=color, lw=lw))


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

# ---------- 2) probe method: how a linear probe is calculated ----------
fig, ax = plt.subplots(figsize=(11, 4.6))
ax.set_xlim(0, 100); ax.set_ylim(0, 44); ax.axis('off')
_box(ax, 2, 16, 20, 14, 'GNN backbone\n(FROZEN — gradients\nblocked)', fs=9.5, bold=True)
_box(ax, 28, 19, 18, 8, 'representation\nh_e / h_v', fs=9.5)
_box(ax, 52, 19, 18, 8, 'linear layer\n(only this\ntrains)', fs=9.5)
_box(ax, 76, 19, 22, 8, 'prediction ŷ\n(e.g. log10 m_ππ)', fs=9.5)
_arrow(ax, 22, 23, 28, 23)
_arrow(ax, 46, 23, 52, 23)
_arrow(ax, 70, 23, 76, 23)
_box(ax, 52, 3, 46, 8, 'R² = fit of ŷ vs the true y\nR² ≈ 1 → quantity readable\nR² ≈ 0 → quantity lost',
     fc='#FDF2E9', ec=ORANGE, fs=9.5, bold=True)
_arrow(ax, 87, 19, 87, 11, color=ORANGE)
fig.suptitle('Linear probe: one linear layer on the frozen representation',
             fontsize=14, fontweight='bold', color='#222222')
plt.tight_layout(rect=[0, 0, 1, 0.93])
save(fig, 'probe_method')

# ---------- 3) probe R2 ----------
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

# ---------- 4) depth: BFS distance from the chain root ----------
fig, ax = plt.subplots(figsize=(11, 4.6))
ax.set_xlim(0, 110); ax.set_ylim(0, 52); ax.axis('off')
nodes = {'B': (55, 40), 'J/psi': (28, 26), 'K': (82, 26), 'mu+': (16, 12), 'mu-': (44, 12)}
tree = [('B', 'J/psi'), ('B', 'K'), ('J/psi', 'mu+'), ('J/psi', 'mu-')]
depth = {'B': 0, 'J/psi': 1, 'K': 1, 'mu+': 2, 'mu-': 2}
fill = {0: '#C6EFCE', 1: '#DDEBF7', 2: '#FDEBD0'}
for a, b in tree:
    (x1, y1), (x2, y2) = nodes[a], nodes[b]
    ax.plot([x1, x2], [y1, y2], color=GREEN, lw=2.2)
for name, (x, y) in nodes.items():
    c = Circle((x, y), 3.4, fc=fill[depth[name]], ec=BLUE, lw=1.6)
    ax.add_patch(c)
    ax.text(x, y + 5.2, f'{name}  d={depth[name]}', ha='center', fontsize=10, color='#222222')
# BFS wavefront hint
ax.annotate('BFS from the root:\nd=0 root → d=1 children → d=2 grandchildren',
            xy=(28, 26), xytext=(1, 38), fontsize=10, color='#222222',
            arrowprops=dict(arrowstyle='->', color=BLUE))
ax.text(55, 3.0, 'depth(node) = BFS distance to the chain root (the B candidate)',
        ha='center', fontsize=10.5, color='#222222')
fig.suptitle('Depth: node position in the tree, by BFS layers', fontsize=14, fontweight='bold', color='#222222')
plt.tight_layout(rect=[0, 0, 1, 0.93])
save(fig, 'depth_calc')

# ---------- 5) RC: rumor centrality = subtree-size product ----------
fig, ax = plt.subplots(figsize=(11, 4.6))
ax.set_xlim(0, 110); ax.set_ylim(0, 52); ax.axis('off')
# star centred at B
nodes = {'B': (55, 40), 'J/psi': (24, 22), 'K': (55, 12), 'pi': (86, 22)}
for a, b in [('B', 'J/psi'), ('B', 'K'), ('B', 'pi')]:
    (x1, y1), (x2, y2) = nodes[a], nodes[b]
    ax.plot([x1, x2], [y1, y2], color=GREEN, lw=2.2)
for name, (x, y) in nodes.items():
    c = Circle((x, y), 3.4, fc='#EAFAF1' if name == 'B' else '#DDEBF7', ec=GREEN if name == 'B' else BLUE, lw=1.8)
    ax.add_patch(c)
    ax.text(x, y + 5.2, name, ha='center', fontsize=10, color='#222222')
# subtree sizes when rooted at B
ax.text(55, 46, 'root the tree at each candidate v → subtree sizes τ(u)', ha='center', fontsize=10.5, color='#222222')
ax.text(55, 30, 'rooted at B:  τ = {1, 1, 1, 4}', ha='center', fontsize=10, color='#222222')
ax.text(55, 25, 'log R(B) = −(log1+log1+log1+log4) = −1.39', ha='center', fontsize=10, color='#222222')
ax.text(55, 15, 'rooted at J/psi: log R = −(log3+log4+log1+log1) = −2.48', ha='center', fontsize=10, color='#222222')
ax.text(55, 6, 'root = argmax log R → B  (log R(v) = −Σ_u log τ_v(u), Shah & Zaman)',
        ha='center', fontsize=10, color='#222222')
fig.suptitle('Rumor Centrality: how the chain root is identified', fontsize=14, fontweight='bold', color='#222222')
plt.tight_layout(rect=[0, 0, 1, 0.93])
save(fig, 'rc_calc')

# ---------- 6) controlled failure: v48 gradient imbalance ----------
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
