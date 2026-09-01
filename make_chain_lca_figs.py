#!/usr/bin/env python3
"""Figures for the "in-chain LCA consistency" part of the deck (S5-S9).
One figure per plot; two-plot slides stack them (top-right / bottom-right).
1) chain_lca_dilution.png          — the problem: class-0 dilution
2) chain_lca_hinge_mechanism.png   — hinge: where confidence comes from / where the reward is paid
3) chain_lca_hinge_curve.png       — hinge penalty vs confidence
4) chain_lca_ce_curve.png          — cross-entropy curve (reward = right class)
5) chain_lca_ce_where.png          — where chain-CE is applied vs the global CE
6) chain_lca_imbalance_dist.png    — edge-class distribution (log scale)
7) chain_lca_imbalance_acc.png     — per-class accuracy
8) chain_lca_before.png            — tree BEFORE chain-CE (one edge misclassified → chain dies)
9) chain_lca_after.png             — tree AFTER chain-CE (all edges correct → chain recovered)
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


def _vline(ax, x, y1, y2, color=BLUE, lw=1.8):
    ax.add_patch(FancyArrowPatch((x, y1), (x, y2), arrowstyle='-|>',
                                 mutation_scale=16, color=color, lw=lw))


# ---------- 1) the problem: class-0 dilution ----------
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
save(fig, 'chain_lca_dilution')

# ---------- 2) hinge: mechanism (confidence source + where the reward is paid) ----------
fig, ax = plt.subplots(figsize=(11, 4.6))
ax.set_xlim(0, 100); ax.set_ylim(0, 44); ax.axis('off')
_box(ax, 8, 34, 84, 6.5, 'LCAG head: per-edge 4-class softmax\n→ (p0, p1, p2, p3)', fs=10, bold=True)
_vline(ax, 50, 33, 29)
_box(ax, 8, 22.5, 84, 6.5, 'confidence = max(p0…p3)\nprobability of the class the model chose', fs=10)
_vline(ax, 50, 21.5, 17.5)
_box(ax, 8, 11, 84, 6.5, 'hinge = max(0, margin − confidence)\nmargin = 0.3 — extra term in the total loss', fc='#FDF2E9', ec=ORANGE, fs=10, bold=True)
_vline(ax, 50, 10, 6.5)
_box(ax, 8, 2, 84, 5.5, 'applied ONLY on truth-chain edges\n(known from MC truth — training only)', fc='#EAFAF1', ec=GREEN, fs=9.5)
fig.suptitle('Hinge (v37): reward for confidence — computed from the head, paid on chain edges',
             fontsize=14, fontweight='bold', color='#222222')
plt.tight_layout(rect=[0, 0, 1, 0.93])
save(fig, 'chain_lca_hinge_mechanism')

# ---------- 3) hinge: penalty vs confidence ----------
fig, ax = plt.subplots(figsize=(11, 4.6))
conf = np.linspace(0, 1, 400)
margin = 0.3
hinge = np.maximum(0, margin - conf)
ax.plot(conf, hinge, lw=2.5, color=ORANGE)
ax.axvline(margin, color=GRAY, ls='--', lw=1.2)
ax.fill_between(conf, 0, hinge, where=hinge > 0, color=ORANGE, alpha=0.15)
ax.set_xlabel('edge confidence = max(pk)', fontsize=12)
ax.set_ylabel('hinge penalty', fontsize=12)
ax.set_ylim(-0.02, 0.35)
ax.grid(alpha=0.3)
ax.text(margin + 0.02, 0.02, 'conf ≥ 0.3 → penalty 0\n(reward kept)', fontsize=11, color='#222222')
ax.text(0.02, 0.24, 'less confident →\npenalty grows linearly', fontsize=11, color='#222222')
fig.suptitle('Hinge: stay confident and the reward is kept', fontsize=14, fontweight='bold', color='#222222')
plt.tight_layout(rect=[0, 0, 1, 0.93])
save(fig, 'chain_lca_hinge_curve')

# ---------- 4) CE: penalty vs p_true ----------
fig, ax = plt.subplots(figsize=(11, 4.6))
p = np.linspace(1e-3, 1, 400)
ce = -np.log(p)
ax.plot(p, ce, lw=2.5, color=BLUE)
ax.set_xlabel('p_true (probability of the TRUE class)', fontsize=12)
ax.set_ylabel('CE = −log(p_true)', fontsize=12)
ax.set_ylim(0, 5)
ax.grid(alpha=0.3)
ax.annotate('confident & right\np_true → 1, penalty → 0', xy=(0.9, -np.log(0.9)), xytext=(0.40, 1.35),
            fontsize=11, color='#222222', arrowprops=dict(arrowstyle='->', color=GREEN))
ax.annotate('unsure / wrong\np_true small, penalty large', xy=(0.05, -np.log(0.05)), xytext=(0.12, 3.6),
            fontsize=11, color='#222222', arrowprops=dict(arrowstyle='->', color=RED))
fig.suptitle('Chain-CE (v38): reward for being RIGHT — CE = −log(p_true)',
             fontsize=14, fontweight='bold', color='#222222')
plt.tight_layout(rect=[0, 0, 1, 0.93])
save(fig, 'chain_lca_ce_curve')

# ---------- 5) CE: where chain-CE is applied ----------
fig, ax = plt.subplots(figsize=(11, 4.6))
ax.set_xlim(0, 100); ax.set_ylim(0, 44); ax.axis('off')
_box(ax, 4, 32, 44, 8, 'global CE\n(ALL edges)', fc='white', ec=GRAY, fs=10, bold=True)
_box(ax, 52, 32, 44, 8, 'chain-CE\n(truth-chain edges, classes 1/2/3)', fc='#EAFAF1', ec=GREEN, fs=10, bold=True)
_vline(ax, 26, 30, 25); _vline(ax, 74, 30, 25)
ax.text(26, 17, 'reward ≈ 0 for structural edges\n(99.9% of the signal is "background")',
        ha='center', fontsize=10, color='#222222')
ax.text(74, 17, 'every chain edge gets a direct\nsignal on its true class\n→ classes 1/2/3 learn',
        ha='center', fontsize=10, color='#222222')
ax.text(50, 3, 'same function −log(p_true) — the change is WHERE it is paid',
        ha='center', fontsize=10, style='italic', color='#222222')
fig.suptitle('Chain-CE: the same −log(p_true), paid only where the chain lives',
             fontsize=14, fontweight='bold', color='#222222')
plt.tight_layout(rect=[0, 0, 1, 0.93])
save(fig, 'chain_lca_ce_where')

# ---------- 6) class imbalance: distribution (log scale) ----------
fig, ax = plt.subplots(figsize=(11, 4.2))
classes = ['class 0', 'class 1', 'class 2', 'class 3']
counts = [49814070, 21786, 19298, 1012]          # from a full eval (20 test files)
frac = np.array(counts) / sum(counts)
bars = ax.bar(classes, frac, color=[GRAY, BLUE, ORANGE, BLUE], width=0.55)
for b, f in zip(bars, frac):
    ax.text(b.get_x() + b.get_width() / 2, f * 1.5, f'{f * 100:.3f}%',
            ha='center', fontsize=11, fontweight='bold', color='#222222')
ax.set_yscale('log'); ax.set_ylim(1e-6, 3)
ax.set_ylabel('fraction of all edges', fontsize=12)
ax.set_title('Edge-class distribution (log scale)', fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
fig.suptitle('Structural edges are ~0.04% of all edges', fontsize=14, fontweight='bold', color='#222222')
plt.tight_layout(rect=[0, 0, 1, 0.92])
save(fig, 'chain_lca_imbalance_dist')

# ---------- 7) class imbalance: per-class accuracy ----------
fig, ax = plt.subplots(figsize=(11, 4.2))
acc = [98.08, 77.43, 49.17, 60.77]               # masshead2 baseline per-class accuracy
bars = ax.bar(classes, acc, color=[GRAY, BLUE, ORANGE, BLUE], width=0.55)
for b, v in zip(bars, acc):
    ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f'{v:.1f}%',
            ha='center', fontsize=11, fontweight='bold', color='#222222')
ax.set_ylim(0, 110)
ax.set_ylabel('accuracy (%)', fontsize=12)
ax.set_title('Per-class accuracy (best model)', fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.annotate('class 2 = rare AND hardest\n→ structural bottleneck', xy=(2, 49.17),
            xytext=(0.55, 22), fontsize=11, color='#222222',
            arrowprops=dict(arrowstyle='->', color=ORANGE))
fig.suptitle('The GNN is weakest exactly where classes are rarest', fontsize=14, fontweight='bold', color='#222222')
plt.tight_layout(rect=[0, 0, 1, 0.92])
save(fig, 'chain_lca_imbalance_acc')


def _draw_tree(ax, bad_edge, title, result):
    ax.set_xlim(0, 105); ax.set_ylim(0, 52); ax.axis('off')
    nodes = {'B': (45, 38), 'J/psi': (24, 24), 'K': (68, 24), 'mu+': (12, 10), 'mu-': (36, 10)}
    tree = [('B', 'J/psi', '1'), ('B', 'K', '1'), ('J/psi', 'mu+', '1'),
            ('J/psi', 'mu-', '1'), ('mu+', 'mu-', '2')]
    bgpts = [(90, 8), (58, 2)]
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
                    fontsize=9, color='#222222', ha='left')
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
    ax.text(50, 49.5, title, ha='center', fontsize=12, fontweight='bold', color='#222222')
    ax.text(50, 1.5, result, ha='center', fontsize=11, color='#222222')


# ---------- 8) before: one misclassified edge kills the chain ----------
fig, ax = plt.subplots(figsize=(11, 4.6))
_draw_tree(ax, ('J/psi', 'mu-'), 'BEFORE — no chain-CE', 'result: chain broken ✗')
fig.suptitle('Why chains die: any structural edge misclassified → chain pruned',
             fontsize=14, fontweight='bold', color='#222222')
plt.tight_layout(rect=[0, 0, 1, 0.92])
save(fig, 'chain_lca_before')

# ---------- 9) after: chain-CE keeps every structural edge correct ----------
fig, ax = plt.subplots(figsize=(11, 4.6))
_draw_tree(ax, None, 'AFTER — with chain-CE', 'result: chain recovered ✓')
fig.suptitle('Chain-CE keeps every structural edge correct', fontsize=14, fontweight='bold', color='#222222')
plt.tight_layout(rect=[0, 0, 1, 0.92])
save(fig, 'chain_lca_after')
