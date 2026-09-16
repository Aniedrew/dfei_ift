#!/usr/bin/env python3
"""Figures for the 2026-09-15 DFEI group meeting (new, stage-based deck).

Generates, into meeting_figs_20260915/:
  stages_tree.png      staged family tree: version-only nodes, colour = PerfectReco,
                       right-hand colour bar, epoch-labelled edges, dashed stage boxes
  metric_fix.png       variable-denominator vs fixed-denominator across thresholds
  scoreboard.png       corrected scoreboard of the key versions
  oracle_decomp.png    oracle decomposition: where the reconstruction loss actually lives
  lr_discovery.png     "frozen" vs "under-trained": the lr ladder
  loss_weights.png     the pruning-loss weight imbalance (node 1 vs edge 33, dead keys)
  delta_stageN.png     before/after "characteristic quantity" panels per stage
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Rectangle
import matplotlib.colors as mcolors

BASE = '/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn'
FIG = '/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn/meeting_figs_20260915'
os.makedirs(FIG, exist_ok=True)

BLUE = '#1F4E79'
DARK = '#333333'
GRAY = '#777777'
RED = '#C0392B'
GREEN = '#2CA02C'
ORANGE = '#E8A33D'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': '#BBBBBB',
                     'axes.labelcolor': DARK, 'text.color': DARK,
                     'xtick.color': DARK, 'ytick.color': DARK, 'figure.dpi': 200})

# ---------------------------------------------------------------- score data (fixed denominator, thr 0.9)
SCORE = {                     # version: (All%, Perfect%)
    31: (22.49, 12.30), 36: (35.17, 18.77), 37: (36.21, 19.50), 38: (37.59, 21.10),
    46: (38.22, 21.28), 47: (39.56, 23.15), 48: (37.36, 21.40), 53: (26.78, 15.84),
    510: (38.01, 21.75), 511: (38.39, 22.11), 512: (24.03, 13.34), 514: (25.44, 15.35),
    515: (39.04, 22.95), 516: (38.95, 22.72), 517: (38.12, 22.00), 518: (38.73, 22.52),
    520: (38.87, 22.69), 500: (33.34, 17.90), 501: (32.70, 17.72), 502: (33.27, 18.01),
    503: (32.78, 17.74), 504: (33.69, 19.13), 505: (33.58, 19.13), 506: (33.20, 19.23),
    507: (34.02, 19.62), 509: (33.27, 17.98), 530: (32.58, 17.71), 549: (33.92, 19.47),
    551: (39.92, 23.19), 553: (37.96, 22.84),
}
VMIN, VMAX = 12.0, 23.5
CMAP = plt.get_cmap('viridis')


def node_color(v):
    p = SCORE.get(v, (None, None))[1]
    return CMAP((p - VMIN) / (VMAX - VMIN)) if p is not None else '#DDDDDD'


def txt_color(v):
    p = SCORE.get(v, (None, None))[1]
    if p is None:
        return '#666666'
    return 'white' if (p - VMIN) / (VMAX - VMIN) < 0.55 else '#222222'


# ---------------------------------------------------------------- 1. staged family tree
def stages_tree():
    fig, ax = plt.subplots(figsize=(13.4, 5.9))
    ax.set_xlim(0, 104); ax.set_ylim(0, 92); ax.axis('off')

    def N(v, x, y, r=2.5, fs=10.0, ghost=False, lw=1.4):
        c = '#F2F2F2' if ghost else node_color(v)
        ec = '#AAB2B8' if ghost else '#2C3E50'
        ax.add_patch(Circle((x, y), r, facecolor=c, edgecolor=ec, lw=lw, zorder=6))
        ax.text(x, y, str(v), ha='center', va='center', fontsize=fs, fontweight='bold',
                color='#999999' if ghost else txt_color(v), zorder=7)

    def E(a, b, lab='', rad=0.0, col='#6E7B87', ls='-', lw=1.3, lx=0, ly=1.6, fs=7.2):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=11, color=col,
                                     lw=lw, linestyle=ls, zorder=3,
                                     connectionstyle=f'arc3,rad={rad}', shrinkA=7, shrinkB=8))
        if lab:
            ax.text((a[0] + b[0]) / 2 + lx, (a[1] + b[1]) / 2 + ly, lab, ha='center', va='center',
                    fontsize=fs, color=col, zorder=8,
                    bbox=dict(fc='white', ec='none', pad=0.8, alpha=0.92))

    def STAGE(x0, x1, y0, y1, title, col):
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                    boxstyle='round,pad=0.7,rounding_size=2.0',
                                    fc=col, ec=col, alpha=0.05, lw=0, zorder=1))
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                    boxstyle='round,pad=0.7,rounding_size=2.0',
                                    fc='none', ec=col, lw=1.5, ls=(0, (5, 3)), zorder=2))
        ax.text(x0 + 1.4, y1 - 1.6, title, ha='left', va='top',
                fontsize=9.2, color=col, fontweight='bold', zorder=9)

    # ---------------- stage bands (disjoint x ranges)
    STAGE(1.5, 20.5, 41.5, 74.0, '① baseline & fine-tuning', '#5B8DB8')
    STAGE(22.0, 44.5, 41.5, 60.5, '② pruning + chain rewards', '#2E7D32')
    STAGE(46.0, 58.5, 41.5, 85.0, '③ physics\nsupervision', '#B8860B')
    STAGE(60.0, 81.0, 60.5, 88.5, '④ capacity & stacking', '#8E44AD')
    STAGE(60.0, 93.0, 28.5, 55.5, '⑤ attention line', '#C2185B')
    STAGE(18.5, 47.0, 3.5, 31.0, '⑥ ablation & combination', '#7A44AA')
    STAGE(49.0, 102.0, 3.0, 26.5, '⑦ NEW — diagnosis & pruning fix', '#0B7285')

    def NOTE(x, y, s, col, fs=6.8, ha='center'):
        ax.text(x, y, s, fontsize=fs, color=col, ha=ha, va='center', zorder=9,
                bbox=dict(fc='white', ec='none', alpha=0.85, pad=1.2))

    # ---------------- ①
    N(31, 5.5, 50); N(32, 12.5, 67); N(35, 18.0, 67)
    E((5.5, 50), (11.0, 64.7), '150 ep', rad=-0.12)
    E((13.9, 67), (16.6, 67), '150 ep', lx=0, ly=1.9)
    NOTE(11.5, 45.6, 'lr 1e-3 → 1e-4,\nno new modules', '#5B8DB8')
    # ---------------- ②
    N(36, 26.0, 49); N(37, 33.5, 49); N(38, 41.0, 49)
    E((18.0, 64.6), (26.0, 51.7), '150 ep', rad=0.10, lx=-1.4, ly=2.0)
    E((26.0, 49), (33.5, 49), '150 ep')
    E((33.5, 49), (41.0, 49), '150 ep')
    NOTE(32.0, 44.2, 'B2 soft mask · cut 0.5→0.85 · class-2 weight · hinge → chain-CE', '#2E7D32')
    # ---------------- ③
    N(46, 52.0, 70); N(47, 52.0, 49)
    E((41.0, 50.6), (52.0, 67.4), '125 ep', rad=0.10, lx=-3.6, ly=1.2)
    E((41.0, 49), (48.8, 49), '125 ep', lx=0, ly=1.9)
    # ---------------- ④
    N(48, 65.0, 83); N(53, 76.0, 83); N(512, 76.0, 68); N(514, 65.0, 68)
    E((41.0, 51.0), (63.0, 81.4), '20–130 ep', rad=-0.12, col='#8E44AD', ls=(0, (4, 2)), lx=-5, ly=-3.0)
    NOTE(70.5, 75.2, 'all worse\nthan v38', '#8E44AD')
    # ---------------- ⑤  (two separate sub-chains: from v38 and from v47)
    N(510, 64.0, 50); N(511, 72.0, 50)
    N(515, 64.0, 34); N(516, 72.0, 34)
    E((41.0, 50.4), (62.2, 49.6), '20 ep', rad=-0.10, col='#C2185B', lx=2.0, ly=2.0, fs=6.8)
    E((64.0, 50), (72.0, 50), '20 ep', col='#C2185B', ly=1.7, fs=6.8)
    E((52.0, 46.9), (62.2, 35.2), '20 ep', rad=0.12, col=RED, ls=(0, (4, 2)), lx=-1.5, ly=2.4, fs=6.8)
    E((64.0, 34), (72.0, 34), '80 ep', col=RED, ls=(0, (4, 2)), ly=1.7, fs=6.8)
    NOTE(80.0, 50, 'from v38\n+1.0 pp', '#C2185B')
    NOTE(80.0, 34, 'from v47\n−0.2 pp', RED)
    # ---------------- ⑥
    N(500, 24.0, 22); N(504, 31.5, 22); N(507, 39.0, 22)
    N(509, 24.0, 9.5); N(530, 31.5, 9.5)
    E((5.5, 47.6), (23.0, 24.4), '20 ep', rad=0.10, col='#7A44AA', lx=7.5, ly=-1.6)
    E((24.0, 22), (31.5, 22), '20 ep', col='#7A44AA', ly=1.7, fs=6.8)
    E((31.5, 22), (39.0, 22), '20 ep', col='#7A44AA', ly=1.7, fs=6.8)
    E((24.0, 19.6), (24.0, 11.6), '20 ep', col='#2CA02C', ls=(0, (3, 2)), lx=4.0, ly=0, fs=6.4)
    E((25.6, 19.8), (30.3, 11.4), '20 ep', col='#2CA02C', ls=(0, (3, 2)), lx=3.6, ly=-1.6, fs=6.4)
    NOTE(32.5, 5.2, 'one factor at a time, kept only\nif AllParticles did not drop', '#7A44AA', fs=6.6)
    # ---------------- ⑦
    N(549, 58.0, 20.0, r=2.3, fs=9.4); N(553, 68.0, 20.0, r=2.3, fs=9.4)
    N(551, 84.0, 20.0, r=3.6, fs=13.0, lw=2.6)
    E((39.0, 20.0), (55.7, 20.0), '20 ep', col='#0B7285', ly=1.7, fs=6.8)
    E((39.0, 22), (80.4, 20.0), '20 ep @ lr 3e-4', col='#0B7285', lw=2.2, lx=-2.0, ly=-3.4, fs=7.6)
    NOTE(68.0, 13.6, 'from v38: pruning-loss weights (point 1 · edge 33)', '#0B7285', fs=6.6)
    for i, v in enumerate([552, 554, 555, 556, 557, 558, 559, 560, 561]):
        N(v, 52.5 + i * 5.4, 8.0, r=1.5, fs=7.0, ghost=True)
    NOTE(51.0, 11.9, '9 arms in flight', '#0B7285', fs=7.0, ha='left')
    orc = FancyBboxPatch((88.6, 15.6), 12.2, 9.0, boxstyle='round,pad=0.5,rounding_size=1.2',
                         fc='#E0F4F7', ec='#0B7285', lw=1.2, zorder=4)
    ax.add_patch(orc)
    ax.text(94.7, 20.1, 'oracle\nupper bounds\n(what pruning\nis worth)', fontsize=7.0,
            color='#0B7285', ha='center', va='center', zorder=5)
    NOTE(84.0, 25.6, 'ties v47 (23.19 vs 23.15)', '#B8860B', fs=6.8)

    # ---------------- colour bar
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=mcolors.Normalize(vmin=VMIN, vmax=VMAX))
    cax = fig.add_axes([0.912, 0.20, 0.012, 0.55])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label('PerfectReco  [%]\n(fixed denominator)', fontsize=8.2, color=DARK)
    cb.ax.tick_params(labelsize=7.5)

    ax.set_position([0.004, 0.01, 0.902, 0.98])
    fig.savefig(FIG + '/stages_tree.png', bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------- 2. metric fix
def metric_fix():
    thr = [0.85, 0.90, 0.95, 0.99]
    oldp = [25.09, 32.70, 36.21, 44.84]
    olda = [44.46, 55.86, 63.43, 80.39]
    newp = [19.09, 23.15, 22.90, 19.10]
    newa = [33.82, 39.56, 40.11, 34.24]
    x = np.arange(len(thr)); w = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.5))
    for ax, (o, n, ttl) in zip(axes, [(oldp, newp, 'PerfectReco'), (olda, newa, 'AllParticles')]):
        b1 = ax.bar(x - w / 2, o, w, label='old (denominator = surviving chains)',
                    color='#E6A6A0', edgecolor='#C0392B')
        b2 = ax.bar(x + w / 2, n, w, label='corrected (denominator = all truth B)',
                    color='#A8CFE8', edgecolor=BLUE)
        for b in (b1, b2):
            ax.bar_label(b, fmt='%.1f', fontsize=7.6, padding=1)
        ax.set_xticks(x); ax.set_xticklabels([f'thr {t}' for t in thr], fontsize=8.5)
        ax.set_title(ttl, fontsize=10, color=BLUE, fontweight='bold')
        ax.set_ylim(0, 88); ax.grid(axis='y', ls=':', color='#DDDDDD')
        ax.set_axisbelow(True)
    axes[0].legend(fontsize=7.4, loc='upper left', frameon=False)
    axes[0].set_ylabel('%', fontsize=9)
    fig.suptitle('The metric was inflating every result: raising the threshold shrank the denominator',
                 fontsize=10.5, color=BLUE, fontweight='bold', y=1.03)
    fig.tight_layout()
    fig.savefig(FIG + '/metric_fix.png', bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------- 3. scoreboard
def scoreboard():
    vers = [31, 36, 37, 38, 47, 500, 504, 507, 553, 551]
    lab = ['v31\nbaseline', 'v36\n+B2+source', 'v37\n+class2+hinge', 'v38\n+chain-CE',
           'v47\n+mass head', 'v500\nablation base', 'v504\n(ab) chain-CE', 'v507\n(ab) struct',
           'v553\nedge w 33→3', 'v551\nlr 3e-4']
    a = [SCORE[v][0] for v in vers]; p = [SCORE[v][1] for v in vers]
    x = np.arange(len(vers)); w = 0.38
    fig, ax = plt.subplots(figsize=(10.6, 3.7))
    b1 = ax.bar(x - w / 2, a, w, label='AllParticles', color='#9DC3E6', edgecolor=BLUE)
    b2 = ax.bar(x + w / 2, p, w, label='PerfectReco', color='#F2C14E', edgecolor='#B8860B')
    for b in (b1, b2):
        ax.bar_label(b, fmt='%.1f', fontsize=7.4, padding=1)
    ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=7.6)
    ax.axhline(SCORE[47][1], ls=':', color='#B8860B', lw=1)
    ax.text(len(vers) - 0.45, SCORE[47][1] + 0.7, 'old best (v47)', fontsize=7.4, color='#B8860B', ha='right')
    ax.set_ylabel('%', fontsize=9); ax.set_ylim(0, 46)
    ax.grid(axis='y', ls=':', color='#DDDDDD'); ax.set_axisbelow(True)
    ax.legend(fontsize=7.8, frameon=False, ncol=2, loc='upper left')
    ax.set_title('Corrected scoreboard (fixed denominator, thr 0.9, 20 test files)',
                 fontsize=10.5, color=BLUE, fontweight='bold')
    fig.tight_layout(); fig.savefig(FIG + '/scoreboard.png', bbox_inches='tight'); plt.close(fig)


# ---------------------------------------------------------------- 4. oracle decomposition
def oracle_decomp():
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.6),
                             gridspec_kw={'width_ratios': [1.25, 1]})
    ax = axes[0]
    names = ['v47\nbaseline', 'point recall\nfixed', 'point precision\nfixed', 'points fully\nperfect',
             'edges\nperfect', 'points +\nedges']
    vals = [39.56, 54.09, 54.68, 71.99, 71.10, 99.75]
    cols = ['#8FA8BF', '#6FA8DC', '#6FA8DC', '#3D85C6', '#C27BA0', '#2CA02C']
    b = ax.bar(range(len(vals)), vals, color=cols, edgecolor='#555555')
    ax.bar_label(b, fmt='%.1f', fontsize=8, padding=1)
    ax.set_xticks(range(len(vals))); ax.set_xticklabels(names, fontsize=7.6)
    ax.set_ylabel('AllParticles  [%]', fontsize=9); ax.set_ylim(0, 112)
    ax.grid(axis='y', ls=':', color='#DDDDDD'); ax.set_axisbelow(True)
    ax.set_title('What each pruning input would be worth', fontsize=10, color=BLUE, fontweight='bold')
    ax = axes[1]
    lab = ['point\nrecall', 'point\nprecision', 'point\nboth', 'edges\nperfect', 'both\nperfect']
    d = [14.5, 15.1, 32.4, 31.5, 60.2]
    b = ax.barh(range(len(d)), d, color=['#3D85C6', '#3D85C6', '#2E5C8A', '#C27BA0', '#2CA02C'],
                edgecolor='#555555')
    ax.bar_label(b, fmt='+%.1f pp', fontsize=8, padding=2)
    ax.set_yticks(range(len(d))); ax.set_yticklabels(lab, fontsize=8)
    ax.invert_yaxis(); ax.set_xlim(0, 72)
    ax.grid(axis='x', ls=':', color='#DDDDDD'); ax.set_axisbelow(True)
    ax.set_xlabel('gain over the v47 baseline  [pp]', fontsize=9)
    ax.set_title('Both point axes are worth ~15 pp', fontsize=10, color=BLUE, fontweight='bold')
    fig.suptitle('Oracle intervention: the reconstruction is fine — the pruning is the bottleneck',
                 fontsize=10.5, color=BLUE, fontweight='bold', y=1.03)
    fig.tight_layout(); fig.savefig(FIG + '/oracle_decomp.png', bbox_inches='tight'); plt.close(fig)


# ---------------------------------------------------------------- 5. lr discovery
def lr_discovery():
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.5))
    ax = axes[0]
    x = np.arange(2); w = 0.36
    base = [SCORE[500][0], SCORE[507][0]]          # 33.34 (v500), 34.02 (v507)
    b1 = ax.bar(x - w / 2, [33.34, 34.02], w, label='before (as-is)', color='#C9D6E3', edgecolor=BLUE)
    b2 = ax.bar(x + w / 2, [33.27, 39.92], w, label='after 20 ep', color='#E8A33D', edgecolor='#B8860B')
    ax.bar_label(b1, fmt='%.1f', fontsize=7.0, padding=1); ax.bar_label(b2, fmt='%.1f', fontsize=7.0, padding=1)
    ax.set_xticks(x); ax.set_xticklabels(['v509: lr 3e-5\n(no change)',
                                          'v551: lr 3e-4\n(v507 base)'], fontsize=7.5)
    ax.set_ylim(0, 47); ax.set_ylabel('AllParticles  [%]', fontsize=9.0)
    ax.grid(axis='y', ls=':', color='#DDDDDD'); ax.set_axisbelow(True)
    ax.legend(fontsize=8.0, frameon=False)
    ax.set_title('Same budget, only the step size differs', fontsize=10.0, color=BLUE,
                 fontweight='bold', pad=6)
    ax.annotate('-0.07 pp  (looks "frozen")', xy=(0 + w / 2, 33.3), xytext=(0.0, 42),
                fontsize=7.0, color=GRAY, arrowprops=dict(arrowstyle='->', color=GRAY, lw=1))
    ax.annotate('+5.90 pp  (ties v47)', xy=(1 + w / 2, 39.9), xytext=(0.55, 46),
                fontsize=7.0, color='#B8860B', arrowprops=dict(arrowstyle='->', color='#B8860B', lw=1.2))
    ax = axes[1]
    lab = ['v31 → v47\n(the whole campaign)', 'v507 → v551\n(20 ep, higher lr)']
    v = [SCORE[47][1] - SCORE[31][1], SCORE[551][1] - SCORE[507][1]]
    b = ax.barh(range(2), v, color=[BLUE, '#E8A33D'], edgecolor='#555555', height=0.5)
    ax.bar_label(b, fmt='+%.1f pp', fontsize=9.0, padding=3)
    ax.set_yticks(range(2)); ax.set_yticklabels(lab, fontsize=7.5)
    ax.invert_yaxis(); ax.set_xlim(0, 13)
    ax.grid(axis='x', ls=':', color='#DDDDDD'); ax.set_axisbelow(True)
    ax.set_xlabel('PerfectReco gain  [pp]', fontsize=9.0)
    ax.set_title('A single run did most of it', fontsize=10.0, color=BLUE,
                 fontweight='bold', pad=6)
    fig.suptitle('Not "frozen": the step was 10x too small',
                 fontsize=11.0, color=BLUE, fontweight='bold', y=1.10)
    fig.tight_layout(); fig.savefig(FIG + '/lr_discovery.png', bbox_inches='tight'); plt.close(fig)


# ---------------------------------------------------------------- 6. loss weights
def loss_weights():
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.5), gridspec_kw={'width_ratios': [1, 1.05]})
    ax = axes[0]
    b = ax.bar(['point head\n(node BCE)', 'edge head\n(edge BCE)'], [1.0, 33.0],
               color=['#C0392B', '#7F8C8D'], edgecolor='#333333')
    ax.bar_label(b, fmt='%.0f', fontsize=9.0, padding=2)
    ax.set_ylim(0, 40); ax.set_ylabel('weight inside the combined loss', fontsize=8.5)
    ax.grid(axis='y', ls=':', color='#DDDDDD'); ax.set_axisbelow(True)
    ax.text(0.5, 37, 'config said\n"rebalance = 10"\nbut was never read',
            fontsize=7.0, color='#C0392B', ha='center', va='top',
            bbox=dict(fc='#FDEDEC', ec='#C0392B', boxstyle='round,pad=0.4'))
    ax.set_title('The loss was never rebalanced', fontsize=10.0, color=BLUE, fontweight='bold', pad=6)
    ax = axes[1]
    cls = ['class 0\n(background)', 'class 1\n(parent–child)', 'class 2\n(sisters)', 'class 3\n(grandparent)']
    acc = [99.9, 76.8, 47.9, 55.8]
    b = ax.bar(cls, acc, color=['#CCCCCC', '#6FA8DC', '#C0392B', '#9DC3E6'], edgecolor='#555555')
    ax.bar_label(b, fmt='%.1f%%', fontsize=8.0, padding=2)
    ax.set_ylim(0, 118); ax.set_ylabel('per-class accuracy  [%]', fontsize=8.5)
    ax.grid(axis='y', ls=':', color='#DDDDDD'); ax.set_axisbelow(True)
    ax.set_title('…so the rare classes stayed weak', fontsize=10.0, color=BLUE,
                 fontweight='bold', pad=6)
    fig.suptitle('The point head carried 1/33 of the edge weight',
                 fontsize=11.0, color=BLUE, fontweight='bold', y=1.10)
    fig.tight_layout(); fig.savefig(FIG + '/loss_weights.png', bbox_inches='tight'); plt.close(fig)


# ---------------------------------------------------------------- 7. before/after delta panels
def delta_panel(fname, title, items, note=''):
    """items: (label, before, after, unit, fmt, direction) direction: 1 better=up, -1 better=down."""
    n = len(items)
    fig, ax = plt.subplots(figsize=(13.8, 0.42 * n + 0.85))
    ax.set_xlim(0, 100); ax.set_ylim(-0.85, n + 0.6); ax.axis('off')
    ax.text(1, n + 0.28, title, fontsize=13.5, fontweight='bold', color=BLUE, va='center')
    for i, (label, b, a, unit, fmt, direc) in enumerate(items):
        yc = n - i - 0.5
        ax.text(2, yc, label, fontsize=11.5, va='center', ha='left')
        ax.add_patch(FancyBboxPatch((40, yc - 0.26), 12, 0.52, boxstyle='round,pad=0.06,rounding_size=0.1',
                                    fc='#E8EEF7', ec='#AAB8C8'))
        ax.text(46, yc, fmt % b, fontsize=12, va='center', ha='center', color='#444444', fontweight='bold')
        good = (a > b) if direc > 0 else (a < b)
        col = GREEN if good else RED
        ax.add_patch(FancyArrowPatch((52.6, yc), (67.4, yc), arrowstyle='-|>', mutation_scale=15,
                                     color=col, lw=2.2))
        ax.add_patch(FancyBboxPatch((68, yc - 0.26), 12, 0.52, boxstyle='round,pad=0.06,rounding_size=0.1',
                                    fc='#EAF7EC' if good else '#FDEDEC', ec=col))
        ax.text(74, yc, fmt % a, fontsize=12, va='center', ha='center', color=col, fontweight='bold')
        d = a - b
        ax.text(81.5, yc, ('%+.2f ' % d) + unit, fontsize=10.5, va='center', color=col, fontweight='bold')
    if note:
        ax.text(1, -0.62, note, fontsize=9.6, color=GRAY, va='center')
    fig.tight_layout(); fig.savefig(FIG + '/' + fname, bbox_inches='tight'); plt.close(fig)


def delta_panels():
    delta_panel('delta_stage1.png', 'Stage 1 — baseline: a silent class-weight bug was fixed',
                [('validation loss, v31 best → v35 best', 36.221, 35.955, '', '%.3f', -1)],
                note='v31 is the reference point of every later number.  The val loss moves little here — that is why the '
                     'later stages are judged on reconstruction, not on the training objective.')
    delta_panel('delta_stage2.png', 'Stage 2 — differentiable pruning + chain rewards (v35 → v38)',
                [('pruning cut, aligned to inference (v36→v38)', 0.50, 0.85, '', '%.2f', 1),
                 ('mask sharpness 1/τ (start → end of a run)', 1.0, 10.0, '', '%.1f', 1),
                 ('class-2 (sister edge) accuracy', 41.3, 47.9, 'pp', '%.1f', 1),
                 ('class-1 (parent–child) accuracy', 67.8, 76.8, 'pp', '%.1f', 1),
                 ('PerfectReco (corrected)', 12.30, 21.10, 'pp', '%.2f', 1)],
                note='Every step is a training-time change; inference is untouched (threshold 0.9 throughout).')
    delta_panel('delta_stage3.png', 'Stage 3 — supervising the representation with physics (v38 → v47)',
                [('edge-mass probe R² (before → after mass head)', 0.003, 0.930, '', '%.3f', 1),
                 ('class-2 (sister edge) accuracy', 44.7, 51.1, 'pp', '%.1f', 1),
                 ('PerfectReco (corrected)', 21.10, 23.15, 'pp', '%.2f', 1)],
                note='Probes first: physics was invisible in the latent space (R² ≈ 0) until it was supervised in. '
                     'The momentum probe stays near zero, which is why the mom head was added later.')
    delta_panel('delta_stage4.png', 'Stage 4 — capacity & stacking attempts (all negative)',
                [('AllParticles, stacking 3 heads (v48)', 39.56, 37.36, 'pp', '%.2f', 1),
                 ('AllParticles, widen latent 32/24 (v512)', 37.59, 24.03, 'pp', '%.2f', 1),
                 ('AllParticles, wider GN 256 (v514)', 37.59, 25.44, 'pp', '%.2f', 1),
                 ('AllParticles, asym from scratch (v53)', 37.59, 26.78, 'pp', '%.2f', 1)],
                note='Revised reading: v48 is a wash (within ±0.4 pp noise), but the widening runs are genuinely destructive.')
    delta_panel('delta_stage5.png', 'Stage 5 — attention line: helps on a plastic base, fails on the best one',
                [('PerfectReco, attention on v38 (v511)', 21.10, 22.11, 'pp', '%.2f', 1),
                 ('PerfectReco, edge-bias attention (v510→v511)', 21.75, 22.11, 'pp', '%.2f', 1),
                 ('PerfectReco, attention on v47 (v515)', 23.15, 22.95, 'pp', '%.2f', 1),
                 ('PerfectReco, struct generation on v47 (v520, 179 ep)', 23.15, 22.69, 'pp', '%.2f', 1)],
                note='Same module, opposite outcome — the base it is grafted onto matters more than the module.')
    delta_panel('delta_stage6.png', 'Stage 6 — ablation chain: kept or dropped, one factor at a time',
                [('chain-CE (v504), events reconstructed', 5854, 5917, '', '%d', 1),
                 ('struct head (v507), events reconstructed', 5917, 5975, '', '%d', 1),
                 ('B2 (v501), events reconstructed', 5854, 5742, '', '%d', 1),
                 ('source head (v505), events reconstructed', 5917, 5897, '', '%d', 1),
                 ('cumulative chain A1 (v530), events', 5854, 5722, '', '%d', 1)],
                note='Noise floor of this pipeline is ±75 events — only chain-CE clears it comfortably.')
    delta_panel('delta_stage7.png', 'Stage 7 — the pruning fix now under test',
                [('edge-loss weight 33 → 3 (v553), AllParticles', 37.59, 37.96, 'pp', '%.2f', 1),
                 ('edge-loss weight 33 → 3 (v553), PerfectReco', 21.10, 22.84, 'pp', '%.2f', 1)],
                note='Still running: point-loss weight 1→5 (v552), focal (v554), chain-level recall (v555), and the '
                     'high-lr stack on v551 (v556–v561).')


if __name__ == '__main__':
    stages_tree(); print('ok stages_tree')
    metric_fix(); print('ok metric_fix')
    scoreboard(); print('ok scoreboard')
    oracle_decomp(); print('ok oracle_decomp')
    lr_discovery(); print('ok lr_discovery')
    loss_weights(); print('ok loss_weights')
    delta_panels(); print('ok delta panels')
    print('all figures -> ' + FIG)
