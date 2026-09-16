#!/usr/bin/env python3
"""Figures for the 2026-09-15 talk: the 0902 deck, brought up to date.

  main_line.png     the v31 -> v47 line, both metrics, fixed denominator
  p0_event.png      one event and the graph the model sees
  p0_stages.png     the two stages end to end
  p0_prune.png      stage 1 in detail
  p0_recon.png      stage 2 in detail
  p0_metric.png     what PerfectReco and AllParticles count
  m_directions.png  the five directions tried after v47
  w_*.png           the "where in the pipeline" strip, one per slide

Everything is drawn with graphviz `dot` or matplotlib; no PowerPoint shapes.
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

BASE = '/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn'
FIG = BASE + '/meeting_figs_20260915'
sys.path.insert(0, BASE)
import make_meeting_maps_dot as D          # dot helpers: _sch, B, E, render

BLUE = '#1F4E79'
DARK = '#2B2B2B'
GRAY = '#777777'
RED = '#C0392B'
GREEN = '#2E8B57'
ORANGE = '#E8A33D'
PURPLE = '#8E44AD'
LIGHT = '#EAEFF5'
GREY = '#F2F2F2'
plt.rcParams.update({'font.size': 12, 'figure.dpi': 200, 'text.color': DARK,
                     'axes.edgecolor': '#BBBBBB', 'xtick.color': DARK, 'ytick.color': DARK})

# fixed denominator: thr 0.9, N = 17 561 truth B candidates
S = {31: (22.49, 12.30), 36: (35.17, 18.77), 37: (36.21, 19.50),
     38: (37.59, 21.10), 46: (38.22, 21.28), 47: (39.56, 23.15)}


def save(fig, name, transparent=False):
    fig.savefig(FIG + '/' + name, bbox_inches='tight', transparent=transparent)
    plt.close(fig)
    print('  ' + name)


# ------------------------------------------------------------ the main line
def main_line():
    vs = [31, 36, 37, 38, 46, 47]
    steps = ['v31\nbaseline', 'v36\nsoft mask', 'v37\n+ hinge', 'v38\nchain-CE', 'v46\nmass head',
             'v47\nlog10 + mask']
    fig, axes = plt.subplots(2, 1, figsize=(6.6, 4.7))
    for ax, key, ttl, col in ((axes[0], 1, 'PerfectReco  [%]', GREEN),
                              (axes[1], 0, 'AllParticles  [%]', BLUE)):
        vals = [S[v][key] for v in vs]
        ax.bar(range(len(vs)), vals, color=col, width=0.62, zorder=3)
        for i, v in enumerate(vals):
            ax.text(i, v + max(vals) * 0.02, '%.2f' % v, ha='center', va='bottom',
                    fontsize=12, fontweight='bold', color=col)
            if i:
                # the step value gets a row of its own at the top: anywhere lower it lands on
                # the bars themselves
                ax.annotate('%+.2f' % (vals[i] - vals[i - 1]), (i - 0.5, max(vals) * 1.30),
                            ha='center', va='center', fontsize=9.5, color=col, fontweight='bold',
                            bbox=dict(fc='white', ec='#DDDDDD', pad=1.2, alpha=0.95))
        ax.set_xticks(range(len(vs)))
        ax.set_xticklabels(steps, fontsize=9.5)
        ax.set_ylim(0, max(vals) * 1.44)
        ax.set_title(ttl, fontsize=12.5, fontweight='bold', color=col, pad=6)
        ax.grid(axis='y', color='#EEEEEE', zorder=0)
        ax.set_axisbelow(True)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis='y', labelsize=10)
    fig.text(0.5, 0.005, 'N = 17 561 truth B candidates  ·  threshold 0.9',
             ha='center', va='bottom', fontsize=10, color=GRAY, style='italic')
    fig.tight_layout()
    save(fig, 'main_line.png')


# ------------------------------------------------------- one event, one graph
def p0_event():
    """Left: the decay as it happens.  Right: the graph the network is handed.

    Only the final-state particles leave hits in the tracker, so only they can be
    nodes: the B and the J/psi decay before the tracker and show up as vertices,
    never as tracks.  Drawing them as graph nodes would misstate the input.
    """
    from itertools import combinations
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.7))
    ax = axes[0]
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')
    ax.set_title('the decay (from MC)', fontsize=12, fontweight='bold', color=BLUE, pad=4)
    pv = (18, 44)
    ax.add_patch(Circle(pv, 3.4, fc='#FDEBD0', ec=ORANGE, lw=1.6, zorder=4))
    ax.text(pv[0], pv[1], 'PV', ha='center', va='center', fontsize=7.5, fontweight='bold', color=DARK)
    # background tracks: straight out of the PV, no relation to the B
    for (dx, dy) in ((-14, 30), (-12, -24), (-2, 34), (10, -26), (-18, 4)):
        ax.add_patch(FancyArrowPatch(pv, (pv[0] + dx, pv[1] + dy), arrowstyle='-',
                                     color='#B9C2CC', lw=1.4, zorder=2))
    # the B chain: B and J/psi are vertices (open), their daughters are tracks (filled)
    b = (pv[0] + 22, pv[1] + 14)
    j = (b[0] + 22, b[1] + 14)
    k = (b[0] + 22, b[1] - 20)
    m1 = (j[0] + 18, j[1] + 12)
    m2 = (j[0] + 18, j[1] - 14)
    for a, c in ((pv, b), (b, j), (b, k), (j, m1), (j, m2)):
        ax.add_patch(FancyArrowPatch(a, c, arrowstyle='-', color=BLUE, lw=2.4, zorder=3))
    for p, lab, is_trk in ((b, 'B', False), (j, 'J/ψ', False), (k, 'K', True),
                           (m1, 'μ+', True), (m2, 'μ−', True)):
        if is_trk:
            ax.add_patch(Circle(p, 4.2, fc='#DCE9F7', ec=BLUE, lw=1.5, zorder=5))
            ax.text(p[0], p[1], lab, ha='center', va='center', fontsize=7.5,
                    fontweight='bold', color=BLUE, zorder=6)
        else:
            ax.add_patch(Circle(p, 4.8, fc='white', ec='#8A8A8A', lw=1.3, ls='--', zorder=5))
            ax.text(p[0], p[1], lab, ha='center', va='center', fontsize=7, color=GRAY, zorder=6)
    ax.text(50, 2, 'filled = a track · open = a vertex\n'
                   'the B and the J/ψ are never tracks',
            ha='center', va='bottom', fontsize=8.0, color=GRAY)

    ax = axes[1]
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')
    ax.set_title('the graph the model sees', fontsize=12, fontweight='bold', color=BLUE, pad=4)
    nodes = {'PV': (12, 56), 'K': (40, 86), 'μ+': (66, 92), 'μ−': (60, 64),
             'bg1': (30, 32), 'bg2': (62, 34), 'bg3': (88, 62)}
    trks = ['K', 'μ+', 'μ−', 'bg1', 'bg2', 'bg3']
    chain = (('K', 'μ+'), ('K', 'μ−'), ('μ+', 'μ−'))
    for a, b in combinations(trks, 2):
        if (a, b) in chain or (b, a) in chain:
            continue
        ax.plot(*zip(nodes[a], nodes[b]), ls=':', color='#C3CBD3', lw=1.3, zorder=2)
    for a in ('K', 'μ+', 'bg1'):                       # track <-> PV edges
        ax.plot(*zip(nodes['PV'], nodes[a]), ls=(0, (5, 3)), color='#E3C79B', lw=1.3, zorder=2)
    for a, b in chain:
        ax.plot(*zip(nodes[a], nodes[b]), color=BLUE, lw=2.6, zorder=4)
    for n, (x, y) in nodes.items():
        if n == 'PV':
            ax.add_patch(Circle((x, y), 4.4, fc='#FDEBD0', ec=ORANGE, lw=1.6, zorder=5))
            ax.text(x, y, 'PV', ha='center', va='center', fontsize=7,
                    fontweight='bold', color=DARK, zorder=6)
            continue
        on = n in ('K', 'μ+', 'μ−')
        ax.add_patch(Circle((x, y), 4.8, fc='#DCE9F7' if on else '#EFEFEF',
                            ec=BLUE if on else '#B9C2CC', lw=1.6, zorder=5))
        ax.text(x, y, n, ha='center', va='center', fontsize=7, fontweight='bold',
                color=BLUE if on else '#8A8A8A', zorder=6)
    ax.text(50, 3, 'nodes = tracks and PVs\n'
                   'edges = every track pair, plus track-PV\n'
                   'blue = the truth tracks K, μ+, μ−',
            ha='center', va='bottom', fontsize=8.0, color=GRAY)
    fig.tight_layout()
    save(fig, 'p0_event.png')


# ------------------------------------------------------- what the metrics count
TREE = {'B': (0.00, 0.95), 'J/ψ': (-0.62, 0.28), 'K': (0.62, 0.28),
        'μ+': (-1.00, -0.48), 'μ−': (-0.28, -0.48)}


def _mtree(ax, cx, cy, scale, edges, missing=(), bad=()):
    """A five-particle chain drawn at (cx, cy); edges = list of node-name pairs.

    B and J/psi are drawn open: they are the mothers (the LCA of a pair), not tracks.
    """
    pos = {k: (cx + scale * v[0], cy + scale * v[1]) for k, v in TREE.items()}
    for a, b in edges:
        if a in missing or b in missing:
            continue
        col = RED if (a, b) in bad else BLUE
        ls = '--' if (a, b) in bad else '-'
        ax.plot(*zip(pos[a], pos[b]), color=col, lw=1.8, ls=ls, zorder=2,
                solid_capstyle='round')
    for n, (x, y) in pos.items():
        if n in missing:
            continue
        if n in ('B', 'J/ψ'):
            ax.add_patch(Circle((x, y), scale * 0.26, fc='white', ec='#8A8A8A', lw=1.1,
                                ls='--', zorder=3))
            ax.text(x, y, n, ha='center', va='center', fontsize=7.4, color=GRAY, zorder=4)
        else:
            ax.add_patch(Circle((x, y), scale * 0.26, fc='#E8EEF4', ec=BLUE, lw=1.2, zorder=3))
            ax.text(x, y, n, ha='center', va='center', fontsize=7.4, color=BLUE, zorder=4)


def p0_metric():
    """What the two flags actually require: particle content vs the tree as well."""
    truth = [('B', 'J/ψ'), ('B', 'K'), ('J/ψ', 'μ+'), ('J/ψ', 'μ−')]
    same_content_other_tree = [('B', 'J/ψ'), ('B', 'K'), ('J/ψ', 'μ+'), ('K', 'μ−')]

    fig = plt.figure(figsize=(6.0, 3.25))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')
    ax.text(13, 97, 'the truth chain', ha='center', va='top', fontsize=10, color=GRAY)
    ax.text(63.5, 97, 'three reconstructed chains', ha='center', va='top', fontsize=10, color=GRAY)

    cols = (13, 39, 64, 88)
    _mtree(ax, cols[0], 64, 11.5, truth)
    _mtree(ax, cols[1], 64, 11.5, truth)
    _mtree(ax, cols[2], 64, 11.5, same_content_other_tree, bad=[('K', 'μ−')])
    _mtree(ax, cols[3], 64, 11.5, truth, missing=['μ−'])
    for x, txt, col in ((cols[1], 'exact match', GRAY), (cols[2], 'other tree', RED),
                        (cols[3], 'one missing', RED)):
        ax.text(x, 42, txt, ha='center', va='center', fontsize=8.2, color=col)

    ax.text(3, 30, 'AllParticles', ha='left', va='center', fontsize=9.6, color=GREEN,
            fontweight='bold')
    ax.text(3, 12, 'PerfectReco', ha='left', va='center', fontsize=9.6, color=BLUE,
            fontweight='bold')
    for x, (a, p) in zip(cols[1:], ((1, 1), (1, 0), (0, 0))):
        for y, ok in ((30, a), (12, p)):
            ax.text(x, y, ('✓ 1' if ok else '✗ 0'), ha='center', va='center', fontsize=10,
                    color=GREEN if ok else RED, fontweight='bold')
    ax.text(50, 0, 'filled = a final-state track  ·  open = a mother particle, the LCA of a pair',
            ha='center', va='bottom', fontsize=8.0, color=GRAY)
    fig.savefig(FIG + '/p0_metric.png', bbox_inches='tight')
    plt.close(fig)
    print('  p0_metric.png')


# ------------------------------------------------- the struct head's two targets
def _chain_nodes(ax, nodes, edges, r=3.2, fs=6.8):
    for a, b in edges:
        ax.plot(*zip(nodes[a], nodes[b]), color=GREEN, lw=1.8, zorder=2)
    for n, (x, y) in nodes.items():
        ax.add_patch(Circle((x, y), r, fc='#EAF7EF', ec=GREEN, lw=1.4, zorder=3))
        ax.text(x, y, n, ha='center', va='center', fontsize=fs, color=DARK, zorder=4)


def depth_calc():
    """depth = BFS layer of a node, counted from the chain root."""
    fig = plt.figure(figsize=(6.05, 1.83))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 36); ax.axis('off')
    ax.set_aspect('equal')
    nodes = {'B': (50, 26), 'J/ψ': (26, 17), 'K': (74, 17), 'μ+': (16, 7), 'μ−': (36, 7)}
    _chain_nodes(ax, nodes, [('B', 'J/ψ'), ('B', 'K'), ('J/ψ', 'μ+'), ('J/ψ', 'μ−')])
    for lab, x, y in (('d = 0', 50, 30.6), ('d = 1', 26, 21.6), ('d = 1', 74, 21.6),
                      ('d = 2', 16, 11.6), ('d = 2', 36, 11.6)):
        ax.text(x, y, lab, ha='center', va='bottom', fontsize=6.6, color=BLUE)
    ax.text(50, 0.5, 'depth(node) = number of BFS layers from the chain root (the B candidate)',
            ha='center', va='bottom', fontsize=7.2, color=GRAY)
    save(fig, 'depth_calc.png')


def rc_calc():
    """Rumor Centrality: root the tree at each candidate, take the largest subtree product."""
    fig = plt.figure(figsize=(6.05, 1.83))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 30); ax.axis('off')
    ax.set_aspect('equal')
    _chain_nodes(ax, {'B': (16, 20), 'J/ψ': (6, 9), 'K': (16, 4), 'π': (26, 9)},
                 [('B', 'J/ψ'), ('B', 'K'), ('B', 'π')], r=3.0, fs=6.2)
    lines = ((26.5, 'root the tree at each candidate v:  subtree sizes τ(u)'),
             (21.0, 'rooted at B:  τ = {1, 1, 1, 4}'),
             (15.5, 'log R(B) = −(log1 + log1 + log1 + log4) = −1.39'),
             (10.0, 'rooted at J/ψ:  log R = −(log3 + log4 + log1 + log1) = −2.48'),
             (4.5, 'root = argmax log R → B   (log R(v) = −Σ_u log τ_v(u); Shah & Zaman)'))
    for y, txt in lines:
        ax.text(36, y, txt, ha='left', va='center', fontsize=6.8, color=DARK)
    save(fig, 'rc_calc.png')


# ------------------------------------------------- the chain as a graph (Part 2)
def chain_tree(name, title, result, bad=False):
    """One decay chain drawn as a graph.  Green = a structural edge; the red dashed one
    was called class 0, so the chain cannot be assembled.  Every label sits in free space.
    """
    fig = plt.figure(figsize=(6.05, 2.00))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 33); ax.axis('off')
    ax.set_aspect('equal')
    nodes = {'B': (42, 24), 'J/ψ': (22, 15), 'K': (68, 15), 'μ+': (12, 6), 'μ−': (33, 6)}
    for bx, by in ((90, 27), (88, 8)):                      # background tracks
        ax.add_patch(Circle((bx, by), 2.2, fc='white', ec='#B9C2CC', lw=0.9, ls=':', zorder=2))
        for a in ('B', 'K', 'μ−'):
            ax.plot([bx, nodes[a][0]], [by, nodes[a][1]], ls=':', color='#C3CBD3', lw=1.0, zorder=1)
    for a, b in (('B', 'J/ψ'), ('B', 'K'), ('J/ψ', 'μ+'), ('J/ψ', 'μ−'), ('μ+', 'μ−')):
        (x1, y1), (x2, y2) = nodes[a], nodes[b]
        if bad and a == 'J/ψ' and b == 'μ−':
            ax.plot([x1, x2], [y1, y2], color=RED, lw=2.4, ls='--', zorder=3)
        else:
            ax.plot([x1, x2], [y1, y2], color=GREEN, lw=2.2, zorder=3)
    for n, (x, y) in nodes.items():
        ax.add_patch(Circle((x, y), 3.1, fc='#EAF7EF', ec=GREEN, lw=1.3, zorder=4))
        ax.text(x, y, n, ha='center', va='center', fontsize=6.6, color=DARK, zorder=5)
    for a, b, tx, ty, ha in (('B', 'J/ψ', 30, 21.5, 'right'), ('B', 'K', 57, 21.5, 'left'),
                             ('J/ψ', 'μ+', 14, 12.4, 'center'), ('μ+', 'μ−', 20.5, 8.6, 'center')):
        ax.text(tx, ty, 'class 2' if (a, b) == ('μ+', 'μ−') else 'class 1',
                ha=ha, va='center', fontsize=7.0, color=DARK)
    if bad:
        ax.text(36, 13.5, 'called class 0 → pruned', ha='left', va='center',
                fontsize=7.0, color=RED)
        ax.annotate('', xy=(28, 11.3), xytext=(35.5, 13.2),
                    arrowprops=dict(arrowstyle='-|>', color=RED, lw=1.1))
    ax.text(50, 31.6, title, ha='center', va='center', fontsize=8.6,
            fontweight='bold', color=BLUE)
    ax.text(50, 0.6, result, ha='center', va='bottom', fontsize=7.4, color=DARK)
    save(fig, name)


# ------------------------------------------------------- the four LCAG classes
def p0_class():
    """One panel per class: what the relation is, pair by pair.

    Grey dashed = a decay vertex (never a track); the solid line is the track pair whose
    class we are looking at.  The head returns four numbers per pair — that is the last line.
    """
    fig = plt.figure(figsize=(6.05, 3.55))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 59); ax.axis('off')
    ax.set_aspect('equal')

    def vtx(p, r=1.8):
        ax.add_patch(Circle(p, r, fc='white', ec='#8A8A8A', lw=1.1, ls='--', zorder=4))

    def trk(p, r=2.5, lab=''):
        ax.add_patch(Circle(p, r, fc='#EAF7EF', ec=GREEN, lw=1.3, zorder=5))
        if lab:
            ax.text(p[0], p[1], lab, ha='center', va='center', fontsize=6.0, color=DARK, zorder=6)

    def link(p, q, col='#8A8A8A', ls='--', lw=1.2, z=3):
        ax.plot(*zip(p, q), color=col, ls=ls, lw=lw, zorder=z)

    def panel(x0, y0, num, name, col):
        ax.add_patch(FancyBboxPatch((x0, y0 + 19.4), 4.3, 3.9, boxstyle='round,pad=0.3',
                                    fc=col, ec=col, lw=1.0, zorder=3))
        ax.text(x0 + 2.15, y0 + 21.35, str(num), ha='center', va='center', fontsize=7.4,
                color='white', fontweight='bold', zorder=4)
        ax.text(x0 + 5.6, y0 + 21.35, name, ha='left', va='center', fontsize=7.4,
                color=col, fontweight='bold')

    def caption(x0, y0, txt):
        ax.text(x0, y0 + 0.4, txt, ha='left', va='bottom', fontsize=6.6, color=GRAY)

    # ---- class 0: two tracks from two different decays
    x0, y0 = 3, 32
    panel(x0, y0, 0, 'background', '#8A8A8A')
    for p in ((x0 + 9, y0 + 16), (x0 + 36, y0 + 16)):
        vtx(p)
    for a, b in (((x0 + 9, y0 + 16), (x0 + 8, y0 + 9)), ((x0 + 36, y0 + 16), (x0 + 37, y0 + 9))):
        link(a, b)
    trk((x0 + 8, y0 + 9), lab='a'); trk((x0 + 37, y0 + 9), lab='b')
    link((x0 + 8, y0 + 9), (x0 + 37, y0 + 9), col='#B9C2CC', ls=':', lw=2.0, z=2)
    ax.text(x0 + 22.5, y0 + 9, '✗', ha='center', va='center', fontsize=8.5, color=RED, zorder=6)
    caption(x0, y0, 'no common ancestor in this event')

    # ---- class 1: the mother is itself a track
    x0, y0 = 53, 32
    panel(x0, y0, 1, 'parent–child', BLUE)
    vtx((x0 + 12, y0 + 17))
    link((x0 + 12, y0 + 17), (x0 + 11, y0 + 11))
    trk((x0 + 11, y0 + 11), lab='a')
    trk((x0 + 28, y0 + 6), lab='b')
    link((x0 + 11, y0 + 11), (x0 + 28, y0 + 6), col=BLUE, ls='-', lw=2.2, z=2)
    caption(x0, y0, 'the mother a is itself a track (rare)')

    # ---- class 2: sisters, one mother
    x0, y0 = 3, 6
    panel(x0, y0, 2, 'sisters', GREEN)
    vtx((x0 + 23, y0 + 16), r=2.0)
    for a in ((x0 + 11, y0 + 8), (x0 + 35, y0 + 8)):
        link((x0 + 23, y0 + 16), a)
    trk((x0 + 11, y0 + 8), lab='a'); trk((x0 + 35, y0 + 8), lab='b')
    link((x0 + 11, y0 + 8), (x0 + 35, y0 + 8), col=GREEN, ls='-', lw=2.4, z=2)
    caption(x0, y0, 'same mother → they mark one decay vertex')

    # ---- class 3: grandparent and grandchild
    x0, y0 = 53, 6
    panel(x0, y0, 3, 'grandparent–grandchild', ORANGE)
    vtx((x0 + 11, y0 + 17)); vtx((x0 + 32, y0 + 13))
    link((x0 + 11, y0 + 17), (x0 + 32, y0 + 13))
    link((x0 + 11, y0 + 17), (x0 + 8, y0 + 8))
    link((x0 + 32, y0 + 13), (x0 + 39, y0 + 7))
    trk((x0 + 8, y0 + 8), lab='a'); trk((x0 + 39, y0 + 7), lab='b')
    link((x0 + 8, y0 + 8), (x0 + 39, y0 + 7), col=ORANGE, ls='-', lw=2.2, z=2)
    caption(x0, y0, 'same chain, two steps apart')

    ax.text(50, 3.4, 'dashed circle = a decay vertex, never a track',
            ha='center', va='bottom', fontsize=6.6, color=GRAY)
    ax.text(50, 1.0, 'for every pair the head returns four numbers — p0 … p3 — and the largest one is the decision',
            ha='center', va='bottom', fontsize=6.6, color=GRAY)
    save(fig, 'p0_class.png')


# --------------------------------------------------- the where-in-the-flow strip
STAGES = [('event\ngraph', 0), ('pruning\nheads', 1), ('GNN\nblocks', 2),
          ('latent\n16-d', 3), ('physics\nheads', 4), ('chain\nassembly', 5)]


def strip(name, hi, note):
    fig = plt.figure(figsize=(6.2, 1.12))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')
    ax.text(0.5, 100, note, fontsize=10, color=BLUE, va='top', ha='left', style='italic')
    w, gap, x = 15.0, 2.0, 0.5
    for i, (lab, _) in enumerate(STAGES):
        on = i in hi
        ax.add_patch(FancyBboxPatch((x, 6), w, 52, boxstyle='round,pad=0.7',
                                    fc=LIGHT if on else '#F2F2F2',
                                    ec=BLUE if on else '#C4C4C4', lw=2.0 if on else 1.2))
        ax.text(x + w / 2, 32, lab, ha='center', va='center', fontsize=8.2,
                fontweight='bold' if on else 'normal', color=BLUE if on else '#8A8A8A')
        if i < len(STAGES) - 1:
            ax.annotate('', xy=(x + w + gap - 0.5, 32), xytext=(x + w + 0.5, 32),
                        arrowprops=dict(arrowstyle='-|>', color='#B0B0B0', lw=1.3))
        x += w + gap
    fig.savefig(FIG + '/' + name, transparent=True, bbox_inches='tight')
    plt.close(fig)
    print('  ' + name)


STRIPS = [
    ('w_ingraph.png', {0}, 'step 1 — what the model is given'),
    ('w_classdef.png', {5}, 'the classes are what the chain assembly reads'),
    ('w_stage1.png', {1}, 'step 2 — decide which tracks and edges survive'),
    ('w_stage2.png', {2, 3, 5}, 'step 3 — rebuild the chain from what survived'),
    ('w_score.png', {5}, 'step 4 — score whole chains, not single edges'),
    ('w_v36.png', {1, 2}, 'this change: how the pruning heads are trained'),
    ('w_edge.png', {5}, 'the object we care about: the assembled chain'),
    ('w_class.png', {1}, 'the classification that decides which edges survive'),
    ('w_hinge.png', {1}, 'a reward added to the per-edge classification'),
    ('w_ce.png', {1}, 'the same head, a different loss'),
    ('w_evid.png', {5}, 'what the two metrics measure'),
    ('w_probe.png', {3}, 'what the representation does and does not contain'),
    ('w_zoo.png', {4}, 'where the new heads attach'),
    ('w_mass.png', {4}, 'a new target for the EDGE representation'),
    ('w_struct.png', {4}, 'a new target for the NODE representation'),
]


# ------------------------------------------------------------- dot schematics
def p0_stages():
    """The two stages, two rows so it fits a slide band.

    What pruning uses is the two binary prune heads — the four-class LCAG head is not
    part of it (checked against the code: the inference cut reads node_weights /
    edge_weights only, while the LCAG logits are read out later, during chain assembly).
    """
    s = D._sch(rankdir='TB', ns=0.30, rs=0.46, fs=11, margin='0.07,0.05', wmin=0.9)
    s += ('  subgraph cluster_s1 {\n    label="stage 1  ·  pruning — decide what survives";\n'
          '    labeljust="l"; fontname="Helvetica-Bold"; fontsize=12;\n'
          '    style="rounded,dashed"; color="#8E44AD"; penwidth=1.5; margin=10;\n')
    s += D.B('g', 'event graph\nnodes = tracks, PVs\nedges = tt pairs')
    s += D.B('n', 'GNN blocks\nweighted message passing\n→ 16-d representation')
    s += D.B('p', 'point / edge\nprune heads\nper-node and per-edge score w')
    s += D.B('c', 'keep if w ≥ 0.9\nat inference\n→ surviving subgraph', fc='#FDEBD0', ec='#C0392B')
    s += '    { rank=same; g; n; p; c; }\n'
    s += '    g -> n -> p -> c;\n  }\n'
    s += ('  subgraph cluster_s2 {\n    label="stage 2  ·  reconstruction — build the chain";\n'
          '    labeljust="l"; fontname="Helvetica-Bold"; fontsize=12;\n'
          '    style="rounded,dashed"; color="#2C7A7B"; penwidth=1.5; margin=10;\n')
    s += D.B('v', 'surviving subgraph', fc='#EAF7EF', ec='#2E8B57')
    s += D.B('l', 'LCAG head\nper-edge class 0–3\n(0 = background)', fc='#E6F4F5', ec='#2C7A7B')
    s += D.B('a', 'chain assembly\nLCA consistency\n→ candidate chains', fc='#E6F4F5', ec='#2C7A7B')
    s += D.B('t', 'compare with\nthe truth chain\n→ the metrics', fc='#E6F4F5', ec='#2C7A7B')
    s += '    { rank=same; v; l; a; t; }\n'
    s += '    v -> l -> a -> t;\n  }\n'
    s += D.E('c', 'v', w=1.8)
    s += '}\n'
    D.render('p0_stages', s)


def p0_prune():
    """Stage 1 in detail — laid out for a single column."""
    s = D._sch(rankdir='TB', ns=0.28, rs=0.40, fs=11, margin='0.08,0.05', wmin=1.0)
    s += D.B('g', 'event graph\nnodes = tracks, PVs  ·  edges = tt pairs')
    s += D.B('n', 'GNN blocks\nweighted message passing\n→ 16-d node / edge representation',
             fc='#E6F4F5', ec='#2C7A7B')
    s += D.B('p', 'point prune head\nkeep / remove a node')
    s += D.B('e', 'edge prune head\nkeep / remove an edge')
    s += D.B('c', 'hard cut at inference\nkeep if w ≥ 0.9', fc='#FDEBD0', ec='#C0392B')
    s += D.B('s', 'surviving subgraph\neverything downstream sees only this',
             fc='#EAF7EF', ec='#2E8B57')
    s += '  g -> n;\n  n -> p;\n  n -> e;\n'
    s += '  { rank=same; p; e; }\n'
    s += '  p -> c [style=invis];\n  e -> c [style=invis];\n  c -> s;\n'
    s += '}\n'
    D.render('p0_prune', s)


def p0_recon():
    """Stage 2 in detail — laid out for a single column."""
    s = D._sch(rankdir='TB', ns=0.28, rs=0.40, fs=11, margin='0.08,0.05', wmin=1.0)
    s += D.B('s', 'surviving subgraph\nonly nodes and edges with w ≥ 0.9', fc='#EAF7EF', ec='#2E8B57')
    s += D.B('z', '16-d node / edge\nrepresentation, from the\nshared GNN blocks',
             fc='#E6F4F5', ec='#2C7A7B')
    s += D.B('l', 'LCAG head\nper-edge class 0–3\n0 = background, 1/2/3 = structural',
             fc='#E6F4F5', ec='#2C7A7B')
    s += D.B('o', 'chain assembly\ntracks are joined while the\nLCA stays consistent\n'
                  '→ candidate B chains', fc='#E6F4F5', ec='#2C7A7B')
    s += '  s -> z -> l -> o;\n'
    s += '}\n'
    D.render('p0_recon', s)


def directions():
    s = D._sch(rankdir='TB', ns=0.30, rs=0.42, fs=11, margin='0.07,0.05', wmin=0.9)
    s += D.B('cap', 'capacity\nwiden the latent space,\nGN hidden 128 → 256\n\n'
                    '✗ 24 – 27 %  against 37.59',
             fc='#FDEDEC', ec=RED, fs=11)
    s += D.B('att', 'context / attention\ntrack-level self-attention,\n+ tt edge-feature bias\n\n'
                    '~ helps only on a plastic base',
             fc='#FEF6E7', ec=ORANGE, fs=11)
    s += D.B('res', 'restructure the event\ntrain on subgraphs,\nsplit the event by PV\n\n'
                    '✗ makes the full-graph pass worse',
             fc='#FDEDEC', ec=RED, fs=11)
    s += D.B('ret', 'retrain the recipe\none change per run\nfrom a common base\n\n'
                    '~ only chain-CE clears the noise',
             fc='#FEF6E7', ec=ORANGE, fs=11)
    s += D.B('pub', 'another dataset\npublic sample, N = 12 774\n\n'
                    '~ negative transfer, then\na layer-by-layer rebuild',
             fc='#FEF6E7', ec=ORANGE, fs=11)
    s += D.B('v', 'after v47, none of the five directions beats the main line'
                  '  →  the headroom is in the pruning heads, not in the network',
             fc='#FFFFFF', ec='#FFFFFF', fs=11)
    s += '  { rank=same; cap; att; res; ret; pub; }\n'
    s += '  cap -> v [style=invis];\n  att -> v [style=invis];\n'
    s += '  res -> v [style=invis];\n  ret -> v [style=invis];\n  pub -> v [style=invis];\n'
    s += '}\n'
    D.render('m_directions', s)


# ------------------------------------------------- the annealing mechanism (v36)
def p1_anneal():
    """v36: the soft mask that replaces the hard cut, and what a message then carries."""
    import numpy as np
    w = np.linspace(0.0, 1.0, 500)
    cut = 0.85
    taus = [(1.0, '#1F4E79', 'tau = 1.0  (start)'), (0.3, '#E8A33D', 'tau = 0.3'),
            (0.1, '#C0392B', 'tau = 0.1  (end)')]
    fig, axes = plt.subplots(1, 2, figsize=(6.05, 2.55))
    for ax, kind in zip(axes, ('mask', 'weff')):
        hard = (w >= cut).astype(float) if kind == 'mask' else np.where(w >= cut, w, 0.0)
        ax.plot(w, hard, lw=2.0, color='#333333', ls='--', label='hard cut (inference)')
        for tau, col, lab in taus:
            m = 1.0 / (1.0 + np.exp(-(w - cut) / tau))
            ax.plot(w, m if kind == 'mask' else w * m, lw=2.2, color=col, label=lab)
        ax.axvline(cut, color='#9AA0A6', ls='--', lw=1.1)
        ax.axvline(0.9, color='#2E8B57', ls=':', lw=1.5)
        ax.set_xlabel('w  (the learned pruning weight, 0–1)', fontsize=10)
        ax.set_ylim(-0.05, 1.08)
        ax.grid(alpha=0.22)
        ax.tick_params(labelsize=9.5)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel('mask', fontsize=10)
    axes[1].set_ylabel('w_eff', fontsize=10)
    axes[0].set_title('mask = σ((w − cut)/τ)', fontsize=11,
                      fontweight='bold', color='#1F4E79', pad=5)
    axes[1].set_title('w_eff = w · mask', fontsize=11,
                      fontweight='bold', color='#1F4E79', pad=5)
    h, l = axes[0].get_legend_handles_labels()
    h += [plt.Line2D([], [], color='#9AA0A6', ls='--', lw=1.1),
          plt.Line2D([], [], color='#2E8B57', ls=':', lw=1.5)]
    l += ['cut = 0.85 (v38)', '0.9 = the inference cut']
    fig.legend(h, l, loc='lower center', ncol=4, fontsize=8.4, frameon=False,
               bbox_to_anchor=(0.5, -0.015), handlelength=1.8, columnspacing=1.4)
    fig.tight_layout(rect=[0, 0.13, 1, 1])
    save(fig, 'p1_anneal.png')


if __name__ == '__main__':
    main_line()
    p1_anneal()
    p0_event()
    p0_metric()
    p0_class()
    depth_calc()
    rc_calc()
    chain_tree('chain_before.png', 'BEFORE — no chain-CE',
               'one structural edge called class 0 → the chain cannot be assembled', bad=True)
    chain_tree('chain_after.png', 'AFTER — with chain-CE',
               'every structural edge classified correctly → the chain is recovered')
    for nm, hi, nt in STRIPS:
        strip(nm, hi, nt)
    p0_stages()
    p0_prune()
    p0_recon()
    directions()
