#!/usr/bin/env python3
"""Part maps for the 2026-09-15 DFEI deck — graph style (nodes / edges / dashed bands).

Design rules (after feedback):
  * nodes are circles that contain ONLY the version number; colour = PerfectReco
  * no paragraphs inside the map — at most one short line per node
  * edges carry the number of epochs of that step
  * dashed boxes mark the parts / groups
  * the sub-maps are VERTICAL (the map sits beside the text on the slide)

  map_master.png   the main line + six group boxes (2 rows of 3)
  map_part1.png    v31 -> v38, vertical chain
  map_part2.png    v38 -> v46 / v47, vertical
  map_part3.png    the six groups, no spine
  map_diag.png     this month, vertical
  plateau.png / works_apart.png   the evidence figures
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

FIG = '/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn/meeting_figs_20260915'
os.makedirs(FIG, exist_ok=True)
BLUE, DARK, GRAY, RED, GREEN, ORANGE = '#1F4E79', '#333333', '#777777', '#C0392B', '#2CA02C', '#E8A33D'
PURPLE, TEAL, PINK, GOLD = '#7A44AA', '#0B7285', '#C2185B', '#B8860B'
plt.rcParams.update({'font.size': 9})

SC = {31: 12.30, 36: 18.77, 37: 19.50, 38: 21.10, 46: 21.28, 47: 23.15, 48: 21.40,
      50: 19.00, 51: 20.00, 52: 20.00, 53: 15.84, 54: 17.78, 510: 21.75, 511: 22.11,
      512: 13.34, 513: 13.00, 514: 15.35, 515: 22.95, 516: 22.72, 517: 22.00, 518: 22.52,
      520: 22.69, 500: 17.90, 504: 19.13, 507: 19.62, 509: 17.98, 530: 17.71, 549: 19.47,
      551: 23.19, 553: 22.84, 27: 22.76, 45: 8.06, 49: 5.95, 60: 23.43, 61: 23.40,
      39: 14.0, 40: 18.45, 41: 18.45, 42: 19.07, 521: 19.10, 540: 22.04}
VMIN, VMAX = 5.0, 23.5
CM = plt.get_cmap('viridis')


def nc(v):
    p = SC.get(v)
    return CM((p - VMIN) / (VMAX - VMIN)) if p is not None else '#DDDDDD'


def tc(v):
    p = SC.get(v)
    if p is None:
        return '#666666'
    return 'white' if (p - VMIN) / (VMAX - VMIN) < 0.55 else '#222222'


class M:
    def __init__(self, figsize, xlim=(0, 100), ylim=(0, 100)):
        self.f, self.a = plt.subplots(figsize=figsize)
        self.a.set_xlim(*xlim); self.a.set_ylim(*ylim); self.a.axis('off')

    def n(self, v, x, y, r=3.0, fs=10, ghost=False, lw=1.4):
        self.a.add_patch(Circle((x, y), r, facecolor='#F2F2F2' if ghost else nc(v),
                                edgecolor='#AAB2B8' if ghost else '#2C3E50', lw=lw, zorder=6))
        self.a.text(x, y, str(v), ha='center', va='center', fontsize=fs, fontweight='bold',
                    color='#9A9A9A' if ghost else tc(v), zorder=7)

    def e(self, p0, p1, lab='', col='#6E7B87', ls='-', lw=1.3, lx=0, ly=0, fs=6.8, rad=0.0):
        self.a.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=11, color=col,
                                         lw=lw, linestyle=ls, zorder=3, shrinkA=6, shrinkB=7,
                                         connectionstyle=f'arc3,rad={rad}'))
        if lab:
            self.a.text((p0[0] + p1[0]) / 2 + lx, (p0[1] + p1[1]) / 2 + ly, lab, ha='center',
                        va='center', fontsize=fs, color=col, zorder=9,
                        bbox=dict(fc='white', ec='none', alpha=0.9, pad=0.7))

    def band(self, x0, y0, x1, y1, title, col, tpos='top'):
        self.a.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                        boxstyle='round,pad=0.6,rounding_size=1.6',
                                        fc=col, ec=col, alpha=0.05, lw=0, zorder=1))
        self.a.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                        boxstyle='round,pad=0.6,rounding_size=1.6',
                                        fc='none', ec=col, lw=1.4, ls=(0, (5, 3)), zorder=2))
        if title:
            tx, ty, va = ((x0 + x1) / 2, y1 + 1.6, 'bottom') if tpos == 'top' else (x0 + 1.4, y1 - 1.4, 'top')
            self.a.text(tx, ty, title, ha='center' if tpos == 'top' else 'left', va=va,
                        fontsize=8.8, color=col, fontweight='bold', zorder=9,
                        bbox=dict(fc='white', ec='none', alpha=0.88, pad=0.9))

    def t(self, x, y, s, col=GRAY, fs=6.8, ha='center', va='center', w='normal'):
        self.a.text(x, y, s, color=col, fontsize=fs, ha=ha, va=va, zorder=9, weight=w,
                    bbox=dict(fc='white', ec='none', alpha=0.88, pad=0.7))

    def save(self, name, cbar=False):
        if cbar:
            sm = plt.cm.ScalarMappable(cmap=CM, norm=mcolors.Normalize(vmin=VMIN, vmax=VMAX))
            cax = self.f.add_axes([0.917, 0.16, 0.012, 0.62])
            cb = self.f.colorbar(sm, cax=cax)
            cb.set_label('PerfectReco  [%]\n(fixed denominator)', fontsize=8, color=DARK)
            cb.ax.tick_params(labelsize=7.5)
        self.f.savefig(FIG + '/' + name, bbox_inches='tight')
        plt.close(self.f)


# ------------------------------------------------------------------ master
def map_master():
    m = M((13.4, 4.9), (0, 122), (23, 106))
    # ---------------- main line
    m.band(2, 76, 86, 103, 'PART 1 + 2   the main line', BLUE, tpos='in')
    for v, x in ((31, 10), (36, 30), (37, 41), (38, 52)):
        m.n(v, x, 88, r=3.0, fs=10.5)
    m.n(47, 73, 88, r=4.2, fs=13, lw=2.3)
    m.n(46, 73, 100, r=2.5, fs=9)
    m.e((10, 88), (30, 88), '3 × 150 ep')
    m.e((30, 88), (41, 88), '150 ep')
    m.e((41, 88), (52, 88), '150 ep')
    m.e((52, 88), (73, 88), '125 ep')
    m.e((52, 90.3), (71, 98.2), '125 ep', rad=0.14, lx=-4.0, ly=1.0)
    m.t(40, 77.6, 'PART 1', BLUE, 8.6, w='bold')
    m.t(73, 77.6, 'PART 2', GOLD, 8.6, w='bold')
    m.t(118, 104.5, 'v47 = the best on this dataset\nAllParticles 39.56  /  PerfectReco 23.15',
        BLUE, 9.0, w='bold', ha='right')
    # ---------------- group boxes: 2 rows x 3, equal spans, never overlapping
    W, GAP, X0 = 37.0, 3.5, 1.0
    rows = [
        (56, 70, [('capacity  (6 runs)', PURPLE, [48, 53, 512, 513, 514]),
                  ('context / attention  (7 runs)', PINK, [510, 511, 515, 516, 517, 520]),
                  ('restructure the event  (4 runs)', '#8E44AD', [39, 40, 41, 42])]),
        (28, 42, [('retrain the recipe  (13 runs)', PURPLE, [500, 504, 507, 509, 530]),
                  ('other dataset  (5 runs)', '#2E8B57', [27, 45, 49, 60, 61]),
                  ('diagnosis  (this month)', TEAL, [549, 551, 553])]),
    ]
    for y0, y1, boxes in rows:
        for i, (title, col, vers) in enumerate(boxes):
            bx0 = X0 + i * (W + GAP)
            bx1 = bx0 + W
            m.band(bx0, y0, bx1, y1, title, col)
            yc = (y0 + y1) / 2
            if len(vers) == 6:
                xs = [bx0 + 5 + j * 5.4 for j in range(6)]
            else:
                xs = [bx0 + (bx1 - bx0) * (j + 1) / (len(vers) + 1) for j in range(len(vers))]
            for v, x in zip(vers, xs):
                m.n(v, x, yc, r=2.4, fs=8.2)
            m.e(((bx0 + bx1) / 2, y1 + 2.2), ((bx0 + bx1) / 2, y1 + 0.4), '', col=col,
                ls=(0, (2, 2)), lw=1.0)
    m.t(60, 25.5, 'circles show representative versions — the count in each title is the total number of runs in '
                  'that group', GRAY, 7.6)
    m.save('map_master.png', cbar=True)


# ------------------------------------------------------------------ part 1 (vertical)
def map_part1():
    m = M((4.6, 6.0), (0, 100), (0, 100))
    seq = [(31, 90), (32, 74), (35, 58), (36, 42), (37, 26), (38, 10)]
    for i, (v, y) in enumerate(seq):
        m.n(v, 46, y, r=4.6, fs=12 if v in (31, 38) else 11,
            lw=2.0 if v in (31, 38) else 1.4)
        if i:
            m.e((46, seq[i - 1][1] - 4.6), (46, y + 4.8), '150 ep', lx=-9.5, ly=0)
    lab = {31: 'class-weight bug fixed', 32: 'fine-tune, lr 1e-4', 35: 'retry of v34',
           36: 'B2 soft mask + source', 37: 'class-2 weight + hinge', 38: 'chain-CE, cut 0.85'}
    for v, y in seq:
        m.t(78, y, lab[v], DARK if v in (36, 37, 38) else GRAY, 7.4, ha='center')
    m.t(50, 99, 'training-time changes only', BLUE, 8.2, w='bold')
    m.save('map_part1.png')


# ------------------------------------------------------------------ part 2 (vertical)
def map_part2():
    m = M((5.4, 5.0), (0, 100), (0, 100))
    m.n(38, 50, 88, r=5.0, fs=13, lw=2.2)
    m.e((44, 86), (26, 52), '125 ep', rad=0.10, lx=-5.5, ly=1.5)
    m.e((56, 86), (74, 52), '125 ep', rad=-0.10, lx=5.5, ly=1.5)
    m.n(46, 20, 44, r=4.0, fs=11)
    m.n(47, 80, 44, r=5.6, fs=15, lw=2.6)
    m.t(20, 33, 'mass head v1\nnot normalised', GRAY, 7.4)
    m.t(80, 33, 'mass head v2\nlog10 + mask fix', GOLD, 7.6, w='bold')
    m.t(50, 14, 'probe R²  0.003 → 0.930', GOLD, 8.0, w='bold')
    m.t(50, 5, 'PerfectReco  21.10 → 23.15', DARK, 8.2)
    m.save('map_part2.png')


# ------------------------------------------------------------------ part 3 (6 groups)
def map_part3():
    m = M((12.6, 3.6), (0, 104), (0, 46))
    boxes = [
        (2, 'capacity  (6 runs)', PURPLE, [(48, 7), (53, 15), (512, 23), (513, 31), (514, 39)]),
        (37, 'context / attention  (7 runs)', PINK, [(510, 42), (511, 50), (515, 58), (516, 66), (517, 74), (520, 82)]),
        (72, 'restructure the event  (4 runs)', '#8E44AD', [(39, 77), (40, 85), (41, 93), (42, 101)]),
        (2, 'retrain the recipe  (13 runs)', PURPLE, [(500, 7), (504, 15), (507, 23), (509, 31), (530, 39)]),
        (37, 'other dataset  (5 runs)', '#2E8B57', [(27, 42), (45, 50), (49, 58), (60, 66), (61, 74), (521, 82)]),
        (72, 'diagnosis  (this month)', TEAL, [(549, 77), (551, 86), (553, 95), (540, 103)]),
    ]
    for i, (x0, title, col, nodes) in enumerate(boxes):
        y0, y1 = (25, 43) if i < 3 else (2, 20)
        xs = [x for _, x in nodes]
        m.band(x0, y0, max(xs) + 3.4, y1, title, col, tpos='in')
        for v, x in nodes:
            m.n(v, x, (y0 + y1) / 2, r=2.3, fs=8.0)
    m.save('map_part3.png')


# ------------------------------------------------------------------ diagnosis (vertical)
def map_diag():
    m = M((5.2, 5.0), (0, 100), (0, 100))
    m.n(38, 14, 88, r=4.4, fs=11.5)
    m.n(507, 72, 88, r=4.4, fs=11.5)
    m.n(553, 14, 56, r=4.4, fs=11.5)
    m.n(549, 66, 56, r=3.6, fs=10)
    m.n(551, 62, 24, r=5.8, fs=15, lw=2.6)
    m.e((14, 83.4), (14, 60.6), '20 ep', lx=10.5, ly=0)
    m.e((72, 83.4), (68, 60.6), '20 ep', lx=10.5, ly=0)
    m.e((67, 53.4), (63, 30.0), '20 ep', col=TEAL, lw=2.0, lx=-10.0, ly=0, fs=7.4)
    for i, v in enumerate([552, 554, 555, 556, 557, 558, 559, 560, 561]):
        m.n(v, 6 + i * 5.4, 6, r=1.5, fs=6.8, ghost=True)
    m.t(50, 96, 'pruning-loss weights', TEAL, 7.4, w='bold')
    m.t(50, 12, '9 arms in flight', TEAL, 7.4, w='bold')
    m.t(78, 24, 'lr 3e-4:\nties v47', GOLD, 7.6, w='bold')
    m.save('map_diag.png')


# ------------------------------------------------------------------ evidence figures
def plateau():
    fig, axes = plt.subplots(2, 1, figsize=(5.7, 5.0), gridspec_kw={'height_ratios': [1.25, 1.0]})
    ax = axes[0]
    fam = [('capacity / width', PURPLE, [48, 53, 54, 512, 514]),
           ('context / attention', PINK, [510, 511, 515, 516, 517, 518, 520]),
           ('from the weak ablation base', '#7A44AA', [500, 504, 507, 509, 530, 549])]
    ALL = {31: 22.49, 36: 35.17, 37: 36.21, 38: 37.59, 46: 38.22, 47: 39.56, 48: 37.36,
           53: 26.78, 54: 30.18, 510: 38.01, 511: 38.39, 512: 24.03, 514: 25.44, 515: 39.04,
           516: 38.95, 517: 38.12, 518: 38.73, 520: 38.87, 500: 33.34, 504: 33.69, 507: 34.02,
           509: 33.27, 530: 32.58, 549: 33.92, 551: 39.92}
    seq = [(v, c) for _, c, vs in fam for v in vs] + [(551, TEAL)]
    x = list(range(len(seq)))
    ax.axhspan(ALL[47] - 0.43, ALL[47] + 0.43, color=GOLD, alpha=0.18, zorder=0)
    ax.axhline(ALL[47], color=GOLD, lw=1.6, zorder=1)
    b = ax.bar(x, [ALL[v] for v, _ in seq], color=[c for _, c in seq], edgecolor='#444444', zorder=2)
    ax.bar_label(b, fmt='%.1f', fontsize=8.0, padding=1, rotation=90)
    ax.text(0.2, 45.5, 'v47 = 39.56  ± 0.43 (run-to-run noise)', fontsize=8.4, color=GOLD,
            ha='left', va='top', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels([str(v) for v, _ in seq], fontsize=8.4, rotation=90)
    for i, (v, c) in enumerate(seq):
        ax.get_xticklabels()[i].set_color(c)
    ax.set_ylabel('AllParticles  [%]', fontsize=9.0)
    ax.set_ylim(0, 47); ax.grid(axis='y', ls=':', color='#DDDDDD'); ax.set_axisbelow(True)
    ax.set_title('16 runs after v47 — one ties it, none passes it', fontsize=12.0, color=BLUE,
                 fontweight='bold', pad=30)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=c) for _, c, _ in fam] +
                      [plt.Rectangle((0, 0), 1, 1, fc=TEAL)],
              labels=[n for n, _, _ in fam] + ['new base + 10× lr'], fontsize=7.2, frameon=False,
              loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.14))
    ax = axes[1]
    items = [('v512: latent 16 → 32/24', 512), ('v514: GN width 128 → 256', 514),
             ('v48: 3 heads at once', 48), ('v516: +60 epochs on v515', 516),
             ('v519/v520: 179 epochs on v47', 520)]
    PERF = {48: 21.40, 512: 13.34, 514: 15.35, 516: 22.72, 520: 22.69, 47: 23.15}
    d = [PERF[v] - PERF[47] for _, v in items]
    b = ax.barh(range(len(d)), d, color=[PURPLE, PURPLE, PURPLE, PINK, PINK], edgecolor='#444444')
    ax.bar_label(b, fmt='%+.2f pp', fontsize=8.0, padding=2, label_type='edge')
    ax.axvline(-0.43, ls='--', color=GRAY, lw=1.1)
    ax.set_yticks(range(len(d))); ax.set_yticklabels([t for t, _ in items], fontsize=7.6)
    ax.invert_yaxis(); ax.set_xlim(-14, 3)
    ax.grid(axis='x', ls=':', color='#DDDDDD'); ax.set_axisbelow(True)
    ax.set_xlabel('PerfectReco relative to v47  [pp]', fontsize=8.5)
    ax.set_title('more epochs, more heads, more width — all below v47', fontsize=12.0, color=BLUE,
                 fontweight='bold', pad=10)
    fig.suptitle('The plateau is real: on this dataset the recipe is exhausted',
                 fontsize=12.5, color=BLUE, fontweight='bold', y=0.99)
    fig.tight_layout(); fig.savefig(FIG + '/plateau.png', bbox_inches='tight', dpi=200); plt.close(fig)


def works_apart():
    P = {38: 21.10, 47: 23.15, 500: 17.90, 504: 19.13, 507: 19.62, 511: 22.11, 551: 23.19, 553: 22.84}
    fig, axes = plt.subplots(2, 1, figsize=(5.7, 5.0), gridspec_kw={'height_ratios': [1.15, 1.0]})
    ax = axes[0]
    lab = ['chain-CE\n(v500 → v504)', 'edge-loss weight 33 → 3\n(v38 → v553)',
           'self-attention + edge bias\n(v38 → v511)', 'learning rate 3e-5 → 3e-4\n(v507 → v551)']
    d = [P[504] - P[500], P[553] - P[38], P[511] - P[38], P[551] - P[507]]
    b = ax.barh(range(4), d, color=[GREEN, GREEN, GREEN, ORANGE], edgecolor='#444444')
    ax.bar_label(b, fmt='+%.2f pp', fontsize=9.0, padding=3)
    ax.axvline(0.43, ls='--', color=GRAY, lw=1.2)
    ax.text(0.48, -0.75, 'noise floor ±0.43 pp', fontsize=7.6, color=GRAY)
    ax.set_yticks(range(4)); ax.set_yticklabels(lab, fontsize=8.0)
    ax.invert_yaxis(); ax.set_xlim(0, 4.6)
    ax.grid(axis='x', ls=':', color='#DDDDDD'); ax.set_axisbelow(True)
    ax.set_xlabel('PerfectReco gain  [pp]', fontsize=8.5)
    ax.set_title('These four DID work — each clears the noise floor', fontsize=12.0, color=BLUE,
                 fontweight='bold', pad=10)
    ax = axes[1]
    ax.axis('off')
    ax.text(0.0, 0.95, 'But they do not add up', fontsize=11.0, color=BLUE, fontweight='bold',
            va='top', transform=ax.transAxes)
    lines = [('chain-CE + struct (v507)', '34.02 / 19.62', '#444444'),
             ('+ mass + mom (v549)', '33.92 / 19.47', RED),
             ('B2 + source (v530)', '32.58 / 17.71', RED),
             ('B2 + cl2 + chain-CE (v521)', '≈ v504', RED),
             ('attention + mass (v517)', '38.12 / 22.00', RED),
             ('the four winners (v556)', 'running', TEAL)]
    y = 0.86
    for t, v, c in lines:
        ax.text(0.0, y, t, fontsize=8.0, color=DARK, va='center', transform=ax.transAxes)
        ax.text(1.0, y, v, fontsize=8.0, color=c, va='center', ha='right', fontweight='bold',
                transform=ax.transAxes)
        y -= 0.145
    ax.text(0.0, 0.05, 'Combining the confirmed winners has never produced a gain.', fontsize=8.0,
            color=GRAY, va='center',
            transform=ax.transAxes, style='italic')
    fig.suptitle('Optimizations do work — just not on top of each other',
                 fontsize=12.5, color=BLUE, fontweight='bold', y=0.99)
    fig.tight_layout(); fig.savefig(FIG + '/works_apart.png', bbox_inches='tight', dpi=200); plt.close(fig)


if __name__ == '__main__':
    map_master(); print('ok map_master')
    map_part1(); print('ok map_part1')
    map_part2(); print('ok map_part2')
    map_part3(); print('ok map_part3')
    map_diag(); print('ok map_diag')
    plateau(); print('ok plateau')
    works_apart(); print('ok works_apart')
