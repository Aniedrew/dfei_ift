#!/usr/bin/env python3
"""PV association accuracy versus version.

The metric is the **mis-association rate** written by
`wmpgnn/performance/reco_accuracy.py::acc_pv_asso` into
`LHCb_logs/DFEI/version_<v>/info_<signal>_<tag>_reco.txt`:

    mis-assoc [%] = 100 - (correctly assigned tracks) / (all tracks) * 100

Lower is better; the minIP baseline (assign every track to the PV with the
smallest impact parameter) is printed alongside in the same file and is the
reference the DFEI paper quotes.

Which eval counts as "the" number for a version
------------------------------------------------
Each version dir can hold several evals, tagged by `evaluate.over_write`
(appended to the file name after `__`).  We take the one whose tag equals the
version's own training-time `over_write` from its `hparams.yaml` -- i.e. the
evaluation the training job ran itself.  When that is empty (older versions
before over_write was used) we fall back to the bare `<signal>` file.

Output
------
  report_figs/pv_asso_vs_version.png   the figure
  report_figs/pv_asso_vs_version.csv   the numbers behind it
"""
import glob
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import yaml

BASE = '/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn'
FIG = BASE + '/report_figs'
SIGNAL = 'inclusive_00342442'

BLUE = '#1F4E79'
DARK = '#2B2B2B'
GRAY = '#777777'
GREEN = '#2E8B57'
ORANGE = '#E8A33D'
RED = '#C0392B'
LIGHT = '#EAEFF5'
plt.rcParams.update({'font.size': 12, 'figure.dpi': 200, 'text.color': DARK,
                     'axes.edgecolor': '#BBBBBB', 'xtick.color': DARK, 'ytick.color': DARK})

# one record per (block, HGNN, minIP); the file has 6-7 blocks per eval
BLOCK = re.compile(
    r'^(\S+):[ \t]*\n'
    r'HGNN association[ ]*:[ ]*([\d.]+)[ ]*\+/-[ ]*([\d.]+)[ ]*\n'
    r'minIP association[ ]*:[ ]*([\d.]+)[ ]*\+/-[ ]*([\d.]+)', re.M)


def parse(path):
    out = {}
    for m in BLOCK.findall(open(path, errors='ignore').read()):
        out[m[0]] = dict(hgnn=float(m[1]), herr=float(m[2]),
                         minip=float(m[3]), iperr=float(m[4]))
    return out


def canonical_file(version):
    """The eval tagged with the version's own training over_write, else the bare one."""
    h = f'{BASE}/LHCb_logs/DFEI/version_{version}/hparams.yaml'
    tag = ''
    if os.path.exists(h):
        tag = ((yaml.safe_load(open(h)) or {}).get('evaluate') or {}).get('over_write') or ''
    if isinstance(tag, list):
        tag = tag[0] if tag else ''
    cands = sorted(glob.glob(f'{BASE}/LHCb_logs/DFEI/version_{version}/info_{SIGNAL}*_reco.txt'))
    for c in cands:
        suffix = os.path.basename(c).replace(f'info_{SIGNAL}', '').replace('_reco.txt', '')
        suffix = suffix[2:] if suffix.startswith('__') else ''
        if suffix == tag:
            return c, tag or 'default'
    return None, None


def collect():
    rows = []
    for d in sorted(glob.glob(f'{BASE}/LHCb_logs/DFEI/version_*')):
        m = re.search(r'version_(\d+)$', d)
        if not m:
            continue
        v = int(m.group(1))
        path, tag = canonical_file(v)
        if not path:
            continue
        blocks = parse(path)
        if 'all_tracks' not in blocks:
            continue
        a = blocks['all_tracks']
        s = blocks.get('sig_AllParticles', {})
        rows.append(dict(version=v, tag=tag,
                         all_hgnn=a['hgnn'], all_err=a['herr'],
                         all_minip=a['minip'], all_iperr=a['iperr'],
                         sig_hgnn=s.get('hgnn'), sig_err=s.get('herr'),
                         sig_minip=s.get('minip')))
    return pd.DataFrame(rows).sort_values('version').reset_index(drop=True)


def save(fig, name):
    fig.savefig(f'{FIG}/{name}', bbox_inches='tight')
    plt.close(fig)
    print('  ' + name)


def plot(df):
    """One figure per metric: broken y (the interesting spread is 1.6-2.1 % while
    minIP sits at 12.7 / 25-26), x = one slot per version so the tightly packed
    v540-v563 block is not squeezed into a few pixels."""
    for key, xkey, mkey, ttl, col, lo, hi in (
            ('all', 'all_minip', 'all_err', 'all tracks in the event', BLUE, (1.4, 3.1), (3.6, 15.5)),
            ('sig', 'sig_minip', 'sig_err', 'tracks from the signal B', GREEN, (1.4, 4.6), (4.85, 30))):
        sub = df.dropna(subset=[f'{key}_hgnn']).reset_index(drop=True)
        x = list(range(len(sub)))
        fig, (axh, axl) = plt.subplots(
            2, 1, figsize=(10.4, 5.4), sharex=True,
            gridspec_kw=dict(height_ratios=[1.05, 2.3], hspace=0.09))

        for ax in (axh, axl):
            ax.set_axisbelow(True)
            ax.grid(axis='y', color='#EEEEEE', zorder=0)
            for s in ('top', 'right'):
                ax.spines[s].set_visible(False)

        # draw each point on the panel that actually contains it, so neither series
        # leaves clipped error-bar caps floating at the break
        for i in range(len(sub)):
            v = sub[xkey][i]
            (axl if v <= lo[1] else axh).plot(
                [i], [v], marker='o', ms=4.0, ls='none', color=GRAY,
                mfc='white', mew=1.2, zorder=2)
            v = sub[f'{key}_hgnn'][i]
            # a bar crossing the break is capped at it; drawing it in full would
            # only get clipped and leave a cap floating on the wrong panel
            err = min(sub[mkey][i], lo[1] - v) if v <= lo[1] else sub[mkey][i]
            (axl if v <= lo[1] else axh).errorbar(
                [i], [v], yerr=[err], marker='o', ms=5.0, lw=0, ls='none',
                color=col, capsize=2.5, elinewidth=1.0, zorder=4)

        axh.set_ylim(*hi)
        axl.set_ylim(*lo)
        axh.spines['bottom'].set_visible(False)
        axl.spines['top'].set_visible(False)
        axh.tick_params(labelbottom=False, bottom=False)
        d = 0.012
        for ax, y in ((axh, -d / 1.0), (axl, 1 - d)):
            ax.plot((-d, +d), (y - d * 0.6, y + d * 0.6), transform=ax.transAxes,
                    color='#BBBBBB', lw=1.1, clip_on=False)
            ax.plot((1 - d, 1 + d), (y - d * 0.6, y + d * 0.6), transform=ax.transAxes,
                    color='#BBBBBB', lw=1.1, clip_on=False)

        axh.set_title(ttl, fontsize=12.5, fontweight='bold', color=col, pad=6)
        axh.annotate('minIP baseline', (len(sub) * 0.45, min(sub[xkey]) - 1.2),
                     xytext=(0, 0), textcoords='offset points', ha='center', va='top',
                     fontsize=10.5, color=GRAY, fontweight='bold')
        axl.set_ylabel('DFEI mis-association  [%]', fontsize=11.5, color=col, fontweight='bold')
        axl.yaxis.set_label_coords(-0.055, 1.09)
        fig.text(0.5, -0.02, 'each point = the version\'s own end-of-training eval  ·  '
                             'threshold 0.9  ·  20 files  ·  bars = binomial error',
                 ha='center', va='top', fontsize=9.5, color=GRAY, style='italic')

        # version numbers below the lower panel; untangle nothing, just rotate
        axl.set_xticks(x)
        axl.set_xticklabels([str(v) for v in sub['version']], rotation=90, fontsize=8.5)
        axl.set_xlim(-1, len(sub))
        axl.set_xlabel('version', fontsize=11)

        # label the turns of the main line
        for v in (31, 47, 557, 563):
            i = sub.index[sub['version'] == v]
            if not len(i):
                continue
            i = i[0]
            val = sub[f'{key}_hgnn'][i]
            ax = axh if val > lo[1] else axl
            if v == 31:
                off, ha = (9, -3), 'left'
            else:
                off, ha = ((0, 10) if val > lo[1] else (0, -18)), 'center'
            ax.annotate(f'v{v}', (i, val), textcoords='offset points', xytext=off,
                        ha=ha, fontsize=10, color=col, fontweight='bold')
        save(fig, f'pv_asso_vs_version_{key}.png')


def main():
    os.makedirs(FIG, exist_ok=True)
    df = collect()
    df.to_csv(f'{FIG}/pv_asso_vs_version.csv', index=False)
    pd.set_option('display.width', 200)
    pd.set_option('display.max_rows', 200)
    print(df.to_string(index=False))
    plot(df)
    print(f'\n{len(df)} versions with a PV-association eval')


if __name__ == '__main__':
    main()
