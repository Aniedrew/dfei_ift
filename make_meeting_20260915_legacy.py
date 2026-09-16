#!/usr/bin/env python3
"""Re-render the legacy 0902 figures at slide size.

The 0902 figures were drawn on an 11-in (or 12.5-in) canvas.  On this deck they
are shown about 5.3 in wide, i.e. at 0.48 of their design size, which shrinks a
11 pt label to 5.3 pt.  Re-running the same script on a smaller canvas with the
fonts scaled by FS makes the figure land on the slide at scale ~1.0 with the
labels at (11 * FS) pt.

Nothing about the content changes: same panels, same data, same wording.

Two font settings are produced for every script (`a` = compact, `b` = roomy) so
the panels can be picked per figure; diagrams need `a`, plain charts are fine
with `b`.
"""
import os
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.axes as maxes
from matplotlib.figure import Figure

BASE = '/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn'
OUT = BASE + '/meeting_figs_20260915/legacy'

_o_subplots = plt.subplots
_o_figure = plt.figure
_o_savefig = Figure.savefig
_o_text = maxes.Axes.text
_o_annotate = maxes.Axes.annotate
_o_set_title = maxes.Axes.set_title
_o_set_xlabel = maxes.Axes.set_xlabel
_o_set_ylabel = maxes.Axes.set_ylabel
_o_legend = maxes.Axes.legend
_o_suptitle = Figure.suptitle
_o_figtext = Figure.text

FS = 1.0


def _kw(kw):
    if 'fontsize' in kw and kw['fontsize'] is not None:
        kw['fontsize'] = kw['fontsize'] * FS
    return kw


def install(k, fs, suptitle=True):
    global FS
    FS = fs

    def subplots(*a, **kw):
        figsize = kw.get('figsize', plt.rcParams['figure.figsize'])
        kw['figsize'] = tuple(v * k for v in figsize)
        return _o_subplots(*a, **kw)

    def figure(*a, **kw):
        figsize = kw.get('figsize', plt.rcParams['figure.figsize'])
        kw['figsize'] = tuple(v * k for v in figsize)
        return _o_figure(*a, **kw)

    def savefig(self, *a, **kw):
        kw['dpi'] = 240
        return _o_savefig(self, *a, **kw)

    plt.subplots = subplots
    plt.figure = figure
    Figure.savefig = savefig
    for name, orig in (('text', _o_text), ('annotate', _o_annotate),
                       ('set_title', _o_set_title), ('set_xlabel', _o_set_xlabel),
                       ('set_ylabel', _o_set_ylabel), ('legend', _o_legend)):
        def make(orig=orig):
            def meth(self, *a, **kw):
                return orig(self, *a, **_kw(kw))
            return meth
        setattr(maxes.Axes, name, make())
    Figure.suptitle = (lambda self, *a, **kw: _o_suptitle(self, *a, **_kw(kw))
                       if suptitle else None)
    Figure.text = lambda self, *a, **kw: _o_figtext(self, *a, **_kw(kw))
    plt.rcParams['xtick.labelsize'] = 10 * fs
    plt.rcParams['ytick.labelsize'] = 10 * fs
    plt.rcParams['axes.labelsize'] = 12 * fs
    plt.rcParams['axes.titlesize'] = 13 * fs


def _tight():
    """Give the axes the whole canvas, minus the strip a suptitle needs."""
    try:
        fig = plt.gcf()
        if getattr(fig, '_suptitle', None) is not None:
            fig.tight_layout(rect=[0, 0, 1, 0.90])
        else:
            fig.tight_layout()
    except Exception:
        pass


# small, purely cosmetic adjustments applied to the *copy* of the script
PATCHES = {
    'make_chain_lca_figs.py': [
        # no room for this annotation once the canvas is at slide size (it lands on the bar
        # labels); the orange class-2 bar and the slide text already say the same thing
        ("ax.annotate('class 2 = rare AND hardest\\n→ structural bottleneck', xy=(2, 49.17),\n"
         "            xytext=(0.55, 22), fontsize=11, color='#222222',\n"
         "            arrowprops=dict(arrowstyle='->', color=ORANGE))",
         "# annotation dropped: at slide size it overlaps the bar labels"),
        # the 99.9% label of class 0 was drawn through the top spine (it sits at f*1.5 on a
        # log axis that tops out at 3): give the axis headroom instead
        ("ax.set_yscale('log'); ax.set_ylim(1e-6, 3)",
         "ax.set_yscale('log'); ax.set_ylim(1e-6, 40)"),
    ],
    'make_pruning_figs.py': [
        ("axes[0].set_title('mask = σ((w − cut)/τ)\\n(how much of w survives)', fontsize=13",
         "axes[0].set_title('mask = σ((w − cut)/τ)\\n(how much of w survives)', fontsize=10"),
        ("axes[1].set_title('w_eff = w · σ((w − cut)/τ)\\n(effective weight used in training)', fontsize=13",
         "axes[1].set_title('w_eff = w · σ((w − cut)/τ)\\n(effective weight used in training)', fontsize=10"),
        ("axes[0].legend(fontsize=11)", "axes[0].legend(fontsize=9, loc='lower right')"),
        ("' cut = 0.85\\n(fixed per run)', color='#222222', fontsize=10",
         "' cut = 0.85\\n(fixed per run)', color='#222222', fontsize=8"),
        ("' 0.9 = inference\\nthreshold', color='#222222', fontsize=10",
         "' 0.9 = inference\\nthreshold', color='#222222', fontsize=8"),
        ("'τ small → behaves like\\nhard pruning at cut', color='#222222', fontsize=11",
         "'τ small → behaves like\\nhard pruning at cut', color='#222222', fontsize=9"),
        ("ax.set_xlabel('w  (node/edge confidence, 0–1)', fontsize=12)",
         "ax.set_xlabel('w  (node/edge confidence, 0–1)', fontsize=10)"),
        ("axes[0].set_ylabel('mask value', fontsize=12)", "axes[0].set_ylabel('mask value', fontsize=10)"),
        ("axes[1].set_ylabel('w_eff', fontsize=12)", "axes[1].set_ylabel('w_eff', fontsize=10)"),
    ],
    'make_meeting_extra_figs.py': [
        # the rotated ylabel is clipped once the canvas is small
        ("ax.set_ylabel('linear-probe R²', fontsize=12)",
         "ax.set_ylabel('linear-probe R²', fontsize=10)"),
        # the 0.930 label of the tall bar ran through the top spine
        ("ax.set_ylim(0, 1.08)", "ax.set_ylim(0, 1.35)"),
        # rc_calc: tree to the left, the annotation block to the right, so nothing overlaps
        ("nodes = {'B': (55, 40), 'J/psi': (24, 22), 'K': (55, 12), 'pi': (86, 22)}",
         "nodes = {'B': (26, 32), 'J/psi': (10, 17), 'K': (26, 6), 'pi': (42, 17)}"),
        ("ax.text(55, 46, 'root the tree at each candidate v → subtree sizes τ(u)', ha='center', fontsize=10.5, color='#222222')",
         "ax.text(52, 44, 'root the tree at each candidate v → subtree sizes τ(u)', ha='left', fontsize=10.5, color='#222222')"),
        ("ax.text(55, 30, 'rooted at B:  τ = {1, 1, 1, 4}', ha='center', fontsize=10, color='#222222')",
         "ax.text(52, 36, 'rooted at B:  τ = {1, 1, 1, 4}', ha='left', fontsize=10, color='#222222')"),
        ("ax.text(55, 25, 'log R(B) = −(log1+log1+log1+log4) = −1.39', ha='center', fontsize=10, color='#222222')",
         "ax.text(52, 28, 'log R(B) = −(log1+log1+log1+log4) = −1.39', ha='left', fontsize=10, color='#222222')"),
        ("ax.text(55, 15, 'rooted at J/psi: log R = −(log3+log4+log1+log1) = −2.48', ha='center', fontsize=10, color='#222222')",
         "ax.text(52, 20, 'rooted at J/psi: log R = −(log3+log4+log1+log1) = −2.48', ha='left', fontsize=10, color='#222222')"),
        ("ax.text(55, 6, 'root = argmax log R → B  (log R(v) = −Σ_u log τ_v(u), Shah & Zaman)',\n        ha='center', fontsize=10, color='#222222')",
         "ax.text(52, 12, 'root = argmax log R → B  (log R(v) = −Σ_u log τ_v(u), Shah & Zaman)',\n        ha='left', fontsize=10, color='#222222')"),
    ],
}


def run(script, k, fs, tag, suptitle=True):
    src = open(os.path.join(BASE, script)).read()
    out = '%s/%s' % (OUT, tag)
    os.makedirs(out, exist_ok=True)
    src = re.sub(r"FIG = '[^']*'", "FIG = %r" % out, src, count=1)
    for old, new in PATCHES.get(script, ()):
        if old not in src:
            raise SystemExit('[!] patch did not match in %s: %r' % (script, old[:60]))
        src = src.replace(old, new)
    src = src.replace('plt.savefig(', '_tight(); plt.savefig(')
    install(k, fs, suptitle)
    exec(compile(src, script, 'exec'), {'__name__': '__main__', '_tight': _tight})


# canvas 11 in -> 6.8 in   (k = 0.62);  canvas 12.5 in -> 6.8 in (k = 0.545)
JOBS = [('make_chain_lca_figs.py', 0.62, 0.90, 'slide_size', False),
        ('make_pruning_figs.py', 0.62, 0.90, 'slide_size', False),
        ('make_meeting_extra_figs.py', 0.62, 0.90, 'slide_size', False),
        ('make_chain_lca_figs.py', 0.85, 0.90, 'wide', False),
        ('make_meeting_extra_figs.py', 0.85, 0.90, 'wide', False)]

for script, k, fs, tag, sup in JOBS:
    print('--- %s  k=%.3f  fs=%.2f  supertitle=%s -> %s' % (script, k, fs, sup, tag))
    run(script, k, fs, tag, sup)
print('[ok] done')
