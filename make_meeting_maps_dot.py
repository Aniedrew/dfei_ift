#!/usr/bin/env python3
"""All deck maps, drawn with graphviz `dot` in the style of docs/version_lineage.pdf.

Every version is its own node; node colour = PerfectReco; node label = version number only.

  m_master.png   the complete lineage, landscape, fills one slide
  m_p1.png       v31 -> v38       the recipe that worked
  m_p2.png       v38 -> v47       physics into the representation
  m_cap.png      capacity
  m_attn.png     context / attention
  m_res.png      restructuring the event
  m_ret.png      retraining the recipe
  m_pub.png      another dataset
  m_diag.png     the diagnosis
  cbar.png       PerfectReco colour bar
"""
import os
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

FIG = '/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn/meeting_figs_20260915'
TMP = '/tmp/dfei_maps'
os.makedirs(FIG, exist_ok=True)
os.makedirs(TMP, exist_ok=True)

# PerfectReco [%] per truth B candidate.  None = not evaluated.
P = {31: 12.30, 36: 18.77, 37: 19.50, 38: 21.10, 46: 21.28, 47: 23.15, 48: 21.40,
     53: 15.84, 512: 13.34, 514: 15.35,
     510: 21.75, 511: 22.11, 515: 22.95, 516: 22.72, 517: 22.00, 518: 22.52, 520: 22.69,
     27: 22.76, 45: 8.06, 49: 5.95, 60: 23.43,
     500: 17.90, 504: 19.13, 505: 19.13, 506: 19.23, 507: 19.62, 509: 17.98, 530: 17.71,
     540: 22.04, 549: 19.47, 551: 23.19, 553: 22.84}
VMIN, VMAX = 5.0, 23.5
CM = plt.get_cmap('viridis')
GREY = '#E4E4E4'


def col(v):
    p = P.get(v)
    return mcolors.to_hex(CM((p - VMIN) / (VMAX - VMIN))) if p is not None else GREY


def tcol(v):
    p = P.get(v)
    if p is None:
        return '#8A8A8A'
    return '#FFFFFF' if (p - VMIN) / (VMAX - VMIN) < 0.55 else '#1A1A1A'


def N(v, fs=12, w=0.30, h=0.22):
    return ('  n%d [label="%d", fillcolor="%s", fontcolor="%s", fontsize=%d, width=%.2f, height=%.2f];\n'
            % (v, v, col(v), tcol(v), fs, w, h))


def E(a, b, lab='', col_='#7C8894', w=1.0, dotted=False, dashed=False):
    st = ''
    if dotted:
        st = ', style=dotted, arrowhead=none'
    elif dashed:
        st = ', style=dashed'
    la = ', label="%s"' % lab if lab else ''
    na = 'n%d' % a if isinstance(a, int) else a
    nb = 'n%d' % b if isinstance(b, int) else b
    return '  %s -> %s [color="%s", penwidth=%.1f%s%s];\n' % (na, nb, col_, w, st, la)


def HDR(ns=0.20, rs=0.55, rankdir='TB', fs=12, nw=0.30, nh=0.22):
    return ('digraph G {\n  bgcolor="transparent";\n  newrank=true;\n  rankdir=%s; nodesep=%.2f; ranksep=%.2f;\n'
            '  graph [fontname="Helvetica", fontsize=11, compound=true];\n'
            '  node [shape=box, style="rounded,filled", fontname="Helvetica-Bold",\n'
            '        width=%.2f, height=%.2f, margin="0.04,0.02", penwidth=1.2, color="#2C3E50",\n'
            '        fontsize=%d];\n'
            '  edge [color="#7C8894", arrowsize=0.6, penwidth=1.0, fontname="Helvetica", fontsize=11,\n'
            '        fontcolor="#5A646E"];\n' % (rankdir, ns, rs, nw, nh, fs))


def CLUSTER(tag, title, c, nodes, same=True, fs=11, rows=None, invis=None):
    """rows: list of version lists, each forced onto one rank (inside the cluster).
    invis: list of (a, b) pairs joined by an invisible edge (inside the cluster)."""
    s = ('  subgraph cluster_%s {\n    label="%s"; labeljust="l"; fontsize=%d;\n'
         '    fontname="Helvetica-Bold"; style="rounded,dashed"; color="%s"; penwidth=1.4; margin=10;\n'
         % (tag, title, fs, c))
    for v, sz in nodes:
        s += N(v, sz)
    if same and len(nodes) > 1:
        s += '    { rank=same; %s }\n' % '; '.join('n%d' % v for v, _ in nodes)
    for row in (rows or []):
        s += '    { rank=same; %s }\n' % '; '.join('n%d' % v for v in row)
    for a, b in (invis or []):
        s += '    n%d -> n%d [style=invis];\n' % (a, b)
    return s + '  }\n'


def render(name, src, dpi=200):
    dp = os.path.join(TMP, name + '.dot')
    open(dp, 'w').write(src)
    out = os.path.join(FIG, name + '.png')
    subprocess.run(['dot', '-Tpng', '-Gdpi=%d' % dpi, dp, '-o', out], check=True)
    from PIL import Image
    w, h = Image.open(out).size
    print('%-12s %5dx%-5d (%.2f:1)' % (name, w, h, w / h))


FAIL = '#C0392B'
NEWC = '#0B7285'
PUB = '#2E8B57'
ABL = '#7A44AA'
MAINC = '#1F4E79'
HEADC = '#B8860B'
CAPC = '#8E44AD'
ATTC = '#C2185B'


# =========================================================== the complete map
def master():
    """Every version, one node each, no verbose annotations."""
    s = HDR(ns=0.10, rs=0.32, rankdir='TB', fs=11, nw=0.28, nh=0.20)
    s += '  ordering=out; splines=true; concentrate=true;\n'

    # ---- early
    s += CLUSTER('early', 'early', '#9AA0A6', [(23, 11), (25, 11), (30, 11)])
    # ---- main line
    s += CLUSTER('main', 'main line', MAINC,
                 [(31, 13), (32, 11), (34, 10), (35, 11), (36, 12), (37, 12), (38, 13),
                  (46, 11), (47, 15)], same=False)
    s += E(31, 32); s += E(32, 34, dashed=True, col_=FAIL); s += E(32, 35)
    s += E(35, 36); s += E(36, 37); s += E(37, 38)
    s += E(38, 46); s += E(38, 47, col_=HEADC, w=2.0); s += E(38, 48, dashed=True, col_=FAIL)
    # ---- public
    s += CLUSTER('pub', 'public data', PUB,
                 [(27, 12), (45, 11), (49, 11), (60, 12), (61, 11), (533, 11)])
    s += E(23, 27, col_=PUB); s += E(27, 60, col_=PUB); s += E(27, 533, col_=PUB)
    s += E(60, 61, col_=PUB); s += E(27, 45, col_=PUB); s += E(45, 49, col_=PUB, dashed=True)
    # ---- restructure
    s += CLUSTER('res', 'restructure', '#9B59B6', [(39, 11), (40, 11), (41, 11), (42, 11)])
    s += E(38, 39, dashed=True, col_=FAIL); s += E(38, 40, dashed=True, col_=FAIL)
    s += E(38, 41, dashed=True, col_=FAIL); s += E(38, 42, dashed=True, col_=FAIL)
    # ---- capacity
    s += CLUSTER('cap', 'capacity', CAPC, [(53, 11), (512, 11), (513, 10), (514, 11)])
    s += E(38, 53, dashed=True, col_=FAIL); s += E(38, 512, dashed=True, col_=FAIL)
    s += E(38, 513, dashed=True, col_=FAIL); s += E(513, 514, dashed=True, col_=FAIL)
    # ---- attention
    s += CLUSTER('attn', 'context / attention', ATTC,
                 [(510, 11), (511, 12), (515, 11), (516, 11), (517, 11), (518, 11),
                  (519, 10), (520, 11), (532, 10)], same=False)
    s += E(38, 510, col_=ATTC); s += E(38, 511, col_=ATTC, w=1.6)
    s += E(38, 517, col_=ATTC); s += E(517, 519, col_=ATTC)
    s += E(47, 515, col_=ATTC, dashed=True); s += E(515, 516, col_=ATTC, dashed=True)
    s += E(47, 518, col_=ATTC, dashed=True); s += E(518, 520, col_=ATTC, dashed=True)
    s += E(511, 532, col_=ATTC)
    # ---- heads tried on top of the frozen model
    s += CLUSTER('heads', 'heads on v47', HEADC, [(48, 11), (50, 11), (51, 11), (52, 11)])
    s += E(47, 50, col_=HEADC); s += E(47, 51, col_=HEADC); s += E(47, 52, col_=HEADC)
    # ---- ablation / retrain
    s += CLUSTER('abl', 'retrain the recipe', ABL,
                 [(500, 12), (501, 10), (502, 10), (503, 10), (504, 12), (505, 10), (506, 10),
                  (507, 12), (508, 10), (509, 10), (521, 10), (530, 10), (531, 10)], same=False)
    s += E(31, 500, col_=ABL, w=1.6)
    s += E(500, 501, dashed=True, col_=FAIL); s += E(500, 502, dashed=True, col_=FAIL)
    s += E(500, 503, dashed=True, col_=FAIL); s += E(500, 504, col_=ABL, w=1.6)
    s += E(500, 509, col_=ABL); s += E(500, 521, col_=ABL); s += E(500, 530, col_=ABL)
    s += E(530, 531, col_=ABL)
    s += E(504, 505, col_=ABL); s += E(504, 506, col_=ABL)
    s += E(504, 507, col_=ABL, w=1.6); s += E(504, 508, col_=ABL)
    # ---- context-aware pruning head
    s += CLUSTER('ctxpr', 'ctx-aware pruning', '#5D6D7E',
                 [(540, 10), (545, 10), (546, 10), (547, 10), (548, 10)])
    s += E(38, 540, col_=FAIL, dashed=True)
    s += E(38, 545, col_=FAIL, dashed=True); s += E(38, 546, col_=FAIL, dashed=True)
    s += E(38, 547, col_=FAIL, dashed=True); s += E(38, 548, col_=FAIL, dashed=True)
    # ---- this month
    s += CLUSTER('new', 'this month', NEWC,
                 [(549, 10), (550, 10), (551, 15), (552, 10), (553, 12), (554, 10), (555, 10),
                  (556, 10), (557, 10), (558, 10), (559, 10), (560, 10), (561, 10)], same=False)
    s += E(507, 549, col_=NEWC, dashed=True)
    s += E(507, 550, col_=NEWC); s += E(507, 551, col_=NEWC, w=2.2)
    s += E(38, 552, col_=NEWC); s += E(38, 553, col_=NEWC); s += E(38, 554, col_=NEWC)
    s += E(38, 555, col_=NEWC)
    s += E(551, 556, col_=NEWC); s += E(551, 557, col_=NEWC); s += E(551, 558, col_=NEWC)
    s += E(551, 559, col_=NEWC)
    s += E(500, 560, col_=NEWC); s += E(500, 561, col_=NEWC)
    # ---- oracle (an evaluation, not a version)
    s += '  oracle [shape=box, style="rounded,filled", fillcolor="#CDEEF3", fontcolor="#0B4F5C",\n'
    s += '          fontsize=11, width=0.9, label="oracle\\n6 variants"];\n'
    s += E(47, 'oracle', dotted=True, col_=NEWC)
    s += '}\n'
    render('m_master', s)


# =========================================================== part 1
def p1():
    s = HDR(ns=0.30, rs=0.58, fs=15)
    s += CLUSTER('p1', 'PART 1   v31 → v38   (training-time changes only)', MAINC,
                 [(31, 16), (32, 12), (34, 11), (35, 12), (36, 15), (37, 15), (38, 17)],
                 same=False, rows=[[34, 35]])
    s += E(31, 32, '150 ep'); s += E(32, 34, '✗', col_=FAIL, dashed=True)
    s += E(32, 35, '150 ep'); s += E(35, 36, '150 ep')
    s += E(36, 37, '150 ep', col_='#2E7D32', w=1.5)
    s += E(37, 38, '150 ep', col_='#2E7D32', w=1.5)
    s += '}\n'
    render('m_p1', s)


# =========================================================== part 2
def p2():
    s = HDR(ns=0.45, rs=1.60, fs=17)
    s += CLUSTER('p2', 'PART 2   v38 → v46 / v47', HEADC,
                 [(38, 22), (46, 15), (47, 23)], same=False, rows=[[46, 47]])
    s += E(38, 47, '125 ep   mass head', col_=HEADC, w=1.9)
    s += E(38, 46, '125 ep   no gain')
    s += '}\n'
    render('m_p2', s)


# =========================================================== chapter: capacity
def cap():
    s = HDR(ns=0.40, rs=0.70, fs=15)
    s += CLUSTER('c', 'capacity', CAPC, [(38, 16), (53, 13), (512, 13), (513, 12), (514, 13)],
                 same=False, rows=[[53, 513], [512, 514]], invis=[(53, 512), (513, 514)])
    s += E(38, 53, 'from scratch', col_=CAPC, dashed=True)
    s += E(38, 512, 'from v38', col_=CAPC, dashed=True)
    s += E(38, 513, 'GN 256', col_=CAPC, dashed=True)
    s += E(513, 514, 'batch 6', col_=CAPC, dashed=True)
    s += '}\n'
    render('m_cap', s)


# =========================================================== chapter: attention
def attn():
    s = HDR(ns=0.40, rs=0.80, fs=15)
    s += CLUSTER('a', 'context / attention', ATTC,
                 [(38, 16), (510, 13), (511, 14), (517, 12), (519, 12), (532, 12),
                  (47, 16), (515, 13), (516, 12), (518, 12), (520, 12)], same=False,
                 rows=[[38, 47]])
    s += E(38, 510, '20 ep', col_=ATTC); s += E(38, 511, '+edge bias', col_=ATTC, w=1.8)
    s += E(38, 517, '150 ep', col_=ATTC); s += E(517, 519, 'resume', col_=ATTC)
    s += E(511, 532, '+mass', col_=ATTC)
    s += E(47, 515, '✗', col_=FAIL, dashed=True); s += E(515, 516, '60 ep', col_=FAIL, dashed=True)
    s += E(47, 518, '✗', col_=FAIL, dashed=True); s += E(518, 520, '179 ep', col_=FAIL, dashed=True)
    s += '}\n'
    render('m_attn', s)


# =========================================================== chapter: restructure
def res():
    s = HDR(ns=0.40, rs=0.72, fs=15)
    s += CLUSTER('r', 'restructure the event', '#9B59B6',
                 [(38, 16), (39, 13), (40, 13), (41, 13), (42, 13)], same=False,
                 rows=[[39, 41], [40, 42]], invis=[(39, 40), (41, 42)])
    s += E(38, 39, 'truth PV', col_='#9B59B6', dashed=True)
    s += E(38, 40, 'cluster head', col_='#9B59B6', dashed=True)
    s += E(38, 41, 'curriculum', col_='#9B59B6', dashed=True)
    s += E(38, 42, '50 files', col_='#9B59B6', dashed=True)
    s += '}\n'
    render('m_res', s)


# =========================================================== chapter: retrain
def ret():
    s = HDR(ns=0.30, rs=0.72, fs=14)
    s += CLUSTER('t', 'retrain the recipe', ABL,
                 [(31, 14), (500, 14), (501, 11), (502, 11), (503, 11), (504, 14), (509, 11),
                  (521, 11), (530, 11), (505, 11), (506, 11), (507, 14), (508, 11)], same=False,
                 rows=[[501, 502, 503], [505, 506]])
    s += E(31, 500, 'from scratch', col_=ABL, w=1.6)
    s += E(500, 501, 'B2 ✗', col_=FAIL, dashed=True)
    s += E(500, 502, 'cl2w ~', col_=ABL, dashed=True)
    s += E(500, 503, 'hinge ✗', col_=FAIL, dashed=True)
    s += E(500, 504, 'chain-CE ✓', col_=ABL, w=1.8)
    s += E(500, 509, 'empty control', col_=ABL, dotted=True)
    s += E(500, 521, 'package', col_=ABL, dotted=True)
    s += E(500, 530, 'A1 ✗', col_=ABL, dashed=True)
    s += E(504, 505, 'source ~', col_=ABL)
    s += E(504, 506, 'mass ~', col_=ABL)
    s += E(504, 507, 'struct ✓', col_=ABL, w=1.8)
    s += E(504, 508, 'mom', col_=ABL)
    s += '}\n'
    render('m_ret', s)


# =========================================================== chapter: public
def pub():
    s = HDR(ns=0.40, rs=0.85, fs=15)
    s += CLUSTER('u', 'another dataset  (public)', PUB,
                 [(27, 15), (60, 14), (61, 12), (533, 12), (45, 13), (49, 12)], same=False,
                 rows=[[60, 533, 45]])
    s += E(27, 60, 'P1  v27+B2+source', col_=PUB, w=1.7)
    s += E(27, 533, 'package', col_=PUB)
    s += E(60, 61, 'P2', col_=PUB)
    s += E(27, 45, 'full CERN stack', col_=PUB, dashed=True)
    s += E(45, 49, '✗', col_=FAIL, dashed=True)
    s += '}\n'
    render('m_pub', s)


# =========================================================== chapter: diagnosis
def diag():
    s = HDR(ns=0.26, rs=0.66, fs=14)
    s += CLUSTER('d', 'the diagnosis', NEWC,
                 [(38, 14), (507, 14), (500, 12), (549, 13), (550, 12), (551, 18)] +
                 [(v, 11) for v in (552, 553, 554, 555, 556, 557, 558, 559, 560, 561)],
                 same=False,
                 rows=[[38, 507, 500], [552, 553, 554, 555], [549, 550],
                       [556, 557, 558, 559], [560, 561]],
                 invis=[(552, 556), (549, 551)])
    s += E(38, 552, col_=NEWC); s += E(38, 553, col_=NEWC, w=1.6)
    s += E(38, 554, col_=NEWC); s += E(38, 555, col_=NEWC)
    s += E(507, 549, col_=NEWC)
    s += E(507, 550, col_=NEWC)
    s += E(507, 551, '20 ep', col_=NEWC, w=2.0)
    s += E(551, 556, col_=NEWC); s += E(551, 557, col_=NEWC)
    s += E(551, 558, col_=NEWC); s += E(551, 559, col_=NEWC)
    s += E(500, 560, col_=NEWC); s += E(500, 561, col_=NEWC)
    s += '}\n'
    render('m_diag', s)


# =========================================================== the branches after v47
def branches():
    """Overview of everything that came after the main line, one row per group."""
    groups = [('bcap', 'capacity', CAPC, [53, 512, 513, 514]),
              ('batt', 'context / attention', ATTC, [510, 511, 517, 519, 532, 515, 516, 518, 520]),
              ('bres', 'restructure the event', '#9B59B6', [39, 40, 41, 42]),
              ('bret', 'retrain the recipe', ABL,
               [500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 521, 530]),
              ('bctx', 'context-aware pruning head', '#5D6D7E', [540, 545, 546, 547, 548]),
              ('bpub', 'another dataset  (public)', PUB, [27, 45, 49, 60, 61, 533])]
    s = HDR(ns=0.16, rs=0.48, fs=12)
    for tag, title, c, vs in groups:
        s += CLUSTER(tag, title, c, [(v, 12) for v in vs], same=True)
    # Stack the groups with an invisible spacer between them: a direct edge
    # between two cluster members makes dot drop a node out of its own box.
    chain = [g[3][0] for g in groups]
    for a, b in zip(chain[:-1], chain[1:]):
        s += '  n%d -> n%d [style=invis];\n' % (a, b)
    s += '}\n'
    render('m_branches', s)


# =========================================================== mechanism schematic
def mech():
    """Two compact rows so the diagram stays readable at slide size."""
    s = _sch(rankdir='TB', ns=0.30, rs=0.34, fs=12, margin='0.06,0.04', wmin=0.95)
    s += B('gn', 'GNN blocks\nnode / edge\nreprs', fs=12)
    s += B('ml', 'weight MLP\n(sigmoid head)', fs=12)
    s += B('w', 'w in [0, 1]', fc='#FFFFFF', fs=12)
    s += B('msk', 'soft mask\nw_eff =\nw·σ((w−cut)/τ)', fc='#FDEBD0', ec='#C0392B', fs=12)
    s += B('mp', 'weighted\nmessage passing', fc='#EAF7EF', ec='#2E7D32', fs=12)
    s += B('ls', 'pruning loss\nvs truth (BCE)', fc='#F2F2F2', ec='#777777', fs=11)
    s += B('cut', 'cut 0.5 → 0.7 → 0.85\nτ annealed 1.0 → 0.1\ninference:\nkeep if w ≥ 0.9',
           fc='#F2F2F2', ec='#777777', fs=11)
    s += '  { rank=same; gn; ml; w; }\n'
    s += '  gn -> ml -> w;\n'
    s += '  w -> msk;\n'
    s += '  { rank=same; msk; mp; cut; }\n'
    s += '  msk -> mp;\n'
    s += '  msk -> ls [color="#C0392B", style=dashed];\n'
    s += '  ls -> cut [style=invis];\n'
    s += '  mp -> gn [constraint=false, style=dashed, color="#2E7D32", label="one pass"];\n'
    s += '}\n'
    render('m_mech', s)


# =========================================================== head zoo
def zoo():
    s = ('digraph G {\n  bgcolor="transparent";\n  rankdir=TB; nodesep=0.18; ranksep=0.34;\n'
         '  graph [fontname="Helvetica", compound=true];\n'
         '  node [shape=box, style="rounded,filled", fontname="Helvetica-Bold", fontsize=12,\n'
         '        width=1.20, height=0.58, margin="0.05,0.03", penwidth=1.2];\n'
         '  edge [color="#1F4E79", arrowsize=0.6, penwidth=1.1];\n')
    s += '  bb [label="shared GNN backbone\\n(16-dim node / edge)", fillcolor="#DCE9F7",\n'
    s += '      color="#1F4E79", fontsize=13, height=0.68, width=2.2];\n'
    orig = [('LCAG', 'edge class 0-3'), ('node prune', 'keep / remove'),
            ('edge prune', 'keep / remove'), ('PV asso', 'track → PV'),
            ('chain\nscorer', 'chain\nconfidence')]
    ours = [('source\nv36', 'chain root (RC)'), ('mass\nv46/47', 'log10 m_ππ'),
            ('struct\nv48/52', 'depth + RC'), ('mom\nv48/51', 'momentum')]
    ids = []
    for k, (n, t) in enumerate(orig + ours):
        new = k >= len(orig)
        i = 'h%d' % k
        ids.append(i)
        s += ('  %s [label="%s\\n%s", fillcolor="%s", color="%s"];\n'
              % (i, n, t, '#EAF7EF' if new else '#EAEFF5', '#2E7D32' if new else '#1F4E79'))
        s += '  bb -> %s;\n' % i
    s += '  { rank=same; %s }\n' % '; '.join(ids[:5])
    s += '  { rank=same; %s }\n' % '; '.join(ids[5:])
    s += '  h0 -> h5 [style=invis];\n'
    s += ('  key [shape=plaintext, fontsize=12, label=<<FONT COLOR="#2E7D32">■</FONT> added by us   '
          '<FONT COLOR="#1F4E79">■</FONT> original DFEI>];\n')
    s += '  h4 -> key [style=invis];\n'
    s += '}\n'
    render('m_zoo', s)


# =========================================================== change schematics
def _sch(rankdir='LR', ns=0.35, rs=0.60, fs=13, margin='0.10,0.07', wmin=0.0):
    nw = ', width=%.2f' % wmin if wmin else ''
    return ('digraph G {\n  bgcolor="transparent";\n  rankdir=%s; nodesep=%.2f; ranksep=%.2f;\n'
            '  graph [fontname="Helvetica", compound=true];\n'
            '  node [shape=box, style="rounded,filled", fontname="Helvetica-Bold", fontsize=%d,\n'
            '        margin="%s"%s, penwidth=1.3];\n'
            '  edge [color="#5A646E", arrowsize=0.7, penwidth=1.3, fontname="Helvetica", fontsize=12,\n'
            '        fontcolor="#5A646E"];\n' % (rankdir, ns, rs, fs, margin, nw))


def B(name, label, fc='#EAEFF5', ec='#1F4E79', fs=13, h=0.0, w=0.0):
    s = '  %s [label="%s", fillcolor="%s", color="%s", fontsize=%d' % (name, label, fc, ec, fs)
    if h:
        s += ', height=%.2f' % h
    if w:
        s += ', width=%.2f' % w
    return s + '];\n'


def s_mass():
    """What the mass head is — two rows."""
    s = _sch(rankdir='TB', ns=0.40, rs=0.50, fs=12)
    s += B('tt', 'tt edge\ntracks i, j', fc='#F2F2F2', ec='#777777')
    s += B('rep', 'edge repr\n16-d')
    s += B('hd', 'mass head\none linear layer', fc='#FDEBD0', ec='#B8860B')
    s += B('out', 'log10 m_ππ on every tt edge')
    s += B('ls', 'loss = MSE vs MC truth\nsentinel edges masked', fc='#FDEDEC', ec='#C0392B', fs=11)
    s += B('ok', 'probe R² 0.003 → 0.930', fc='#EAF7EF', ec='#2E7D32', fs=11)
    s += '  { rank=same; tt; rep; hd; }\n'
    s += '  tt -> rep -> hd;\n'
    s += '  hd -> out;\n'
    s += '  { rank=same; out; ls; }\n'
    s += '  hd -> ls [color="#C0392B", style=dashed];\n'
    s += '  out -> ok [color="#2E7D32"];\n'
    s += '  { rank=same; ok; }\n'
    s += '}\n'
    render('s_mass', s)


def s_capacity():
    """What the capacity experiments changed (before → after, one row per idea)."""
    s = _sch(rankdir='TB', ns=0.35, rs=0.42, fs=13)
    s += B('a1', 'node repr  16-d', fc='#F2F2F2', ec='#777777')
    s += B('a2', 'tracks 32-d\\ntt edges 24-d', fc='#F3ECFF', ec='#8E44AD')
    s += B('b1', 'GN hidden  128', fc='#F2F2F2', ec='#777777')
    s += B('b2', 'GN hidden  256', fc='#F3ECFF', ec='#8E44AD')
    s += B('c1', 'one 16-d space\\nfor nine heads', fc='#F2F2F2', ec='#777777')
    s += B('c2', 'three heads\\nat once', fc='#F3ECFF', ec='#8E44AD')
    s += '  { rank=same; a1; a2; }\n'
    s += '  { rank=same; b1; b2; }\n'
    s += '  { rank=same; c1; c2; }\n'
    s += '  a1 -> a2 [color="#8E44AD"];\n'
    s += '  b1 -> b2 [color="#8E44AD"];\n'
    s += '  c1 -> c2 [color="#8E44AD"];\n'
    s += '  a1 -> b1 [style=invis];\n  b1 -> c1 [style=invis];\n'
    s += B('r', 'all four:  24 – 27 %  AllParticles\\nagainst 37.59 for v38',
           fc='#FDEDEC', ec='#C0392B', fs=12)
    s += '  c1 -> r [style=invis];\n'
    s += '}\n'
    render('s_capacity', s)


def s_attention():
    """Where the context module is inserted."""
    s = _sch(rankdir='TB', ns=0.30, rs=0.40, fs=12)
    s += B('gn', 'GNN blocks')
    s += B('att', 'track-level self-attention\\nafter the last GN block', fc='#FFE3EC', ec='#C2185B')
    s += B('hd', 'the nine heads')
    s += B('b1', 'v510  pure content', fc='#F2F2F2', ec='#777777', fs=12)
    s += B('b2', 'v511  + tt edge features\\nas attention bias (ParT-style)', fc='#FFE3EC', ec='#C2185B', fs=12)
    s += '  gn -> att -> hd;\n'
    s += '  att -> b1 [style=invis];\n'
    s += '  b1 -> b2 [style=invis];\n'
    s += '  { rank=same; b1; }\n'
    s += '}\n'
    render('s_attention', s)


def s_restructure():
    """What the restructuring attempts did."""
    s = _sch(rankdir='TB', ns=0.32, rs=0.38, fs=13)
    s += B('ev', 'event graph\\n~1000 edges per track', fc='#F2F2F2', ec='#777777')
    s += B('cl', 'split the tracks\\nby truth PV (v39)', fc='#F3ECFF', ec='#9B59B6')
    s += B('hd', 'or learn the split\\nwith a head (v40/v41)', fc='#F3ECFF', ec='#9B59B6')
    s += B('rc', 'reconstruct each\\ncluster separately', fc='#F3ECFF', ec='#9B59B6')
    s += B('bad', 'worse on the full-graph pass\\n50 files:  26.28  vs  29.26',
           fc='#FDEDEC', ec='#C0392B', fs=12)
    s += '  ev -> cl -> rc;\n  hd -> rc [style=dashed, color="#9B59B6"];\n'
    s += '  rc -> bad [color="#C0392B"];\n'
    s += '  ev -> hd [style=invis];\n'
    s += '  { rank=same; cl; hd; }\n'
    s += '}\n'
    render('s_restructure', s)


def s_retrain():
    """The ablation chain: one change at a time from a common base."""
    s = _sch(rankdir='TB', ns=0.22, rs=0.34, fs=12)
    s += B('b', 'v500   base   33.34  AllParticles', fc='#E9DDFF', ec='#7A44AA', fs=13)
    labels = [('c1', 'B2'), ('c2', 'class-2 w'), ('c3', 'hinge'), ('c4', 'chain-CE ✓'),
              ('c5', 'source'), ('c6', 'mass'), ('c7', 'struct'), ('c8', 'empty'), ('c9', 'mom')]
    for n, t in labels:
        s += B(n, t, fc='#F3ECFF', ec='#7A44AA')
    s += '  { rank=same; c1; c2; c3; }\n'
    s += '  { rank=same; c4; c5; c6; }\n'
    s += '  { rank=same; c7; c8; c9; }\n'
    for n, _ in labels:
        s += '  b -> %s;\n' % n
    s += '  c1 -> c4 [style=invis];\n  c4 -> c7 [style=invis];\n'
    s += B('v', 'keep a change only if AllParticles\\ndoes not drop over 20 epochs',
           fc='#F2F2F2', ec='#777777', fs=12)
    s += B('w', 'only chain-CE clears the noise floor\\n(+63 events against ±75)',
           fc='#EAF7EF', ec='#2E7D32', fs=12)
    s += '  c7 -> v [style=invis];\n  c8 -> w [style=invis];\n'
    s += '  { rank=same; v; w; }\n'
    s += '}\n'
    render('s_retrain', s)


def s_public():
    """The public-data rebuild path."""
    s = _sch(rankdir='TB', ns=0.30, rs=0.40, fs=12)
    s += B('cern', 'CERN v38 stack   37.59 / 21.10')
    s += B('bad', 'transfer the whole stack to public data\\n(no PID, 4–14× more class 2/3)\\nv45  19.70   ·   v49  14.15', fc='#FDEDEC', ec='#C0392B', fs=12)
    s += B('v27', 'v27   simple stack   51.66 / 22.76', fc='#E4F6E8', ec='#2E8B57')
    s += B('v60', 'v60   + B2 + source   54.52 / 23.43', fc='#D7F2DC', ec='#2E8B57')
    s += B('v61', 'v61   + class-2 w + hinge   (in flight)', fc='#E4F6E8', ec='#2E8B57', fs=12)
    s += '  cern -> bad [color="#C0392B", style=dashed];\n'
    s += '  v27 -> v60 -> v61 [color="#2E8B57"];\n'
    s += '  bad -> v27 [style=invis];\n'
    s += '  { rank=same; bad; }\n'
    s += '}\n'
    render('s_public', s)


def s_oracle():
    """How the oracle intervention works: the weights are frozen, only the decisions change."""
    s = _sch(rankdir='LR', ns=0.30, rs=0.62, fs=13)
    s += B('w', 'v47 weights\\nfrozen, not retrained', fc='#F2F2F2', ec='#777777')
    s += B('pt', 'point pruning\\ndecisions', fc='#FDEBD0', ec='#C0392B')
    s += B('ed', 'edge pruning\\ndecisions', fc='#FDEBD0', ec='#C0392B')
    s += B('rc', 'reconstruction')
    s += B('sc', 'score')
    s += B('tr', 'MC truth chain\\n(known from simulation)', fc='#EAF7EF', ec='#2E7D32', fs=12)
    s += B('var', 'six variants:\\nfix recall · fix precision · fix both\\nfix edges · fix all · add fakes',
           fc='#F2F2F2', ec='#777777', fs=12)
    s += '  w -> pt -> ed -> rc -> sc;\n'
    s += '  tr -> pt [color="#2E7D32", style=dashed, constraint=false];\n'
    s += '  tr -> ed [color="#2E7D32", style=dashed, constraint=false];\n'
    s += '  var -> w [style=invis];\n'
    s += '  { rank=same; tr; var; }\n'
    s += '  { rank=same; w; var; }\n'
    s += '}\n'
    render('s_oracle', s)


# =========================================================== compact mechanism cards
def d_hinge():
    """v37 hinge: where the confidence comes from and where the reward is paid."""
    s = _sch(rankdir='TB', ns=0.34, rs=0.38, fs=12, margin='0.08,0.05', wmin=1.0)
    s += B('a', 'LCAG head:\nper-edge 4-class\nsoftmax\n→ (p0, p1, p2, p3)', fs=12)
    s += B('b', 'confidence =\nmax(p0…p3)\nprobability of the\nclass the model chose', fs=12)
    s += B('c', 'hinge =\nmax(0, margin − confidence)\nmargin = 0.3\nextra term in the total loss',
           fc='#FDEBD0', ec='#E67E22', fs=12)
    s += B('d', 'applied ONLY on\ntruth-chain edges\n(known from MC truth —\ntraining only)',
           fc='#EAF7EF', ec='#2E7D32', fs=12)
    s += '  { rank=same; a; b; }\n  { rank=same; c; d; }\n'
    s += '  a -> b;\n  b -> c;\n  c -> d;\n'
    s += '}\n'
    render('d_hinge', s)


def d_cewhere():
    """Chain-CE: the same −log(p_true), paid only where the chain lives."""
    s = _sch(rankdir='TB', ns=0.40, rs=0.36, fs=12, margin='0.08,0.05', wmin=1.0)
    s += B('g1', 'global CE\n(ALL edges)', fc='#F2F2F2', ec='#777777', fs=12)
    s += B('c1', 'chain-CE\n(truth-chain edges, classes 1/2/3)',
           fc='#EAF7EF', ec='#2E7D32', fs=12)
    s += B('g2', 'reward ≈ 0 for\nstructural edges\n(99.9% of the signal\nis \\"background\\")',
           fc='#FFFFFF', ec='#BBBBBB', fs=11)
    s += B('c2', 'every chain edge gets a direct\nsignal on its true class\n→ classes 1/2/3 learn',
           fc='#FFFFFF', ec='#BBBBBB', fs=11)
    s += B('n', 'same function −log(p_true)\n— the change is WHERE it is paid',
           fc='#FFFFFF', ec='#FFFFFF', fs=11)
    s += '  { rank=same; g1; c1; }\n  { rank=same; g2; c2; }\n'
    s += '  g1 -> g2;\n  c1 -> c2;\n'
    s += '  g2 -> n [style=invis];\n  c2 -> n [style=invis];\n'
    s += '}\n'
    render('d_cewhere', s)


def d_probe():
    """Linear probe: one linear layer on the frozen representation."""
    s = _sch(rankdir='TB', ns=0.26, rs=0.40, fs=11, margin='0.07,0.05', wmin=1.0)
    s += B('bb', 'GNN backbone\n(FROZEN,\ngradients\nblocked)', fs=11)
    s += B('r', 'representation\nh_e / h_v', fs=11)
    s += B('l', 'linear layer\n(only this\ntrains)', fs=11)
    s += B('p', 'prediction ŷ\n(log10 m_ππ)', fs=11)
    s += B('n', 'R² = fit of ŷ vs the true y\nR² ≈ 1 → quantity readable    ·    R² ≈ 0 → quantity lost',
           fc='#FDEBD0', ec='#E67E22', fs=11)
    s += '  { rank=same; bb; r; l; p; }\n'
    s += '  bb -> r -> l -> p;\n'
    s += '  p -> n [color="#E67E22", style=dashed];\n'
    s += '}\n'
    render('d_probe', s)


# =========================================================== colour bar
def cbar():
    fig = plt.figure(figsize=(1.0, 4.6))
    ax = fig.add_axes([0.42, 0.02, 0.26, 0.96])
    sm = plt.cm.ScalarMappable(cmap=CM, norm=mcolors.Normalize(vmin=VMIN, vmax=VMAX))
    cb = fig.colorbar(sm, cax=ax)
    cb.set_label('PerfectReco [%]', fontsize=9, color='#333333', labelpad=3)
    cb.ax.tick_params(labelsize=8)
    cb.ax.yaxis.set_label_position('right')
    cb.ax.yaxis.set_ticks_position('right')
    fig.savefig(FIG + '/cbar.png', transparent=True, bbox_inches='tight')
    plt.close(fig)
    print('cbar.png     ok')


if __name__ == '__main__':
    master(); p1(); p2(); cap(); attn(); res(); ret(); pub(); diag(); branches()
    mech(); zoo(); s_mass(); s_capacity(); s_attention(); s_restructure(); s_retrain(); s_public()
    s_oracle()
    d_hinge(); d_cewhere(); d_probe()
    cbar()
