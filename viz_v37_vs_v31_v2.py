#!/usr/bin/env python3
"""v37 vs v31 优化 3D 可视化 v3 (事件 92441664) —— 结构强化版。

v2 问题: 产生点都在束流附近几 mm 内 + 动量方向一致(喷注), 纯几何下链不可辨。
v3 改动:
- 背景 track 合并为单个 trace (图例可一键隐藏), 更淡
- 链 track / 树边 / 顶点 / 层次标签 各自合并为独立 trace
- 树边加粗显眼, 边中点标注衰变层次 (L1/L2/... 含粒子名)
- 次级顶点 marker 加大
"""
import os
import io
import zstandard as zstd
import torch
import numpy as np
import plotly.graph_objects as go

DATA_FILE = "/lzufs/user/guoqingxiang/DFEI_IFT_20260702/data/MC_normed/inclusive_00342442/tst_data_00231000_00231999.pt.zst"
EVT_IDX = 747
NORM_DICT = "/lzufs/user/guoqingxiang/DFEI_IFT_20260702/dfei_repo/preprocessing/normalization_dict.pt"
OUT_HTML = "/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn/viz_v37_vs_v31_evt92441664_v2.html"

TRACK_LEN_CM = 0.10  # track 线段长度 (m) —— 产生点沿动量方向一小段

COLOR_CHAIN0 = "#e63946"
COLOR_CHAIN1 = "#1d6fd6"
COLOR_BKG = "#9aa0a6"
COLOR_PV = "#111111"

# PDG id -> 粒子符号 (简化)
PDG_NAME = {211: "π⁺", -211: "π⁻", 321: "K⁺", -321: "K⁻", 2212: "p", -2212: "p̄",
            11: "e⁻", -11: "e⁺", 13: "μ⁻", -13: "μ⁺", 111: "π⁰", 311: "K⁰",
            313: "K*⁰", 323: "K*⁺", 443: "J/ψ", 411: "D⁺", 421: "D⁰",
            511: "B⁰", 521: "B⁺", 5122: "Λb⁰", 333: "φ"}


def load_event():
    dctx = zstd.ZstdDecompressor()
    with open(DATA_FILE, "rb") as f:
        with dctx.stream_reader(f) as r:
            data = torch.load(io.BytesIO(r.read()), weights_only=False)
    return data[EVT_IDX]


def build_decay_tree(evt, idx_of_key):
    """衰变树边 + 每条链的根。"""
    tt = evt[("tracks", "to", "tracks")]
    s, r = tt["senders"].tolist(), tt["receivers"].tolist()
    edges = []
    for sk, rk in zip(s, r):
        if sk >= 0 and rk >= 0 and sk in idx_of_key and rk in idx_of_key:
            edges.append((idx_of_key[sk], idx_of_key[rk]))
    has_parent = {c for _, c in edges}
    roots = sorted({p for p, _ in edges if p not in has_parent})
    return edges, roots


def tree_levels(edges, roots):
    """根 -> 叶的衰变层次 (每 track 的 level)。"""
    children = {}
    for p, c in edges:
        children.setdefault(p, []).append(c)
    level = {}
    stack = [(rt, 0) for rt in roots]
    while stack:
        node, lv = stack.pop()
        if node in level:
            continue
        level[node] = lv
        for ch in children.get(node, []):
            stack.append((ch, lv + 1))
    return level


def pv_of_track(evt, track_idx):
    trpv = evt[("tracks", "to", "pvs")]
    ei, y = trpv.edge_index, trpv.y
    m = (ei[0] == track_idx) & (y == 1)
    return int(ei[1][m][0]) if m.any() else None


def main():
    evt = load_event()
    x = evt["tracks"]["x"].numpy()
    n = x.shape[0]
    tr = evt["tracks"]
    norm = torch.load(NORM_DICT, map_location="cpu", weights_only=False)
    c, s = norm["center"], norm["scale"]
    names = ["px_reco", "py_reco", "pz_reco", "xProd_reco", "yProd_reco", "zProd_reco"]
    cols = {nm: i for i, nm in enumerate(names)}
    phys = np.zeros((n, 6))
    for nm in names:
        phys[:, cols[nm]] = x[:, cols[nm]] * s[nm] + c[nm]
    px, py, pz = phys[:, 0], phys[:, 1], phys[:, 2]
    x0, y0, z0 = phys[:, 3] / 1000, phys[:, 4] / 1000, phys[:, 5] / 1000

    part_keys = tr["part_keys"].tolist()
    idx_of_key = {k: i for i, k in enumerate(part_keys)}
    edges, roots = build_decay_tree(evt, idx_of_key)
    levels = tree_levels(edges, roots)

    pvx = evt["pvs"]["x"].numpy()
    pv_phys = np.zeros_like(pvx)
    for i, nm in enumerate(["xPV_reco", "yPV_reco", "zPV_reco"]):
        pv_phys[:, i] = pvx[:, i] * s[nm] + c[nm]
    pv_pos = pv_phys / 1000.0

    # 链归属 (head_keys)
    head_keys = tr["head_keys"].tolist()
    chain_of = {}
    for i, hk in enumerate(head_keys):
        chain_of.setdefault(hk, []).append(i)
    chain_of_lists = list(chain_of.values())
    chain_color = {t: (COLOR_CHAIN0 if ci == 0 else COLOR_CHAIN1)
                   for ci, tracks in enumerate(chain_of_lists) for t in tracks}
    # 每条链的根 track
    root_of_chain = []
    for tracks in chain_of_lists:
        rt = [t for t in tracks if t in roots]
        root_of_chain.append(rt[0] if rt else tracks[0])

    # 粒子名
    pids = tr["part_ids"].tolist()
    def pname(i):
        return PDG_NAME.get(pids[i], str(pids[i]))

    # 线段终点
    def seg(i):
        d = np.array([px[i], py[i], pz[i]], dtype=float)
        dn = d / (np.linalg.norm(d) + 1e-9)
        end = np.array([x0[i], y0[i], z0[i]]) + dn * TRACK_LEN_CM
        return [x0[i], end[0]], [y0[i], end[1]], [z0[i], end[2]]

    fig = go.Figure()

    # ===== 束管 =====
    th = np.linspace(0, 2 * np.pi, 40)
    zt = np.linspace(-0.10, 0.50, 8)
    TT, ZZ = np.meshgrid(th, zt)
    fig.add_trace(go.Surface(
        x=(0.01 * np.cos(TT)).tolist(), y=(0.01 * np.sin(TT)).tolist(), z=ZZ.tolist(),
        opacity=0.12, colorscale=[[0, "#555"], [1, "#555"]], showscale=False,
        name="束管", hoverinfo="skip",
    ))

    # ===== 背景 track (合并单 trace, 图例可隐藏) =====
    bkg_x, bkg_y, bkg_z = [], [], []
    for i in range(n):
        if i in chain_color:
            continue
        xx, yy, zz = seg(i)
        bkg_x += [xx[0], xx[1], None]
        bkg_y += [yy[0], yy[1], None]
        bkg_z += [zz[0], zz[1], None]
    fig.add_trace(go.Scatter3d(
        x=bkg_x, y=bkg_y, z=bkg_z, mode="lines",
        line=dict(color=COLOR_BKG, width=2), opacity=0.25,
        name="背景 track (可点击隐藏)", hoverinfo="skip",
    ))

    # ===== 链 track + 树边 + 顶点 (按链) =====
    for ci, (tracks, col) in enumerate(zip(chain_of_lists, [COLOR_CHAIN0, COLOR_CHAIN1])):
        # 链 track 线段 (合并)
        tx, ty, tz = [], [], []
        for i in tracks:
            xx, yy, zz = seg(i)
            tx += [xx[0], xx[1], None]
            ty += [yy[0], yy[1], None]
            tz += [zz[0], zz[1], None]
        fig.add_trace(go.Scatter3d(
            x=tx, y=ty, z=tz, mode="lines",
            line=dict(color=col, width=3), opacity=1.0,
            name=f"链{ci} track ({len(tracks)}粒子)", hoverinfo="skip",
        ))
        # 树边 (粗, 中点标层次) —— 只画链内边
        ex, ey, ez, labels, lx, ly, lz = [], [], [], [], [], [], []
        for p, c_ in edges:
            if p not in tracks or c_ not in tracks:
                continue
            ex += [x0[p], x0[c_], None]
            ey += [y0[p], y0[c_], None]
            ez += [z0[p], z0[c_], None]
            lx.append((x0[p] + x0[c_]) / 2)
            ly.append((y0[p] + y0[c_]) / 2)
            lz.append((z0[p] + z0[c_]) / 2)
            labels.append(f"L{levels.get(c_, 0)} {pname(p)}→{pname(c_)}")
        fig.add_trace(go.Scatter3d(
            x=ex, y=ey, z=ez, mode="lines",
            line=dict(color=col, width=6), opacity=0.95,
            name=f"链{ci} 衰变级联", hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter3d(
            x=lx, y=ly, z=lz, mode="text", text=labels,
            textfont=dict(size=10, color=col), hoverinfo="skip",
            name=f"链{ci} 层次", showlegend=False,
        ))
        # 顶点 (产生点 marker) + hover
        vx, vy, vz, vtext = [], [], [], []
        for i in tracks:
            vx.append(x0[i]); vy.append(y0[i]); vz.append(z0[i])
            vtext.append(f"#{i} {pname(i)}")
        fig.add_trace(go.Scatter3d(
            x=vx, y=vy, z=vz, mode="markers+text",
            text=vtext, textposition="top center",
            marker=dict(size=8, color=col, symbol="circle",
                        line=dict(width=1.5, color="white")),
            textfont=dict(size=9, color="#222"),
            name=f"链{ci} 次级顶点", hoverinfo="skip", showlegend=False,
        ))

    # ===== PV + B 飞行线 =====
    for iv, pv in enumerate(pv_pos):
        fig.add_trace(go.Scatter3d(
            x=[pv[0]], y=[pv[1]], z=[pv[2]], mode="markers",
            marker=dict(size=10, color=COLOR_PV, symbol="x", line=dict(width=2)),
            name=f"PV{iv} (主顶点)", showlegend=(iv == 0), hoverinfo="skip",
        ))
    for ci, rt in enumerate(root_of_chain):
        col = [COLOR_CHAIN0, COLOR_CHAIN1][ci]
        pvi = pv_of_track(evt, rt)
        if pvi is None:
            continue
        pv = pv_pos[pvi]
        fig.add_trace(go.Scatter3d(
            x=[pv[0], x0[rt]], y=[pv[1], y0[rt]], z=[pv[2], z0[rt]],
            mode="lines", line=dict(color=col, width=4, dash="dot"),
            opacity=0.9, hoverinfo="skip", showlegend=False,
            name=f"B 飞行 (链{ci})",
        ))
        fig.add_trace(go.Scatter3d(
            x=[(pv[0] + x0[rt]) / 2], y=[(pv[1] + y0[rt]) / 2], z=[(pv[2] + z0[rt]) / 2],
            mode="text", text=["B 介子"], textfont=dict(size=11, color=col),
            hoverinfo="skip", showlegend=False,
        ))

    fig.update_layout(
        title=dict(
            text=("DFEI 重建优化 v37 vs v31 —— 衰变级联结构视图<br>"
                  "<sup>事件 92441664 · 红=链0 (v31失败/v37成功), 蓝=链1 · 粗线=衰变级联, "
                  "细线=track 方向 · 点击图例可隐藏背景</sup>"),
            font=dict(size=15),
        ),
        scene=dict(
            xaxis_title="x (m)", yaxis_title="y (m)", zaxis_title="z (束流, m)",
            xaxis_range=[-0.06, 0.06], yaxis_range=[-0.06, 0.06], zaxis_range=[-0.10, 0.50],
            aspectmode="manual", aspectratio=dict(x=1.0, y=1.0, z=1.4),
            camera=dict(eye=dict(x=1.8, y=1.4, z=1.2)),
        ),
        legend=dict(font=dict(size=12), x=0.02, y=0.98),
        width=1150, height=900,
    )

    os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)
    fig.write_html(OUT_HTML)
    print(f"[ok] 已生成: {OUT_HTML} | 树边 {len(edges)} | 根 {roots}")


if __name__ == "__main__":
    main()
