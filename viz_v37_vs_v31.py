#!/usr/bin/env python3
"""v37 vs v31 优化 3D 可视化 (事件 92441664)。

选中事件: 双B、长衰变链 (链0: 7 粒子, 链1: 4 粒子), 共 74 tracks (11 信号 + 63 背景)。
- 链0: v31 重建失败 (Perfect=0), v37 重建成功 (Perfect=1)  <-- 本图核心
- 链1: v31 与 v37 都重建成功

画法:
- 每条 track: 从产生点 (xProd,yProd,zProd) 沿动量 (px,py,pz) 方向画 3D 线段
- 链内 truth 边 (LCA y>0): 同色细线连接, 展示衰变链父子结构
- 背景 track: 灰色高透明
"""
import os
import io
import zstandard as zstd
import torch
import numpy as np
import plotly.graph_objects as go

DATA_FILE = "/lzufs/user/guoqingxiang/DFEI_IFT_20260702/data/MC_normed/inclusive_00342442/tst_data_00231000_00231999.pt.zst"
EVT_IDX = 747
OUT_HTML = "/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn/viz_v37_vs_v31_evt92441664.html"

# truth 链 (连通分量, 来自 LCA y>0 边)
CHAIN0 = [1, 13, 14, 24, 51, 52, 69]   # 长链 7 粒子: v31 失败 / v37 成功
CHAIN1 = [7, 19, 46, 58]               # 短链 4 粒子: v31 & v37 都成功

TRACK_LEN = 1.8   # 线段长度 (归一化单位, 沿动量方向)

COLOR_CHAIN0 = "#e63946"   # 红: v37 成功 / v31 失败
COLOR_CHAIN1 = "#1d6fd6"   # 蓝: 都成功
COLOR_BKG = "#9aa0a6"      # 灰: 背景 track


def load_event():
    dctx = zstd.ZstdDecompressor()
    with open(DATA_FILE, "rb") as f:
        with dctx.stream_reader(f) as r:
            data = torch.load(io.BytesIO(r.read()), weights_only=False)
    return data[EVT_IDX]


def main():
    evt = load_event()
    x = evt["tracks"].x.numpy()  # [N, 8]: px,py,pz,xProd,yProd,zProd,Charge,ghost
    n = x.shape[0]
    px, py, pz = x[:, 0], x[:, 1], x[:, 2]
    x0, y0, z0 = x[:, 3], x[:, 4], x[:, 5]
    charge = x[:, 6]

    # track 归属
    chain_of = {i: 0 for i in CHAIN0}
    chain_of.update({i: 1 for i in CHAIN1})
    color_of = {i: COLOR_CHAIN0 for i in CHAIN0}
    color_of.update({i: COLOR_CHAIN1 for i in CHAIN1})

    # truth 链内边 (LCA y>0)
    tt = evt[("tracks", "to", "tracks")]
    ei, y = tt.edge_index.numpy(), tt.y.numpy()
    pos_edges = ei[:, y > 0]

    fig = go.Figure()

    # ---- 1. 背景 track: 灰色高透明 ----
    for i in range(n):
        if i in chain_of:
            continue
        d = np.array([px[i], py[i], pz[i]])
        d = d / (np.linalg.norm(d) + 1e-9)
        end = np.array([x0[i], y0[i], z0[i]]) + d * TRACK_LEN
        fig.add_trace(go.Scatter3d(
            x=[x0[i], end[0]], y=[y0[i], end[1]], z=[z0[i], end[2]],
            mode="lines", line=dict(color=COLOR_BKG, width=1.5),
            opacity=0.12, hoverinfo="skip", showlegend=False,
        ))

    # ---- 2. 链内 truth 边 (细线, 展示链结构) ----
    for a, b in pos_edges.T.tolist():
        if a not in chain_of or b not in chain_of:
            continue
        c = color_of[a]
        fig.add_trace(go.Scatter3d(
            x=[x0[a], x0[b]], y=[y0[a], y0[b]], z=[z0[a], z0[b]],
            mode="lines", line=dict(color=c, width=1.0, dash="dot"),
            opacity=0.55, hoverinfo="skip", showlegend=False,
        ))

    # ---- 3. 链 track (实线) ----
    for i in CHAIN0:
        d = np.array([px[i], py[i], pz[i]])
        d = d / (np.linalg.norm(d) + 1e-9)
        end = np.array([x0[i], y0[i], z0[i]]) + d * TRACK_LEN
        fig.add_trace(go.Scatter3d(
            x=[x0[i], end[0]], y=[y0[i], end[1]], z=[z0[i], end[2]],
            mode="lines", line=dict(color=COLOR_CHAIN0, width=5),
            opacity=1.0, showlegend=False,
            customdata=[[i, 0]] * 2,
            hovertemplate=(
                "track #%{customdata[0]}<br>链0 (7粒子, B衰变链)" +
                "<extra></extra>"),
        ))
    for i in CHAIN1:
        d = np.array([px[i], py[i], pz[i]])
        d = d / (np.linalg.norm(d) + 1e-9)
        end = np.array([x0[i], y0[i], z0[i]]) + d * TRACK_LEN
        fig.add_trace(go.Scatter3d(
            x=[x0[i], end[0]], y=[y0[i], end[1]], z=[z0[i], end[2]],
            mode="lines", line=dict(color=COLOR_CHAIN1, width=5),
            opacity=1.0, showlegend=False,
            customdata=[[i, 1]] * 2,
            hovertemplate=(
                "track #%{customdata[0]}<br>链1 (4粒子, B衰变链)" +
                "<extra></extra>"),
        ))

    # ---- 4. 产生点 marker ----
    for i in CHAIN0:
        fig.add_trace(go.Scatter3d(
            x=[x0[i]], y=[y0[i]], z=[z0[i]], mode="markers",
            marker=dict(size=5, color=COLOR_CHAIN0, symbol="circle"),
            hoverinfo="skip", showlegend=False,
        ))
    for i in CHAIN1:
        fig.add_trace(go.Scatter3d(
            x=[x0[i]], y=[y0[i]], z=[z0[i]], mode="markers",
            marker=dict(size=5, color=COLOR_CHAIN1, symbol="circle"),
            hoverinfo="skip", showlegend=False,
        ))

    # ---- 5. 图例 (用空 trace 占位) ----
    for label, color in [
        ("链0 (7粒子): v31 重建失败 → v37 重建成功", COLOR_CHAIN0),
        ("链1 (4粒子): v31 & v37 都成功", COLOR_CHAIN1),
        ("背景 track (无关)", COLOR_BKG),
    ]:
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None], mode="lines",
            line=dict(color=color, width=5), name=label, showlegend=True,
            opacity=0.15 if "背景" in label else 1.0,
        ))

    fig.update_layout(
        title=dict(
            text=("DFEI 重建优化: v37 vs v31<br>"
                  "<sup>事件 92441664 · 双B衰变 · 74 tracks · "
                  "蓝/红 = 信号 track, 灰 = 背景 (低透明度)</sup>"),
            font=dict(size=16),
        ),
        scene=dict(
            xaxis_title="x (归一化)", yaxis_title="y (归一化)", zaxis_title="z (束流方向)",
            aspectmode="cube",
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.1)),
        ),
        legend=dict(font=dict(size=13), x=0.02, y=0.98),
        width=1100, height=850,
    )

    os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)
    fig.write_html(OUT_HTML)
    print(f"[ok] 已生成: {OUT_HTML}")


if __name__ == "__main__":
    main()
