"""前置验证: 核对归一化常数 + 检查哨兵掩码 + 看 ππ 不变质量分布 (CPU, 快)。"""
import io
import zstandard as zstd
import torch

NORM = "/lzufs/user/guoqingxiang/DFEI_IFT_20260702/dfei_repo/preprocessing/normalization_dict.pt"
DATA = "/lzufs/user/guoqingxiang/DFEI_IFT_20260702/data/MC_normed/inclusive_00342442/tst_data_00220000_00220999.pt.zst"

# 1) 归一化字典核对
d = torch.load(NORM, weights_only=False)
c, s = d["center"], d["scale"]
print("normalization_dict keys:", list(d.keys()))
for k in ["px_reco", "py_reco", "pz_reco"]:
    print(f"{k}: center={c[k]:.4f} scale={s[k]:.4f}")

# 反归一化常数 (center, scale) —— 与 mass head 中一致
NORM_C = {
    "px": (c["px_reco"], s["px_reco"]),
    "py": (c["py_reco"], s["py_reco"]),
    "pz": (c["pz_reco"], s["pz_reco"]),
}

# 2) 读一个事件, 检查 denorm 动量与哨兵
dctx = zstd.ZstdDecompressor()
with open(DATA, "rb") as f:
    with dctx.stream_reader(f) as reader:
        data = torch.load(io.BytesIO(reader.read()), weights_only=False)
print(f"\n{len(data)} events in file")
evt = data[0]
x = evt["tracks"].x
print(f"tracks.x shape: {x.shape}")   # 应为 [N, 8]
px = x[:, 0] * NORM_C["px"][1] + NORM_C["px"][0]
py = x[:, 1] * NORM_C["py"][1] + NORM_C["py"][0]
pz = x[:, 2] * NORM_C["pz"][1] + NORM_C["pz"][0]
pt = torch.sqrt(px ** 2 + py ** 2)
p = torch.sqrt(px ** 2 + py ** 2 + pz ** 2)
print(f"N tracks: {len(px)}")
print(f"pT: min={pt.min():.1f} p50={pt.median():.1f} max={pt.max():.1f} MeV")
print(f"|p|: max={p.max():.1f} MeV")
print(f"px<0 比例: {(px < 0).float().mean():.3f}  (真实负 px 轨迹比例)")

# 哨兵检测: 三动量同时≈-1 (未重建径迹)
sent = (px > -1.5) & (px < -0.5) & (py > -1.5) & (py < -0.5) & (pz > -1.5) & (pz < -0.5)
print(f"哨兵(px≈py≈pz≈-1) 比例: {sent.float().mean():.4f}")
# 单看 px≈-1 (含真实负px的误伤风险)
sent_px = (px > -1.5) & (px < -0.5)
print(f"px∈(-1.5,-0.5) 比例: {sent_px.float().mean():.4f}  (若显著>哨兵比例, 说明会误伤真实轨迹)")

# 3) ππ 不变质量分布 (验证目标量纲 + log10 尺度)
m_pi = 139.570
valid = ~sent
if valid.sum() >= 2:
    idx = valid.nonzero(as_tuple=False).flatten()[:50]
    # 取前 50 个有效轨迹的两两组合 (稀疏采样)
    ms = []
    for i in range(0, min(len(idx), 20), 2):
        j = i + 1
        if j >= len(idx):
            break
        p1 = torch.stack([px[idx[i]], py[idx[i]], pz[idx[i]]])
        p2 = torch.stack([px[idx[j]], py[idx[j]], pz[idx[j]]])
        E1 = torch.sqrt((p1 ** 2).sum() + m_pi ** 2)
        E2 = torch.sqrt((p2 ** 2).sum() + m_pi ** 2)
        m2 = (E1 + E2) ** 2 - ((p1 + p2) ** 2).sum()
        ms.append(torch.sqrt(torch.clamp(m2, min=1.0)))
    ms = torch.stack(ms)
    print(f"\nππ 不变质量 (20 对采样): min={ms.min():.0f} med={ms.median():.0f} max={ms.max():.0f} MeV")
    print(f"log10(m) 范围: {torch.log10(ms).min():.2f} ~ {torch.log10(ms).max():.2f}")
