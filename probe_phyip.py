#!/usr/bin/env python3
"""PhyIP 式线性探针: 验证 mass head 是否让主干表征更"物理"。

冻结主干, 用线性回归 (Ridge) 从表示预测物理量, 对比 R²:
  - 边表征 latent_edges (16维) -> log10(ππ 不变质量): 核心对比 (mass head 直接监督的目标)
  - 节点表征 tracks.x (16维) -> px/py/pz: sanity (两者都应可预测, 验证方法有效)
对比: v38 (version_38 ep105) vs masshead2 (version_47 ep113 best)
"""
import sys, io, argparse
sys.path.append('/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn')

import yaml
import torch
import zstandard as zstd
from torch_geometric.data import Batch
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

from wmpgnn.analysis.config_adjusting import adjust_config_training
from wmpgnn.analysis.load_module import load_module
from wmpgnn.data_loader.weights_calculator import transform_pos_weight

torch.manual_seed(0)

# 与 _mass_loss 完全一致的反归一化常数 (center, scale)
NORM = {
    "px": (-4.1619, 470.8137),
    "py": (0.7674, 597.9097),
    "pz": (7117.4619, 10077.2412),
}
M_PI = 139.570

DATA = '/lzufs/user/guoqingxiang/DFEI_IFT_20260702/data/MC_normed/inclusive_00342442/tst_data_00220000_00220999.pt.zst'
CKPTS = {
    "v38":      '/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn/LHCb_logs/DFEI/version_38/checkpoints/best-epoch=105-val_combined_loss=35.921.ckpt',
    "masshead2": '/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn/LHCb_logs/DFEI/version_47/checkpoints/best-epoch=113-val_combined_loss=35.677.ckpt',
}


def load_events(path, n=4):
    dctx = zstd.ZstdDecompressor()
    with open(path, 'rb') as f:
        with dctx.stream_reader(f) as reader:
            data = torch.load(io.BytesIO(reader.read()), weights_only=False)
    evts = data[:n]
    for evt in evts:  # 双向化 (与 load_dataset 一致)
        store = evt[('tracks', 'to', 'tracks')]
        store.edge_index = torch.cat([store.edge_index, store.edge_index.flip(0)], dim=1)
        store.edges = store.edges.repeat(2, 1)
        if getattr(store, 'y', None) is not None:
            store.y = store.y.repeat(2)
        if getattr(store, 'lca', None) is not None:
            store.lca = store.lca.repeat(2, 1)
    return evts


def denorm_p(p_norm):
    px = p_norm[:, 0] * NORM["px"][1] + NORM["px"][0]
    py = p_norm[:, 1] * NORM["py"][1] + NORM["py"][0]
    pz = p_norm[:, 2] * NORM["pz"][1] + NORM["pz"][0]
    return px, py, pz


def pi_pi_logm(p1, p2):
    """两条径迹 (反归一化动量 [E,3]) 的 ππ 不变质量 (log10 MeV)。"""
    E1 = torch.sqrt((p1 ** 2).sum(-1) + M_PI ** 2)
    E2 = torch.sqrt((p2 ** 2).sum(-1) + M_PI ** 2)
    m2 = (E1 + E2) ** 2 - ((p1 + p2) ** 2).sum(-1)
    m = torch.sqrt(torch.clamp(m2, min=1.0))
    return torch.log10(m)


def extract(module, events, max_edges_per_evt=10000):
    """提取节点/边表征与物理真值。返回 dict of numpy 数组。"""
    X_node, y_px, y_py, y_pz, y_px_n = [], [], [], [], []
    X_edge, y_logm = [], []
    for evt in events:
        batch = Batch.from_data_list([evt])
        p_norm = batch['tracks'].x[:, :3].clone()      # 原始归一化动量 (forward 会覆盖 x)
        ei = batch[('tracks', 'to', 'tracks')].edge_index
        with torch.no_grad():
            outputs = module.forward(batch)             # concat pid + model forward
        # 节点: 表征 -> px/py/pz (MeV) 与归一化输入 (encoder 直接输入)
        nx = outputs['tracks'].x.cpu()
        px, py, pz = denorm_p(p_norm.cpu())
        X_node.append(nx)
        y_px.append(px); y_py.append(py); y_pz.append(pz)
        y_px_n.append(p_norm[:, 0].cpu())
        # 边: latent_edges -> log10(m)
        ex = outputs[('tracks', 'to', 'tracks')].latent_edges.cpu()
        a, b = ei[0].cpu(), ei[1].cpu()
        p1 = torch.stack([px[a], py[a], pz[a]], dim=-1)
        p2 = torch.stack([px[b], py[b], pz[b]], dim=-1)
        m = pi_pi_logm(p1, p2)
        # 采样 (边太多, 均匀抽 max_edges)
        if ex.shape[0] > max_edges_per_evt:
            idx = torch.randperm(ex.shape[0])[:max_edges_per_evt]
            ex, m = ex[idx], m[idx]
        X_edge.append(ex)
        y_logm.append(m)
    return {
        "X_node": torch.cat(X_node).numpy(),
        "y_px": torch.cat(y_px).numpy(), "y_py": torch.cat(y_py).numpy(), "y_pz": torch.cat(y_pz).numpy(),
        "y_px_n": torch.cat(y_px_n).numpy(),
        "X_edge": torch.cat(X_edge).numpy(), "y_logm": torch.cat(y_logm).numpy(),
    }


def ridge_r2(X, y):
    reg = Ridge(alpha=1.0)
    reg.fit(X, y)
    pred = reg.predict(X)
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot


def mlp_r2(X, y):
    """非线性探针: 判断信息是否只是"非线性编码" (线性 R² 低但 MLP R² 高 -> 信息在)。"""
    reg = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu',
                       max_iter=1000, random_state=0)
    reg.fit(X, y)
    pred = reg.predict(X)
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nevents', type=int, default=4)
    ap.add_argument('--which', nargs='+', default=list(CKPTS.keys()))
    args = ap.parse_args()

    with open('config_files/train_CERN_v38_masshead.yaml') as f:
        configs = yaml.safe_load(f)
    configs = adjust_config_training(configs)
    pos_weights = transform_pos_weight(None, None, mode="eval")

    events = load_events(DATA, n=args.nevents)
    print(f"[probe] 事件数: {len(events)}, 模型: {args.which}")

    results = {}
    for name in args.which:
        cfg = yaml.safe_load(open('config_files/train_CERN_v38_masshead.yaml'))
        cfg = adjust_config_training(cfg)
        cfg['DFEI']['cpt'] = CKPTS[name]
        module = load_module(cfg, pos_weights)
        module.eval()
        r = extract(module, events)
        # 节点表征 -> 动量: MeV 值 (跨事件) 与归一化输入 (encoder 直接输入)
        r2_px = ridge_r2(r["X_node"], r["y_px"])
        r2_pz = ridge_r2(r["X_node"], r["y_pz"])
        r2_px_n = ridge_r2(r["X_node"], r["y_px_n"])
        r2_px_n_mlp = mlp_r2(r["X_node"], r["y_px_n"])
        # 核心: 边表征 -> log10(m)
        r2_m = ridge_r2(r["X_edge"], r["y_logm"])
        results[name] = (r2_px, r2_pz, r2_px_n, r2_m, r2_px_n_mlp)
        print(f"\n[{name}]")
        print(f"  节点表征 -> px(MeV):  线性 R² = {r2_px:.4f} | pz: {r2_pz:.4f}")
        print(f"  节点表征 -> px(归一化输入): 线性 R² = {r2_px_n:.4f} | MLP R² = {r2_px_n_mlp:.4f}")
        print(f"  边表征 -> log10(m_ππ): 线性 R² = {r2_m:.4f}  (核心)")

    print("\n========== 对比表 ==========")
    print(f"  {'模型':<12} {'边->m 线性':>10} {'节点->px_n 线性':>12} {'节点->px_n MLP':>12}")
    for name in results:
        print(f"  {name:<12} {results[name][3]:>10.4f} {results[name][2]:>12.4f} {results[name][4]:>12.4f}")


if __name__ == '__main__':
    main()
