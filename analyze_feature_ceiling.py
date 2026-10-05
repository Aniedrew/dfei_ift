"""特征上界分析: "现有可测量特征"到底能不能分辨难负例?

动机 (2026-09-30)
    这一轮所有模型侧手段 (加头/加损失/调采样/换排序/加结构) 全部无效或为负, 唯一有效的
    是"给剪枝 MLP 喂更好的物理输入"。于是在动结构之前必须先回答一个更基本的问题:

        那条缺失的信息, 到底**在不在现有输入里**?

    做法: 不训练 GNN, 只取每条 tt 边的**特征**与**真值类别**, 用一个**简单可解释**的分类器
    (梯度提升树) 去分 "真边 vs fake_inter"。于是:
      - 若简单分类器也 ≈0.6  -> 特征里就没有那条信息 -> 必须换输入 (更物理的量/上游要量);
      - 若简单分类器远高于 GNN (0.72) -> 信息在, 是模型/训练没用上 -> 该动结构。
    这也直接给出了"顶点路线"的特征侧天花板。

关键设计 (避免自欺)
    - **按事件切分** train/test: 同事件内的边高度相关, 按边随机切会严重高估泛化。
    - 特征分 4 组, 逐组累加, 看每加一组涨多少 -> 定位"信息在哪一组里/根本不在"。
    - 任务有两个: 难池 (真边 vs fake_intra+inter) 与 最难的 true vs fake_inter。

用法 (CPU 即可, 不需要 GPU):
    python3 analyze_feature_ceiling.py --config config_files/eval_iso_v601_k0_n95.yaml \
        --version 601 --events 400 --out report_figs/feat_ceiling_601.npz
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import yaml

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from wmpgnn.analysis.config_adjusting import adjust_config_evaluation        # noqa: E402
from wmpgnn.data_loader.get_data_loader import load_tst_loader                # noqa: E402
from wmpgnn.lightning_module.dfei_lightning_module import derive_pruning_features  # noqa: E402
from wmpgnn.reconstruction.reco_helper import (lca_truth_matrix, get_truth_part_keys,  # noqa: E402
                                              get_truth_part_ids, reconstruct_decay, particle_name)

NORM = "/lzufs/user/guoqingxiang/DFEI_IFT_20260904/dfei_repo/preprocessing/normalization_dict.pt"

RAW_NAMES = ["fspv", "theta", "trdist", "dz0", "logDOCA"]
DER_NAMES = ["dR", "m", "absSumQ", "dpT", "dIP", "rank", "dz", "sup", "rk_aff"]
VRT_NAMES = ["zcpa", "flight", "collin"]
NODE_NAMES = [f"n{i}" for i in range(8)]        # 两端各 8 维节点特征
# [2026-09-30] 修正几何 (绕开上游 calculate_doca 的 t1 符号 bug, 用标准最小二乘解现算)
GEO_NAMES = ["doca", "logdoca", "dstart"]
# [2026-09-30] 上下文特征 (含**第三方**信息): 以事件内两两 |Δ起点| 定义亲和度 A=exp(-d/50),
#   再看"这一对在两端径迹各自的所有候选伙伴里排第几 / 离最优差多少 / 是否互为最优"。
#   动机: 之前的 BDT 上界 (0.70~0.75) 只用"这一对 + 两端特征", **没有任何第三方信息**;
#   而"上下文能否突破这个上限"从未被检验过 -> 这是"要不要上注意力/匹配"的闸门。
#   注意: 全部只用输入量, 不含任何真值。
CTX_NAMES = ["rk_s", "gap_s", "ratio_s", "hi_s", "deg_s",
             "rk_t", "gap_t", "ratio_t", "hi_t", "deg_t", "mutual"]


def collect(a):
    cfg = yaml.safe_load(open(a.config))
    cfg["settings"]["model"] = a.version
    cfg = adjust_config_evaluation(cfg)
    _, _, ckl = load_tst_loader(cfg)
    nd = torch.load(NORM, map_location="cpu")
    nc = {k: float(v) for k, v in nd["center"].items()}
    ns = {k: (float(v) or 1.0) for k, v in nd["scale"].items()}

    # [2026-10-03] 额外导出"每条边两端径迹的事件内局部 id"(t0/t1)与事件径迹数(ntr):
    #   这是**全局匹配(Sinkhorn/匈牙利)**与 **track-token 注意力** 的前提 —— 没有端点身份,
    #   就无法构建"同事件径迹 x 径迹"的稠密矩阵, 也无法把 token 从"边"换成"径迹"。
    rows = {k: [] for k in ("X", "y", "grp", "t0", "t1", "ntr")}
    n_evt = 0
    for batch in ckl.test_dataloader():
        # 原始 tt 边特征 (模型的输入; 注意不能在模型前向之后取, 会被覆盖)
        x_raw = batch[("tracks", "to", "tracks")].edges.detach().cpu().numpy()
        ei = batch[("tracks", "to", "tracks")].edge_index.detach().cpu()
        tb = batch["tracks"].batch.detach().cpu()
        ft = batch["tracks"].ft.detach().cpu().numpy().astype(int)
        y_tt = batch[("tracks", "to", "tracks")].y.detach().cpu().numpy().reshape(-1)
        x_node = batch["tracks"].x.detach().cpu().numpy()
        # 派生特征 (与训练时同一函数、同一归一化字典): 9 维三角版 + 12 维含次级顶点版
        _, der9 = derive_pruning_features(batch, nc, ns, True, False)
        _, der12 = derive_pruning_features(batch, nc, ns, True, True)
        der9 = der9.detach().cpu().numpy()
        der12 = der12.detach().cpu().numpy()
        graphs = [g for g in batch.to_data_list()]

        for gid in range(int(tb.max().item()) + 1 if tb.numel() else 0):
            if n_evt >= a.events:
                break
            n_evt += 1
            tm = (tb == gid)
            em = tm[ei[0]] & tm[ei[1]]
            g = graphs[gid]
            try:
                tl = lca_truth_matrix(g)
                keys = get_truth_part_keys(g).tolist()
                ids = list(map(particle_name, get_truth_part_ids(g).numpy()))
                tc, _, _ = reconstruct_decay(tl, keys, particle_ids=ids, truth_level_simulation=1)
            except Exception:
                tc = {}
            pk = g["tracks"].part_keys.numpy().tolist()
            pk2i = {int(k): i for i, k in enumerate(pk)}
            cl_id = np.full(len(pk), -1, dtype=int)
            for ci, (ck, cl) in enumerate(tc.items()):
                nodes = [pk2i[int(k)] for k in cl["node_keys"] if int(k) in pk2i]
                if len(nodes) < 2:
                    continue
                cl_id[nodes] = ci
            gt = tm.nonzero()[:, 0].numpy()
            g2l = {int(x): i for i, x in enumerate(gt)}
            ei_g = ei[:, em].numpy()
            loc = np.array([[g2l[int(x)] for x in ei_g[0]], [g2l[int(x)] for x in ei_g[1]]], dtype=int)
            ys = (y_tt[em.numpy()] > 0).astype(int)
            is_sig = (ft[gt] != 1).astype(int)
            # 类别: true / fake_intra / fake_inter / 其它(易例, 本分析丢弃)
            cls = np.array(["" for _ in range(loc.shape[1])], dtype=object)
            for k in range(loc.shape[1]):
                i, j = int(loc[0, k]), int(loc[1, k])
                si, sj = is_sig[i], is_sig[j]
                ci_, cj_ = cl_id[i], cl_id[j]
                if ys[k] == 1:
                    cls[k] = "true"
                elif si and sj and ci_ >= 0 and ci_ == cj_:
                    cls[k] = "fake_intra"
                elif si and sj:
                    cls[k] = "fake_inter"
                else:
                    cls[k] = "easy"
            keep = (cls == "true") | (cls == "fake_intra") | (cls == "fake_inter")
            if not keep.any():
                continue
            kk = np.nonzero(keep)[0]
            # ---- 修正几何: 两条径迹所在直线的最近距离(DOCA) + 两端生产顶点距离(|Δ起点|) ----
            xt = x_node[gt]                                     # 该事件节点特征 (归一化)
            def _rc(nm, col, _xt=xt):
                return _xt[:, col] * float(ns.get(nm, 1.0)) + float(nc.get(nm, 0.0))
            _P1 = np.stack([_rc("px_reco", 0)[loc[0]], _rc("py_reco", 1)[loc[0]], _rc("pz_reco", 2)[loc[0]]], 1)
            _P2 = np.stack([_rc("px_reco", 0)[loc[1]], _rc("py_reco", 1)[loc[1]], _rc("pz_reco", 2)[loc[1]]], 1)
            _A1 = np.stack([_rc("xProd_reco", 3)[loc[0]], _rc("yProd_reco", 4)[loc[0]], _rc("zProd_reco", 5)[loc[0]]], 1)
            _B1 = np.stack([_rc("xProd_reco", 3)[loc[1]], _rc("yProd_reco", 4)[loc[1]], _rc("zProd_reco", 5)[loc[1]]], 1)
            _u1 = _P1 / np.clip(np.linalg.norm(_P1, axis=1, keepdims=True), 1e-9, None)
            _u2 = _P2 / np.clip(np.linalg.norm(_P2, axis=1, keepdims=True), 1e-9, None)
            _w = _A1 - _B1
            _cc = (_u1 * _u2).sum(1)
            _dd = (_u1 * _w).sum(1)
            _ee = (_u2 * _w).sum(1)
            _den = 1.0 - _cc * _cc
            _ok = _den > 1e-6
            _t1 = np.where(_ok, (_cc * _ee - _dd) / np.where(_ok, _den, 1.0), 0.0)
            _t2 = np.where(_ok, (_ee - _cc * _dd) / np.where(_ok, _den, 1.0), 0.0)
            _doca = np.linalg.norm(_w + _t1[:, None] * _u1 - _t2[:, None] * _u2, axis=1)
            _dstart = np.linalg.norm(_w, axis=1)
            geo = np.stack([_doca / 100.0, np.log(_doca + 1e-5), _dstart / 100.0], 1)
            # ---- 上下文: 事件内亲和度 -> 排名 / 离最优的差 / 度 / 是否互为最优 ----
            _ae = np.exp(-_dstart / 50.0)
            _df = pd.DataFrame({"s": loc[0], "t": loc[1], "a": _ae})
            _df["hi"] = (_df["a"] > 0.5).astype(float)
            def _ctx(keycol):
                grp = _df.groupby(keycol)["a"]
                mx = grp.transform("max").values
                cnt = grp.transform("size").values.astype(float)
                rk = grp.rank(ascending=False, method="average").values
                hi = _df.groupby(keycol)["hi"].transform("sum").values
                return np.stack([(rk - 1.0) / np.maximum(cnt - 1.0, 1.0), mx - _df["a"].values,
                                 _df["a"].values / (mx + 1e-9), hi / 10.0, cnt / 50.0], 1)
            _cs, _ct = _ctx("s"), _ctx("t")
            _mut = ((_cs[:, 0] == 0) & (_ct[:, 0] == 0)).astype(np.float32)[:, None]
            ctx = np.concatenate([_cs, _ct, _mut], 1)
            node_pair = np.concatenate([x_node[gt[loc[0]]], x_node[gt[loc[1]]]], axis=1)   # [E,16]
            X = np.concatenate([x_raw[kk], der9[kk], der12[kk][:, 9:], node_pair[kk],
                                geo[kk], ctx[kk]], axis=1)
            rows["X"].append(X.astype(np.float32))
            rows["y"].append((cls[kk] == "true").astype(np.int8))
            rows["grp"].append(np.full(len(kk), n_evt, dtype=np.int32))
            rows["t0"].append(loc[0][kk].astype(np.int32))      # 起点径迹的事件内局部 id
            rows["t1"].append(loc[1][kk].astype(np.int32))      # 终点径迹的事件内局部 id
            rows["ntr"].append(np.full(len(kk), int(len(gt)), dtype=np.int32))   # 该事件径迹数
        if n_evt >= a.events:
            break

    X = np.concatenate(rows["X"])
    y = np.concatenate(rows["y"])
    grp = np.concatenate(rows["grp"])
    names = (RAW_NAMES + DER_NAMES + VRT_NAMES + [f"{p}_{n}" for p in ("a", "b") for n in NODE_NAMES]
             + GEO_NAMES + CTX_NAMES)
    assert X.shape[1] == len(names), (X.shape, len(names))
    t0 = np.concatenate(rows["t0"]); t1 = np.concatenate(rows["t1"]); ntr = np.concatenate(rows["ntr"])
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez_compressed(a.out, X=X, y=y, grp=grp, names=np.array(names),
                        t0=t0, t1=t1, ntr=ntr)
    print(f"[feat] 写出 {a.out}: X={X.shape} 真边={int(y.sum())} 难负例={int((1-y).sum())} "
          f"事件={len(np.unique(grp))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_files/eval_iso_v601_k0_n95.yaml")
    ap.add_argument("--version", type=int, default=601)
    ap.add_argument("--events", type=int, default=400)
    ap.add_argument("--out", default="report_figs/feat_ceiling_601.npz")
    a = ap.parse_args()
    collect(a)


if __name__ == "__main__":
    main()
