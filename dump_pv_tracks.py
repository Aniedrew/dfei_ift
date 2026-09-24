"""逐径迹 dump: 真值 PV / 我们的判定 / minIP 判定 + 物理特征, 供"错配率 vs 物理量"分析。

关键量:
  ip_rank = 把各 PV 按 log_minIP 升序排好后 **真值 PV 的名次** (0 = 真值 PV 恰好是 IP 最小的那个)
  ip_gap  = log_minIP(真值 PV) - min(log_minIP)  (名次>0 时为正, 越大说明真值 PV 越"不像")
  dz_pred / dz_minip = z(判定 PV) - z(真值 PV)     (错配方向的物理量, 带符号; z 为归一化值)
用法: python dump_pv_tracks.py --config <cfg> --version <v> --max-events 300 --tag x --norm old|new
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

from wmpgnn.analysis.config_adjusting import adjust_config_evaluation          # noqa: E402
from wmpgnn.analysis.load_module import load_module                            # noqa: E402
from wmpgnn.data_loader.get_data_loader import load_tst_loader                 # noqa: E402
from wmpgnn.data_loader.weights_calculator import transform_pos_weight         # noqa: E402
from wmpgnn.util.pruners import true_node_pruning, edge_pruning                # noqa: E402
from wmpgnn.reconstruction.reco_helper import (lca_truth_matrix, get_truth_part_keys,  # noqa: E402
                                              get_truth_part_ids, reconstruct_decay, particle_name)

NORMS = {"new": "/lzufs/user/guoqingxiang/DFEI_IFT_20260904/dfei_repo/preprocessing/normalization_dict.pt",
         "old": "/lzufs/user/guoqingxiang/DFEI_IFT_20260702/dfei_repo/preprocessing/old_norm_ported.pt"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--version", type=int, required=True)
    p.add_argument("--max-events", type=int, default=300)
    p.add_argument("--tag", default="dump")
    p.add_argument("--norm", default="new", choices=["new", "old"])
    p.add_argument("--out", default="report_figs")
    return p.parse_args()


def main():
    a = parse_args()
    cfg = yaml.safe_load(open(a.config))
    cfg["settings"]["model"] = a.version
    cfg = adjust_config_evaluation(cfg)
    thr_n = float(cfg["inference"].get("node_prune_thr", 0.9))
    thr_e = float(cfg["inference"].get("edge_prune_thr", 0.9))
    module = load_module(cfg, transform_pos_weight(None, None, mode="eval"))
    module.eval()
    dev = next(module.parameters()).device
    _, _, chunkloader = load_tst_loader(cfg)

    nd = torch.load(NORMS[a.norm], map_location="cpu")
    c = {k: float(nd["center"][k + "_reco"]) for k in ("px", "py", "pz")}
    s = {k: float(nd["scale"][k + "_reco"]) for k in ("px", "py", "pz")}

    rows, n_evt = [], 0
    for batch in chunkloader.test_dataloader():
        batch = batch.to(dev)
        ei_tt = batch[("tracks", "to", "tracks")].edge_index.detach().cpu()
        ei_tp = batch[("tracks", "to", "pvs")].edge_index.detach().cpu()
        minip_tp = batch[("tracks", "to", "pvs")].edges.detach().cpu().numpy().reshape(-1)
        y_tp = batch[("tracks", "to", "pvs")].y.detach().cpu().numpy().reshape(-1)
        tp_store = batch[("tracks", "pvs")]
        filt = (tp_store.filter.detach().cpu().numpy().reshape(-1) > 0) if "filter" in tp_store \
            else np.ones(y_tp.shape[0], dtype=bool)
        tb = batch["tracks"].batch.detach().cpu()
        pb = batch["pvs"].batch.detach().cpu()
        raw_x = batch["tracks"].x.detach().cpu().clone()
        ft = batch["tracks"].ft.detach().cpu().numpy().astype(int)
        assert y_tp.shape[0] == minip_tp.shape[0] == filt.shape[0], (y_tp.shape, minip_tp.shape, filt.shape)
        if module.use_pid == "true":
            batch["tracks"].x = torch.cat([batch["tracks"].x, batch["tracks"].pid], dim=1)
        with torch.no_grad():
            out = module.model(batch)
        blk = module.model._blocks[-1]
        nw = blk.node_weights["tracks"].detach().cpu().squeeze(-1).numpy()
        ew = blk.edge_weights[("tracks", "to", "tracks")].detach().cpu().squeeze(-1).numpy()
        pvsc_all = blk.edge_weights[("tracks", "to", "pvs")].detach().cpu().squeeze(-1).numpy()
        out[("tracks", "to", "tracks")].lca = out[("tracks", "to", "tracks")].edges
        graphs = out.to_data_list()

        for gid in range(int(tb.max().item()) + 1 if tb.numel() else 0):
            if n_evt >= a.max_events:
                break
            n_evt += 1
            gt_idx = (tb == gid).nonzero()[:, 0].numpy()       # 本事件径迹的全局索引
            gp_idx = (pb == gid).nonzero()[:, 0].numpy()
            tr_loc = {int(k): i for i, k in enumerate(gt_idx)}
            pv_loc = {int(k): i for i, k in enumerate(gp_idx)}
            nt, npv = len(gt_idx), len(gp_idx)
            g = graphs[gid]
            assert int(g["tracks"].num_nodes) == nt and int(g["pvs"].num_nodes) == npv, \
                (int(g["tracks"].num_nodes), nt, int(g["pvs"].num_nodes), npv)
            pvz = g["pvs"].x[:, 2].numpy().astype(float)
            # 逐径迹 × 逐 PV 矩阵 (用 store 自己的 edge_index 散射, 不假设顺序)
            ipmat = np.full((nt, npv), np.nan)
            pvmat = np.full((nt, npv), -np.inf)
            ymat = np.zeros((nt, npv), dtype=int)
            em = (tb[ei_tp[0]].numpy() == gid) & (pb[ei_tp[1]].numpy() == gid) & filt
            ii = np.array([tr_loc[int(k)] for k in ei_tp[0].numpy()[em]])
            jj = np.array([pv_loc[int(k)] for k in ei_tp[1].numpy()[em]])
            ipmat[ii, jj] = minip_tp[em]
            pvmat[ii, jj] = pvsc_all[em]
            ymat[ii, jj] = y_tp[em].astype(int)
            # 剪枝(与管线一致), 用于取"存活径迹"与真值链信息
            node_sel = nw[gt_idx] > thr_n
            g2 = g.clone()
            emask = true_node_pruning(torch.tensor(node_sel), g2, "tracks", [('tracks', 'to', 'tracks')])
            emask = emask.numpy() if torch.is_tensor(emask) else emask
            emm = (tb[ei_tt[0]] == gid) & (tb[ei_tt[1]] == gid)
            es = ew[emm.numpy()][emask] > thr_e
            edge_pruning(torch.tensor(es), g2, ('tracks', 'to', 'tracks'))
            try:
                tl = lca_truth_matrix(g2)
                keys = get_truth_part_keys(g2).tolist()
                ids = list(map(particle_name, get_truth_part_ids(g2).numpy()))
                tc, _, _ = reconstruct_decay(tl, keys, particle_ids=ids, truth_level_simulation=1)
            except Exception:
                tc = {}
            clen, cpos = {}, {}
            for ck, cl in tc.items():
                for pos, k in enumerate(cl["node_keys"]):
                    clen[int(k)] = len(cl["node_keys"]); cpos[int(k)] = pos

            for i, gix in enumerate(gt_idx):
                gix = int(gix)
                px = raw_x[gix, 0].item() * s["px"] + c["px"]
                py = raw_x[gix, 1].item() * s["py"] + c["py"]
                pz = raw_x[gix, 2].item() * s["pz"] + c["pz"]
                pt = float(np.hypot(px, py)); pm = float(np.sqrt(px * px + py * py + pz * pz))
                # 与管线一致: filter 通过的边里, true = 标签 1 的第一条; pred = 得分最大; minIP = IP 最小
                if np.isfinite(ipmat[i]).any():
                    t_pv = int(np.argmax(ymat[i])) if ymat[i].any() else -1
                    pred = int(np.argmax(np.where(np.isfinite(ipmat[i]), pvmat[i], -np.inf)))
                    mip = int(np.nanargmin(ipmat[i]))
                else:
                    t_pv = pred = mip = -1
                order = np.argsort(ipmat[i])
                rank = int(np.where(order == t_pv)[0][0]) if 0 <= t_pv < npv else -1
                zt = float(pvz[t_pv]) if 0 <= t_pv < npv else np.nan
                rows.append(dict(evt=n_evt, npvs=npv, true_pv=t_pv, pred_pv=pred, minip_pv=mip,
                                 ip_rank=rank, ip_true=float(ipmat[i, t_pv]) if 0 <= t_pv < npv else np.nan,
                                 ip_min=float(np.min(ipmat[i])), ip_gap=float(ipmat[i, t_pv] - np.min(ipmat[i]))
                                 if 0 <= t_pv < npv else np.nan,
                                 z_true=zt, dz_pred=float(pvz[pred]) - zt, dz_minip=float(pvz[mip]) - zt,
                                 node_w=float(nw[gix]), node_alive=bool(node_sel[i]), is_sig=int(ft[gix] != 1),
                                 pt=pt, p=pm, eta=float(np.arctanh(np.clip(pz / max(pm, 1e-9), -.999, .999))),
                                 charge=float(raw_x[gix, 6]), ghost=float(raw_x[gix, 7]),
                                 chain_len=clen.get(gix, -1), chain_pos=cpos.get(gix, -1)))
        if n_evt >= a.max_events:
            break

    df = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    p = f"{a.out}/pv_tracks_{a.tag}.csv"
    df.to_csv(p, index=False)
    ok = df[df["true_pv"] >= 0]
    print(f"\n[{a.tag}] 事件 {n_evt} / 径迹 {len(df)} (有真值 PV {len(ok)}) -> {p}")
    print(f"  正确率: 我们 {100*(ok.pred_pv==ok.true_pv).mean():.2f}% | minIP {100*(ok.minip_pv==ok.true_pv).mean():.2f}%")
    print(f"  真值 PV 恰好就是 min-IP PV 的比例: {100*(ok.ip_rank==0).mean():.2f}%")
    hard = ok[ok.ip_rank > 0]
    print(f"  在 ip_rank>0 (minIP 必错) 的 {len(hard)} 条径迹上: 我们救回 {100*(hard.pred_pv==hard.true_pv).mean():.2f}%")
    print(f"  存活径迹比例 {100*ok.node_alive.mean():.2f}% | 平均 pT {ok.pt.mean():.0f} MeV | 平均 |η| {ok.eta.abs().mean():.2f}")


if __name__ == "__main__":
    main()
