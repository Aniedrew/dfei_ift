"""阈值选择: 在本地 CPU 上复现完整重建指标, 扫描 thr, 分母用"未剪枝真值链总数 N_total"。

对每个阈值:
  1) 用管线自己的剪枝函数 (true_node_pruning + edge_pruning) 剪枝;
  2) 在剪枝后的图上做真值解码 (truth) 与模型解码 (reco);
  3) 逐真值链判定 AllParticles / PerfectReco / NoneIso / PartReco / NotFound
     (逻辑与 reconstruction.py 一致), 分母固定为 **未剪枝** 图上 200/300 事件的真值链总数;
  4) 覆盖率 = 剪枝后仍存在的真值链数 / N_total。

另外输出 pruning ROC (node/edge) 数据与 PNG。

用法: python thr_scan_metric.py --config config_files/eval_armC_renorm_v557.yaml --max-events 300 --tag armC_0904
"""
import argparse
import os
import sys

import numpy as np
import torch
import yaml

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from wmpgnn.analysis.config_adjusting import adjust_config_evaluation          # noqa: E402
from wmpgnn.analysis.load_module import load_module                            # noqa: E402
from wmpgnn.data_loader.get_data_loader import load_tst_loader                 # noqa: E402
from wmpgnn.data_loader.weights_calculator import transform_pos_weight         # noqa: E402
from wmpgnn.util.pruners import true_node_pruning, edge_pruning                # noqa: E402
from wmpgnn.reconstruction.reco_helper import (lca_truth_matrix, lca_reco_matrix,  # noqa: E402
                                               get_truth_part_keys, get_truth_part_ids,
                                               get_final_keys, reconstruct_decay, particle_name)

GRID = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.88, 0.9, 0.92, 0.95, 0.97]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--version", type=int, default=557)
    p.add_argument("--max-events", type=int, default=300)
    p.add_argument("--tag", default="scan")
    p.add_argument("--out", default="report_figs")
    return p.parse_args()


def truth_clusters(g):
    try:
        tl = lca_truth_matrix(g)
        keys = get_truth_part_keys(g).tolist()
        ids = list(map(particle_name, get_truth_part_ids(g).numpy()))
        tc, _, _ = reconstruct_decay(tl, keys, particle_ids=ids, truth_level_simulation=1)
        return tc
    except Exception:
        return {}


def reco_clusters(g):
    try:
        rl = lca_reco_matrix(g, mode="reco")
        keys = get_final_keys(g).tolist()
        rc, _, _ = reconstruct_decay(rl, keys)
        return rc
    except Exception:
        return {}


def judge(tc, rc):
    """返回 (All, Perf, NoneIso, Part, NotFound) 计数, 口径同 reconstruction.py。"""
    a = p = ni = pr = nf = 0
    for tc_key, t in tc.items():
        hit_all = hit_perf = False
        for rc_key, r in rc.items():
            if r["node_keys"] == t["node_keys"]:
                hit_all = True
                if r["LCA_values"] == t["LCA_values"]:
                    hit_perf = True
                break
            true_in = np.sum(np.isin(t["node_keys"], r["node_keys"])) / max(1, len(t["node_keys"]))
            if true_in == 1 and len(r["node_keys"]) > len(t["node_keys"]):
                ni += 1
                hit_all = hit_perf = None
                break
            if 0.2 <= true_in < 1:
                pr += 1
                hit_all = hit_perf = None
                break
        if hit_all is True:
            a += 1
            if hit_perf:
                p += 1
        elif hit_all is None:
            pass
        else:
            nf += 1
    return a, p, ni, pr, nf


def main():
    a = parse_args()
    cfg = yaml.safe_load(open(a.config))
    cfg["settings"]["model"] = a.version
    cfg = adjust_config_evaluation(cfg)
    module = load_module(cfg, transform_pos_weight(None, None, mode="eval"))
    module.eval()
    dev = next(module.parameters()).device
    _, _, chunkloader = load_tst_loader(cfg)

    evts = []
    n_evt = 0
    for batch in chunkloader.test_dataloader():
        batch = batch.to(dev)
        tt_ei_in = batch[("tracks", "to", "tracks")].edge_index.detach().cpu().clone()
        tb = batch["tracks"].batch.cpu().clone()
        if module.use_pid == "true":
            batch["tracks"].x = torch.cat([batch["tracks"].x, batch["tracks"].pid], dim=1)
        with torch.no_grad():
            module.model(batch)
        blk = module.model._blocks[-1]
        nw = blk.node_weights["tracks"].detach().cpu().squeeze(-1)
        ew = blk.edge_weights[("tracks", "to", "tracks")].detach().cpu().squeeze(-1)
        lca_logits = batch[("tracks", "to", "tracks")].edges.detach().cpu()   # op_trafo 覆盖后的 4 类 logits
        graphs = batch.to_data_list()
        for gid in range(int(tb.max().item()) + 1 if tb.numel() else 0):
            if n_evt >= a.max_events:
                break
            n_evt += 1
            tm = tb == gid
            em = tm[tt_ei_in[0]] & tm[tt_ei_in[1]]
            g = graphs[gid].clone()
            g[("tracks", "to", "tracks")].lca = lca_logits[em].clone()
            evts.append(dict(nw=nw[tm].clone(), ew=ew[em].clone(), ft=g["tracks"].ft.numpy().astype(int),
                             y=g[("tracks", "to", "tracks")].y.numpy(), g=g))
        if n_evt >= a.max_events:
            break

    # N_total: 未剪枝图上的真值链总数 (固定分母)
    n_total = 0
    for d in evts:
        n_total += len(truth_clusters(d["g"]))
    ns = np.concatenate([d["nw"].numpy() for d in evts])
    nl = np.concatenate([(d["ft"] != 1).astype(int) for d in evts])
    es = np.concatenate([d["ew"].numpy() for d in evts])
    el = np.concatenate([(d["y"] > 0).astype(int) for d in evts])

    def auc(score, label):
        o = np.argsort(score); r = np.empty(len(o), float); r[o] = np.arange(1, len(o) + 1)
        p = label.sum(); n = len(label) - p
        return (r[label == 1].sum() - p * (p + 1) / 2) / (p * n) if p and n else float("nan")

    print(f"\n=== {a.tag} · v{a.version} · 事件 {len(evts)}")
    print(f"  信号节点 {int(nl.sum())}/{len(nl)}  结构边(双向) {int(el.sum())}/{len(es)}")
    print(f"  未剪枝真值链总数 N_total = {n_total}   AUC(node)={auc(ns, nl):.5f}  AUC(edge)={auc(es, el):.5f}")

    print(f"\n{'thr':>6}{'覆盖率%':>9}{'All/Ntot%':>11}{'Perf/Ntot%':>12}{'NoneIso%':>10}{'Part%':>7}{'NotFnd%':>9}"
          f"{'节点召回%':>10}{'背景保留%':>10}")
    out = []
    for thr in GRID:
        A = P = NI = PR = NF = 0
        n_surv = 0
        k_rec = n_rec = b_keep = b_tot = 0
        for ei, d in enumerate(evts):
            g = d["g"].clone()
            nw, ew = d["nw"], d["ew"]
            nlab = d["ft"] != 1
            node_selbool = nw > thr
            k_rec += int((node_selbool & torch.tensor(nlab)).sum()); n_rec += int(nlab.sum())
            b_keep += int((node_selbool & torch.tensor(~nlab)).sum()); b_tot += int((~nlab).sum())
            edge_mask = true_node_pruning(node_selbool, g, "tracks", [('tracks', 'to', 'tracks')])
            edge_selbool = ew[edge_mask] > thr
            edge_pruning(edge_selbool, g, ('tracks', 'to', 'tracks'))
            tc = truth_clusters(g)
            n_surv += len(tc)
            rc = reco_clusters(g)
            if ei == 0 and thr == GRID[0]:
                print(f"  [诊断] thr={thr} 事件0: 真值链 {len(tc)} 条, reco 链 {len(rc)} 条, "
                      f"剪枝后 tt 边 {g[('tracks','to','tracks')].edge_index.shape[1]}, "
                      f"tt keys={list(g[('tracks','to','tracks')].keys())}")
                if not rc:
                    import traceback
                    try:
                        rl = lca_reco_matrix(g, mode="reco")
                        print(f"    (reco 直调: 矩阵 {len(rl)} 行)")
                    except Exception as e:
                        print("    !! reco 失败:", type(e).__name__, e)
                        traceback.print_exc()
            a_, p_, ni_, pr_, nf_ = judge(tc, rc)
            A += a_; P += p_; NI += ni_; PR += pr_; NF += nf_
        N = max(1, n_total)
        row = dict(thr=thr, coverage=100 * n_surv / N, All=100 * A / N, Perf=100 * P / N,
                   NoneIso=100 * NI / N, Part=100 * PR / N, NotFound=100 * NF / N,
                   node_recall=100 * k_rec / max(1, n_rec), node_bkg_keep=100 * b_keep / max(1, b_tot))
        out.append(row)
        print(f"{thr:>6.2f}{row['coverage']:>9.2f}{row['All']:>11.2f}{row['Perf']:>12.2f}{row['NoneIso']:>10.2f}"
              f"{row['Part']:>7.2f}{row['NotFound']:>9.2f}{row['node_recall']:>10.2f}{row['node_bkg_keep']:>10.2f}")
    os.makedirs(a.out, exist_ok=True)
    import csv
    p_csv = f"{a.out}/thr_scan_metric_{a.tag}_v{a.version}.csv"
    with open(p_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    np.save(f"{a.out}/roc_{a.tag}_v{a.version}_node.npy", np.stack([ns, nl]))
    np.save(f"{a.out}/roc_{a.tag}_v{a.version}_edge.npy", np.stack([es, el]))
    best = max(out, key=lambda r: r["All"])
    print(f"\nAllParticles 最大: thr={best['thr']:.2f} → All {best['All']:.2f}% / Perf {best['Perf']:.2f}% "
          f"(覆盖率 {best['coverage']:.2f}%, 背景保留 {best['node_bkg_keep']:.2f}%)")
    print("写出", p_csv)


if __name__ == "__main__":
    main()
