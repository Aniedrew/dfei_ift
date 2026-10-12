"""方向3 离线评测: "连通分量" vs "seqrec-learned" vs "seqrec-geom" (用已训练 ckpt, 无需重训)。

对同一批真实事件、同一候选边集 (v654 工作点剪枝后、且模型判 LCA>0 的 tt 边), 只替换
"最后一步: 连通分量 -> 顺序重组合" 的聚合方式, 与同一份真值链比较:

  1. 连通分量基线 (existing: reconstruct_decay 的 connected components, 等价于
     single-linkage + cut=+inf; 用 sequential_recombine(cut=None) 复现, 已在单测里回归核对);
  2. seqrec + learned 度量  d = -ln(p_e + eps);
  3. seqrec + geom    度量  d = 纯几何距离 (DOCA/100, 无学习对照)。

指标 (逐事件, 事件级聚合):
  - chain_full  整链完整恢复比例: 真值链中能被某条预测链**完全精确**复现的比例 (AllParticles 口径);
  - chain_P / chain_R  链级 precision / recall (IoU >= 0.5 贪心二部匹配);
  - dn_chains   平均链数误差 |n_pred - n_truth|; 另给 mean(n_pred)-mean(n_truth);
  - overlap     被 >=2 条预测链同时占用的径迹比例 (排他性; 划分型解码器应 ~0)。

用法:
  python eval_seqrec_offline.py --version 654 --events 300 --thr 0.9 \
      --cut 1.0 --metric mix --lam 1.0 --out report_figs
"""
import argparse
import os
import sys
import time

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
from wmpgnn.reconstruction.reco_helper import (lca_truth_matrix, get_truth_part_keys,  # noqa: E402
                                              get_truth_part_ids, reconstruct_decay,
                                              get_final_keys, particle_name)
from wmpgnn.reconstruction.sequential_recombine import (sequential_recombine,  # noqa: E402
                                                        metric_distance, geom_edge_features,
                                                        combine_geom)


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------
def iou(a, b):
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def greedy_match(truth, pred, thr=0.5):
    """按 IoU>=thr 贪心二部匹配 (确定性: (iou, i, j) 字典序)。返回 (matches, pairs)。"""
    pairs = []
    for ti, ts in enumerate(truth):
        for pi, ps in enumerate(pred):
            v = iou(ts, ps)
            if v >= thr:
                pairs.append((v, ti, pi))
    pairs.sort(key=lambda x: (-x[0], x[1], x[2]))
    used_t, used_p, matches = set(), set(), []
    for v, ti, pi in pairs:
        if ti in used_t or pi in used_p:
            continue
        used_t.add(ti); used_p.add(pi); matches.append((ti, pi))
    return matches


def exact_recovered(truth, pred):
    pred_sets = {frozenset(s) for s in pred}
    return sum(1 for ts in truth if frozenset(ts) in pred_sets)


def overlap_rate(pred, n_tracks):
    """被 >=2 条预测链同时占用的径迹比例 (分母 = 出现在 >=1 条链里的径迹数)。"""
    cnt = np.zeros(n_tracks, dtype=int)
    for s in pred:
        for x in s:
            cnt[x] += 1
    occupied = cnt > 0
    denom = int(occupied.sum())
    if denom == 0:
        return 0.0, 0
    return float((cnt > 1).sum()) / denom, denom


def make_chains_from_labels(labels, min_size=2):
    out = []
    for l in np.unique(labels[labels >= 0]):
        m = np.nonzero(labels == l)[0]
        if m.size >= min_size:
            out.append(set(int(x) for x in m.tolist()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_files/eval_bench_v3_n90.yaml")
    ap.add_argument("--version", type=int, default=654)
    ap.add_argument("--events", type=int, default=300)
    ap.add_argument("--thr", type=float, default=None, help="node/edge 剪枝阈值 (缺省取配置)")
    ap.add_argument("--cuts", default="0.02,0.05,0.1",
                    help="learned / mix 度量的 cut 扫描 (距离); 候选边 p>thr 故 d<-ln(thr)")
    ap.add_argument("--cuts_geom", default="0.0005,0.001,0.002,0.005",
                    help="geom 度量的 cut 扫描 (DOCA/100 的尺度)")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--linkage", default="nearest", choices=["nearest", "centroid"])
    ap.add_argument("--nclusters", type=int, default=None, help="若给定, 不按 cut 而固定缩并到 N 簇")
    ap.add_argument("--seqrec_input", default="all", choices=["lca", "all"],
                    help="seqrec 的输入边集: lca=与 CC 完全同一集合(LCA>0); all=全部剪枝候选边")
    ap.add_argument("--out", default="report_figs")
    ap.add_argument("--tag", default="seqrec_offline")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    cfg["settings"]["model"] = a.version
    cfg = adjust_config_evaluation(cfg)
    thr = a.thr if a.thr is not None else float(cfg["inference"].get("edge_prune_thr", 0.9))
    node_thr = thr
    edge_thr = thr
    cuts = [float(x) for x in a.cuts.split(",")]
    cuts_geom = [float(x) for x in a.cuts_geom.split(",")]
    print(f"[seqrec-offline] version={a.version} thr={thr} input={a.seqrec_input} "
          f"cuts={cuts} cuts_geom={cuts_geom} lambda={a.lam} linkage={a.linkage}")

    module = load_module(cfg, transform_pos_weight(None, None, mode="eval"))
    module.eval()
    dev = next(module.parameters()).device
    nc = getattr(module, "_nc", {}) or {}
    ns = getattr(module, "_ns", {}) or {}
    _, _, ckl = load_tst_loader(cfg)

    # 每个 (decoder, cut) 的逐事件记录
    plan = ["cc"] + [f"learned@{c}" for c in cuts] + [f"mix@{c}" for c in cuts] \
        + [f"geom@{c}" for c in cuts_geom]
    rows = {name: dict(n_truth=0, n_pred=0, full=0, match_p=0, match_t=0,
                       n_evt=0, dn=0, occ=0, occ_denom=0, dn_b=0, n_evt_b=0)
            for name in plan}
    n_evt = 0
    t0 = time.time()
    for batch in ckl.test_dataloader():
        if n_evt >= a.events:
            break
        batch = batch.to(dev)
        ei = batch[("tracks", "to", "tracks")].edge_index.detach().cpu()
        tb = batch["tracks"].batch.detach().cpu()
        ft = batch["tracks"].ft.detach().cpu().numpy().astype(int)
        if module.use_pid == "true":
            batch["tracks"].x = torch.cat([batch["tracks"].x, batch["tracks"].pid], dim=1)
        with torch.no_grad():
            out = module.model(batch)
        blk = module.model._blocks[-1]
        nw = blk.node_weights["tracks"].detach().cpu().squeeze(-1).numpy()
        ew = blk.edge_weights[("tracks", "to", "tracks")].detach().cpu().squeeze(-1).numpy()
        lca_logits = out[("tracks", "to", "tracks")].edges.detach().cpu().numpy()   # [E,4]
        x_all = batch["tracks"].x.detach().cpu()                                     # [N_all, d]
        graphs = [g.cpu() for g in out.to_data_list()]

        for gid in range(int(tb.max().item()) + 1 if tb.numel() else 0):
            if n_evt >= a.events:
                break
            n_evt += 1
            tm = (tb == gid)
            em = tm[ei[0]] & tm[ei[1]]
            gt = tm.nonzero()[:, 0]
            g = graphs[gid]
            n_loc = int(tm.sum())
            g2l = {int(x): i for i, x in enumerate(gt.tolist())}
            ii = np.array([g2l[int(x)] for x in ei[0][em].tolist()], dtype=int)
            jj = np.array([g2l[int(x)] for x in ei[1][em].tolist()], dtype=int)
            sc = ew[em.numpy()]
            arg = lca_logits[em.numpy()].argmax(-1)
            nloc_w = nw[gt.numpy()]
            x_loc = x_all[gt]

            # ---- 候选边: 两端点存活 + 边分 > thr (与管线剪枝一致) ----
            surv = (nloc_w[ii] > node_thr) & (nloc_w[jj] > node_thr) & (sc > edge_thr)
            e_all = np.stack([ii[surv], jj[surv]], 1) if surv.any() else np.zeros((0, 2), int)
            p_all = sc[surv]
            # 管线 CC 只用模型判 LCA>0 的边
            cand_lca = surv & (arg > 0)
            e_lca = np.stack([ii[cand_lca], jj[cand_lca]], 1) if cand_lca.any() \
                else np.zeros((0, 2), int)
            p_lca = sc[cand_lca]

            def _dgeom(edges):
                if len(edges) == 0:
                    return np.zeros(0)
                return combine_geom(geom_edge_features(x_loc, edges, nc, ns)).numpy()

            dgeom_all = _dgeom(e_all)
            dgeom_lca = _dgeom(e_lca)

            # ---- 真值链 ----
            try:
                tl = lca_truth_matrix(g)
                tkeys = get_truth_part_keys(g).tolist()
                ids = list(map(particle_name, get_truth_part_ids(g).numpy()))
                tc, _, _ = reconstruct_decay(tl, tkeys, particle_ids=ids, truth_level_simulation=1)
            except Exception:
                tc = {}
            pk = get_final_keys(g).numpy().tolist()
            pk2i = {int(k): i for i, k in enumerate(pk)}
            truth = []
            for ck, cl in tc.items():
                nodes = [pk2i[int(k)] for k in cl["node_keys"] if int(k) in pk2i]
                if len(nodes) >= 2:
                    truth.append(set(nodes))

            # seqrec 输入边集 (可切换 lca / all)
            e_in, p_in, dg_in = (e_lca, p_lca, dgeom_lca) if a.seqrec_input == "lca" \
                else (e_all, p_all, dgeom_all)
            if len(e_in):
                d_learn = metric_distance(p_in, metric="learned").numpy()
                d_mix = metric_distance(p_in, metric="mix", lam=a.lam, d_geom=dg_in).numpy()
                d_geom = dg_in
            else:
                d_learn = d_mix = d_geom = np.zeros(0)

            specs = [("cc", e_lca, np.zeros(len(e_lca)), None)]
            specs += [(f"learned@{c}", e_in, d_learn, c) for c in cuts]
            specs += [(f"mix@{c}", e_in, d_mix, c) for c in cuts]
            specs += [(f"geom@{c}", e_in, d_geom, c) for c in cuts_geom]

            for name, e_, d_, cut_ in specs:
                lab = sequential_recombine(n_loc, e_, d_, cut=cut_,
                                           linkage=a.linkage, n_clusters=a.nclusters)
                pred = make_chains_from_labels(lab)
                r = rows[name]
                r["n_evt"] += 1
                r["n_truth"] += len(truth)
                r["n_pred"] += len(pred)
                r["full"] += exact_recovered(truth, pred)
                mt = greedy_match(truth, pred, 0.5)
                r["match_t"] += len(mt)
                r["match_p"] += len(mt)
                r["dn"] += abs(len(pred) - len(truth))
                if len(truth) > 0:                    # 只含 B 衰变的事件 (链数误差在其中才算得准)
                    r["dn_b"] += abs(len(pred) - len(truth))
                    r["n_evt_b"] += 1
                ov, denom = overlap_rate(pred, n_loc)
                r["occ"] += ov * denom
                r["occ_denom"] += denom
        if n_evt % 50 == 0:
            print(f"  ... {n_evt} events, {time.time()-t0:.1f}s", flush=True)

    # ---- 汇总 ----
    def summ(name):
        r = rows[name]
        fail = r["n_truth"] if r["n_truth"] else 1
        fpred = r["n_pred"] if r["n_pred"] else 1
        return dict(decoder=name, n_evt=r["n_evt"], n_truth=r["n_truth"], n_pred=r["n_pred"],
                    chain_full=round(100 * r["full"] / fail, 2),
                    chain_P=round(100 * r["match_p"] / fpred, 2),
                    chain_R=round(100 * r["match_t"] / fail, 2),
                    mean_chains_truth=round(r["n_truth"] / max(r["n_evt"], 1), 3),
                    mean_chains_pred=round(r["n_pred"] / max(r["n_evt"], 1), 3),
                    dn_chains=round(r["dn"] / max(r["n_evt"], 1), 3),
                    dn_chains_B=round(r["dn_b"] / max(r["n_evt_b"], 1), 3),
                    overlap=round(100 * r["occ"] / max(r["occ_denom"], 1), 3))

    df = pd.DataFrame([summ(n) for n in plan])
    os.makedirs(a.out, exist_ok=True)
    csv = f"{a.out}/{a.tag}_v{a.version}_thr{thr}_{a.seqrec_input}.csv"
    df.to_csv(csv, index=False)
    print("\n===== 方向3 离线评测 (连通分量 vs seqrec-learned/geom/mix) =====")
    print(f"    版本 v{a.version} | 事件 {n_evt} | 阈值 thr={thr} | seqrec输入={a.seqrec_input} "
          f"| linkage={a.linkage} | lambda={a.lam}")
    print("    (cc: 连通分量基线=cut=None; learned/mix/geom: 不同 cut 的 seqrec)")
    print(df.to_string(index=False))
    print(f"\n[已写出] {csv}  (耗时 {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
