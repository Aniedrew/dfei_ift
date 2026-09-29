"""剪枝损失的两个测量 (服务"理解扔掉了哪些" + 判断注意力/上下文是否有空间):
  A. 假边构成: 每条被保留的 tt 边按两端真值身份分类, 量出模型分数对"真边 vs 各类假边"的可分度 (AUC)
     - fake_inter (两端都是真值径迹但属**不同**真值链) 是唯一"必须靠链级/上下文信息"才能分辨的类别
       -> 若它的 AUC 明显 > 0.5 说明信息已在逐边特征里 (改阈值/校准即可); 若 ~0.5 才说明需要上下文/注意力
  B. 剪枝效率 vs 物理量: 真值链的存活率按 pT / |η| / 链长 / 事件 PV 数 / 是否多 B 事件分箱 -> 服务 reweight
用法: python analyze_prune_loss.py --config config_files/eval_v601_inc51_0904.yaml --version 601 --events 300
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
from wmpgnn.reconstruction.reco_helper import (lca_truth_matrix, get_truth_part_keys,  # noqa: E402
                                              get_truth_part_ids, reconstruct_decay, particle_name)

NEW = "/lzufs/user/guoqingxiang/DFEI_IFT_20260904/dfei_repo/preprocessing/normalization_dict.pt"
OLD = "/lzufs/user/guoqingxiang/DFEI_IFT_20260702/dfei_repo/preprocessing/old_norm_ported.pt"


def ap_score(score, label):
    """average precision (正类率极低时比 AUC 有意义)"""
    o = np.argsort(-score); l = label[o]
    tp = np.cumsum(l); prec = tp / np.arange(1, len(l) + 1)
    return float((prec * l).sum() / max(1, l.sum()))


def prec_at_recall(score, label, r):
    o = np.argsort(-score); l = label[o]
    tp = np.cumsum(l); rec = tp / max(1, l.sum())
    idx = np.searchsorted(rec, r)
    return float(tp[min(idx, len(tp) - 1)] / (min(idx, len(tp) - 1) + 1))


# ==== [2026-09-29] v3: 分层池 + 事件级 bootstrap ====
HARD_CLASSES = ("fake_intra", "fake_inter")     # 两端都是真值径迹的假边 = 难池


def _boot_ci(score, label, groups, fn, n_boot=200, seed=0):
    """事件级 bootstrap 的 ±1σ 区间。

    边/点**不是独立样本**: 同一事件内的边高度相关 (共享 PV / 径迹)。按边做 bootstrap 会
    严重低估方差, 所以必须**按事件重采样**。返回 (lo, hi) = 16%/84% 分位。
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    if len(uniq) < 3:
        return np.nan, np.nan
    idx_by_g = {g: np.nonzero(groups == g)[0] for g in uniq}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_g[g] for g in pick])
        v = fn(score[idx], label[idx])
        if np.isfinite(v):
            vals.append(v)
    if len(vals) < 10:
        return np.nan, np.nan
    return float(np.percentile(vals, 16)), float(np.percentile(vals, 84))


def auc(score, label):
    o = np.argsort(score); r = np.empty(len(o), float); r[o] = np.arange(1, len(o) + 1)
    p = int(label.sum()); n = len(label) - p
    return (r[label == 1].sum() - p * (p + 1) / 2) / (p * n) if p and n else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--version", type=int, default=601)
    ap.add_argument("--events", type=int, default=300)
    ap.add_argument("--tag", default="v601_0904")
    ap.add_argument("--norm", default="new", choices=["new", "old"])
    ap.add_argument("--thr", type=float, default=0.9)
    ap.add_argument("--out", default="report_figs")
    # [2026-09-29 v3] 汇总表文件名 (v3 列与 v2 不同, 不能追加到同一个 csv)
    ap.add_argument("--csv_name", default="bench_auc_v3.csv")
    ap.add_argument("--n_boot", type=int, default=200, help="事件级 bootstrap 次数 (0=关闭)")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    cfg["settings"]["model"] = a.version
    cfg = adjust_config_evaluation(cfg)
    module = load_module(cfg, transform_pos_weight(None, None, mode="eval"))
    module.eval()
    dev = next(module.parameters()).device
    _, _, ckl = load_tst_loader(cfg)
    nd = torch.load(NEW if a.norm == "new" else OLD, map_location="cpu")
    C = {k: float(nd["center"][k + "_reco"]) for k in ("px", "py", "pz")}
    S = {k: float(nd["scale"][k + "_reco"]) for k in ("px", "py", "pz")}

    e_rows, c_rows = [], []
    nsc_acc, nlab_acc = [], []      # [benchmark] 节点头累积
    lca_s_acc, lca_y_acc, frag_rows = [], [], []   # [benchmark] 非剪枝轴: LCA 头 + 链碎裂度
    pv_rows = []                    # [v3] PV 关联组: 逐 (事件, 径迹) 的关联对错
    n_evt = 0
    for batch in ckl.test_dataloader():
        batch = batch.to(dev)
        ei = batch[("tracks", "to", "tracks")].edge_index.detach().cpu()
        tb = batch["tracks"].batch.detach().cpu()
        raw_x = batch["tracks"].x.detach().cpu().clone()
        ft = batch["tracks"].ft.detach().cpu().numpy().astype(int)
        y_tt = batch[("tracks", "to", "tracks")].y.detach().cpu().numpy().reshape(-1)
        # [2026-09-24] benchmark: 真值 PV 关联 (用于把 fake_inter 拆成 同PV / 跨PV 两个子类)
        ei_tp = batch[("tracks", "to", "pvs")].edge_index.detach().cpu()
        pb = batch["pvs"].batch.detach().cpu()
        try:
            y_tp = batch[("tracks", "to", "pvs")].y.detach().cpu().numpy().reshape(-1)
        except Exception:
            y_tp = None
        # [v3] tr-pv 候选边掩码 (与 reconstruction.py 的 edge_filter 同源) + minIP 基线特征
        ef_all = batch[("tracks", "pvs")].filter.detach().cpu().numpy().reshape(-1) == 1
        minip_all = batch[("tracks", "to", "pvs")].edges.detach().cpu().numpy().reshape(-1)
        if module.use_pid == "true":
            batch["tracks"].x = torch.cat([batch["tracks"].x, batch["tracks"].pid], dim=1)
        with torch.no_grad():
            out = module.model(batch)
        blk = module.model._blocks[-1]
        nw = blk.node_weights["tracks"].detach().cpu().squeeze(-1).numpy()
        ew = blk.edge_weights[("tracks", "to", "tracks")].detach().cpu().squeeze(-1).numpy()
        try:    # [v3] PV 关联头分数 (tr-pv 边)
            ew_pv = blk.edge_weights[("tracks", "to", "pvs")].detach().cpu().squeeze(-1).numpy()
        except Exception:
            ew_pv = None
        out[("tracks", "to", "tracks")].lca = out[("tracks", "to", "tracks")].edges
        graphs = out.to_data_list()
        for gid in range(int(tb.max().item()) + 1 if tb.numel() else 0):
            if n_evt >= a.events:
                break
            n_evt += 1
            tm = (tb == gid)
            em = tm[ei[0]] & tm[ei[1]]
            gt = tm.nonzero()[:, 0].numpy()
            g = graphs[gid]
            npv = int(g["pvs"].num_nodes)
            # 真值链 (未剪枝图) -> 每径迹的簇 id
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
            chains = []
            for ci, (ck, cl) in enumerate(tc.items()):
                nodes = [pk2i[int(k)] for k in cl["node_keys"] if int(k) in pk2i]
                if len(nodes) < 2:
                    continue
                cl_id[nodes] = ci
                s = set(nodes)
                eix = [i for i in range(int(em.sum())) if False]   # 占位
                chains.append(nodes)
            # 本事件 tt 边 -> 全部转成"事件内局部索引"
            ei_g = ei[:, em.numpy()].numpy()
            g2l = {int(x): i for i, x in enumerate(gt)}
            ei_e = np.array([[g2l[int(x)] for x in ei_g[0]], [g2l[int(x)] for x in ei_g[1]]], dtype=int)
            ys = (y_tt[em.numpy()] > 0).astype(int)
            scs = ew[em.numpy()]
            is_sig = (ft[gt] != 1).astype(int)
            # [v3] 节点存活 (与剪枝同一 thr): 决定"工作点 population"(两端点都活下来的边)
            ns_surv = nw[gt] > a.thr
            multi_b = int(len(chains) > 1)
            # 真值 PV (局部索引): 取该径迹所有 tr-pv 边中 y 最大的 PV, 无则 -1
            truth_pv = np.full(len(gt), -1, dtype=int)
            if y_tp is not None and len(y_tp):
                m_tp = (tb[ei_tp[0]].numpy() == gid) & (pb[ei_tp[1]].numpy() == gid)
                for kt in np.nonzero(m_tp)[0]:
                    if y_tp[kt] > 0:
                        ti = g2l.get(int(ei_tp[0][kt]))
                        if ti is not None:
                            truth_pv[ti] = int(ei_tp[1][kt])
            # [benchmark] 节点头: 累积 (分数, 是否真值径迹, 事件是否多 B)
            nsc_acc.append(nw[gt]); nlab_acc.append(is_sig)
            for k in range(ei_e.shape[1]):
                i, j = int(ei_e[0, k]), int(ei_e[1, k])
                si, sj = is_sig[i], is_sig[j]
                ci_, cj_ = cl_id[i], cl_id[j]
                sub = ""
                if ys[k] == 1:
                    cls = "true"
                elif si and sj and ci_ >= 0 and ci_ == cj_:
                    cls = "fake_intra"          # 同链但非真值结构边 (深度错)
                elif si and sj and (ci_ < 0 or cj_ < 0 or ci_ != cj_):
                    cls = "fake_inter"          # 两端都是真值径迹但不同链 -> 需上下文
                    if truth_pv[i] < 0 or truth_pv[j] < 0:
                        sub = "pv_unk"
                    else:
                        sub = "same_pv" if truth_pv[i] == truth_pv[j] else "diff_pv"
                elif si or sj:
                    cls = "fake_sigbkg"
                else:
                    cls = "fake_bkgbkg"
                e_rows.append(dict(evt=n_evt, cls=cls, sub=sub, multi_b=multi_b,
                                   dp=int(bool(ns_surv[i]) and bool(ns_surv[j])), score=float(scs[k])))
            # ==== [2026-09-26] 非剪枝轴: (a) LCA 头判别力; (b) 真值链在剪枝图上的碎裂度 ====
            # 动机: v631 剪枝曲线最差却 All_fix 最高(24.46) -> All_fix 主要由链装配/解码决定,
            #       必须把"非剪枝头"也纳入 benchmark, 否则无法解释端到端指标。
            _ns = ns_surv                 # 存活节点 (与剪枝用同一 thr)
            _es = scs > a.thr             # 存活边
            try:
                _ll = out[("tracks", "to", "tracks")].edges[em.numpy()]
                if _ll.dim() == 2 and _ll.shape[1] >= 2:
                    lca_s_acc.append((_ll[:, 1] - _ll[:, 0]).numpy())   # 类1 - 类0
                    lca_y_acc.append(ys)
            except Exception:
                pass
            _par = list(range(len(gt)))

            def _find(x, _par=_par):
                while _par[x] != x:
                    _par[x] = _par[_par[x]]
                    x = _par[x]
                return x
            for k in range(ei_e.shape[1]):
                if not _es[k]:
                    continue
                i, j = int(ei_e[0, k]), int(ei_e[1, k])
                if _ns[i] and _ns[j]:
                    ra, rb = _find(i), _find(j)
                    if ra != rb:
                        _par[ra] = rb
            for c in np.unique(cl_id[cl_id >= 0]):
                mem = np.nonzero(cl_id == c)[0]
                surv = mem[_ns[mem]]
                comps = len({_find(int(x)) for x in surv})
                frag_rows.append(dict(evt=n_evt, n_tracks=len(mem), n_surv=len(surv), n_comp=comps,
                                      full=int(len(surv) == len(mem)), single=int(comps == 1)))
            # ==== [v3] PV 关联组: 逐径迹的关联对错 ====
            # 口径同 reconstruction.py: 候选边掩码 filter 全通过才判该 track; pred = 候选边里
            # 分数最大者对应的 PV; true = 标签为 1 的那条边的 PV。这是与剪枝**不同类**的量,
            # 方向相反 (越低越好), 单独成组, 绝不并入剪枝分。
            if ew_pv is not None and y_tp is not None and len(y_tp):
                m_tp = (tb[ei_tp[0]].numpy() == gid) & (pb[ei_tp[1]].numpy() == gid)
                if m_tp.any():
                    t_gl = ei_tp[0].numpy()[m_tp]
                    pv_gl = ei_tp[1].numpy()[m_tp]
                    y_tp_e = y_tp[m_tp]
                    sc_pv = ew_pv[m_tp]
                    ef_e = ef_all[m_tp]
                    for t in np.unique(t_gl):
                        sel = np.nonzero(t_gl == t)[0]
                        if not ef_e[sel].all():
                            continue                       # 有候选边被 filter 剔除 -> 不判
                        pos = sel[y_tp_e[sel] > 0]
                        if pos.size == 0:
                            continue                       # 无真值 PV (ghost 等) -> 不判
                        pred = int(pv_gl[sel][np.argmax(sc_pv[sel])])
                        pv_rows.append(dict(evt=n_evt, sig=int(ft[int(t)] != 1),
                                            miss=int(pred != int(pv_gl[pos[0]]))))
            # 事件内数组 (按局部索引)
            xg = raw_x[gt]; nwg = nw[gt]; sigg = is_sig
            # 链级物理量
            for nodes in chains:
                nn = np.array(nodes)
                px = xg[nn, 0].numpy() * S["px"] + C["px"]
                py = xg[nn, 1].numpy() * S["py"] + C["py"]
                pz = xg[nn, 2].numpy() * S["pz"] + C["pz"]
                pt = np.hypot(px, py)
                px_t, py_t, pz_t = px.sum(), py.sum(), pz.sum()
                pt_tot = float(np.hypot(px_t, py_t))
                pm_tot = float(np.sqrt(px_t ** 2 + py_t ** 2 + pz_t ** 2)) + 1e-9
                eta = float(np.arctanh(np.clip(pz_t / pm_tot, -.999, .999)))
                # 生存判定: 该链的真值结构边 (两端都在簇内且 y>0) 与所有节点是否都被保留
                n_ok = bool((nwg[nn] > a.thr).all())
                # 真值结构边: 局部索引里两端都在本簇内且 y>0
                s_set = set(nodes)
                eids = [k for k in range(ei_e.shape[1])
                        if int(ei_e[0, k]) in s_set and int(ei_e[1, k]) in s_set and ys[k] == 1]
                e_ok = bool(all(scs[k] > a.thr for k in eids)) if eids else True
                # [v3] AND 语义的连续版: 该链"最弱环节"的分数 (全部节点 + 全部真值结构边取 min)。
                # 一个环节掉下去整链就断 -> 链的"余量"应该由 min 描述, 而不是逐边平均。
                _mins = [float(nwg[nn].min())] + [float(scs[k]) for k in eids]
                chain_min_score = float(min(_mins)) if _mins else np.nan
                c_rows.append(dict(evt=n_evt, n_daughters=len(nodes), sum_pt=float(pt.sum()),
                                   chain_pt=pt_tot, chain_abs_eta=abs(eta), npvs=npv,
                                   n_chains_in_evt=len(chains), multi_b=int(len(chains) > 1),
                                   n_true_edges=len(eids), node_surv=int(n_ok), edge_surv=int(e_ok),
                                   chain_min_score=chain_min_score, surv=int(n_ok and e_ok)))
        if n_evt >= a.events:
            break

    e = pd.DataFrame(e_rows); ch = pd.DataFrame(c_rows)
    os.makedirs(a.out, exist_ok=True)
    e.to_csv(f"{a.out}/prune_loss_edges_{a.tag}.csv", index=False)
    ch.to_csv(f"{a.out}/prune_loss_chains_{a.tag}.csv", index=False)

    print(f"\n=== {a.tag}: {n_evt} 事件 | tt 边 {len(e)} | 真值链 {len(ch)} (thr={a.thr})")
    print(f"  链存活: 全部 {100*ch.surv.mean():.1f}%  只卡点 {100*ch.node_surv.mean():.1f}%  只卡边 {100*ch.edge_surv.mean():.1f}%")
    print(f"\n--- 边头整体判别力 (真实正类率 {100*(e.cls=='true').mean():.2f}%) ---")
    lab_all = (e.cls == "true").astype(int).values
    sc_all = e.score.values
    print(f"  AUC={auc(sc_all, lab_all):.4f}  AP={ap_score(sc_all, lab_all):.4f}  "
          f"precision@recall90={prec_at_recall(sc_all, lab_all, 0.90):.4f}  "
          f"@recall99={prec_at_recall(sc_all, lab_all, 0.99):.4f}")
    print("\n--- 边的构成与可分辨度 (AUC: 真边 vs 该类假边, 0.5=完全不可分) ---")
    tr = e[e.cls == "true"].score.values
    print(f"{'类别':>14}{'边数':>8}{'占比%':>8}{'score中位':>10}{'@thr0.9保留%':>13}{'AUC vs真边':>12}")
    for cls in ["true", "fake_intra", "fake_inter", "fake_sigbkg", "fake_bkgbkg"]:
        s = e[e.cls == cls].score.values
        if len(s) == 0:
            continue
        keep = 100 * (s > a.thr).mean()
        if cls == "true":
            print(f"{cls:>14}{len(s):>8}{100*len(s)/len(e):>8.1f}{np.median(s):>10.4f}{keep:>13.1f}{'-':>12}")
        else:
            lab = np.r_[np.ones(len(tr)), np.zeros(len(s))]
            au = auc(np.r_[tr, s], lab)
            print(f"{cls:>14}{len(s):>8}{100*len(s)/len(e):>8.1f}{np.median(s):>10.4f}{keep:>13.1f}{au:>12.4f}")
    # ==== [2026-09-24] pruning AUC benchmark (阈值无关, 供跨版本/跨方案比较) ====
    nsc = np.concatenate(nsc_acc) if nsc_acc else np.zeros(0)
    nlab = np.concatenate(nlab_acc) if nlab_acc else np.zeros(0, dtype=int)
    print("\n--- 节点头 (真值径迹 vs 背景/ghost) ---")
    if len(nsc):
        print(f"  AUC={auc(nsc, nlab):.4f}  AP={ap_score(nsc, nlab):.4f}  "
              f"precision@r90={prec_at_recall(nsc, nlab, 0.90):.4f}  @r99={prec_at_recall(nsc, nlab, 0.99):.4f}"
              f"  (正类率 {100*nlab.mean():.2f}%)")
    fi = e[e.cls == "fake_inter"]
    print("\n--- fake_inter 子类 (最该被抬高的那一类) ---")
    print(f"{'子类':>10}{'边数':>8}{'':>8}{'score中位':>10}{'@thr保留%':>13}{'AUC vs真边':>12}")
    for sb in ["same_pv", "diff_pv", "pv_unk"]:
        s = fi[fi["sub"] == sb].score.values
        if len(s) == 0:
            continue
        print(f"{sb:>10}{len(s):>8}{'':>8}{np.median(s):>10.4f}{100*(s>a.thr).mean():>13.1f}"
              f"{auc(np.r_[tr, s], np.r_[np.ones(len(tr)), np.zeros(len(s))]):>12.4f}")
    print("\n--- 按事件类型拆 (单 B vs 多 B) ---")
    for mb, nm in [(0, "1-B"), (1, "multi-B")]:
        se = e[e.multi_b == mb]
        if len(se) == 0 or (se.cls == "true").sum() == 0:
            continue
        print(f"{nm:>8}: 边 AUC={auc(se.score.values, (se.cls=='true').astype(int).values):.4f} "
              f"AP={ap_score(se.score.values, (se.cls=='true').astype(int).values):.4f} "
              f"(边数 {len(se)}, 真边率 {100*(se.cls=='true').mean():.2f}%)")

    # ==================== [v3] 分层池 / 工作点池 / 链级 AND / PV 组 ====================
    # 动机(2026-09-29): 全局 edge AUC 里 99.7% 是"背景-背景"这类闭眼可分的负例 -> AUC≈常数,
    # 被易例撑起。必须把评价挪到 (a) 难池 (b) 推理真正起作用的工作点池 (c) 链级的 AND 语义。
    ev = e.evt.values
    tr_mask = (e.cls == "true").values
    hard_mask = tr_mask | e.cls.isin(HARD_CLASSES).values
    dp_mask = e.dp.values == 1
    res = {}
    hs, hl = e.score.values[hard_mask], tr_mask[hard_mask].astype(int)
    res["ap_hardpool"] = ap_score(hs, hl)
    res["p_at_r90_hardpool"] = prec_at_recall(hs, hl, 0.90)
    res["hardpool_pos_rate"] = 100.0 * float(hl.mean()) if len(hl) else np.nan
    res["hardpool_n"] = int(len(hs))
    ds, dl = e.score.values[dp_mask], tr_mask[dp_mask].astype(int)
    _dp_ok = bool(dl.sum() and dl.sum() < len(dl))
    res["ap_dp"] = ap_score(ds, dl) if _dp_ok else np.nan
    res["dp_pos_rate"] = 100.0 * float(dl.mean()) if len(dl) else np.nan
    res["dp_n"] = int(len(ds))
    for cls in ("fake_intra", "fake_inter", "fake_sigbkg", "fake_bkgbkg"):
        s = e[e.cls == cls].score.values
        res[f"ap_{cls}"] = (ap_score(np.r_[tr, s], np.r_[np.ones(len(tr)), np.zeros(len(s))])
                            if len(s) and len(tr) else np.nan)
        res[f"n_{cls}"] = int(len(s))
    if a.n_boot > 0:
        res["ap_hardpool_lo"], res["ap_hardpool_hi"] = _boot_ci(hs, hl, ev[hard_mask], ap_score,
                                                               n_boot=a.n_boot, seed=0)
        if _dp_ok:
            res["ap_dp_lo"], res["ap_dp_hi"] = _boot_ci(ds, dl, ev[dp_mask], ap_score,
                                                       n_boot=a.n_boot, seed=1)
        else:
            res["ap_dp_lo"] = res["ap_dp_hi"] = np.nan
    print("\n===== [v3] 分层池 / 工作点池 (主判据) =====")
    print(f"  难池  (真边 vs fake_intra+inter): 边数 {res['hardpool_n']} (真边率 {res['hardpool_pos_rate']:.2f}%)")
    print(f"        AP={res['ap_hardpool']:.4f} "
          f"[{res.get('ap_hardpool_lo', np.nan):.4f}, {res.get('ap_hardpool_hi', np.nan):.4f}]"
          f"   p@r90={res['p_at_r90_hardpool']:.4f}")
    print(f"  工作点池 (两端点均过 thr={a.thr}): 边数 {res['dp_n']} (真边率 {res['dp_pos_rate']:.2f}%)")
    print(f"        AP={res['ap_dp']:.4f} "
          f"[{res.get('ap_dp_lo', np.nan):.4f}, {res.get('ap_dp_hi', np.nan):.4f}]")
    print("  分类别 AP (正例池=全部真边): " + "  ".join(
        f"{c.replace('fake_', '')}={res['ap_' + c]:.4f}(n={res['n_' + c]})"
        for c in ("fake_intra", "fake_inter", "fake_sigbkg", "fake_bkgbkg")))

    # ---- 链层: AND 语义 (按链长分箱 + min 分数 + 反解单环节存活率) ----
    chs = {}
    for lo_, hi_, nm in ((0, 3, "2"), (3, 4, "3"), (4, 5, "4"), (5, 1e9, "5p")):
        m = (ch.n_daughters >= lo_) & (ch.n_daughters < hi_)
        S = 100.0 * float(ch.surv[m].mean()) if m.any() else np.nan
        chs[f"chain_surv_{nm}"] = S
        # 若单环节存活率 p 均匀, 长 n 链存活 = p^n -> 反解 p = S^(1/n)。跨链长若 p 一致,
        # 说明"长链更差"纯粹是长度效应; 若 p 随 n 下降, 才是模型对长链真的更弱。
        chs[f"per_link_p_{nm}"] = float((S / 100.0) ** (1.0 / min(lo_ + 2, 5))) if np.isfinite(S) else np.nan
    _ms = ch.chain_min_score.dropna().values if "chain_min_score" in ch else np.zeros(0)
    chs["chain_minscore_p10"] = float(np.percentile(_ms, 10)) if len(_ms) else np.nan
    chs["chain_minscore_p50"] = float(np.percentile(_ms, 50)) if len(_ms) else np.nan
    print("\n===== [v3] 链层 AND 语义 (一失毁全链) =====")
    print("  按链长存活率: " + "  ".join(f"n={nm}:{chs['chain_surv_' + nm]:.1f}%" for nm in ("2", "3", "4", "5p")))
    print("  反解单环节存活率 p: " + "  ".join(f"n={nm}:{chs['per_link_p_' + nm]:.3f}" for nm in ("2", "3", "4", "5p"))
          + "   (各 n 一致=纯长度效应; 随 n 下降=长链真的更弱)")
    print(f"  链 min-score 分位: p10={chs['chain_minscore_p10']:.4f}  p50={chs['chain_minscore_p50']:.4f}"
          f"  (AND 余量, 越接近 thr={a.thr} 越好)")

    # ---- PV 关联组 (独立, 方向相反: 越低越好) ----
    pvs = {}
    pvd = pd.DataFrame(pv_rows)
    if len(pvd):
        pvs["pv_miss_all"] = 100.0 * float(pvd.miss.mean())
        pvs["pv_miss_sig"] = 100.0 * float(pvd[pvd.sig == 1].miss.mean()) if (pvd.sig == 1).any() else np.nan
        pvs["pv_miss_bkg"] = 100.0 * float(pvd[pvd.sig == 0].miss.mean()) if (pvd.sig == 0).any() else np.nan
        pvs["pv_n_tracks"] = int(len(pvd))
        lo, hi = _boot_ci(pvd.miss.values.astype(float), np.ones(len(pvd)), pvd.evt.values,
                          lambda s, l: float(s.mean()), n_boot=max(a.n_boot, 0))
        pvs["pv_miss_all_lo"] = 100.0 * lo if np.isfinite(lo) else np.nan
        pvs["pv_miss_all_hi"] = 100.0 * hi if np.isfinite(hi) else np.nan
        print("\n===== [v3] PV 关联组 (per-track 错误率, **越低越好**, 不与剪枝分合并) =====")
        print(f"  径迹数 {pvs['pv_n_tracks']}: miss_all={pvs['pv_miss_all']:.2f}% "
              f"[{pvs['pv_miss_all_lo']:.2f}, {pvs['pv_miss_all_hi']:.2f}]"
              f"  sig={pvs['pv_miss_sig']:.2f}%  bkg={pvs['pv_miss_bkg']:.2f}%")
    else:
        print("\n===== [v3] PV 关联组: 无样本 (pv_asso 未开或该数据无 tr-pv 标签) =====")


    def _auc_of(s):
        return auc(np.r_[tr, s], np.r_[np.ones(len(tr)), np.zeros(len(s))]) if len(s) and len(tr) else np.nan
    print("\n--- 非剪枝轴 (链装配 / LCA 头) ---")
    if lca_s_acc:
        _ls, _ly = np.concatenate(lca_s_acc), np.concatenate(lca_y_acc)
        print(f"  LCA 头 (类1-类0, 二分类口径): AUC={auc(_ls, _ly):.4f} AP={ap_score(_ls, _ly):.4f}")
    if frag_rows:
        fr = pd.DataFrame(frag_rows)
        print(f"  真值链 {len(fr)} 条: 全存活 {100*fr.full.mean():.1f}% | 单连通分量 {100*fr.single.mean():.1f}%"
              f" | 平均碎裂分量数 {fr.n_comp.mean():.2f} | 平均存活占比 {100*(fr.n_surv/fr.n_tracks).mean():.1f}%")
        print(f"  存活链里 单分量 占比: {100*fr[fr.n_surv>0].single.mean():.1f}%"
              f" | 平均分量数(存活链) {fr[fr.n_surv>0].n_comp.mean():.2f}")

    row = dict(tag=a.tag, version=a.version, events=n_evt,
               pos_rate=round(100 * float((e.cls == "true").mean()), 3),
               edge_auc=round(auc(sc_all, lab_all), 4), edge_ap=round(ap_score(sc_all, lab_all), 4),
               edge_p_at_r90=round(prec_at_recall(sc_all, lab_all, 0.90), 4),
               edge_p_at_r99=round(prec_at_recall(sc_all, lab_all, 0.99), 4),
               node_auc=round(auc(nsc, nlab), 4) if len(nsc) else np.nan,
               node_ap=round(ap_score(nsc, nlab), 4) if len(nsc) else np.nan,
               auc_fake_inter=round(_auc_of(fi.score.values), 4),
               auc_fake_inter_samepv=round(_auc_of(fi[fi["sub"] == "same_pv"].score.values), 4),
               auc_fake_inter_diffpv=round(_auc_of(fi[fi["sub"] == "diff_pv"].score.values), 4),
               auc_fake_sigbkg=round(_auc_of(e[e.cls == "fake_sigbkg"].score.values), 4),
               auc_fake_bkgbkg=round(_auc_of(e[e.cls == "fake_bkgbkg"].score.values), 4),
               chain_surv=round(100 * float(ch.surv.mean()), 2), thr=a.thr,
               lca_auc=round(auc(np.concatenate(lca_s_acc), np.concatenate(lca_y_acc)), 4) if lca_s_acc else np.nan,
               lca_ap=round(ap_score(np.concatenate(lca_s_acc), np.concatenate(lca_y_acc)), 4) if lca_s_acc else np.nan,
               chain_full_surv=round(100 * float(np.mean([r["full"] for r in frag_rows])), 2) if frag_rows else np.nan,
               chain_single_comp=round(100 * float(np.mean([r["single"] for r in frag_rows])), 2) if frag_rows else np.nan,
               mean_comp=round(float(np.mean([r["n_comp"] for r in frag_rows])), 3) if frag_rows else np.nan)
    # [v3] 追加分层池 / 工作点池 / 链级 AND / PV 组
    for _k, _v in {**res, **chs, **pvs}.items():
        row[_k] = round(float(_v), 4) if isinstance(_v, (float, np.floating)) else _v
    bf = f"{a.out}/{a.csv_name}"
    pd.DataFrame([row]).to_csv(bf, mode="a", header=not os.path.exists(bf), index=False)
    print(f"\n[benchmark] 汇总行已追加 -> {bf}")

    print("\n--- 剪枝效率 vs 物理量 (存活率 %, 括号内为该箱链数) ---")
    for col, bins in [("chain_pt", [0, 5, 10, 20, 30, 50, 1e9]), ("sum_pt", [0, 10, 20, 40, 80, 1e9]),
                      ("chain_abs_eta", [0, 2, 3, 3.5, 4, 1e9]), ("n_daughters", [0, 3, 4, 5, 6, 1e9]),
                      ("npvs", [0, 4, 6, 8, 10, 1e9]), ("n_chains_in_evt", [0, 1, 2, 1e9])]:
        if col not in ch:
            continue
        b = pd.cut(ch[col], bins)
        g = ch.groupby(b, observed=True)
        txt = "  ".join(f"{int(mid.left if hasattr(mid,'left') else 0)}-{int(mid.right if hasattr(mid,'right') else 0)}:"
                        f"{100*v:.0f}%({n})" for mid, v, n in zip(g.size().index, g.surv.mean(), g.size()))
        print(f"  {col:>16}: {txt}")


if __name__ == "__main__":
    main()
