#!/usr/bin/env python3
"""训练"链打分器" (候选链 -> 信号链概率), 供重建侧候选链过滤/排序。

数据管线复用 evaluate_chain_auc.py:
  - 正样本 = truth 链 (y>0 连通分量)
  - 负样本 = 剪枝图分量 (模型误判信号的结构) + 随机组合
  - 特征   = conf / struct_conf / struct_frac / class2/3占比 / edge_w / node_w / 链大小 / 链内边数

与 evaluate_chain_auc 的区别:
  1. 按事件拆分 train/val (同链特征不跨集泄漏, 检验真实泛化)
  2. 训练 LogisticRegression 并保存 (joblib), 供重建侧调用

用法:
  python3 train_chain_scorer.py --nevents 40 --out models/chain_scorer.joblib
"""
import sys, io, argparse
sys.path.append('/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn')

import yaml
import torch
import torch.nn.functional as F
import zstandard as zstd
import numpy as np
from torch_geometric.data import Batch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import joblib

from wmpgnn.analysis.config_adjusting import adjust_config_training
from wmpgnn.analysis.load_module import load_module
from wmpgnn.data_loader.weights_calculator import transform_pos_weight
from evaluate_chain_auc import (load_events, truth_chains, chain_features,
                                pruned_components)

torch.manual_seed(0)
np.random.seed(0)

DATA = '/lzufs/user/guoqingxiang/DFEI_IFT_20260702/data/MC_normed/inclusive_00342442/tst_data_00220000_00220999.pt.zst'
CKPT = '/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn/LHCb_logs/DFEI/version_47/checkpoints/best-epoch=113-val_combined_loss=35.677.ckpt'
FEAT_NAMES = ['conf', 'struct_conf', 'struct_frac', 'class2/3占比',
              '平均edge_w', '平均node_w', '链大小', '链内边数']


def collect_event_features(module, evt):
    """单事件: 提取正样本 (truth 链) 与负样本 (剪枝分量 + 随机组合) 特征."""
    batch = Batch.from_data_list([evt])
    with torch.no_grad():
        outputs = module.forward(batch)
    lca_probs = F.softmax(outputs[('tracks', 'to', 'tracks')].edges, dim=-1).cpu()
    block = module.model._blocks[-1]
    edge_w = block.edge_weights[('tracks', 'to', 'tracks')].squeeze(-1).cpu()
    node_w = block.node_weights['tracks'].squeeze(-1).cpu()
    tt_ei = batch[('tracks', 'to', 'tracks')].edge_index.cpu()
    y = batch[('tracks', 'to', 'tracks')].y.cpu()
    n_nodes = batch['tracks'].x.shape[0]

    chains = truth_chains(y, tt_ei, n_nodes)
    truth_sets = [set(c) for c in chains]
    Xp, Xn = [], []
    for c in chains:
        f = chain_features(c, lca_probs, edge_w, node_w, tt_ei)
        if f is not None:
            Xp.append(f)
    for comp in pruned_components(node_w, edge_w, tt_ei, n_nodes):
        if any(set(comp) == ts for ts in truth_sets):
            continue
        f = chain_features(comp, lca_probs, edge_w, node_w, tt_ei)
        if f is not None:
            Xn.append(f)
    return Xp, Xn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nevents', type=int, default=40, help='总事件数')
    ap.add_argument('--out', default='models/chain_scorer.joblib')
    args = ap.parse_args()

    with open('config_files/train_CERN_v38_masshead.yaml') as f:
        configs = yaml.safe_load(f)
    configs = adjust_config_training(configs)
    configs['DFEI']['cpt'] = CKPT
    pos_weights = transform_pos_weight(None, None, mode="eval")
    module = load_module(configs, pos_weights)
    module.eval()

    events = load_events(DATA, n=args.nevents)
    n_val = max(1, int(args.nevents * 0.25))   # 按事件拆 25% 做 val
    trn_evts, val_evts = events[n_val:], events[:n_val]
    print(f"[scorer] 训练事件 {len(trn_evts)} + 验证事件 {len(val_evts)}")

    Xtr, ytr, Xva, yva = [], [], [], []
    for evt in trn_evts:
        Xp, Xn = collect_event_features(module, evt)
        Xtr += Xp + Xn
        ytr += [1] * len(Xp) + [0] * len(Xn)
    for evt in val_evts:
        Xp, Xn = collect_event_features(module, evt)
        Xva += Xp + Xn
        yva += [1] * len(Xp) + [0] * len(Xn)
    Xtr, ytr = np.array(Xtr), np.array(ytr)
    Xva, yva = np.array(Xva), np.array(yva)
    print(f"[scorer] train: 正{int(ytr.sum())} 负{int((ytr == 0).sum())} | val: 正{int(yva.sum())} 负{int((yva == 0).sum())}")

    reg = LogisticRegression(max_iter=3000)
    reg.fit(Xtr, ytr)
    auc_tr = roc_auc_score(ytr, reg.predict_proba(Xtr)[:, 1])
    auc_va = roc_auc_score(yva, reg.predict_proba(Xva)[:, 1])
    print("\n========== 链打分器 (LogReg) ==========")
    print(f"  train AUC = {auc_tr:.4f}")
    print(f"  val   AUC = {auc_va:.4f}   ← 跨事件泛化")
    print("  特征系数:")
    for nm, c in zip(FEAT_NAMES, reg.coef_[0]):
        print(f"    {nm:<14}: {c:+.4f}")

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump({'model': reg, 'feat_names': FEAT_NAMES}, args.out)
    print(f"\n[scorer] 已保存 -> {args.out}")
    print(f"[scorer] 用法: scorer.predict_proba(features)[:,1] > thr 过滤候选链")


if __name__ == '__main__':
    main()
