#!/usr/bin/env python3
"""生成"逐项消融链"配置 (ab01..ab09): 从 v31 基线出发, 按 PPT 顺序每次只加一个优化。

协议 (每步):
  - 权重: DFEI.cpt = 上一步版本号 (int -> get_bis_model 自动选 best) 加载, 全新微调
  - 20 epoch, lr 3e-5, 200 files, 与 v31-v48 同口径; 训练结束自动 thr0.9 评估 (trainer.py)
  - 每步写独立版本目录 (settings.log_version), 版本号从 500 起, 避免与现有作业冲突
  - 评估/剪枝阈值固定 0.9; 非 PPT 主题的 loss 再平衡 (node/lca weight 10) 作为 ab01 单独一步,
    之后每步都是单一 PPT 优化, before/after 归因干净

DFEI 架构模板 = version_31 的输入配置 (标准 16 维), 保证加载 v31 best 时 state_dict 严格匹配。
"""
import yaml
import copy
import os

BASE = '/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn'
OUT_DIR = os.path.join(BASE, 'config_files')
V31_CFG = os.path.join(BASE, 'LHCb_logs/DFEI/version_31/input_config.yaml')

# ---------- 从 v31 input_config 提取确切架构模板 ----------
with open(V31_CFG) as f:
    v31 = yaml.safe_load(f)
ARCH = copy.deepcopy(v31['DFEI'])          # 标准 16 维架构 (与 v31 训练的完全一致)
ARCH.pop('cpt', None)                       # cpt 由每步生成时指定

# ---------- 公共 settings (v31 架构 + 微调协议) ----------
COMMON_SETTINGS = {
    'data_dir': "/lzufs/user/guoqingxiang/DFEI_IFT_20260702/data/MC_normed",
    'sample': ["inclusive_00342442"],
    'nfiles': [200],
    'precision': 32,
    'batch_size': 8,
    'lr': 3e-5,          # 微调 lr (同 v38)
    'weight_decay': 1e-5,
    'epochs': 20,        # 每步 20 epoch
    'gacc': 2,
    'ngpu': 1,
    'ncpu': 2,
    'graph_mode': "None",
    'node_sel': "true",
    'pv_model': "None",
    'calibration': False,
}

# ---------- 公共 inference 底 (同 v31; thr 固定 0.9) ----------
COMMON_INF = {
    'LCA': True,
    'LCA_weights': True,
    'node_prune': True,
    'node_prune_weights': True,
    'edge_prune': True,
    'edge_prune_weights': True,
    'pv_asso': True,
    'pv_asso_weights': True,
    'edge_prune_thr': 0.9,
    'node_prune_thr': 0.9,
}

# ---------- 步骤定义 (PPT 顺序) ----------
# name / log_version / cpt(上一步) / over_write / inference 增量 / GNblocks 增量 / 说明
STEPS = [
    dict(name='ab01_rebal', ver=500, cpt=31, tag='ab01_rebal',
         note='loss 再平衡 (node/lca weight 10) — v36 配方, 非 PPT 主题, 单独隔离',
         inf_add={'node_prune_weight': 10.0, 'lca_weight': 10.0}, gn_add={}),
    dict(name='ab02_b2', ver=501, cpt=500, tag='ab02_b2',
         note='Part1: B2 可微剪枝+软掩码 (cut 0.85)。每步独立微调, 故 τ 恒为终点 sharpness 0.1',
         inf_add={},
         gn_add={'b2': True, 'b2_cut': 0.85, 'b2_k': 0,
                 'b2_tau_start': 0.1, 'b2_tau_end': 0.1, 'b2_tau_epochs': 1}),
    dict(name='ab03_cl2w', ver=502, cpt=501, tag='ab03_cl2w',
         note='Part2: class2 (同母) 加权 2.0',
         inf_add={'lca_class2_weight': 2.0}, gn_add={}),
    dict(name='ab04_hinge', ver=503, cpt=502, tag='ab04_hinge',
         note='Part2: 链内 LCA 一致性 hinge (保持置信, margin 0.3)',
         inf_add={'chain_lca_loss': True, 'chain_lca_loss_weight': 2.0,
                  'chain_lca_margin': 0.3, 'chain_lca_filter': False}, gn_add={}),
    dict(name='ab05_ce', ver=504, cpt=503, tag='ab05_ce',
         note='Part2: 链内边"类别正确" CE (y>0 结构边直接监督)',
         inf_add={'chain_lca_ce': True, 'chain_lca_ce_weight': 1.0}, gn_add={}),
    dict(name='ab06_source', ver=505, cpt=504, tag='ab06_source',
         note='Part3: 源检测头 (Rumor Centrality 找根训练化)',
         inf_add={'source_head': True, 'source_loss_weight': 5.0}, gn_add={}),
    dict(name='ab07_mass', ver=506, cpt=505, tag='ab07_mass',
         note='Part3: 边级 mass head (log10 m_ππ)',
         inf_add={'mass_head': True, 'mass_loss_weight': 1.0}, gn_add={}),
    dict(name='ab08_struct', ver=507, cpt=506, tag='ab08_struct',
         note='Part3: 节点 struct head (depth+RC), 低权重 0.3 (v48 教训)',
         inf_add={'struct_head': True, 'struct_head_weight': 0.3}, gn_add={}),
    dict(name='ab09_mom', ver=508, cpt=507, tag='ab09_mom',
         note='Part3: 节点 mom head (归一化动量), 低权重 0.2 — 终点=全优化模型',
         inf_add={'mom_head': True, 'mom_loss_weight': 0.2}, gn_add={}),
]


def build(step):
    s = copy.deepcopy(COMMON_SETTINGS)
    s['log_version'] = step['ver']
    inf = copy.deepcopy(COMMON_INF)
    inf.update(step['inf_add'])
    arch = copy.deepcopy(ARCH)
    arch['cpt'] = step['cpt']
    if step['gn_add']:
        arch['GNblocks'].update(step['gn_add'])
    cfg = {
        'settings': s,
        'inference': inf,
        'evaluate': {'sample': ['inclusive_00342442'], 'nfiles': [20],
                     'over_write': step['tag']},
        'DFEI': arch,
    }
    return cfg


def main():
    for step in STEPS:
        cfg = build(step)
        path = os.path.join(OUT_DIR, f"train_CERN_{step['name']}.yaml")
        with open(path, 'w') as f:
            f.write(f"# DFEI 逐项消融链 — {step['ver']}: {step['note']}\n")
            yaml.safe_dump(cfg, f, default_flow_style=None, sort_keys=False)
        # 校验可解析 + 关键字段
        with open(path) as f:
            c2 = yaml.safe_load(f)
        assert c2['settings']['log_version'] == step['ver']
        assert c2['DFEI']['cpt'] == step['cpt']
        print(f"[ok] {path.split('/')[-1]}  log_version={step['ver']}  cpt={step['cpt']}")


if __name__ == '__main__':
    main()
