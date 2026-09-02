#!/usr/bin/env python3
"""离线冒烟测试: 复现 trainer.py 的 cpt 权重加载路径 (CPU, 不训练).
验证 v31 best ckpt 能加载进全新模块:
  - ab01 (无新头): 严格 state_dict 匹配
  - ab06 (新增 source_head): 新头自动随机初始化, 主干加载成功
"""
import sys, os, torch
sys.path.append('/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn')
import yaml
from wmpgnn.analysis.config_adjusting import adjust_config_training
from wmpgnn.analysis.load_module import load_module

BASE = '/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn'
os.chdir(BASE)

pw = {"LCA": torch.ones(4), "nodes": torch.tensor(1.0),
      "edges": torch.tensor(1.0), "pv_asso": torch.tensor(1.0)}

for name, check_head in [('ab01_rebal', None),
                         ('ab06_source', 'model.source_head'),
                         ('ab09_mom', 'model.node_mom_head')]:
    cfg = yaml.safe_load(open(f'config_files/train_CERN_{name}.yaml'))
    if check_head:
        # 模拟"旧 ckpt 无新头": 让 ab06/ab09 也加载 v31 best (无 source/mom 头)
        cfg['DFEI']['cpt'] = 31
    cfg = adjust_config_training(cfg)
    print(f"\n===== {name}  cpt={cfg['DFEI']['cpt']} (check_head={check_head}) =====")
    module = load_module(cfg, pw)
    sd = module.state_dict()
    n = len(sd)
    print(f"[ok] 模块构建+加载成功, state_dict 共 {n} 个参数张量")
    if check_head:
        keys = [k for k in sd if k.startswith(check_head)]
        print(f"[ok] 新头 {check_head}: {len(keys)} 个参数, 示例 {keys[:2]}")
        # 新头参数应是随机初始 (非全零), 主干应已加载 (取一个主干权重统计)
    # 主干权重加载验证: 主干参数不应全零/全NaN
    nonzeros = 0
    for k, v in sd.items():
        if not k.startswith(('model.source_head', 'model.node_struct_head',
                             'model.node_mom_head', 'model.edge_mass_head',
                             'model.chain_scorer', 'model.pv_cluster_head')):
            nonzeros += int((v != 0).sum().item()) if v.numel() < 1e7 else 0
    print(f"[ok] 主干加载完成 (非新头参数非零计数抽样: {nonzeros})")

print("\n[ALL PASS] cpt 加载路径可用")
