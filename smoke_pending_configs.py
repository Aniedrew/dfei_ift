#!/usr/bin/env python3
"""CPU 冒烟: 把排队中的训练配置逐个建模型, 验证"配置 -> 模型"接线与维度断言。

动机 (2026-10-08): 集群好卡被占满, 5 个训练作业只能排在 gpu08 上空转; 一次 GPU 训练要
3-4 天, 如果配置本身有接线错误 (比如 extra_edge_dim 与派生特征维数不一致、line_graph 的
bias 列越界), 那就是白等几天。这里在**前台 CPU** 上把它们先建一遍:
  - derived_pair_sym (v645/v647) 是唯一从未跑过的派生特征路径;
  - line_graph_bias_cols 只在 v649 上跑过 (v649 在训练中 = 该路径已端到端验证);
  - 建模型会触发 dfei_lightning_module 里的 assert (派生输入维数 vs GNblocks 配置),
    以及 LineGraphAttention 的构造打印 (rounds/heads/hidden/bias_cols)。

用法: python3 smoke_pending_configs.py [v645 v647 ...]   (默认全部排队配置)
"""
import os
import sys

import torch
import yaml

BASE = "/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn"
sys.path.append(BASE)
os.chdir(BASE)

from wmpgnn.analysis.config_adjusting import adjust_config_training   # noqa: E402
from wmpgnn.analysis.load_module import load_module                    # noqa: E402

CFGS = ["v645", "v647", "v649", "v650", "v651"]

PW = {"LCA": torch.ones(4), "nodes": torch.tensor(1.0),
      "edges": torch.tensor(1.0), "pv_asso": torch.tensor(1.0)}

FAIL = 0
for v in (sys.argv[1:] or CFGS):
    path = f"config_files/train_CERN_0904_{v}.yaml"
    print("=" * 70)
    print(f"[smoke] {v}")
    try:
        cfg = adjust_config_training(yaml.safe_load(open(path)))
        mod = load_module(cfg, PW)
        sd = mod.state_dict()
        lg = sorted({k.split("_line_attn.")[-1].split(".")[0] for k in sd if "_line_attn." in k})
        bias = [k for k in sd if "_line_attn.bias_mlp" in k]
        # GNblocks 里声明的派生输入维 (会与模块内部算出的维数做 assert)
        gn = cfg["DFEI"]["GNblocks"]
        print(f"[smoke]   extra_node_dim={gn.get('extra_node_dim')} extra_edge_dim={gn.get('extra_edge_dim')} "
              f"line_graph_attn={gn.get('line_graph_attn', False)} bias_cols={gn.get('line_graph_bias_cols')}")
        print(f"[smoke]   line_attn 子模块: {lg}")
        if gn.get("line_graph_attn"):
            assert "_line_attn" in lg or "q" in lg, "line_graph_attn=True 但模型里没有 _line_attn 参数!"
            if gn.get("line_graph_bias_cols"):
                assert bias, "指定了 line_graph_bias_cols 但 bias_mlp 参数不存在!"
                ncol = len(gn["line_graph_bias_cols"])
                hl = int(gn.get("line_graph_bias_hidden", 32))
                w0 = sd[[k for k in bias if k.endswith("weight")][0]]
                assert tuple(w0.shape) == (hl, 4 * ncol), f"bias_mlp 第一层 {tuple(w0.shape)} != ({hl},{4*ncol})"
                wl = sd[[k for k in bias if k.endswith("weight")][-1]]
                heads = int(gn.get("line_graph_heads", 4))
                assert tuple(wl.shape) == (heads, hl), f"bias_mlp 末层 {tuple(wl.shape)} != ({heads},{hl})"
                # 末层必须零初始化 -> 起始与"无 bias"逐位一致
                assert float(wl.abs().sum()) == 0.0, "bias_mlp 末层非零初始化 (会破坏单变量对照)!"
                print(f"[smoke]   bias_mlp OK: 输入 {4*ncol} 维 <- {ncol} 列, 末层零初始化 ✓")
            else:
                assert not bias, "未指定 bias_cols 却存在 bias_mlp!"
        print(f"[smoke] {v} OK ✓")
    except Exception as e:      # noqa: BLE001
        FAIL += 1
        print(f"[smoke] {v} 失败 ✗ : {type(e).__name__}: {e}")

print("=" * 70)
print(f"[smoke] 共 {len(sys.argv[1:] or CFGS)} 个配置, 失败 {FAIL} 个")
sys.exit(1 if FAIL else 0)
