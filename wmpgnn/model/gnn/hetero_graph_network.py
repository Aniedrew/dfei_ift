import pytorch_lightning as pl

import torch
from torch.nn import Sigmoid

from wmpgnn.model.blocks.hetero_edge_block import HeteroEdgeBlock
from wmpgnn.model.blocks.hetero_global_block import HeteroGlobalBlock
from wmpgnn.model.blocks.hetero_node_block import HeteroNodeBlock
from wmpgnn.model.mlp_class import create_mlp
from wmpgnn.util.pruners import *


class HeteroGraphNetwork(pl.LightningModule):
    def __init__(self, config, node_types, edge_types, FT_layer=False):
        super().__init__()
        self.edge_types = edge_types
        self.node_types = node_types
        self.FT = FT_layer
        self._use_globals = config["use_globals"]
        self._use_node_weights = config["use_node_weights"]
        self._use_edge_weights = config["use_edge_weights"]
        self._weighted_pass = False
        if any([config["use_node_weights"], config["use_edge_weights"]]) and config["weighted_pass"]:
            self._weighted_pass = config["weighted_pass"]

        # ==== 不对称 latent 维度扩展 ====
        # MLP_forward_dim: {type_key: dim} 覆盖对应类型 MLP 的输出维度 (默认保持 MLP_forward 最后一维),
        # 例: {"tracks": 32, "tracks_tracks": 24} -> 节点 32 维 / tt 边 24 维, 其余 16。
        # 依据: 物理自由度分析——节点需承载 ~12-14 自由度 + 9 头竞争, 16 贴下限; 边 ~7-9。
        self._mlp_forward = config["MLP_forward"]
        dim_override = config.get("MLP_forward_dim", {})
        def _fw(type_key):
            if type_key in dim_override:
                fw = dict(self._mlp_forward)
                layers = list(fw.get("layers", []))
                layers[-1] = dim_override[type_key]
                fw["layers"] = layers
                return fw
            return self._mlp_forward
        edge_configs = {et: _fw(f"{et[0]}_{et[2]}") for et in edge_types}
        node_configs = {nt: _fw(nt) for nt in node_types}

        # Edge, Node, Global block
        self._edge_block = HeteroEdgeBlock(config["MLP_forward"], edge_types, configs_per_type=edge_configs)
        self._node_block = HeteroNodeBlock(config["MLP_forward"], node_types, edge_types,
                                           configs_per_type=node_configs)
        if self._use_globals:
            self._global_block = HeteroGlobalBlock(config["MLP_forward"], node_types, edge_types,
                                                   weighted_mp=self._weighted_pass)
        # Inference layers
        self._node_mlps = {}
        self._edge_mlps = {}
        if config["use_node_weights"]:
            for edge_type in edge_types:
                self._edge_mlps[edge_type] = create_mlp(config["MLP_infer"])

        if config["use_edge_weights"]:
            self._node_mlps['tracks'] = create_mlp(config["MLP_infer"])

        if self.FT:
            self._node_mlps['ft'] = create_mlp(config["MLP_infer"], outdim=3)

        self._edge_models_model_dict = torch.nn.ModuleDict({str(i): j for i, j in self._edge_mlps.items()})
        self._node_models_model_dict = torch.nn.ModuleDict({str(i): j for i, j in self._node_mlps.items()})

        self._sigmoid = Sigmoid()
        # nodes
        self.node_weights = {}
        self.node_logits = {}
        # edges
        self.edge_weights = {}
        self.edge_logits = {}

        # Pruning cuts for evaluate
        self.edge_prune = False
        self.node_prune = False
        self.prune_by_cut = False
        self.k_edges = 20
        self.k_nodes = 70
        self.edge_weight_cut = 0.001
        self.node_weight_cut = 0.001

        # ==== B2: 可微剪枝训练 (软掩码模拟剪枝, 消除 train-inference gap) ====
        # 训练时对 node/edge weight 施加可微软掩码 mask = σ((w - cut) / τ),
        # 让消息传递在"被剪的图"上进行, 梯度经掩码流回主干; τ 由 lightning module 退火。
        # 推理时不启用 (edge_prune/node_prune 硬剪枝在 reconstruct 阶段做)。
        # 注意: 只在**最后一个 GN block** 启用 (与推理剪枝作用于最终输出权重的位置一致),
        #       前面 block 正常学习全图表征。DFEI_HGNN.forward 通过 _b2_active 控制。
        self._b2 = bool(config.get("b2", False))
        self._b2_cut = float(config.get("b2_cut", 0.5))
        self._b2_tau = float(config.get("b2_tau_start", 1.0))  # 当前温度, 由外部按 epoch 更新
        self._b2_active = False  # 是否为本图网络的最后一个 block (由外层 forward 设置)

        self.edge_indices = {}
        self.node_indices = {}
        self.edge_node_pruning_indices = {}

    def _b2_mask(self, w):
        """B2 可微软掩码: mask = σ((w - cut) / τ), 返回与 w 同形的连续掩码 [0,1]。"""
        tau = max(float(self._b2_tau), 1e-3)
        return torch.sigmoid((w - self._b2_cut) / tau)

    def forward(self, graph, pid_nodes):
        # Applying edge update
        node_input = self._edge_block(graph)

        # Infer edges
        for edge_type in self.edge_types:
            if self._use_edge_weights:
                graph_batch = node_input[edge_type[0]].batch[node_input[edge_type].edge_index[0]]
                self.edge_logits[edge_type] = self._edge_mlps[edge_type](node_input[edge_type].edges, graph_batch)
                self.edge_weights[edge_type] = self._sigmoid(self.edge_logits[edge_type])
            else:
                self.edge_weights[edge_type] = torch.ones((graph[edge_type].edges.shape[0], 1)).to(self.device)

        # ==== B2: 训练时对边权重施加软掩码, 模拟剪枝后的图 (消息传递在软剪枝图上进行) ====
        # 仅在最后一个 GN block 启用 (与推理时剪枝作用于最终输出权重的位置对齐)
        if self._b2 and self._b2_active and self.training:
            for edge_type in self.edge_types:
                self.edge_weights[edge_type] = self.edge_weights[edge_type] * self._b2_mask(self.edge_weights[edge_type])

        if self.edge_prune:
            for edge_type in self.edge_types:
                if edge_type == ('tracks', 'to', 'tracks'):
                    mask = self.edge_weights[edge_type] > self.edge_weight_cut
                    edge_indices = torch.nonzero(mask, as_tuple=True)[0]
                    self.edge_indices[edge_type] = edge_indices
                    self.edge_weights[edge_type] = self.edge_weights[edge_type][edge_indices, :]
                    edge_pruning(edge_indices, node_input, edge_type)

        # Node update
        global_input = self._node_block(node_input, self.edge_weights)

        # Node infer
        for node_type in self.node_types:
            if self._use_node_weights and node_type != "pvs":
                self.node_logits[node_type] = self._node_mlps[node_type](global_input[node_type].x,
                                                                         global_input[node_type].batch)
                self.node_weights[node_type] = self._sigmoid(self.node_logits[node_type])
            else:
                self.node_weights[node_type] = torch.ones((graph[node_type].x.shape[0], 1)).to(self.device)

        # ==== B2: 节点权重软掩码 (全局聚合前, 与边掩码同理, 模拟剪枝后的节点集) ====
        if self._b2 and self._b2_active and self.training:
            for node_type in self.node_types:
                if node_type != "pvs" and self._use_node_weights:
                    self.node_weights[node_type] = self.node_weights[node_type] * self._b2_mask(self.node_weights[node_type])

        if self.FT:
            # self.node_logits["frag"] = self._node_mlps["frag"](global_input["tracks"].x, global_input["tracks"].batch)
            # self.node_weights["frag"] = self._sigmoid(self.node_logits["frag"])
            # FT, catting pid information before pass, as well as the nodes itself
            combined_graph = torch.cat([global_input["tracks"].x, pid_nodes], dim=1)
            combined_graph = torch.cat([combined_graph, self.node_weights['tracks']], dim=1)
            self.node_logits["ft"] = self._node_mlps["ft"](combined_graph, global_input["tracks"].batch)
            self.node_weights["ft"] = torch.softmax(self.node_logits["ft"], dim=1)

        if self.node_prune:
            for node_type in self.node_types:
                if node_type == "tracks":
                    mask = self.node_weights[node_type] > self.node_weight_cut
                    node_indices = torch.nonzero(mask, as_tuple=True)[0]
                    self.node_indices[node_type] = node_indices
                    edge_index = faster_node_pruning(mask, global_input, node_type,
                                                     [('tracks', 'to', 'tracks')],
                                                     device=self.device)
                    self.edge_node_pruning_indices[node_type] = edge_index
                    for key in edge_index.keys():
                        self.edge_weights[key] = self.edge_weights[key][edge_index[key]]

        # Global update
        if self._use_globals:
            return self._global_block(global_input, self.edge_weights, self.node_weights)
        else:
            return global_input
