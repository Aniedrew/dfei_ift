import pytorch_lightning as pl

import torch
from torch.nn import Sigmoid

from wmpgnn.model.blocks.hetero_edge_block import HeteroEdgeBlock
from wmpgnn.model.blocks.hetero_global_block import HeteroGlobalBlock
from wmpgnn.model.blocks.hetero_node_block import HeteroNodeBlock
from wmpgnn.model.mlp_class import create_mlp
from wmpgnn.model.context_prune import ContextPruneHead
from wmpgnn.model.line_graph_attn import LineGraphAttention
from wmpgnn.util.pruners import *


class HeteroGraphNetwork(pl.LightningModule):
    def __init__(self, config, node_types, edge_types, FT_layer=False, context_last=False):
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
        self.edge_dir_logits = {}
        self._dir_head_edges = set()
        # ==== [2026-09-23] 剪枝 MLP 的第二输出: 方向头 ====
        # MLP_infer_dir_head=true 时, tt 边的剪枝 MLP 多一个输出, 预测"sender 是否更靠上游"。
        # 只对 ('tracks','to','tracks') 生效; 其余边/节点 MLP 完全不变。
        self._dir_head_on = bool(config.get("MLP_infer_dir_head", False))
        if config["use_node_weights"]:
            for edge_type in edge_types:
                _dir = 1 if (self._dir_head_on and edge_type == ('tracks', 'to', 'tracks')) else 0
                self._edge_mlps[edge_type] = create_mlp(config["MLP_infer"], dir_outdim=_dir)
                if _dir:
                    self._dir_head_edges.add(edge_type)
                    print(f"[dir_head] 剪枝 MLP 追加方向输出: {edge_type} (与剪枝头共享 trunk)")

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

        # ==== 方案 A: 上下文感知剪枝头 (默认 off -> 零影响) ====
        # 挂在最后一个 GN block (与 B2/推理剪枝作用位置一致, 由外层 forward 置 _context_active)。
        self._context_head = None
        self._context_active = False
        cp_cfg = config.get("context_prune", {}) or {}
        if context_last and str(cp_cfg.get("mode", "off")) not in ("off", "local", ""):
            node_dim = node_configs["tracks"]["layers"][-1]
            tt_key = ('tracks', 'to', 'tracks')
            edge_dim = edge_configs[tt_key]["layers"][-1] if tt_key in edge_configs \
                else self._mlp_forward["layers"][-1]
            self._context_head = ContextPruneHead(node_dim, edge_dim, cp_cfg)

        # ==== [2026-09-23] 剪枝 MLP 的派生输入: 零初始化残差适配器 ====
        # 由 lightning module 从原始特征现算物理派生量 (pT/|p|/minIP/隔离度/ΔR/m(ππ)/…),
        # 挂到 graph 的 x_der / der_edges 上; 这里用 x' = x + W·der 注入。
        # W,b 初始化为 0 -> **起点与旧模型逐位等价**(不破坏已继承的剪枝头与主干),
        # 模型只在派生量确有增益时才学出非零映射。
        self._node_der_adapter = None
        self._edge_der_adapter = None
        _kn = int(config.get("extra_node_dim", 0))
        _ke = int(config.get("extra_edge_dim", 0))
        if _kn > 0:
            self._node_der_adapter = torch.nn.Linear(_kn, node_configs["tracks"]["layers"][-1])
            torch.nn.init.zeros_(self._node_der_adapter.weight)
            torch.nn.init.zeros_(self._node_der_adapter.bias)
            print(f"[der_input] 节点剪枝 MLP 追加 {_kn} 维派生输入 (零初始化适配器)")
        if _ke > 0:
            _ekey = ('tracks', 'to', 'tracks')
            _edim = edge_configs[_ekey]["layers"][-1] if _ekey in edge_configs \
                else self._mlp_forward["layers"][-1]
            self._edge_der_adapter = torch.nn.Linear(_ke, _edim)
            torch.nn.init.zeros_(self._edge_der_adapter.weight)
            torch.nn.init.zeros_(self._edge_der_adapter.bias)
            print(f"[der_input] 边剪枝 MLP 追加 {_ke} 维派生输入 (零初始化适配器)")

        # ==== [2026-09-24] 三个并行方向 (各自独立开关, 默认全关 -> 与旧模型逐位等价) ====
        # (1) pv_overlap_inject: 把 PV 关联头给出的"两条径迹是否同 PV"软重叠喂进 tt 剪枝 MLP 输入。
        #     直接打通"跨 PV 的边"这条通道 (剪枝此前完全看不到 PV 关联头的信息)。
        #     PV 头 logits 在同一 block 内晚于 tt 计算, 故用**前一个 block** 的 logits (延迟一层),
        #     经零初始化适配器加入 tt 边表征。要求边缘不变 (edge_prune=False)。
        # (2) event_bias: 事件级自适应偏置 —— 事件内节点表征均值 -> 一个标量, 加到最后一层的
        #     点/边剪枝 logits 上, 让"该事件有多挤"自行决定剪枝工作点 (端到端, 无标签)。
        # (3) event_count_head: 事件级计数辅助头 (预测该事件有几条真值链), 迫使全局表征编码事件结构。
        _ekey_tt = ('tracks', 'to', 'tracks')
        _edim = edge_configs[_ekey_tt]["layers"][-1] if _ekey_tt in edge_configs \
            else self._mlp_forward["layers"][-1]
        _ndim = node_configs["tracks"]["layers"][-1]

        self._pv_inject = bool(config.get("pv_overlap_inject", False))
        self._pv_ov_adapter = None
        self._pv_ov_cache = None      # 本 block 算出的 (递给下一个 block, 由外层 forward 转交)
        self._pv_ov_in = None         # 上一个 block 在同一次 forward 内算出的 (本 block 消耗)
        if self._pv_inject:
            self._pv_ov_adapter = torch.nn.Linear(3, _edim)
            torch.nn.init.zeros_(self._pv_ov_adapter.weight)
            torch.nn.init.zeros_(self._pv_ov_adapter.bias)
            print("[pv_inject] tt 剪枝 MLP 追加 PV 软重叠 3 维 (零初始化适配器, 用前一层 logits)")

        self._evt_bias_on = bool(config.get("event_bias", False))
        self._evt_bias = None
        if self._evt_bias_on:
            self._evt_bias = torch.nn.Linear(_ndim, 1)
            torch.nn.init.zeros_(self._evt_bias.weight)
            torch.nn.init.zeros_(self._evt_bias.bias)
            print("[event_bias] 事件级自适应剪枝偏置启用 (零初始化)")

        self._evt_count_on = bool(config.get("event_count_head", False))
        self._evt_count = None
        self._evt_count_logits = None
        self._last_node_emb = None
        if self._evt_count_on:
            self._evt_count = torch.nn.Linear(_ndim, 4)   # 0/1/2/>=3 条真值链
            print("[event_count] 事件级链数辅助头启用")

        # ==== [2026-09-26] tt 边图 (line-graph) 注意力: 让 tt 边之间互相传消息 ====
        # 动机: 三角传递性 —— 若边 (i,j) 与 (j,k) 都很强, 则 (i,k) 通常也应为强边 (同一条链)。
        # 与 pv_overlap_inject 同位置: 挂在**最后一个 GN block** (context_last),
        # 在边剪枝 MLP 打分之前对 tt 边做 line_graph_rounds 轮注意力; 输出经零初始化投影
        # 残差加回边表征 -> 未训练时严格恒等 (关闭开关时逐位一致)。
        self._line_attn = None
        self._line_active = False
        if context_last and bool(config.get("line_graph_attn", False)):
            # [2026-10-08 FIX S3] bias_cols 的越界与"隐性耦合"守卫。
            #   越界时偏置**永远用不上**却静默失效; 且 9/10/11 这三列的含义依赖 derived_vertex:
            #   一旦同时打开 derived_vertex, 列序前移 -> 指向 zcpa/flight/collin 而非
            #   doca/logdoca/|Δ起点|, 不报错但语义完全变了 (v651 恰好没开 derived_vertex 才对)。
            _bcols = [int(c) for c in (config.get("line_graph_bias_cols", []) or [])]
            _ke_bias = int(config.get("extra_edge_dim", 0))
            assert not _bcols or max(_bcols) < _ke_bias, (
                f"line_graph_bias_cols={_bcols} 越界: extra_edge_dim={_ke_bias} -> 偏置永远用不上 (静默失效)")
            if _bcols and bool(config.get("derived_vertex", False)) and max(_bcols) <= 11:
                print("[line_graph] WARN: bias_cols 含 9..11 且 derived_vertex=True -> "
                      "这几列实际是 zcpa/flight/collin, 不是 doca/logdoca/|Δ起点| (列索引被前移)")
            self._line_attn = LineGraphAttention(
                _edim,
                n_rounds=int(config.get("line_graph_rounds", 1)),
                n_heads=int(config.get("line_graph_heads", 4)),
                hidden=int(config.get("line_graph_hidden", 32)),
                max_neighbors=int(config.get("line_graph_max_neighbors", 32)),
                # [2026-10-05] 几何 pair-bias: 用 der_edges 的这几列 (如顶点几何 doca/|Δ起点|)
                #   导出每头的 attention logit 偏置; 空/未给 -> 不加偏置 (与 v637 逐位一致)
                bias_cols=list(config.get("line_graph_bias_cols", []) or []),
                bias_hidden=int(config.get("line_graph_bias_hidden", 32)),
            )

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
                _e_in = node_input[edge_type].edges
                _de = getattr(graph[edge_type], "der_edges", None)     # 派生输入 (可选)
                if self._edge_der_adapter is not None and _de is not None:
                    _e_in = _e_in + self._edge_der_adapter(_de)
                # (1) PV 软重叠注入: 用**上一个 block 在同一次 forward 内**算出的 (E_tt,3) 特征
                if (self._pv_ov_adapter is not None and edge_type == ('tracks', 'to', 'tracks')
                        and self._pv_ov_in is not None
                        and self._pv_ov_in.shape[0] == _e_in.shape[0]):
                    _e_in = _e_in + self._pv_ov_adapter(
                        self._pv_ov_in.to(dtype=_e_in.dtype, device=_e_in.device))
                # (4) [2026-09-26] tt 边图注意力: 共享端点的边之间传消息 (仅最后一个 block)。
                #     在边剪枝 MLP 打分之前精修边表征; 零初始化投影 -> 起始恒等。
                #     graph_batch = 每条边的事件 id (由 sender 节点取), 用于事件隔离。
                if (self._line_attn is not None and edge_type == ('tracks', 'to', 'tracks')
                        and getattr(self, "_line_active", False)):
                    # bias_x=_de: 几何 pair-bias 的输入列 (无 bias_cols 时被模块内部忽略)
                    _e_in = self._line_attn(_e_in, node_input[edge_type].edge_index, graph_batch,
                                            bias_x=_de)
                self.edge_logits[edge_type] = self._edge_mlps[edge_type](_e_in, graph_batch)
                self.edge_weights[edge_type] = self._sigmoid(self.edge_logits[edge_type])
                # 方向头 (剪枝 MLP 的第二输出), 只在最后一个 block 取值
                if edge_type in self._dir_head_edges and getattr(self, "_dir_active", False):
                    self.edge_dir_logits[edge_type] = self._edge_mlps[edge_type].dir_logits
            else:
                self.edge_weights[edge_type] = torch.ones((graph[edge_type].edges.shape[0], 1)).to(self.device)

        # (1) 缓存本 block 的 PV 关联软重叠, 供下一个 block 的 tt 剪枝使用
        if self._pv_ov_adapter is not None:
            _tpv = ('tracks', 'to', 'pvs')
            _tt = ('tracks', 'to', 'tracks')
            if _tpv in self.edge_logits and _tt in self.edge_logits:
                from wmpgnn.model.pv_overlap import pv_same_soft
                _n_tr = node_input['tracks'].x.shape[0]
                _b = node_input['tracks'].batch if 'batch' in node_input['tracks'] \
                    else torch.zeros(_n_tr, dtype=torch.long, device=node_input['tracks'].x.device)
                self._pv_ov_cache = pv_same_soft(self.edge_logits[_tpv],
                                                 node_input[_tpv].edge_index,
                                                 node_input[_tt].edge_index,
                                                 int(_n_tr), _b)

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
                _x_in = global_input[node_type].x
                _dx = getattr(graph[node_type], "x_der", None)         # 派生输入 (可选)
                if self._node_der_adapter is not None and _dx is not None:
                    _x_in = _x_in + self._node_der_adapter(_dx)
                self.node_logits[node_type] = self._node_mlps[node_type](_x_in,
                                                                         global_input[node_type].batch)
                self.node_weights[node_type] = self._sigmoid(self.node_logits[node_type])
            else:
                self.node_weights[node_type] = torch.ones((graph[node_type].x.shape[0], 1)).to(self.device)

        # ==== 方案 A: 上下文感知剪枝头 (仅最后一个 block; 用邻域点/边分数精修裁决) ====
        # γ 初始 0 -> 训练初期恒等; 精修后的 logits/weights 写回 block, 供 BCE 监督与推理剪枝使用。
        if (self._context_head is not None and getattr(self, "_context_active", False)
                and not (self.edge_prune or self.node_prune) and self._use_node_weights
                and ("tracks" in self.node_logits)):
            tt = ('tracks', 'to', 'tracks')
            if tt in self.edge_logits:
                s_node = self.node_logits["tracks"]
                s_edge = self.edge_logits[tt]
                s_node_new, s_edge_new = self._context_head(
                    global_input["tracks"].x, s_node,
                    node_input[tt].edge_index,
                    node_input[tt].edges, s_edge)
                self.node_logits["tracks"] = s_node_new
                self.node_weights["tracks"] = self._sigmoid(s_node_new)
                self.edge_logits[tt] = s_edge_new
                self.edge_weights[tt] = self._sigmoid(s_edge_new)

        # ==== [2026-09-24] (2) 事件级自适应剪枝偏置 + (3) 事件级链数辅助头 (仅最后一个 block) ====
        # 事件级表征 = 事件内 tracks 节点表征均值; 用它预测标量偏置/链数。
        # 偏置与计数头都是零初始化/无真值输入 -> 起点恒等, 且偏置端到端学不依赖真值。
        if (self._evt_bias is not None or self._evt_count is not None
                or getattr(self, "_stash_emb", False)) \
                and getattr(self, "_evt_active", False) \
                and ("tracks" in self.node_logits) and self._use_node_weights:
            _tt = ('tracks', 'to', 'tracks')
            _x = global_input["tracks"].x
            _b = global_input["tracks"].batch
            _n_ev = int(_b.max().item()) + 1
            _sum = torch.zeros(_n_ev, _x.shape[-1], device=_x.device, dtype=_x.dtype).index_add(0, _b, _x)
            _cnt = torch.bincount(_b, minlength=_n_ev).clamp_min(1).to(_x.dtype).unsqueeze(1)
            _evt = _sum / _cnt                                    # (n_ev, D) 事件级表征
            self._last_node_emb = _x                              # 供链级对比损失使用
            if self._evt_bias is not None:
                _bias = self._evt_bias(_evt)                      # (n_ev, 1)
                _nl = self.node_logits["tracks"]
                self.node_logits["tracks"] = _nl + _bias[_b].to(_nl.dtype)
                self.node_weights["tracks"] = self._sigmoid(self.node_logits["tracks"])
                if _tt in self.edge_logits:
                    _el = self.edge_logits[_tt]
                    _eb = _b[global_input[_tt].edge_index[0]]
                    self.edge_logits[_tt] = _el + _bias[_eb].to(_el.dtype)
                    self.edge_weights[_tt] = self._sigmoid(self.edge_logits[_tt])
            if self._evt_count is not None:
                self._evt_count_logits = self._evt_count(_evt)    # (n_ev, 4) 链数分类

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
