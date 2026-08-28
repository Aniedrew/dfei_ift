import pytorch_lightning as L

from collections import defaultdict

import torch
import torch.nn as nn

from wmpgnn.lightning_module.lightning_helper import *
from wmpgnn.reconstruction.reconstruction import EventReconstruction
from wmpgnn.reconstruction.topk_selection import CandidateScorer, build_chain_samples
from wmpgnn.performance.plotter import *
from wmpgnn.performance.reco_accuracy import acc_four_class, obtain_reco_accuracy, acc_pv_asso
from wmpgnn.performance.plot_results import plot_sig_pv_missasso, plot_sig_b_system_pv_missasso


class DFEILightningModule(L.LightningModule):
    def __init__(self, model, optimizer_class, optimizer_params, configs, pos_weights):
        super().__init__()
        if "model" in configs["settings"]:
            self.version = configs["settings"]["model"]
        else:
            self.version = None
            self.save_hyperparameters({
                **configs,
                "pos_weights": make_loggable(pos_weights)
            })

        self.signal = "_".join(configs["evaluate"]["sample"])
        if configs["evaluate"]["over_write"] != "":
            self.signal += "__" + configs["evaluate"]["over_write"]

        self.configs = configs["inference"]
        self.model = model
        self.use_pid = configs["DFEI"]["use_pid"]  # str holding what to do with pid information for DFEI
        # GN blocks 配置 (B2 温度退火参数读取; 兼容无该段的旧配置)
        self.configs_gn = configs.get("DFEI", {}).get("GNblocks", {})

        # ==== 候选衰变链选择 MLP (第5个监督头, 与主干联合训练) ====
        # 启用条件: config inference 段 selection_mlp 非空 ("builtin" 或 ckpt 路径)
        # scorer 只注册为 model 子模块 (model.chain_scorer), 随 state_dict 保存/加载;
        # 本模块通过只读 property self.chain_scorer 代理访问, 避免重复注册。
        self.chain_loss_weight = float(self.configs.get("chain_select_loss_weight", 10.0))
        self.chain_select_on = bool(self.configs.get("selection_mlp", "")) and self.configs.get("selection_mlp", "") != "None"
        if self.chain_select_on:
            node_dim = int(self.configs.get("selection_mlp_node_dim", 1 + 16))   # CERN use_pid: tracks.x=16 (encoder 输出)
            edge_dim = int(self.configs.get("selection_mlp_edge_dim", 1 + 4 + 5))  # 1 + 4 LCA + 5 物理边特征
            model.chain_scorer = CandidateScorer(node_dim, edge_dim)
            self.chain_criterion = nn.BCEWithLogitsLoss()

        # ==== 第6个监督头: 源检测 (Rumor Centrality 训练化) ====
        # 监督 GNN 预测每条 truth 链的"根节点" (B 介子候选): 节点级二分类
        # 标签 = truth 链内 rumor centrality 最大的节点 (truth_chain_roots, 无噪声)
        # 作用: 让主干显式学"衰变链的根-叶结构", 服务 LCA 结构分类;
        #       与推理侧 RC (找根) 对齐: 训练时学找根, 推理时 RC 用根。
        self.source_head_on = bool(self.configs.get("source_head", False))
        if self.source_head_on:
            # 输入 = 节点特征 (tracks.x, 16) + node_weight (1) -> [17]
            src_node_dim = int(self.configs.get("source_head_node_dim", 16))
            model.source_head = nn.Sequential(
                nn.Linear(src_node_dim + 1, 64), nn.ReLU(),
                nn.Linear(64, 1),
            )
            self.source_criterion = nn.BCEWithLogitsLoss()
            self.source_loss_weight = float(self.configs.get("source_loss_weight", 5.0))

        # ==== 方案5: 链内 LCA 一致性辅助损失 (chain_lca_filter 训练化) ====
        # 鼓励 truth 链内边 (y>0) 的 LCA 预测"高置信" (被判类别 softmax 概率高),
        # 让模型主动产出物理自洽的链 (推理侧 chain_lca_filter 的判据前移到训练)。
        self.chain_lca_on = bool(self.configs.get("chain_lca_loss", False))
        self.chain_lca_loss_weight = float(self.configs.get("chain_lca_loss_weight", 2.0))
        self.chain_lca_margin = float(self.configs.get("chain_lca_margin", 0.3))
        # ==== 方案6: 链内边"类别正确"监督 (chain_lca 升级) ====
        # 在"高置信"基础上, 额外对链内边 (y>0, 仅 ~0.1% 的边) 施加 LCA 类别 CE,
        # 专门放大链内结构边 (class1/2/3) 的分类监督, 对抗 class0 绝对数量对主
        # LCA loss 的稀释; 链内边分类错误 (尤其 class2<->class1) 直接破坏链结构。
        self.chain_lca_ce = bool(self.configs.get("chain_lca_ce", False))
        self.chain_lca_ce_weight = float(self.configs.get("chain_lca_ce_weight", 1.0))

        # ==== 方案7b (v40): 可训练 PV 分簇 MLP 头 (pv_cluster_head) ====
        # 用户反馈: "cluster 本身就该是一个带训练的 MLP, 不然你要怎么 cluster; 温度退火也得改"。
        # 设计: pv_cluster_head = 独立分簇器, 输入 concat(tracks.x 原始8, pvs.x 3, trpv.edges 1)=12 维,
        #       输出 track->PV 归属 logit, BCE 监督 (与 pv_asso 同标签, 各自独立训练)。
        #       训练时用它的预测 + 温度退火 (Gumbel 噪声随 tau 退火: 高 tau 探索采样 ->
        #       低 tau 收敛 argmax) 给 track 分配 PV -> 切子图; 推理时 reconstruction 用
        #       同一头分簇 -> 训练/推理严格对齐。单 B 子图的处理路径 (GNN/loss/重建) 完全不变。
        self.pv_cluster_on = bool(self.configs.get("pv_cluster", False))
        if self.pv_cluster_on:
            pvclu_in = int(self.configs.get("pv_cluster_head_input_dim", 12))
            pvclu_hid = int(self.configs.get("pv_cluster_head_hidden", 64))
            model.pv_cluster_head = nn.Sequential(
                nn.Linear(pvclu_in, pvclu_hid), nn.ReLU(),
                nn.Linear(pvclu_hid, 1),
            )
            self.pv_cluster_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights["pv_asso"])
            self.pv_cluster_loss_weight = float(self.configs.get("pv_cluster_loss_weight", 1.0))
            # 推理侧 track->PV 分配来源: "cluster_head"(推荐, 与本头对齐) | "pred"(pv_asso) | "true"
            self.pv_cluster_assign = self.configs.get("pv_cluster_assign", "cluster_head")
            # 温度退火 (与 B2 tau 同哲学): 1.0(探索,Gumbel采样) -> 0.1(收敛 argmax)
            # ⚠️ 相对本次训练起点计 (续训时 current_epoch 从 v38 ep101 起, 不能用绝对 epoch)
            self.pv_tau_start = float(self.configs.get("pv_cluster_tau_start", 1.0))
            self.pv_tau_end = float(self.configs.get("pv_cluster_tau_end", 0.1))
            self.pv_tau_epochs = max(int(self.configs.get("pv_cluster_tau_epochs", 100)), 1)
            self.pv_tau = self.pv_tau_start
            # ==== v41 修复②: truth->cluster 课程式过渡 ====
            # alpha: 1.0(用 truth PV 分簇, 稳定热启动) -> 0.0(用 cluster 头分簇)。
            # 解决 v40 随机 cluster 头 + tau=0 直接扰动 v38 权重导致发散的问题;
            # 相对本次训练起点计, curriculum 周期内每个 track 以概率 alpha 用 truth 分配。
            self.pv_cluster_curriculum_epochs = max(
                int(self.configs.get("pv_cluster_curriculum_epochs", 30)), 1)
            self.pv_curriculum_alpha = 1.0
            # ==== v41 修复③: 子图数量上限 (训练提速, 0=不限) ====
            # batch 从 8 个全图 -> ~50 子图导致 ~2h/epoch; 训练侧随机保留 cap 个子图,
            # val/test 不受限 (val 每 epoch 一次, 开销可忽略)。
            self.pv_cluster_max_subgraphs = int(self.configs.get("pv_cluster_max_subgraphs", 0))
            # 本次训练起点计数 (续训时 current_epoch 是 v38 的绝对 epoch, 退火/课程需相对本 run)
            self._pv_run_epoch = 0
            self._test_pv_cluster_logits = None  # test 时存全图 cluster logits, 供重建分簇

        # ==== 第7个监督头: 边级不变质量回归 (输出侧物理监督) ====
        # 用 π-π 不变质量 (log10 尺度) 作为物理真值, 监督 tt 边表征携带"动量-夹角"物理信息。
        # 目标 m_ij 由 batch 轨迹动量 (px,py,pz 归一化) 反归一化后计算, 无需额外标签;
        # 输入 = decoder 边表征 (latent_edges, 16维, model 在 op_trafo 前保存)。
        # 纯辅助监督: 不改主任务端到端目标, 梯度经 head 流回主干。
        self.mass_head_on = bool(self.configs.get("mass_head", False))
        if self.mass_head_on:
            mass_edge_dim = int(self.configs.get("mass_head_edge_dim", 16))
            model.edge_mass_head = nn.Sequential(
                nn.Linear(mass_edge_dim, 64), nn.ReLU(),
                nn.Linear(64, 1),
            )
            self.mass_criterion = nn.SmoothL1Loss(beta=0.3)
            self.mass_loss_weight = float(self.configs.get("mass_loss_weight", 1.0))
            # 反归一化常数 (center, scale), 与 normalization_dict.pt (LHCb 通道) 逐位核对
            self._mass_norm = {
                "px": (torch.tensor(-4.1619), torch.tensor(470.8137)),
                "py": (torch.tensor(0.7674), torch.tensor(597.9097)),
                "pz": (torch.tensor(7117.4619), torch.tensor(10077.2412)),
            }
            self._m_pi = 139.570  # MeV

        # ==== 第8个监督头: 节点结构监督 (深度 + Rumor Centrality 回归) ====
        # 用户提议 (2026-08-26): source_head 只预测"根"(1 bit) 信息量低;
        # 升级为连续结构监督, 让主干学"节点在衰变树中的位置/层级":
        #   ① depth 主头: 节点到链根的拓扑距离 (归一化 [0,1])
        #   ② rc 辅头:   节点 logR 链内 min-max 归一化 [0,1]
        # 与 LCA 边分类形成全局一致性约束 (class1 边 depth 差1, class2 差0, class3 差>=2)。
        # 标签由 truth_chain_structure 计算 (纯 truth, 无模型噪声)。
        self.struct_head_on = bool(self.configs.get("struct_head", False))
        if self.struct_head_on:
            struct_in = int(self.configs.get("struct_head_node_dim", 16))
            model.node_struct_head = nn.Sequential(
                nn.Linear(struct_in + 1, 64), nn.ReLU(),
                nn.Linear(64, 2),   # [depth_pred, rc_pred]
            )
            self.struct_criterion = nn.SmoothL1Loss(beta=0.1)
            self.struct_head_weight = float(self.configs.get("struct_head_weight", 1.0))
            self.rc_head_weight = float(self.configs.get("rc_head_weight", 0.5))

        # ==== 第9个监督头: 节点级动量回归 (mom_head) ====
        # PhyIP 探针发现: encoder 的 graph_norm+ReLU 把输入动量信息打散,
        # 节点表征几乎不携带线性可读的动量 (线性/MLP 探针 R²≈0)。
        # 方案 A (用户确认): 像 mass head 对边表征那样, 监督节点表征输出
        # 归一化输入动量 [px_n, py_n, pz_n] (encoder 的直接输入), 强制节点
        # 表征线性携带运动学信息 -> 未来节点级物理任务 (如 trigger pT) 可用。
        self.mom_head_on = bool(self.configs.get("mom_head", False))
        if self.mom_head_on:
            mom_in = int(self.configs.get("mom_head_node_dim", 16))
            model.node_mom_head = nn.Sequential(
                nn.Linear(mom_in, 64), nn.ReLU(),
                nn.Linear(64, 3),   # [px_n, py_n, pz_n]
            )
            self.mom_criterion = nn.SmoothL1Loss(beta=0.3)
            self.mom_loss_weight = float(self.configs.get("mom_loss_weight", 1.0))
            # 哨兵检测用反归一化常数 (与 _mass_loss 一致)
            self._mom_norm = {
                "px": (torch.tensor(-4.1619), torch.tensor(470.8137)),
                "py": (torch.tensor(0.7674), torch.tensor(597.9097)),
                "pz": (torch.tensor(7117.4619), torch.tensor(10077.2412)),
            }

        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params
        # 续训时希望使用的初始学习率 (从 settings.lr 读取; None 表示沿用 checkpoint 中的 lr)
        self.resume_lr = configs.get("settings", {}).get("lr", None)

        # Loss functions + associated inference class for plotting
        if self.configs["LCA"]:
            lca_w = pos_weights["LCA"].clone().float()
            # ==== 方案4: class2 (同B边) 专项加权 ====
            # class2 决定链内父子结构, 准确率仅 ~36% 是 Perfect 的最大结构瓶颈
            # (class1 已 75%+ 保证连通, class2 是短板)。lca_class2_weight>1 放大其 loss。
            c2 = float(self.configs.get("lca_class2_weight", 1.0))
            if c2 != 1.0:
                lca_w[2] = lca_w[2] * c2
            self.lca_criterion = nn.CrossEntropyLoss(weight=lca_w)
        if self.configs["node_prune"]:
            self.node_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights["nodes"])
        if self.configs["edge_prune"]:
            self.edge_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights["edges"])
        if self.configs["pv_asso"]:
            self.pv_asso_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights["pv_asso"])

        self.trn_log, self.val_log = init_logs(configs)
        self.tst_log = init_logs(configs, mode="test")
        # init event reconstruction class
        self.evt_reco = EventReconstruction(configs)

        # Pruning threshold for reco
        self.edge_prune = configs["inference"]["edge_prune_thr"]
        self.node_prune = configs["inference"]["node_prune_thr"]

        self.log_dir = configs["log_dir"]

    @property
    def chain_scorer(self):
        """候选衰变链 scorer (只读代理到 model.chain_scorer; 未启用时为 None)。"""
        return getattr(self.model, "chain_scorer", None)

    def on_load_checkpoint(self, checkpoint):
        """兼容旧 checkpoint 续训: 无 chain_scorer/source_head 头时, 重置 optimizer/lr_scheduler 状态。

        新头使 optimizer 参数数增加 (如 CERN 297 -> 309), 旧 checkpoint 的
        optimizer_states 会因 param group 大小不匹配而 load 失败; 用新结构重建
        空状态, optimizer 从当前 lr 重新起步 (新头本无历史动量)。
        """
        new_heads = []
        if self.chain_scorer is not None:
            has_cs = any(k.startswith("model.chain_scorer.") for k in checkpoint.get("state_dict", {}))
            if not has_cs:
                new_heads.append("chain_scorer")
        if self.source_head_on:
            has_sh = any(k.startswith("model.source_head.") for k in checkpoint.get("state_dict", {}))
            if not has_sh:
                new_heads.append("source_head")
        if self.pv_cluster_on:
            has_pch = any(k.startswith("model.pv_cluster_head.") for k in checkpoint.get("state_dict", {}))
            if not has_pch:
                new_heads.append("pv_cluster_head")
        if self.mass_head_on:
            has_mh = any(k.startswith("model.edge_mass_head.") for k in checkpoint.get("state_dict", {}))
            if not has_mh:
                new_heads.append("edge_mass_head")
        if self.struct_head_on:
            has_sh2 = any(k.startswith("model.node_struct_head.") for k in checkpoint.get("state_dict", {}))
            if not has_sh2:
                new_heads.append("node_struct_head")
        if self.mom_head_on:
            has_mh2 = any(k.startswith("model.node_mom_head.") for k in checkpoint.get("state_dict", {}))
            if not has_mh2:
                new_heads.append("node_mom_head")
        if new_heads:
            print(f"[heads] 旧 checkpoint 无 {new_heads} 头: "
                  "重置 optimizer/lr_scheduler 状态 (新头无历史动量, 从当前 lr 重新起步)")
            opt = self.optimizer_class(self.model.parameters(), **self.optimizer_params)
            checkpoint["optimizer_states"] = [opt.state_dict()]
            sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="min", factor=0.5, patience=5, min_lr=1e-6)
            checkpoint["lr_schedulers"] = [sch.state_dict()]

    def load_state_dict(self, state_dict, strict=True):
        """兼容旧 checkpoint 续训: 旧模型无 chain_select/source_head 头
        (model.chain_scorer / model.source_head)。

        trainer.fit(ckpt_path=...) 与 load_from_checkpoint 最终都经 load_state_dict;
        当新模块启用了新头而 checkpoint 缺少其参数时, 允许缺失 (随机初始化),
        其余参数保持严格匹配。
        """
        if strict:
            missing = [k for k in self.state_dict() if k not in state_dict
                       and (k.startswith("model.chain_scorer") or k.startswith("model.source_head")
                            or k.startswith("model.pv_cluster_head") or k.startswith("model.edge_mass_head")
                            or k.startswith("model.node_struct_head") or k.startswith("model.node_mom_head"))]
            if missing:
                print(f"[heads] 旧 checkpoint 无新头参数 ({len(missing)} 个: "
                      f"chain_scorer/source_head/pv_cluster_head/edge_mass_head/node_struct_head/node_mom_head), 新头随机初始化续训")
                return super().load_state_dict(state_dict, strict=False)
        return super().load_state_dict(state_dict, strict=strict)

    def forward(self, batch):
        if self.use_pid == "realistic":  # only for pythia
            batch["tracks"].x = torch.cat([batch["tracks"].x, batch["tracks"].real_pid], dim=1)
        elif self.use_pid == "true":  # mc response for lhcb or onehot for pythia
            batch["tracks"].x = torch.cat([batch["tracks"].x, batch["tracks"].pid], dim=1)
        return self.model(batch)

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.model.parameters(), **self.optimizer_params)


        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,  # Reduce LR after 5 epochs of no improvement
            min_lr=1e-6,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_combined_loss",
                "interval": "epoch",
                "frequency": 1,
                "strict": True,
            },
        }

    def on_train_start(self):
        # 从 checkpoint 续训时, 若配置了新学习率, 覆盖 checkpoint 中保存的 lr
        # (checkpoint 恢复会还原 optimizer 状态, 包括旧 lr)
        if self.resume_lr is not None:
            for g in self.optimizers().param_groups:
                g["lr"] = float(self.resume_lr)
            print(f"[resume] overwrite lr -> {self.resume_lr}")

        # ==== v41 修复: 重置 EarlyStopping (子图训练 val 口径变化) ====
        # v38 切到子图训练后, val_combined_loss 起点会远高于 v38 全图口径的 35.562;
        # 若继承 checkpoint 里的 best_score/wait_count, 15 epoch 内没跌破旧 best 就会误停
        # (v40 就是被继承的 wait_count 在 4 epoch 后停掉的)。子图训练应只跟本次 run 的 best 比。
        if self.pv_cluster_on:
            try:
                for cb in self.trainer.callbacks:
                    if cb.__class__.__name__ == "EarlyStopping":
                        dev = cb.best_score.device if cb.best_score is not None else self.device
                        cb.best_score = torch.tensor(float("inf"), device=dev)
                        cb.wait_count = 0
                        print("[pv_cluster] 重置 EarlyStopping (子图训练 val 口径变化, 只跟本次 run 比)")
            except Exception as e:
                print(f"[pv_cluster] 重置 EarlyStopping 失败(跳过): {e}")

    def _pv_cluster_scores(self, batch):
        """pv_cluster_head 前向: 全图 tr-pv 边 -> 归属 logits [E, 1]。

        特征 = concat(tracks.x[src] (原始, pid concat 前), pvs.x[dst], trpv.edges),
        训练/测试都在同一特征空间 (与 GNN 编码无关), 保证 train/infer 对齐。
        """
        trpv = batch[('tracks', 'to', 'pvs')]
        ei = trpv.edge_index
        feat = torch.cat([
            batch['tracks'].x[ei[0]],
            batch['pvs'].x[ei[1]],
            trpv.edges,
        ], dim=-1)
        return self.model.pv_cluster_head(feat)

    def _pv_cluster_assign(self, batch, logits, sample=True):
        """用 pv_cluster_head 预测 (+ 温度退火 Gumbel 路由) 给每个 track 分配 PV (全局索引, -1=无)。

        分配 = argmax((logit + Gumbel噪声) / tau) (Gumbel-Softmax 标准形式):
          - sample=True (训练): tau 高(早期) 近似按 softmax 概率采样(探索); tau 低(后期) 收敛 argmax
          - sample=False (val): 确定性 argmax(logit), 与推理侧分簇一致
        """
        trpv = batch[('tracks', 'to', 'pvs')]
        ei = trpv.edge_index
        s = logits.squeeze(-1)
        if sample:
            tau = max(self.pv_tau, 1e-3)
            u = torch.rand_like(s).clamp(1e-8, 1.0 - 1e-8)
            g = -torch.log(-torch.log(u))  # Gumbel(0,1)
            score = (s + g) / tau
        else:
            score = s
        track_pv = torch.full((batch['tracks'].x.shape[0],), -1, dtype=torch.long, device=s.device)
        for t in torch.unique(ei[0]):
            m = (ei[0] == t)
            if not m.any():
                continue
            track_pv[int(t.item())] = ei[1][m][score[m].argmax()]
        return track_pv

    def _truth_pv_assign(self, batch):
        """truth PV 分配 (v39 方式): 每条 track 取其 y==1 关联边的第一个 PV; 无关联 -> -1。"""
        t_batch = batch['tracks'].batch
        trpv = batch[('tracks', 'to', 'pvs')]
        tr_pv = trpv.edge_index[:, trpv.y == 1]
        track_pv = torch.full((t_batch.numel(),), -1, dtype=torch.long, device=t_batch.device)
        for t, p in zip(tr_pv[0].tolist(), tr_pv[1].tolist()):
            if track_pv[t].item() == -1:
                track_pv[t] = p
        return track_pv

    def _split_by_pv(self, batch, logits=None, sample=True, cap=0):
        """方案7 (v39/v40/v41): 训练/验证时按 PV 分簇子图 (pv_cluster 纳入训练)。

        把 batch 中每个事件按 PV 拆成子图, 再用 Batch.from_data_list 重拼。
        子图内仅含该 PV 的 tracks + 该 PV 节点 + 簇内边 (tt 两端同簇 + tr-pv)
        —— 模型只见到"簇内低连通小图", 与推理时分簇重建完全对齐 (消除 train-inference gap)。

        track->PV 分配来源 (v41 课程式过渡):
          - alpha = 1.0 (run 起点): 全部用 truth 分配 (稳定热启动, v38 权重不被随机头扰动)
          - alpha -> 0.0: 逐步切换到 cluster 头分配 (训练 MLP + 温度退火)
          - logits=None: 回退纯 truth 分配
        碎片簇 (< pv_cluster_min_tracks) 并入无PV簇, 与推理侧同规则。
        cap>0 时训练侧随机保留 cap 个子图 (提速; val/test 传 cap=0 全量)。
        """
        try:
            from torch_geometric.data import Batch
            t_batch = batch['tracks'].batch              # [n_tr] 全局事件索引
            p_batch = batch['pvs'].batch                 # [n_pv]
            n_evt = int(t_batch.max()) + 1
            if logits is not None:
                truth_pv = self._truth_pv_assign(batch)
                cluster_pv = self._pv_cluster_assign(batch, logits, sample=sample)
                # 课程式过渡: 以概率 alpha 用 truth, 否则用 cluster 头 (逐 track)
                alpha = self.pv_curriculum_alpha
                if alpha <= 0.0:
                    track_pv = cluster_pv
                elif alpha >= 1.0:
                    track_pv = truth_pv
                else:
                    use_truth = torch.rand(truth_pv.shape, device=truth_pv.device) < alpha
                    track_pv = torch.where(use_truth, truth_pv, cluster_pv)
            else:
                track_pv = self._truth_pv_assign(batch)

            min_tracks = int(self.configs.get("pv_cluster_min_tracks", 3))
            subgraphs = []
            for ev in range(n_evt):
                ev_tracks = (t_batch == ev).nonzero().flatten().tolist()
                ev_pvs = (p_batch == ev).nonzero().flatten().tolist()
                groups = {}
                for t in ev_tracks:
                    groups.setdefault(int(track_pv[t].item()), []).append(t)
                # 碎片 PV 簇并入无PV簇 (防碎片链, 与推理侧同规则)
                for p in list(groups.keys()):
                    if p != -1 and len(groups[p]) < min_tracks:
                        groups.setdefault(-1, []).extend(groups.pop(p))
                for p, tl in groups.items():
                    if not tl:
                        continue
                    # 无PV (-1) 簇保留该事件全部 pvs (维持 pv_asso 监督); 有 PV 簇只留该 PV
                    pvs_keep = ev_pvs if p == -1 else [p]
                    if not pvs_keep:
                        continue
                    subgraphs.append(self._build_pv_subgraph(batch, tl, pvs_keep, ev))
            if not subgraphs:
                return batch
            # v41 修复③: 训练侧子图数量上限 (随机保留, 提速; 每 batch 随机 -> 全覆盖)
            if cap > 0 and len(subgraphs) > cap:
                keep = torch.randperm(len(subgraphs))[:cap].tolist()
                subgraphs = [subgraphs[i] for i in keep]
            return Batch.from_data_list(subgraphs)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[pv_cluster_train] WARN: {type(e).__name__}: {e} -> 用原 batch")
            return batch

    def _build_pv_subgraph(self, batch, tl, pvs_keep, ev):
        """手动构造 PV 子图 (普通 HeteroData, 无 batch 残留):
        选中 tracks/pvs 节点 + 两端均在簇内的 tt/tr-pv 边 + 节点/边/全局属性。
        """
        from torch_geometric.data import HeteroData
        dev = batch['tracks'].x.device  # 所有构造张量与 batch 同设备 (GPU 上 isin/indexing 需同设备)
        tl_t = torch.tensor(tl, dtype=torch.long, device=dev)
        pv_t = torch.tensor(pvs_keep, dtype=torch.long, device=dev)
        old2new = {old: i for i, old in enumerate(tl)}
        pv_old2new = {old: i for i, old in enumerate(pvs_keep)}
        sub = HeteroData()
        n_tr = batch['tracks'].x.shape[0]
        # 节点属性 (排除 batch/ptr; 只复制节点级属性, 信号级属性如 sig_keys 长度 != 节点数, 训练不需要)
        for k, v in batch['tracks'].items():
            if k in ('batch', 'ptr'):
                continue
            if v.shape[0] == n_tr:
                sub['tracks'][k] = v[tl_t]
        for k, v in batch['pvs'].items():
            if k in ('batch', 'ptr'):
                continue
            sub['pvs'][k] = v[pv_t]
        # tt 边: 两端同簇
        tt = batch[('tracks', 'to', 'tracks')]
        n_e = tt.edge_index.shape[1]
        m = torch.isin(tt.edge_index[0], tl_t) & torch.isin(tt.edge_index[1], tl_t)
        sub_tt_ei = tt.edge_index[:, m]
        sub_tt_ei = torch.tensor(
            [[old2new[int(a)] for a in sub_tt_ei[0].tolist()],
             [old2new[int(b)] for b in sub_tt_ei[1].tolist()]],
            dtype=torch.long, device=dev)
        sub[('tracks', 'to', 'tracks')].edge_index = sub_tt_ei
        for k, v in tt.items():
            # 只复制边级属性 (长度==边数; senders/receivers/sig_y 是稀疏衰变边属性, 训练不需要)
            if k != 'edge_index' and v.shape[0] == n_e:
                sub[('tracks', 'to', 'tracks')][k] = v[m]
        # tr-pv 边: 簇内 track + 保留的 PV
        trpv = batch[('tracks', 'to', 'pvs')]
        n_ep = trpv.edge_index.shape[1]
        m2 = torch.isin(trpv.edge_index[0], tl_t) & torch.isin(trpv.edge_index[1], pv_t)
        sub_trpv_ei = trpv.edge_index[:, m2]
        sub_trpv_ei = torch.tensor(
            [[old2new[int(a)] for a in sub_trpv_ei[0].tolist()],
             [pv_old2new[int(b)] for b in sub_trpv_ei[1].tolist()]],
            dtype=torch.long, device=dev)
        sub[('tracks', 'to', 'pvs')].edge_index = sub_trpv_ei
        for k, v in trpv.items():
            if k != 'edge_index' and v.shape[0] == n_ep:
                sub[('tracks', 'to', 'pvs')][k] = v[m2]
        # 全局属性 (Batch 无 .store property, 用 _global_store)
        gstore = getattr(batch, '_global_store', None)
        if gstore is not None:
            for k, v in gstore.items():
                sub[k] = v
        # globals: Batch 中堆叠为 [n_evt, dim], 单子图取本事件 (模型 encoder 需要 globals.x)
        try:
            gx = batch['globals'].x
            sub['globals'].x = gx[ev].unsqueeze(0)
        except Exception:
            pass
        return sub

    def shared_step(self, batch, batch_idx, log, mode="train"):
        loss = init_loss(self.device)

        # ==== 方案7b (v40/v41): 可训练 PV 分簇头 ====
        # cluster 头吃原始特征 (pid concat / GNN 前), 训练与推理同一头同一特征 -> 严格对齐。
        # 训练时: 头 loss (BCE, 全图 tr-pv 边) + 温度退火 Gumbel 分配 -> 切子图;
        # 验证时: 同样切子图 (v41 修复①: 消除 train/val 图结构 gap), 确定性分配;
        # 测试时: logits 存起来供 reconstruction 用同一头分簇 (pv_cluster_assign=cluster_head)。
        pv_cluster_logits = None
        if self.pv_cluster_on and mode in ("train", "val", "test"):
            try:
                trpv = batch[('tracks', 'to', 'pvs')]
                pv_cluster_logits = self._pv_cluster_scores(batch)      # [E, 1]
                if mode == "train":
                    y_pvclu = trpv.y.to(torch.float32).view(-1, 1)
                    pvclu_filter = (trpv.filter == 1) if hasattr(trpv, 'filter') else None
                    if pvclu_filter is not None and pvclu_filter.any():
                        loss["pv_cluster"] = self.pv_cluster_criterion(
                            pv_cluster_logits[pvclu_filter], y_pvclu[pvclu_filter])
                        log["pv_cluster_loss"].append(loss["pv_cluster"].item())
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[pv_cluster] WARN: {type(e).__name__}: {e} -> 回退 truth 分簇 / 跳过 cluster 头")
                pv_cluster_logits = None

        # ==== 方案7 (v39/v41): 训练/验证时 PV 分簇子图 (训练/推理图结构对齐) ====
        # v41 修复①: val 也切子图 (与 train 同结构, 消除 train/val gap; 确定性分配, 与推理一致)
        # v41 修复③: 训练侧 cap 子图数提速 (val/test 全量, val 每 epoch 一次开销可忽略)
        if mode in ("train", "val") and self.configs.get("pv_cluster", False):
            cap = self.pv_cluster_max_subgraphs if mode == "train" else 0
            batch = self._split_by_pv(batch, logits=pv_cluster_logits,
                                      sample=(mode == "train"), cap=cap)

        # modify batch to include pid information depending on use_pid or not
        if self.use_pid == "realistic":
            batch["tracks"].x = torch.cat([batch["tracks"].x, batch["tracks"].real_pid], dim=1)
        elif self.use_pid == "true":
            batch["tracks"].x = torch.cat([batch["tracks"].x, batch["tracks"].pid], dim=1)
        if mode == "test" and self.configs["pv_asso"]:
            minip = batch[("tracks", "to", "pvs")].edges.flatten()

        # 保存原始 tt 物理边特征 (model forward 会原地修改 batch edges -> 必须提前 clone)
        orig_tt_edges = batch[('tracks', 'to', 'tracks')].edges.clone()
        # 保存原始轨迹动量 (px,py,pz, 归一化) —— model forward 会原地覆盖 tracks.x
        # 为 encoder 表征, mass head 的物理真值 (ππ 不变质量) 需在覆盖前取出。
        orig_tracks_p = batch['tracks'].x[:, :3].clone()

        outputs = self.model(batch)
        # 将原始物理边特征挂到模型输出上 (model 会覆盖 edges 为 LCA 输出, 物理特征需保留)
        outputs[('tracks', 'to', 'tracks')].phys_edges = orig_tt_edges
        if self.configs["LCA"]:
            y_LCA = batch[('tracks', 'to', 'tracks')].y.to(torch.int64)
            outputs[('tracks', 'to', 'tracks')].lca = outputs[('tracks', 'to', 'tracks')].edges
            loss["LCA"] = self.lca_criterion(outputs[('tracks', 'to', 'tracks')].lca, y_LCA)
            log["LCA_loss"].append(loss["LCA"].item())
            acc_LCA = acc_four_class(outputs[('tracks', 'to', 'tracks')].lca, y_LCA)
            for key, values in acc_LCA.items():
                log[key].append(values)
        if self.configs["node_prune"]:
            y_nodes = (batch["tracks"].ft != 1).to(torch.float32).unsqueeze(-1)
        if self.configs["edge_prune"]:
            y_edges = batch[('tracks', 'to', 'tracks')].y > 0
            y_edges = y_edges.to(torch.float32).unsqueeze(-1)
        if self.configs["pv_asso"]:
            y_pv_asso = batch[("tracks", "to", "pvs")].y.to(torch.float32).view(-1, 1)
            pv_filter = batch[('tracks', 'pvs')].filter == 1

        for i, block in enumerate(self.model._blocks):
            if self.configs["node_prune"]:
                loss["t_nodes"] += self.node_criterion(block.node_logits['tracks'], y_nodes)
                if mode == "test" and self.configs["plt_nodes"]:
                    get_block_score(log, block.node_weights['tracks'].squeeze(), y_nodes, i, var="nodes")

            if self.configs["edge_prune"]:
                loss["tt_edges"] += self.edge_criterion(block.edge_logits[('tracks', 'to', 'tracks')], y_edges)
                if mode == "test" and self.configs["plt_edges"]:
                    get_block_score(log, block.edge_weights[('tracks', 'to', 'tracks')].squeeze(), y_edges, i,
                                    var="edges")
            if self.configs["pv_asso"]:
                loss["pv_asso"] += self.pv_asso_criterion(block.edge_logits[("tracks", "to", "pvs")][pv_filter],
                                                          y_pv_asso[pv_filter])
                if mode == "test" and self.configs["plt_pvs"]:
                    get_block_score(log, block.edge_weights[("tracks", "to", "pvs")].squeeze(), y_pv_asso, i,
                                    var="pv_asso")

        # Cap each loss component to prevent FP16 overflow before combination.
        # The 33x multiplier on tt_edges can push values beyond FP16 max (65504).
        # Per-component protection means only the overflowed component is zeroed,
        # while other loss signals still produce gradients for this batch.
        # clamp(max=1e3) 额外防止有限但极大的 loss (如 v31 ep68-70 出现 ~1e17 的数值爆炸)
        for k in loss:
            loss[k] = torch.nan_to_num(loss[k], nan=0.0, posinf=1e6, neginf=0.0).clamp(max=1e3)

        # ==== 第5个监督头: 候选衰变链选择 (train 时与主干联合训练) ====
        if mode == "train" and self.chain_scorer is not None:
            chain_loss = self._chain_select_loss(batch, outputs, block)
            if chain_loss is not None:
                loss["chain_select"] = chain_loss
                log["chain_select_loss"].append(chain_loss.item())

        # ==== 第6个监督头: 源检测 (Rumor Centrality 训练化, train 时) ====
        if mode == "train" and self.source_head_on:
            src_loss = self._source_loss(batch, outputs, block)
            if src_loss is not None:
                loss["source"] = src_loss
                log["source_loss"].append(src_loss.item())

        # ==== 方案5: 链内 LCA 一致性辅助损失 (chain_lca_filter 训练化, train 时) ====
        if mode == "train" and self.chain_lca_on:
            cl_loss = self._chain_lca_loss(batch, outputs, block)
            if cl_loss is not None:
                loss["chain_lca"] = cl_loss
                log["chain_lca_loss"].append(cl_loss.item())

        # ==== 第7个监督头: 边级不变质量回归 (输出侧物理监督, train 时) ====
        if mode == "train" and self.mass_head_on:
            m_loss = self._mass_loss(batch, outputs, orig_tracks_p)
            if m_loss is not None:
                loss["mass"] = m_loss
                log["mass_loss"].append(m_loss.item())

        # ==== 第8个监督头: 节点结构监督 (depth + RC 回归, train 时) ====
        if mode == "train" and self.struct_head_on:
            st_loss = self._struct_loss(batch, outputs, block)
            if st_loss is not None:
                loss["struct"] = st_loss
                log["struct_loss"].append(st_loss.item())

        # ==== 第9个监督头: 节点级动量回归 (mom_head, train 时) ====
        if mode == "train" and self.mom_head_on:
            mo_loss = self._mom_loss(batch, outputs, block, orig_tracks_p)
            if mo_loss is not None:
                loss["mom"] = mo_loss
                log["mom_loss"].append(mo_loss.item())

        combined_loss = loss["LCA"] + loss["t_nodes"] + 33*  loss["tt_edges"] + loss["pv_asso"]
        if "chain_select" in loss and self.chain_scorer is not None:
            combined_loss = combined_loss + self.chain_loss_weight * loss["chain_select"]
        if "source" in loss and self.source_head_on:
            combined_loss = combined_loss + self.source_loss_weight * loss["source"]
        if "chain_lca" in loss and self.chain_lca_on:
            combined_loss = combined_loss + self.chain_lca_loss_weight * loss["chain_lca"]
        if "mass" in loss and self.mass_head_on:
            combined_loss = combined_loss + self.mass_loss_weight * loss["mass"]
        if "struct" in loss and self.struct_head_on:
            combined_loss = combined_loss + self.struct_head_weight * loss["struct"]
        if "mom" in loss and self.mom_head_on:
            combined_loss = combined_loss + self.mom_loss_weight * loss["mom"]
        if "pv_cluster" in loss and self.pv_cluster_on:
            combined_loss = combined_loss + self.pv_cluster_loss_weight * loss["pv_cluster"]

        # 极端防御: 组合 loss 仍非有限或异常巨大时, 置为 0 损失, 避免梯度爆炸污染训练
        if not torch.isfinite(combined_loss) or combined_loss > 1e5:
            combined_loss = torch.zeros((), device=combined_loss.device, requires_grad=True)

        # Apply reco
        if mode == "test":
            # Attaching pruning information to graph
            if self.configs["node_prune"]:
                outputs["node_weights"] = block.node_weights["tracks"].squeeze()
            if self.configs["edge_prune"]:
                outputs["edge_weights"] = block.edge_weights[('tracks', 'to', 'tracks')].squeeze()
            # Getting the PV decisions
            if self.configs["pv_asso"]:
                pv_asso_des = {"pred": block.edge_weights[('tracks', 'to', 'pvs')].squeeze(), "minIP": minip,
                               "true": y_pv_asso.squeeze(), "pv_filter": pv_filter}
                # 方案7b: 用同一 pv_cluster_head 的得分供重建分簇 (train/infer 对齐)
                if self.pv_cluster_on and pv_cluster_logits is not None:
                    pv_asso_des["cluster_pred"] = torch.sigmoid(pv_cluster_logits).squeeze(-1)
            else:
                pv_asso_des = None
            # 注入内置 scorer (联合训练挂载在 model 上), 供链级选择使用
            if self.chain_scorer is not None:
                self.evt_reco.chain_scorer = self.chain_scorer
            self.evt_reco.reconstruct_heavyhadrons(outputs, pv_des=pv_asso_des)

        """Logging"""
        log = loss_logging(log, loss, self.configs, mode="DFEI")

        log["combined_loss"].append(combined_loss.item())
        return combined_loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, self.trn_log, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, self.val_log, mode="val")
        return loss

    def test_step(self, batch, batch_idx):
        _ = self.shared_step(batch, batch_idx, self.tst_log, mode="test")
        return {}

    def _chain_select_loss(self, batch, outputs, block):
        """第5个监督头: 候选衰变链选择 loss (train 时调用)。

        truth 正边连通分量 -> 真链 (正样本); 随机节点组合 -> 假链 (负样本);
        scorer 对每条链打分, BCE 到真/假标签。梯度经链内 node/edge 权重流回主干。
        """
        try:
            node_w = block.node_weights['tracks']                                  # [N_all, 1]
            edge_w = block.edge_weights[('tracks', 'to', 'tracks')]                # [E_all, 1]
            lca = outputs[('tracks', 'to', 'tracks')].edges                        # [E_all, 4] logits
            # 节点物理特征用模型输出 (encoder 后, 16 维), 与推理时链特征空间一致
            x = outputs['tracks'].x                                                # [N_all, 16]
            # 边物理特征必须用原始特征 (model 覆盖了 edges 为 LCA 输出)
            edges_phys = outputs[('tracks', 'to', 'tracks')].phys_edges            # [E_all, d_edge_phys]
            ei = batch[('tracks', 'to', 'tracks')].edge_index                      # [2, E_all]
            y = batch[('tracks', 'to', 'tracks')].y                                # [E_all] 或 [E_all,4]
            ft = batch['tracks'].ft                                                # [N_all]

            track_batch = batch['tracks'].batch                                    # [N_all]
            edge_batch = track_batch[ei[0]]                                        # [E_all] 边归属事件

            # 边 truth 二元 (兼容 one-hot 与标量)
            if y.dim() == 2 and y.shape[-1] > 1:
                y_bin = y.argmax(dim=-1) > 0
            else:
                y_bin = y > 0
            node_sig = ft != 1                                                      # 信号节点掩码

            n_evts = int(track_batch.max().item()) + 1
            scores, labels = [], []
            for evt_id in range(n_evts):
                tm = track_batch == evt_id
                em = edge_batch == evt_id
                if not tm.any():
                    continue
                nf = torch.cat([node_w[tm], x[tm]], dim=-1)                          # [N_i, 1+d_phys] (node_w 已含权重列)
                ef = torch.cat([edge_w[em], lca[em], edges_phys[em]], dim=-1)        # [E_i, 1+4+d_e]
                samples = build_chain_samples(ei[:, em], y_bin[em], node_sig[tm],
                                              int(tm.sum()), nf, ef)
                if samples is None:
                    continue
                pos_n, pos_e, pos_y, neg_n, neg_e, neg_y = samples
                dev = nf.device
                # 正样本链打分
                for nfe, efe in zip(pos_n, pos_e):
                    scores.append(self.chain_scorer(nfe.to(dev), efe.to(dev)))
                    labels.append(1.0)
                # 负样本链打分
                for nfe, efe in zip(neg_n, neg_e):
                    scores.append(self.chain_scorer(nfe.to(dev), efe.to(dev)))
                    labels.append(0.0)

            if not scores:
                return None
            s = torch.stack(scores)
            t = torch.tensor(labels, device=s.device)
            return self.chain_criterion(s, t)
        except Exception as e:
            print(f"[chain_select] WARN: {type(e).__name__}: {e}")
            return None

    def _source_loss(self, batch, outputs, block):
        """第6个监督头: 源检测 loss (Rumor Centrality 训练化)。

        truth 链内 rumor centrality 最大的节点 = 链根 (B 介子候选) -> 节点级标签;
        source_head 从节点特征 (tracks.x + node_weight) 预测"是否为根",
        BCE 到 truth 标签。梯度经 head 流回主干, 让主干学"根-叶结构"。
        """
        try:
            from wmpgnn.reconstruction.topk_selection import truth_chain_roots
            ei = batch[('tracks', 'to', 'tracks')].edge_index                      # [2, E_all]
            y = batch[('tracks', 'to', 'tracks')].y                                # [E_all] 或 [E_all,4]
            track_batch = batch['tracks'].batch                                    # [N_all]
            # 边 truth 二元 (兼容 one-hot 与标量)
            if y.dim() == 2 and y.shape[-1] > 1:
                y_bin = y.argmax(dim=-1)
            else:
                y_bin = y

            # 根节点标签: 每条 truth 链 (非背景边连通分量) 的 rumor centrality argmax
            roots = truth_chain_roots(y_bin, ei, track_batch, self.device)          # [N_all] 0/1

            # 节点特征: 最终 block 的 node_weight + 节点表征 -> [N_all, 1+16]
            node_w = block.node_weights['tracks']                                  # [N_all, 1]
            x = outputs['tracks'].x                                                # [N_all, 16]
            feat = torch.cat([node_w, x], dim=-1)
            logits = self.model.source_head(feat).squeeze(-1)                      # [N_all]
            return self.source_criterion(logits, roots)
        except Exception as e:
            print(f"[source_head] WARN: {type(e).__name__}: {e}")
            return None

    def _mass_loss(self, batch, outputs, orig_tracks_p):
        """第7个监督头: 边级 π-π 不变质量回归 (输出侧物理监督, train 时)。

        物理真值完全由数据自带 (无需额外标签): 从每条 tt 边两端的轨迹动量
        (orig_tracks_p, 归一化) 反归一化, 按双 π 假设 (E = sqrt(p² + m_pi²)) 计算
        不变质量 m_ij, 目标 = log10(m_ij/MeV) (动态范围 2.4~4.3, 匹配 SmoothL1 β=0.3);
        edge_mass_head 从 decoder 边表征 (latent_edges, 16维) 回归该目标。

        作用: 让 tt 边表征显式携带"动量-夹角"物理信息 (同母 ππ 对 -> 共振峰),
        辅助主干学习物理结构; 纯辅助监督, 不改主任务端到端目标。

        掩码: 任一端为未重建径迹 (px≈py≈pz≈-1 哨兵, VALUE_OR(-1)) 的边跳过。
        单维 px<0 不能做掩码 (真实轨迹 px<0 占比 ~48%)。
        """
        try:
            px = orig_tracks_p[:, 0] * self._mass_norm["px"][1] + self._mass_norm["px"][0]
            py = orig_tracks_p[:, 1] * self._mass_norm["py"][1] + self._mass_norm["py"][0]
            pz = orig_tracks_p[:, 2] * self._mass_norm["pz"][1] + self._mass_norm["pz"][0]

            # 未重建径迹哨兵: 三动量同时≈-1
            sentinel = ((px > -1.5) & (px < -0.5) & (py > -1.5) & (py < -0.5)
                        & (pz > -1.5) & (pz < -0.5))
            valid = ~sentinel

            ei = batch[('tracks', 'to', 'tracks')].edge_index
            a, b = ei[0], ei[1]
            edge_valid = valid[a] & valid[b]
            if not edge_valid.any():
                return None

            p1 = torch.stack([px[a], py[a], pz[a]], dim=-1)          # [E, 3]
            p2 = torch.stack([px[b], py[b], pz[b]], dim=-1)
            E1 = torch.sqrt((p1 ** 2).sum(-1) + self._m_pi ** 2)
            E2 = torch.sqrt((p2 ** 2).sum(-1) + self._m_pi ** 2)
            m2 = (E1 + E2) ** 2 - ((p1 + p2) ** 2).sum(-1)
            m = torch.sqrt(torch.clamp(m2, min=1.0))                 # MeV, 下限防 sqrt(0)
            target = torch.log10(m)                                  # ~2.4-4.3

            feat = outputs[('tracks', 'to', 'tracks')].latent_edges  # [E_all, 16]
            pred = self.model.edge_mass_head(feat).squeeze(-1)       # [E_all]
            return self.mass_criterion(pred[edge_valid], target[edge_valid])
        except Exception as e:
            print(f"[mass_head] WARN: {type(e).__name__}: {e}")
            return None

    def _struct_loss(self, batch, outputs, block):
        """第8个监督头: 节点结构监督 (深度 + Rumor Centrality 回归, train 时)。

        用户提议: source_head 只预测"链根"(1 bit), 信息量低; 改为连续结构监督,
        让主干学"节点在衰变树中的位置/层级":
          - depth 主头: 节点到链根的拓扑距离 (链内归一化 [0,1], 根=0)
          - rc 辅头:    节点 logR 链内 min-max 归一化 [0,1] (质心=1, 叶子=0)
        与 LCA 边分类 (class1 母子/class2 同母/class3 祖孙) 形成一致性约束:
        class1 边 depth 差 1, class2 差 0, class3 差 >=2 -> 模型被迫产出全局自洽的树。

        标签由 truth_chain_structure 计算 (纯 truth, 无模型噪声), 只监督链内节点。
        """
        try:
            from wmpgnn.reconstruction.topk_selection import truth_chain_structure
            ei = batch[('tracks', 'to', 'tracks')].edge_index            # [2, E_all]
            y = batch[('tracks', 'to', 'tracks')].y                      # [E_all] 或 [E_all,4]
            track_batch = batch['tracks'].batch                          # [N_all]
            if y.dim() == 2 and y.shape[-1] > 1:
                y_bin = y.argmax(dim=-1)
            else:
                y_bin = y
            st = truth_chain_structure(y_bin, ei, track_batch, self.device)
            m = st['in_chain']
            if not m.any():
                return None

            node_w = block.node_weights['tracks']                        # [N_all, 1]
            x = outputs['tracks'].x                                      # [N_all, 16]
            feat = torch.cat([node_w, x], dim=-1)
            out = self.model.node_struct_head(feat)                      # [N_all, 2]
            d_loss = self.struct_criterion(out[:, 0][m], st['depth'][m])
            r_loss = self.struct_criterion(out[:, 1][m], st['rc'][m])
            return d_loss + self.rc_head_weight * r_loss
        except Exception as e:
            print(f"[struct_head] WARN: {type(e).__name__}: {e}")
            return None

    def _mom_loss(self, batch, outputs, block, orig_tracks_p):
        """第9个监督头: 节点级动量回归 (mom_head, train 时)。

        PhyIP 探针发现: encoder 的 graph_norm+ReLU 把输入动量打散, 节点表征
        几乎不携带线性可读的动量 (线性/MLP 探针 R²≈0)。方案 A: 监督节点表征
        输出归一化输入动量 [px_n, py_n, pz_n] (encoder 的直接输入), 强制节点
        表征线性携带运动学信息。目标纯由数据自带 (tracks.x 前3维), 无需标签。

        掩码: 未重建径迹 (px≈py≈pz≈-1 哨兵) 跳过。
        """
        try:
            px = orig_tracks_p[:, 0] * self._mom_norm["px"][1] + self._mom_norm["px"][0]
            py = orig_tracks_p[:, 1] * self._mom_norm["py"][1] + self._mom_norm["py"][0]
            pz = orig_tracks_p[:, 2] * self._mom_norm["pz"][1] + self._mom_norm["pz"][0]
            sentinel = ((px > -1.5) & (px < -0.5) & (py > -1.5) & (py < -0.5)
                        & (pz > -1.5) & (pz < -0.5))
            valid = ~sentinel
            if not valid.any():
                return None

            x = outputs['tracks'].x                                # [N_all, 16]
            pred = self.model.node_mom_head(x)                     # [N_all, 3]
            target = orig_tracks_p                                 # [N_all, 3] 归一化输入动量
            return self.mom_criterion(pred[valid], target[valid])
        except Exception as e:
            print(f"[mom_head] WARN: {type(e).__name__}: {e}")
            return None

    def _chain_lca_loss(self, batch, outputs, block):
        """方案5: 链内 LCA 一致性辅助损失 (chain_lca_filter 训练化)。

        "最物理"判据训练化: 真链的链内边应是模型**高置信**的非背景边。
        truth 链内边 = LCA y>0 的边; 对该边被判类别 (argmax) 的 softmax 概率
        施加 hinge loss (低于 margin 则惩罚) -> 模型主动产出物理自洽的链,
        推理时 chain_lca_filter 的 conf 阈值自然更高、过滤更准。

        方案6 (chain_lca_ce=true): 额外对链内边施加 LCA 类别 CE, 直接监督
        链内边类别正确性 (class1/2/3), 对抗 class0 对主 LCA loss 的稀释。
        """
        try:
            import torch.nn.functional as F
            lca_logits = outputs[('tracks', 'to', 'tracks')].lca              # [E_all, 4]
            y = batch[('tracks', 'to', 'tracks')].y                            # [E_all] 或 [E_all,4]
            if y.dim() == 2 and y.shape[-1] > 1:
                y_cat = y.argmax(dim=-1)
            else:
                y_cat = y.long()
            y_bin = y_cat > 0                                                  # 链内边 (真类别 1/2/3)
            if not y_bin.any():
                return None
            probs = F.softmax(lca_logits[y_bin], dim=-1)                       # [n_chain_e, 4]
            conf = probs.max(dim=-1).values                                    # 被判类别概率
            margin = self.chain_lca_margin
            # hinge: conf < margin 的边受罚 (0.5*|conf-margin|² 让<margin的边远离)
            gap = torch.clamp(margin - conf, min=0.0)
            loss = (gap ** 2).mean()
            if self.chain_lca_ce:
                # 链内边类别 CE (target 1/2/3 合法), 放大结构边分类监督
                ce = F.cross_entropy(lca_logits[y_bin], y_cat[y_bin], reduction='mean')
                loss = loss + self.chain_lca_ce_weight * ce
            return loss
        except Exception as e:
            print(f"[chain_lca] WARN: {type(e).__name__}: {e}")
            return None

    def on_train_epoch_start(self):
        # ==== B2: 温度退火 (epoch 0: tau_start -> epoch b2_tau_epochs: tau_end) ====
        if getattr(self.model, "_b2_enable", False):
            gn = self.configs_gn
            tau_start = float(gn.get("b2_tau_start", 1.0))
            tau_end = float(gn.get("b2_tau_end", 0.1))
            tau_epochs = max(int(gn.get("b2_tau_epochs", 100)), 1)
            frac = min(float(self.current_epoch) / tau_epochs, 1.0)
            tau = tau_start + (tau_end - tau_start) * frac
            self.model.set_b2_tau(tau)
            if self.current_epoch % 10 == 0:
                print(f"[B2] epoch {self.current_epoch}: tau = {tau:.4f}", flush=True)

        # ==== 方案7b (v41): PV 分簇温度退火 + truth->cluster 课程式过渡 ====
        # 均相对本次训练起点计 (续训时 current_epoch 是 v38 的绝对 epoch, 不能直接用)
        if self.pv_cluster_on:
            self._pv_run_epoch += 1
            re = float(self._pv_run_epoch)
            # 温度退火: 1.0(探索,Gumbel采样) -> 0.1(收敛 argmax), 与 B2 同谱系
            frac = min(re / self.pv_tau_epochs, 1.0)
            self.pv_tau = self.pv_tau_start + (self.pv_tau_end - self.pv_tau_start) * frac
            # 课程式过渡: alpha 1.0(truth 分簇, 稳定热启动) -> 0.0(cluster 头)
            self.pv_curriculum_alpha = max(1.0 - re / self.pv_cluster_curriculum_epochs, 0.0)
            if self.current_epoch % 10 == 0 or self._pv_run_epoch == 1:
                print(f"[pv_cluster] run_epoch {self._pv_run_epoch}: tau={self.pv_tau:.3f} "
                      f"alpha={self.pv_curriculum_alpha:.3f}", flush=True)

    def on_train_epoch_end(self):
        avg_losses = epoch_end_loggable(self.trn_log)
        for key, val in avg_losses.items():
            self.log(f"train_{key}", val, prog_bar=(key == "combined_loss"), on_epoch=True, on_step=False)
        self.trn_log = defaultdict(list)

        optimizer = self.optimizers()
        current_lr = optimizer.param_groups[0]["lr"]
        self.log("lr", current_lr, prog_bar=False, on_epoch=True, on_step=False)

    def on_validation_epoch_end(self):
        avg_losses = epoch_end_loggable(self.val_log)
        for key, val in avg_losses.items():
            self.log(f"val_{key}", val, prog_bar=(key == "combined_loss"), on_epoch=True, on_step=False)
        self.val_log = defaultdict(list)

    def on_test_epoch_end(self):
        if self.version is None:
            self.version = self.logger.version
        # grab from the class and save to disk
        sig_df, evt_df = self.evt_reco.collect_results()
        sig_df.to_csv(f'{self.log_dir}/DFEI/version_{self.version}/signal_reco_df_{self.signal}.csv', index=False)
        evt_df.to_csv(f'{self.log_dir}/DFEI/version_{self.version}/event_reco_df_{self.signal}.csv', index=False)
        if self.configs["LCA"]:
            obtain_reco_accuracy(sig_df, self.version, self.signal, self.log_dir, model="DFEI")
            # LCAG 分类准确率 (论文 Table1), 追加到 info txt
            lca_num_keys = [k for k in self.tst_log if k.startswith("LCA_class") and k.endswith("_num")]
            if lca_num_keys:
                info_path = f"{self.log_dir}/DFEI/version_{self.version}/info_{self.signal}_reco.txt"
                with open(info_path, "a") as f:
                    f.write("=" * 50 + "\n")
                    f.write("LCAG classification accuracy (Table1):\n")
                    for i in range(4):
                        nums = list(self.tst_log.get(f"LCA_class{i}_num", []))
                        total_num = sum(nums)
                        if total_num == 0:
                            f.write(f"  LCA_class{i}: n=0\n")
                            continue
                        preds = []
                        for j in range(4):
                            fracs = list(self.tst_log.get(f"LCA_class{i}_pred_class{j}", []))
                            if not fracs:
                                continue
                            # 按每批数量加权
                            weighted = sum(n * p for n, p in zip(nums, fracs)) / total_num
                            preds.append(f"pred{j}={weighted*100:.2f}%")
                        f.write(f"  LCA_class{i} (n={total_num}): " + " ".join(preds) + "\n")

        if self.configs["plt_nodes"]:
            for i in range(len(self.model._blocks)):
                plot_weights(self.tst_log[f"sig_nodes_score_{i}"], self.tst_log[f"bkg_nodes_score_{i}"],
                             [f"NN_nodes_{i}_decision", "sig", "bkg"], self.version,
                             model="DFEI", channel=self.signal, log_dir=self.log_dir)
                plot_roc_curve(self.tst_log[f"sig_nodes_score_{i}"], self.tst_log[f"bkg_nodes_score_{i}"],
                               [f"NN_nodes_{i}_roc", "sig", "bkg"], self.version,
                               model="DFEI", channel=self.signal, log_dir=self.log_dir)
        if self.configs["plt_edges"]:
            for i in range(len(self.model._blocks)):
                plot_weights(self.tst_log[f"sig_edges_score_{i}"], self.tst_log[f"bkg_edges_score_{i}"],
                             [f"NN_edges_{i}_decision", "sig", "bkg"], self.version,
                             model="DFEI", channel=self.signal, log_dir=self.log_dir)
                plot_roc_curve(self.tst_log[f"sig_edges_score_{i}"], self.tst_log[f"bkg_edges_score_{i}"],
                               [f"NN_edges_{i}_roc", "sig", "bkg"], self.version,
                               model="DFEI", channel=self.signal, log_dir=self.log_dir)
        if self.configs["plt_pvs"]:
            for i in range(len(self.model._blocks)):
                plot_weights(self.tst_log[f"sig_pv_asso_score_{i}"], self.tst_log[f"bkg_pv_asso_score_{i}"],
                             [f"NN_pv_asso_{i}_decision", "correct", "false"], self.version,
                             model="DFEI", channel=self.signal, log_dir=self.log_dir)
                plot_roc_curve(self.tst_log[f"sig_pv_asso_score_{i}"], self.tst_log[f"bkg_pv_asso_score_{i}"],
                               [f"NN_pv_asso_{i}_roc", "sig", "bkg"], self.version,
                               model="DFEI", channel=self.signal, log_dir=self.log_dir)
            # Get the PV association performance
            log = self.evt_reco.log
            pv_perf = {}
            pv_perf["all_tracks"] = plot_pv_missasso(log["pv_corr_ml"], log["pv_corr_ip"], log["pv_total"], log["npvs"],
                                                     self.version, self.signal, log_dir=self.log_dir)
            pv_sig_tracks = plot_sig_pv_missasso(sig_df, self.version, self.signal, log_dir=self.log_dir)
            pv_perf.update(pv_sig_tracks)
            pv_perf["sig_b_system"] = plot_sig_b_system_pv_missasso(sig_df, self.version, self.signal,
                                                                    log_dir=self.log_dir)
            acc_pv_asso(pv_perf, self.version, self.signal, self.log_dir, model="DFEI")
