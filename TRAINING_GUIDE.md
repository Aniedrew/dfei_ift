# DFEI 模型训练完全指南

## 目录

1. [背景介绍](#1-背景介绍)
2. [核心概念：图与图神经网络](#2-核心概念图与图神经网络)
3. [数据：粒子对撞事件如何表示为图](#3-数据粒子对撞事件如何表示为图)
4. [模型架构：DFEI_HGNN](#4-模型架构dfei_hgnn)
5. [多任务学习](#5-多任务学习)
6. [训练流程](#6-训练流程)
7. [评估方法](#7-评估方法)
8. [配置文件详解](#8-配置文件详解)
9. [完整操作指南](#9-完整操作指南)
10. [常见问题](#10-常见问题)

---

## 1. 背景介绍

### 1.1 粒子物理中的事件重建问题

在高能物理实验中（如 LHCb 实验），质子以极高速度对撞，产生大量次级粒子。物理学家感兴趣的是那些包含 **b 夸克**（底夸克）的对撞事件，因为 b 夸克会衰变成其他粒子，这些衰变过程携带着重要的物理信息。

**事件重建（Event Reconstruction）** 的任务是：根据探测器记录的大量粒子径迹（tracks）和对撞顶点（vertices），还原出完整的物理过程——即哪些粒子来自同一个母粒子的衰变。

### 1.2 DFEI 做什么

**DFEI（Deep-learning-based Full Event Interpretation）** 是一个基于深度学习的全事件重建模型。它接收一个对撞事件中的所有径迹和顶点信息，然后同时完成以下四个子任务：

| 任务 | 简称 | 做什么 |
|------|------|--------|
| 径迹-径迹关联分类 | LCA | 判断任意两条径迹是否来自同一个母粒子的衰变（4类） |
| 径迹筛选（节点剪枝） | Node Pruning | 判断每条径迹是否属于感兴趣物理过程 |
| 径迹-径迹边筛选 | Edge Pruning | 判断径迹间的关联是否真实（二分类） |
| 径迹-顶点关联 | PV Association | 判断每条径迹属于哪一个对撞顶点 |

### 1.3 为什么用图神经网络？

粒子对撞事件天然就是**图结构**的数据：
- **节点（Nodes）** = 径迹（tracks）和顶点（PVs）
- **边（Edges）** = 径迹之间的可能关联、径迹与顶点的关联

传统神经网络（如 CNN、MLP）处理的是规则的网格或向量数据，无法有效建模这种不规则的结构化关系。图神经网络（GNN）正是为这种结构化数据设计的。

---

## 2. 核心概念：图与图神经网络

### 2.1 什么是图？

图由两部分组成：

```
图 = 节点（Nodes）+ 边（Edges）

例子：社交网络
  - 节点 = 人
  - 边 = 朋友关系
```

### 2.2 异构图（Heterogeneous Graph）

本项目中使用的图是**异构图**，意味着有不同类型的节点和边：

```
节点类型：
  - "tracks"（径迹）← 粒子穿过探测器留下的轨迹
  - "pvs"（顶点）  ← 粒子对撞发生的位置

边类型：
  - "tracks → tracks"（径迹之间的关联）
  - "tracks → pvs"（径迹与顶点的关联）

此外还有全局特征 "globals"，描述整个事件的整体信息
```

### 2.3 图神经网络（GNN）的基本原理

GNN 的核心思想是**消息传递（Message Passing）**：

```
1. 每条边根据两端节点的特征，计算出"边消息"
2. 每个节点收集与它相连的所有边的消息，更新自己的特征
3. 重复以上步骤多次（多个"消息传递层"）
4. 最终每个节点和边的特征包含了整个图的结构信息
```

可以这样理解：**每个节点通过不断与邻居"对话"，逐步了解整个图的情况。**

### 2.4 本项目使用的 GNN 构架

本项目遵循 **Graph Network（GN）框架**，每个 GN 块包含三个子模块：

```
GN 块 = 边更新 → 节点更新 → 全局更新
   ↑                        |
   └────────────────────────┘ (残差连接)
```

1. **边更新（Edge Block）**：用两端节点的特征更新边的特征
2. **节点更新（Node Block）**：用相连边的特征更新节点的特征
3. **全局更新（Global Block）**：用所有节点和边的特征更新全局特征

---

## 3. 数据：粒子对撞事件如何表示为图

### 3.1 原始数据格式

每个对撞事件被保存为一对 `.npy` 文件：
- `input_XXXXX.npy`：输入特征（探测器测量值）
- `target_XXXXX.npy`：真实标签（Truth / Monte Carlo truth）

### 3.2 转换后的数据格式

经过转换后，每个事件是一个 `HeteroData` 对象（PyTorch Geometric 格式），使用 zstd 压缩保存为 `.zst` 文件。

#### 径迹节点（tracks）的特征

每条径迹有 10 个输入特征（`track_x[:, :10]`），来自探测器测量：

| 特征索引 | 含义 | 物理量 |
|---------|------|--------|
| 0-2 | 动量 p<sub>x</sub>, p<sub>y</sub>, p<sub>z</sub> | GeV/c |
| 3 | 能量 E | GeV |
| 4-6 | 对撞参数 IP<sub>x</sub>, IP<sub>y</sub>, IP<sub>z</sub> | mm |
| 7 | χ²（径迹拟合质量） | 无量纲 |
| 8 | 径迹 χ² 的概率 | 0~1 |
| 9 | 电荷（charge） | -1, 0, 或 +1 |

当启用 PID（粒子识别）时（`use_pid: true`），还会额外拼接 5 个 PID 特征（用于区分不同粒子种类，如 π、K、p、e、μ）。

#### 顶点节点（pvs）的特征

PV 节点只包含三维坐标 `(x, y, z)`，单位是 mm。

#### 径迹-径迹边（tracks→tracks）的特征

每条边连接两条径迹，有 16 个特征，描述两条径迹之间的几何关系（如夹角、距离等）。

#### 全局特征（globals）

每个事件有一个全局特征向量，描述整个事件的整体性质。

### 3.3 数据文件格式

转换后的数据目录结构如下：

```
{data_dir}/
  └── 00342442_inclusive/
        ├── trn_data_000.zst    ← 训练数据（约800个事件/文件）
        ├── trn_data_001.zst
        ├── ...
        ├── val_data_000.zst    ← 验证数据（约200个事件/文件）
        ├── ...
        ├── tst_data_000.zst    ← 测试数据（约200个事件/文件）
        └── ...
```

文件使用 **zstd** 压缩。读取时先解压，然后用 `torch.load` 加载包含 HeteroData 对象列表的 PyTorch 数据。

### 3.4 数据加载器（ChunkLoader）

由于数据集很大（数万事件），项目使用**分块加载（Chunk Loading）**策略：

```
1. 将所有训练文件分成多个"块"（chunks），每个块约含 8 个文件
2. 训练时，每次只加载一个块到内存
3. 当前块训练完后，释放内存，加载下一个块
4. 这样即使数据集很大，内存占用也有限
```

工作流程：

```
文件列表 → 分组(chunks) → 依次加载每个chunk → 打乱事件顺序 → 按batch喂给模型
```

---

## 4. 模型架构：DFEI_HGNN

### 4.1 整体结构

```
输入图（HeteroData）
    │
    ▼
┌─────────────────────┐
│   编码器（Encoder）   │  ← 将原始特征映射到隐空间（latent space）
│   HeteroGraphCoder   │     维度: 10 + (PID) → 16
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  GN Block 1         │  ← 消息传递（边→节点→全局）
│   ├─ Edge Block     │     每块都学习和更新特征
│   ├─ Node Block     │
│   └─ Global Block   │
└─────────────────────┘
    │  ↑ 残差连接（与编码器输出拼接）
    ▼  │
┌─────────────────────┐
│  GN Block 2         │
│   ├─ Edge Block     │
│   ├─ Node Block     │
│   └─ Global Block   │
└─────────────────────┘
    │  ↑ 残差连接
    ▼  │
┌─────────────────────┐
│  GN Block 3         │
│   ├─ Edge Block     │
│   ├─ Node Block     │
│   └─ Global Block   │
└─────────────────────┘
    │  ↑ 残差连接
    ▼  │
┌─────────────────────┐
│  GN Block 4         │
│   ├─ Edge Block     │
│   ├─ Node Block     │
│   └─ Global Block   │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│   解码器（Decoder）   │  ← 从隐空间映射回输出空间
│   HeteroGraphCoder   │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ 输出变换（Output     │  ← 产生最终预测结果
│   Transformation）   │     径迹-径迹边: 4维(LCAG分类)
└─────────────────────┘
    │
    ▼
    输出图（包含各任务的预测结果）
```

### 4.2 编码器（Encoder）

编码器由 4 个独立的 MLP（多层感知机）组成，分别处理不同类型的数据：

```
Encoder:
  ├── global MLP:    全局特征  → 16维
  ├── tracks MLP:    径迹特征  → 16维（含PID时为16维）
  ├── pvs MLP:       顶点特征  → 16维
  ├── tracks_tracks MLP: 边特征 → 16维
  └── tracks_pvs MLP:    边特征 → 16维
```

**什么是 MLP？**

MLP（多层感知机）是最基本的神经网络类型，由多个"层"堆叠而成：

```
输入 → [线性变换 → 激活函数 → Dropout] → [线性变换 → 激活函数 → Dropout] → 输出
```

每一层都将输入数据做线性变换（乘以权重矩阵 + 加偏置），然后通过非线性激活函数（如 ReLU）引入非线性能力。

本项目中的 MLP 结构（以 `[-1, 128, 128, 16]` 为例）：
- `-1`：自动匹配输入维度
- `128`：第一个隐藏层，128个神经元
- `128`：第二个隐藏层，128个神经元
- `16`：输出维度，16维

### 4.3 GN 块（消息传递层）

每个 GN 块执行一轮完整消息传递：

#### 边更新（Edge Block）

```
对每条边 (i→j)：
  1. 获取源节点特征 h_i、目标节点特征 h_j、当前边特征 e_ij
  2. 将三者拼接 → [h_i, h_j, e_ij]
  3. 通过 MLP 更新边特征 → e'_ij
```

#### 节点更新（Node Block）

```
对每个节点 i：
  1. 收集所有以 i 为目标的边特征 e'_ji
  2. 对收集到的边特征求和（或取平均）→ 聚合特征
  3. 将聚合特征与当前节点特征拼接
  4. 通过 MLP 更新节点特征 → h'_i
```

#### 全局更新（Global Block）

```
对整个图：
  1. 收集所有节点特征和边特征
  2. 对所有特征求和聚合
  3. 与当前全局特征拼接
  4. 通过 MLP 更新全局特征
```

#### 残差连接（Residual / Skip Connection）

```
每个 GN 块输出 = GN块(当前图) + 编码器输出（按特征维度拼接）
```

这种设计让模型既可以学习新的复杂特征，又能保留原始的底层信息，有效防止深层网络中的梯度消失问题。

### 4.4 解码器（Decoder）

解码器的结构与编码器完全相同，但独立参数，负责将隐空间特征映射到适合产生最终预测的表示空间。

### 4.5 输出变换（Output Transformation）

输出变换层将解码器的输出转换为各任务的预测结果格式。关键配置：

```yaml
op_trafo:
  tr_tr: 4  # 径迹-径迹边输出4维 → LCAG 4类分类
```

其他输出维度设为 `None` 表示不使用。

### 4.6 模型总参数量

完整模型约 **569,000 个可训练参数**（约 0.57M），模型文件约 **9 MB**。

---

## 5. 多任务学习

### 5.1 什么是多任务学习？

DFEI 模型同时学习多个相关任务，共享底层的图特征表示。这类似于一个人同时学习多项相关技能，各技能之间可以互相促进。

### 5.2 四个学习任务

#### 任务1：LCAG 分类（径迹-径迹关联）

```
输入：每条 tracks→tracks 边的特征
输出：4类概率分布
类别：
  0 = 无关联（background）
  1 = 来自同一个 b 夸克的衰变产物
  2 = 来自同一条 b 夸克链
  3 = 其他关联
损失函数：交叉熵损失 (CrossEntropyLoss)
```

#### 任务2：径迹筛选（Node Pruning）

```
输入：每个 tracks 节点的特征
输出：该径迹是否属于感兴趣的物理过程
        (0 = 属于感兴趣过程, 1 = 背景噪声)
损失函数：二分类交叉熵 (BCEWithLogitsLoss)
```

#### 任务3：边筛选（Edge Pruning）

```
输入：每条 tracks→tracks 边的特征
输出：该边是否真实（是否代表真实的物理关联）
        (0 = 假边, 1 = 真边)
损失函数：二分类交叉熵 (BCEWithLogitsLoss)
注意：该损失的权重乘以 33（提高对边筛选的重视）
```

#### 任务4：径迹-顶点关联（PV Association）

```
输入：每条 tracks→pvs 边的特征
输出：该径迹是否属于该对撞顶点 (0或1)
损失函数：二分类交叉熵 (BCEWithLogitsLoss)
```

### 5.3 联合损失函数（Combined Loss）

四个任务的损失函数加权求和，形成最终的优化目标：

```
combined_loss = LCA_loss + node_prune_loss + 33 × edge_prune_loss + pv_asso_loss
```

边筛选损失的权重为 33，因为该任务的重要性较高且正负样本极度不平衡。

### 5.4 每个 GN 块都做预测

值得注意的是，**每个 GN 块**都有自己的预测头（MLP_infer），独立输出节点权重和边权重。这些预测用于：
1. **训练时**：所有块的预测都参与损失计算
2. **推理时**：最后一个块的预测用于事件重建

---

## 6. 训练流程

### 6.1 训练流程图

```
开始
 │
 ├─ 1. 读取配置文件 (YAML)
 │
 ├─ 2. 加载训练/验证数据 (ChunkLoader)
 │    ├─ 扫描文件列表
 │    ├─ 分组为 Chunks
 │    └─ 计算类别权重 (pos_weights)
 │
 ├─ 3. 创建模型实例 (DFEI_HGNN)
 │    └─ 根据配置构建编码器、GN块、解码器、输出变换
 │
 ├─ 4. 配置优化器
 │    ├─ 优化器: Adam (lr=0.001, weight_decay=1e-5)
 │    └─ 学习率调度器: ReduceLROnPlateau
 │         └─ 验证损失5个epoch不下降 → 学习率减半
 │
 ├─ 5. 开始训练 (最多100个Epoch)
 │    │
 │    ├─ 每个Epoch:
 │    │   ├─ 训练阶段 (training_step)
 │    │   │   ├─ 加载一个Batch的图数据
 │    │   │   ├─ 前向传播 → 得到预测
 │    │   │   ├─ 计算四个任务的损失
 │    │   │   ├─ 计算联合损失
 │    │   │   └─ 反向传播 + 梯度更新
 │    │   │
 │    │   └─ 验证阶段 (validation_step)
 │    │       ├─ 前向传播 (不计算梯度)
 │    │       ├─ 计算损失
 │    │       └─ 记录验证损失
 │    │
 │    ├─ 每个Epoch结束:
 │    │   ├─ 记录训练/验证指标
 │    │   ├─ 检查是否减少学习率
 │    │   ├─ 保存当前epoch的checkpoint
 │    │   └─ 如果验证损失新低 → 保存最佳checkpoint
 │    │
 │    └─ 停止条件:
 │        ├─ 达到最大epoch数 (通常100)
 │        └─ 提前停止 (验证损失连续15个epoch不下降)
 │
 ├─ 6. 测试阶段
 │    ├─ 加载测试数据
 │    ├─ 在测试集上前向传播
 │    ├─ 执行事件重建
 │    └─ 保存重建结果到CSV
 │
 └─ 7. 生成损失曲线图
      └─ 保存 metrics.csv 和 loss 曲线
```

### 6.2 优化器

使用 **Adam 优化器**，一种自适应学习率的优化算法：

| 参数 | 值 | 说明 |
|------|-----|------|
| 初始学习率 (lr) | 1e-3 (0.001) | 每次参数更新的步长 |
| 权重衰减 (weight_decay) | 1e-5 | L2正则化，防止过拟合 |

### 6.3 学习率调度（Learning Rate Schedule）

使用 **ReduceLROnPlateau** 策略：

```
监视指标：val_combined_loss（验证集联合损失）
触发条件：连续5个epoch验证损失没有下降
调度动作：学习率 × 0.5（减半）
最低学习率：1e-6
```

典型的学习率衰减过程：

```
Epoch  0-27:  lr = 1.0e-3  (初始学习率)
Epoch 28-37:  lr = 5.0e-4  (第1次减半)
Epoch 38-49:  lr = 2.5e-4  (第2次减半)
Epoch 50-65:  lr = 1.25e-4
Epoch 66-73:  lr = 6.25e-5
Epoch 74-81:  lr = 3.125e-5
Epoch 82-99:  lr = 1.5625e-5
```

### 6.4 提前停止（Early Stopping）

- **监视指标**：验证集联合损失 (val_combined_loss)
- **耐心值**：15 个 epoch
- **行为**：验证损失连续 15 个 epoch 没有下降时，自动停止训练

### 6.5 Checkpoint 保存

训练过程中自动保存模型 checkpoint：

| 类型 | 保存规则 | 文件名示例 |
|------|---------|-----------|
| 最佳模型 (Best) | 验证损失最低的前15个 | `best-epoch=93-val_combined_loss=0.923.ckpt` |
| 每轮模型 (All) | 每个epoch都保存 | `epoch_epoch=42.ckpt` |

最佳模型按验证损失排序，保存最低的前 15 个。这能确保即使后期过拟合，也能找到最好的模型。

### 6.6 梯度裁剪（Gradient Clipping）

- **阈值**：1.0
- **目的**：防止梯度爆炸，使训练更稳定

### 6.7 训练中的关键超参数

| 参数 | 典型值 | 说明 |
|------|--------|------|
| batch_size | 16 | 每次更新使用的样本数 |
| gacc (梯度累积) | 1 | 梯度累积步数 |
| epoch | 100 | 最大训练轮数 |
| ngpu | 1 | GPU数量 |
| ncpu | 8 | CPU线程数（数据加载） |

### 6.8 训练输出文件

训练完成后，在 `{log_dir}/DFEI/version_{N}/` 目录下生成：

```
version_{N}/
  ├── checkpoints/
  │   ├── best-epoch=93-val_combined_loss=0.923.ckpt  ← 最佳模型
  │   ├── epoch_epoch=00.ckpt                          ← 每轮模型
  │   ├── epoch_epoch=01.ckpt
  │   └── ...
  ├── metrics.csv         ← 所有epoch的训练/验证指标
  ├── hparams.yaml        ← 超参数记录
  ├── input_config.yaml   ← 原始输入配置
  └── events.out.*        ← TensorBoard日志（用于可视化）
```

`metrics.csv` 包含每一轮的详细指标：

| 列名 | 含义 |
|------|------|
| epoch | 轮数 |
| lr | 当前学习率 |
| train_combined_loss | 训练集联合损失 |
| val_combined_loss | 验证集联合损失 |
| train_LCA_loss | 训练集LCA损失 |
| val_LCA_loss | 验证集LCA损失 |
| train_t_nodes_loss | 训练集节点剪枝损失 |
| val_t_nodes_loss | 验证集节点剪枝损失 |
| train_tt_edges_loss | 训练集边剪枝损失 |
| val_tt_edges_loss | 验证集边剪枝损失 |
| val_LCA_class*_num | 验证集LCA各类样本数 |
| val_LCA_class*_pred_class* | 验证集LCA分类准确率 |

---

## 7. 评估方法

### 7.1 评估流程

评估（Evaluation）可以独立于训练运行，使用训练好的最佳 checkpoint 对测试数据进行评估：

```
1. 加载训练好的模型（从 checkpoint）
2. 加载测试数据
3. 在测试数据上执行前向传播
4. 执行事件重建（从网络输出中重建物理对象）
5. 输出重建结果到 CSV 文件
6. 生成评估图表（权重分布、ROC曲线等）
```

### 7.2 评估指标

#### 损失指标

与训练时相同的四大损失函数，在测试集上计算：

- **LCA Loss**：径迹-径迹关联分类的交叉熵损失
- **Node Pruning Loss**：径迹筛选的二分类损失
- **Edge Pruning Loss**：边筛选的二分类损失
- **PV Association Loss**：径迹-顶点关联的二分类损失
- **Combined Loss**：以上四项的加权和

#### 分类准确率

对于 LCAG 四分类任务，报告每个类别的：
- **预测准确率**：正确分类的样本比例
- **混淆矩阵**：每个真实类别的预测分布

#### 事件重建准确率

通过网络输出的权重和分类结果，执行完整的物理事件重建，评估：
- **B 强子重建效率**：正确重建的 B 强子比例
- **PV 关联准确率**：径迹-顶点关联的正确率
- **误关联率**：错误关联的比例

### 7.3 输出文件

评估完成后，生成：

```
version_{N}/
  ├── signal_reco_df_{sample}.csv    ← 信号事件重建结果表
  ├── event_reco_df_{sample}.csv     ← 全部事件重建结果表
  ├── NN_*_decision.png              ← 神经网络决策分布图
  └── NN_*_roc.png                   ← ROC曲线图
```

---

## 8. 配置文件详解

### 8.1 完整配置项说明

```yaml
settings:
  # ──── 数据路径 ────
  data_dir: "/path/to/converted_data"  # 转换后数据目录
  sample: [ "00342442_inclusive" ]     # 数据样本名称
  nfiles: [ 50 ]                       # 每个样本使用的trn文件数

  # ──── 训练参数 ────
  batch_size: 16        # 批次大小
  lr: 1e-3              # 初始学习率
  weight_decay: 1e-5    # 权重衰减（L2正则化）
  epochs: 100           # 最大训练轮数
  gacc: 1               # 梯度累积步数
  ngpu: 1               # GPU数量
  ncpu: 8               # CPU线程数（数据加载用）

  # ──── 数据预处理 ────
  graph_mode: "None"    # 是否使用真实标签剪枝
  node_sel: "true"      # 剪枝模式
  pv_model: "None"      # PV关联预处理模型
  calibration: False    # 是否做校准

inference:
  LCA: true             # 是否执行LCA任务
  LCA_weights: true     # 是否计算LCA权重
  node_prune: true      # 是否执行节点剪枝
  node_prune_weights: true
  edge_prune: true      # 是否执行边剪枝
  edge_prune_weights: true
  pv_asso: true         # 是否执行PV关联
  pv_asso_weights: true
  edge_prune_thr: 0.9   # 边剪枝阈值
  node_prune_thr: 0.9   # 节点剪枝阈值

evaluate:
  sample: [ "00342442_inclusive" ]
  nfiles: [ 50 ]        # 测试文件数
  over_write: ""        # 输出文件名覆盖

DFEI:
  use_pid: "true"       # 是否使用PID信息
                        #   "true" - 使用MC PID
                        #   "realistic" - 使用真实PID
                        #   "None" - 不使用
  cpt: None             # 加载checkpoint继续训练
  node_types: [ 'tracks', 'pvs' ]  # 图节点类型
  edge_types: [ "tracks_tracks", "tracks_pvs" ]  # 图边类型

  # ──── 编码器 ────
  encoder:
    usage: true
    global:       { layers: [ -1, 128, 128, 16 ], norm: "graph_norm", dropout: 0.01 }
    tracks:       { layers: [ -1, 128, 128, 16 ], norm: "graph_norm", dropout: 0.01 }
    pvs:          { layers: [ -1, 128, 128, 16 ], norm: "graph_norm", dropout: 0.01 }
    tracks_tracks:{ layers: [ -1, 128, 128, 16 ], norm: "graph_norm", dropout: 0.01 }
    tracks_pvs:   { layers: [ -1, 128, 128, 16 ], norm: "graph_norm", dropout: 0.01 }

  # ──── 消息传递层 ────
  GNblocks:
    nBlocks: 4            # GN块数量
    use_globals: true     # 是否使用全局特征更新
    use_node_weights: true
    use_edge_weights: true
    weighted_pass: true
    MLP_forward: { layers: [ -1, 128, 128, 16 ], norm: "None", dropout: 0.01 }
    MLP_infer:   { layers: [ -1, 16, 16, 16, 1 ], norm: "None", dropout: 0.01 }

  # ──── 解码器 ────
  decoder:  # 结构与编码器相同

  # ──── 输出变换 ────
  op_trafo:
    usage: true
    tr_tr: 4              # 径迹-径迹边输出维度（4类LCAG）
    tr_pv: None
    tr: None
    pv: None
    global: None
    MLP: { layers: [ -1, -1 ], norm: "None", dropout: 0.01 }
```

### 8.2 配置文件的关键字段

#### `layers` 参数格式

`layers: [ -1, 128, 128, 16 ]` 表示：

| 位置 | 值 | 含义 |
|------|-----|------|
| 第1层 | -1 | 自动匹配输入维度 |
| 第2层 | 128 | 第一个隐藏层，128个神经元 |
| 第3层 | 128 | 第二个隐藏层，128个神经元 |
| 第4层 | 16 | 输出维度，16维 |

#### `norm` 参数

- `"graph_norm"`：图归一化（推荐用于编码器/解码器）
- `"batch_norm"`：批归一化
- `"None"`：不使用归一化（通常用于GN块内部）

#### `dropout` 参数

取值 0~1，表示训练时随机丢弃神经元比例。`0.01` 表示1%的丢弃率，是较轻的正则化。

---

## 9. 完整操作指南

### 9.1 环境准备

```bash
# 进入项目目录
cd /lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn

# 激活conda环境
source ~/.bashrc
export PATH=$HOME/miniconda3/envs/dfei/bin:$PATH
```

### 9.2 数据转换

如果原始数据是 `.npy` 格式，需要先转换为 `.zst` chunk 格式：

```bash
python3 convert_all_data.py
```

转换后验证：

```bash
python3 convert_all_data.py --validate-only
```

默认输出到 `/lzufs/user/guoqingxiang/DFEI_data/converted/00342442_inclusive/`。

### 9.3 本地测试训练

提交小规模测试作业，验证流程是否正常：

```bash
# 使用小配置文件（2个文件，2个epoch）
hep_sub submit_train_dfei_small.sh -g lzuhep -gpu 1 -cpu 4 -m 16000
```

### 9.4 正式训练作业

```bash
# 完整训练（100 epoch，全部数据）
hep_sub submit_train_dfei.sh -g ghigh -gpu 1 -cpu 8 -m 32000 -wt long \
  -o logs/train_dfei.out -e logs/train_dfei.err
```

### 9.5 监控训练

```bash
# 查看作业状态
hep_q -u guoqingxiang

# 查看训练日志
tail -f logs/train_dfei.out

# 查看已生成的checkpoint
ls LHCb_logs/DFEI/version_{N}/checkpoints/
```

### 9.6 评估已训练模型

创建评估配置文件（指向要评估的版本号），然后运行：

```bash
python3 wmpgnn/analysis/evaluate.py --config config_files/eval_config.yaml
```

或通过 HTCondor 提交评估作业：

```bash
hep_sub submit_eval.sh -g lzuhep -gpu 1 -cpu 4 -m 16000 -wt short
```

### 9.7 查看评估结果

```bash
# 训练指标
cat LHCb_logs/DFEI/version_{N}/metrics.csv

# 重建结果CSV
cat LHCb_logs/DFEI/version_{N}/signal_reco_df_{sample}.csv

# 评估图片（需查看目录下的PNG文件）
ls LHCb_logs/DFEI/version_{N}/*.png
```

### 9.8 两个模型对比

本项目中存在两个可用于对比的模型：

| 比较项 | CERN MC 数据模型 | 公开 LHCb 数据模型 |
|--------|----------------|------------------|
| 数据来源 | CERN 生成的 MC 模拟数据 | 公开发布的 LHCb 碰撞数据 |
| 训练文件数 | 219 个 trn 文件 | 50 个 trn 文件 |
| 是否含 PID | 是 (`use_pid: true`) | 否 (`use_pid: None`) |
| 版本目录 | version_8 | version_9（待训练完成） |
| 最佳验证损失 | ~0.923 | 待定 |

---

## 10. 常见问题

### Q1: 训练时出现 "CUDA out of memory"

**原因**：GPU 显存不足。
**解决方案**：
1. 减小 `batch_size`（如从16减到8）
2. 减少 `ncpu`（如从8减到4）
3. 提交作业时增加内存限制：`-m 32000`（32GB）

### Q2: 训练时出现 "AttributeError: NodeStorage has no attribute 'pid'"

**原因**：配置中设置了 `use_pid: true`，但数据中不包含 PID 信息。
**解决方案**：将 `use_pid` 改为 `"None"`。

### Q3: 验证损失不再下降但训练损失还在降

**原因**：模型开始过拟合。
**解决方案**：
1. 检查是否有早停机制（EarlyStopping）生效
2. 降低学习率或增加 weight_decay
3. 增加 dropout 率

### Q4: 测试阶段一直卡住或超时

**原因**：测试数据加载器批大小不匹配或显存不足。
**解决方案**：
- 测试阶段使用的 `batch_size` 为 512，可考虑降低
- 确保测试文件数配置正确

### Q5: 数据路径中包含哪些关键字才能使用 ChunkLoader？

ChunkLoader 通过检测 `data_dir` 路径中的关键字来启用：
- `"nu7p6"`：Pythia 模拟数据
- `"LHCbcollision"`：LHCb 碰撞数据
- `"LHCb"`：通用 LHCb 数据
- `"CERN_data"`：CERN MC 数据

如果路径中不包含以上关键字，将使用默认数据加载器（一次性加载所有数据到内存）。

### Q6: 如何从 checkpoint 继续训练？

在配置文件中设置：
```yaml
DFEI:
  cpt: None  # 设置为版本号或 checkpoint 路径
```
- `cpt: 8`：加载 `LHCb_logs/DFEI/version_8/` 中最好的 checkpoint
- `cpt: "/path/to/model.ckpt"`：加载指定路径的 checkpoint
