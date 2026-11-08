# DFEI 组会：逐页设计稿 v2（中文规划，页面文字为英文）

2026-10-26 · 只做规划，未改 PPT

参考（用于用词与句式，不引用其结论）：
- DFEI/HGNN 论文：<https://arxiv.org/html/2504.21844>
- 新一代 FEI 模型（metric learning + latent clustering）：<https://arxiv.org/html/2610.16021v1>
- LHCb 报告 "GNN event interpretations at LHCb and SHIP"：<https://indico.global/event/9778/contributions/93200/attachments/42696/80362/GNN_event_interpretations.pdf>
- DFEI 早期报告：<https://indico.cern.ch/event/1159913/contributions/5062747/attachments/2541356/4374977/DFEI_LHCb_JGP_041122.pdf>

---

## 0. 排版规则 v2（比上一版放宽，但更集中）

1. **每页只有一个文本框**。正文页 2–4 句（≤ 45 词），附录页 4–6 句（≤ 90 词）。禁止把说明拆成标题旁、图旁、页脚三处。
2. 图 ≥ 55% 版面；**页底只留一行结论**（≤ 12 词，粗体），其余全部进那个文本框。
3. 允许正文出现连续句子，但必须"一段话讲完一件事"；不写斜体注脚、不写重复口径。
4. 附录可以详细：一段话 + 一张图 + 一个公式，允许 6 句，用来应付提问。

一句话：**一页 = 一张图 + 一段话 + 一句结论。**

---

## 1. 英文为什么之前读着别扭（诊断）

| 毛病 | 例子 | 改法 |
|---|---|---|
| 中文直译、主语缺失 | "What each one is: next page." | "The next page describes each change in one line." |
| 营销/评价词 | "This is the honest size of the ranking term." | 删掉评价，只给数字与条件 |
| 一句塞三个动作 | "Edges become nodes, pass messages and add geometry to the score." | 拆成两句，每句一个动作 |
| 隐喻代替实现 | "let edges compete / talk to each other" | 直接说输入多了什么、消息在谁之间传 |
| 名词堆叠 | "geometry-aware attention with vertex-geometry bias" | "attention with a geometric bias on the pair" |

论文里的标准句式（照抄结构即可）：
- "The node-pruning GNN filters out background nodes based on kinematic and topological features."（arXiv 2504.21844）
- "The edge-pruning GNN removes unlikely associations and simplifies the event graph."（同上）
- "Edge pruning: the GNN classifies whether the edges connect particles from the same b-hadron."（LHCb 报告）
- "We add / We replace / We keep … frozen."（方法段常用主动式）

---

## 2. 结构总览

正文 17 页（约 22 分钟）+ 附录 7 页（不讲）。顺序：先便宜且大，再贵且小。

| # | 页 | 一句话作用 |
|---|---|---|
| 1 | 封面 | — |
| 2 | 目录 | 两条修复 + 一条优化 |
| 3–4 | Part 1 | PV 关联 |
| 5–6 | Part 2 | DOCA 符号 |
| 7 | 0702 饱和 | 为什么转 pruning |
| 8 | 账本 | 14.5 → 28.0 |
| 9 | 决策链地图 | 五个旋钮，按性价比 |
| 10 | 工作点 | +3.5，零训练 |
| 11 | 顶点几何 | +3.4，三列输入 |
| 12 | 排序损失 | +1.8 / +2.6 |
| 13 | 邻域特征 | +1.0，六个输入 |
| 14 | attention | +1.0，归因于几何 |
| 15 | 四个小项 | 合计 +2.3 |
| 16 | 输出结构 | 目标与现状 |
| 17 | 总结 | 3 个数字 + 2 个下一步 |

---

## 3. 逐页设计

> 版式：图放左或放满宽，**右侧一个文本框**（宽 4.6 in），页底结论固定在 y = 6.45 in。

### P1 封面
- 文字（居中三行）：`DFEI: bug fixes and pruning` / `0904 sample · 14.5 → 28.0%` / 姓名 · 单位 · 日期。
- 不加任何说明句。

### P2 目录
- 三条：`Part 1 · PV association` / `Part 2 · DOCA` / `Part 3 · Pruning and output`。
- 文本框（2 句）：`Two fixes change the inputs. One round of changes moves the pruning output from 14.5 to 28.0%.`

### P3 Part 1 · PV association
- 图：现有柱状图（旧模型 4 根、minIP 虚线、重训蓝柱）。
- 文本框（3 句）：`Four models trained on a sample with PV leakage give 13.5% misassociation on corrected data. The minIP baseline gives 10.15%. Retraining on the corrected sample brings them to 8.6%.`
- 结论：**Leakage inflated the gain; the corrected models still beat minIP.**
- 讲：重点在"旧模型还不如传统方法"这一句。
- 问：为什么回不到 2.0%？→ 那是泄漏样本上的数。

### P4 Part 1 · PV head
- 图：一条径迹 → 三个 PV 概率（0.85 / 0.10 / 0.05）。
- 文本框（3 句）：`An auxiliary head predicts which PV each surviving track comes from. It shares the trunk with the pruning tasks and is not used at inference. The head forces the trunk to encode event-level geometry.`
- 公式：`L_PV = −log p(true PV)`。
- 结论：**One auxiliary task recovers most of the loss.**
- 问：增益归头还是归重训？→ 同时引入，无法分离。

### P5 Part 2 · DOCA
- 图：左右两栏公式，红线标出唯一差异。
- 文本框（2 句）：`Both panels minimise the same distance between two straight tracks. The code differs in one sign in t₁, so the stationary point is wrong.`
- 公式：左 `t₁ = (c·e − d)/(1 − c²)`；右 `t₁ = −(d + c·e)/(1 − c²)`。
- 结论：**One wrong sign, same inputs, same algebra.**
- 问：定义不同？→ 不是，A、u 完全相同。

### P6 Part 2 · what it cost
- 图：两根柱 14.5 / 14.3，旁标 AUC 0.972 / 0.974。
- 文本框（3 句）：`Replacing the stored column moves the chain yield from 14.5 to 14.3% and the edge AUC from 0.972 to 0.974. The reconstruction never used this column in a way that exposed the error. The corrected distance matters only as an input, in Part 3.`
- 结论：**The fix alone buys nothing.**
- 问：那为什么修？→ 见 P11，+3.4 来自把它当输入。

### P7 0702 saturation
- 图：阶梯线 22.5 → 36.7 后拉平。
- 文本框（3 句）：`On the 0702 sample we tried nine physics heads, wider layers, attention and event restructuring. Reconstruction rose from 22.5 to 36.7% and then stopped. Adding capacity to the GNN no longer moves chains.`
- 结论：**The GNN side is saturated.**
- 注：口径（N = 17 561）只在此页出现一次。

### P8 账本
- 图：5 根柱 14.5 / 22.6 / 26.0 / 27.0 / 28.0；柱下标签 ≤ 2 词。
- 文本框（2 句）：`Fixed denominator, 13 255 signal chains. The largest single step is the cut, not a model change.`
- 结论：**14.5 to 28.0% in four steps.**
- 问：能相加吗？→ 不能，单变量效果有交互。

### P9 决策链地图（新页）
- 图：`candidate pair → score → cut → partition → chain`，每个方块下挂"改了什么 + Δ + 成本圆点"。
- 文本框（3 句）：`Each box is a place where the number can move. The labels give the measured gain and the cost. We start with the cheapest knob.`
- 结论：**Five knobs, cheapest first.**
- 讲：这一页只做导航，不展开。

### P10 工作点
- 图：三阈柱 + 三行表。
- 文本框（3 句）：`The weights are unchanged; only the cut on the edge score moves. At 0.90, 0.95 and 0.97 the chain yield is 14.5, 17.9 and 18.5%. All later numbers use 0.95.`
- 结论：**+3.5 points, no retraining.**
- 问：算不算调参？→ 是评测侧选择，所以单独声明。

### P11 顶点几何
- 图：两柱 22.6 → 26.0；三行特征表。
- 文本框（4 句）：`We add three inputs per candidate pair: the distance of closest approach between the two tracks, its logarithm, and the distance between the two production vertices. The first two are recomputed; the stored column is noise, with a correlation of 0.019 to the correct value. The third is a new absolute quantity. Fixing the column alone gave nothing (P6).`
- 结论：**+3.4 points from three input columns.**
- 问：这就是修 DOCA 吗？→ 修列是 0.0，这 +3.4 是特征块。

### P12 排序损失
- 图：左公式卡，右两组柱。
- 文本框（4 句）：`The edge loss is dominated by the positive weight of about 700, which lifts signal edges but does not order them against background. We add a pairwise term: every signal edge is compared with the 64 highest-scoring background edges of the same event. The network, the inputs and the schedule are unchanged.`
- 公式：`L = L_BCE + 10 · mean max(0, τ + s_bkg − s_sig)`，`τ = 1.0`。
- 结论：**+1.8 points at 0.90, +2.6 at 0.95.**
- 问：是构造假边吗？→ 不是，取同事件最难的 64 条真背景边。

### P13 邻域特征
- 图：一条边 (i,j)，两端各 3 条其它候选；右侧两柱。
- 文本框（4 句）：`For each end of a pair we ask two questions: how good a partner is this pair for that track, and how many near-equal alternatives does the track have. We add three numbers per end and keep the minimum and maximum over the two ends, so the six inputs do not depend on the order of the pair. The edge vector grows from 12 to 18.`
- 公式：`A = exp(−d₃/50)`，`r_t = A_e / max_{f∈C(t)} A_f`，`d_t = #{A_f > 0.5}/10`。
- 结论：**+1.0 point from six inputs.**
- 问：为什么取 min/max？→ 母/子方向不可靠，对称量避免方向泄漏。

### P14 attention
- 图：line graph 小图；右侧三行对照。
- 文本框（4 句）：`Candidate pairs become the nodes of a line graph, and pairs that share a track exchange messages in one round of multi-head attention. A small network maps the pair geometry into a bias on the attention logits. Attention alone is neutral (21.8 → 21.6); the gain appears only with this bias. There is no bias-only run, so we attribute the gain to the geometry, not to attention.`
- 公式：`α_ef = softmax(q_e·k_f/√8 + b_ef)`。
- 结论：**Geometry in the score, not attention, gives the point.**
- 问：attention 有用吗？→ 作为载体，单独 −0.2。

### P15 四个小项
- 图：4 行两列表。
- 文本框（2 句）：`Four low-cost changes; none of them exceeds one point on its own. Mechanics are in the appendix.`
- 结论：**About +2.3 points together.**
- 行：`minIP input +0.2` / `event bias +0.5` / `derived and triangle inputs +0.8` / `checkpoint on pruning AP +0.8`。

### P16 输出结构
- 图：左"pair → LCAG 0–3 → connected components"，右"track → one chain or background"。
- 文本框（4 句）：`Today each candidate pair is classified as LCAG 0–3 and chains are built from connected components. A track can end up in more than one chain, and there is no background class. The next step is an assignment: each track goes to one chain or to background. The previous attempt did not run; it is fixed and rerunning.`
- 结论：**Change the output, not the score.**
- 问：为什么能突破上限？→ 现在的输出形式与物理对象不对应。

### P17 总结
- 图：无；三个大数字。
- 文本框（3 句）：`The cut gives 3.5 points without retraining. Inputs and the loss give the rest. The next ceiling is the output format.`
- 结论：**14.5 to 28.0%; next, an exclusive output.**
- 下一步两条：`exclusive assignment` / `channel-level benchmark`。

---

## 4. 附录（可以不讲，一页一段话 + 一图 + 一公式）

- **A1 顶点几何**：段落说明三列定义、0.019 的来源、|Δstart| 与旧 `d_perp/(|Δz|+1)` 的区别、块内不可拆；配 stored-vs-correct 散点。
- **A2 邻域特征**：段落定义 `C(t)`、`A`、`r`、`d`、`c`，解释 min/max 的对称性、12→18 维、新增列零初始化；配端点示意图。
- **A3 排序损失**：段落写清"全部真边 vs 64 条最难背景边"、τ 与权重；配一条正边对多条负边的示意。
- **A4 attention**：段落写 line graph 构造、4 头 × 8 维、一轮、mask 规则、bias 形式；配 line graph 图与 `b_ef` 公式。
- **A5 四个小项**：表格 + 每行一句机制；末尾一句"event-count head −2.8、PV-overlap −0.4 已删"。
- **A6 输出结构诊断**：三行 bug 表（症状 / 根因 / 修法）+ 上一轮 7 个 job 小表；一段话说明为什么本轮不报数字。
- **A7 指标与 ROC**：ROC 图 + 5 行定义；一段话解释两套分母不可互换。

---

## 5. 中英用词对照（按论文/报告改）

| 中文 | 不再用 | 改成（出处） |
|---|---|---|
| 节点剪枝 / 边剪枝 | "prune nodes / cut edges" 混用 | `node pruning` / `edge pruning`（2504.21844） |
| 边分类 | "classify edges" | `the GNN classifies whether the edges connect particles from the same b-hadron`（LHCb 报告） |
| 衰变链 | "signal chain"（口语） | `decay chain`；指标处用 `signal decay chain` |
| 候选对 | "pair" 单独出现 | `candidate track pair`，首次出现后简称 `pair` |
| 工作点 | "working point 0.95" 混用 | `working point`；阈值处说 `the cut on the edge score` |
| 阈值扫描 | "threshold move" | `threshold scan` |
| 排他分配 | "exclusive output" 含糊 | `each track is assigned to at most one chain` / `exclusive assignment` |
| 背景类 | `dustbin` | `background class`（首次可括注 dustbin） |
| 目标凝聚 | "OC" | `object condensation`；`condensation points` → `cluster centres` |
| 槽位 | "slot" 不解释 | `a fixed set of object slots`；`slot attention` |
| 度量学习 | "latent clustering" | `metric learning in a latent space`（2610.16021） |
| 几何偏置 | "geometry-aware" | `a geometric bias on the attention logits` |
| 只改推理 | "inference-side" | `inference-only change` |
| 单变量效果 | "single-variable effect"（生硬） | `measured one change at a time` |
| 打包 | "bundle" | `several changes applied together` |

句式清单（直接套）：
- 我们加了什么：`We add three inputs per candidate pair.`
- 我们换了什么：`We replace the stored column with a locally recomputed distance.`
- 我们没动什么：`The network, the inputs and the schedule are unchanged.`
- 归因边界：`There is no bias-only run, so we attribute the gain to the geometry.`

---

## 6. 预判提问

1. **28% vs 论文 60%？** 分母不同：论文用剪枝后幸存链，我们用 13 255 条总信号链。
2. **AUC 降了不是变差？** AUC 是全局平均；重建只用一个阈值，取决于难例附近的排序。
3. **+3.4 是修 DOCA 吗？** 只换存的那一列是 0.0；+3.4 是重算距离 + |Δstart| 作为输入。
4. **attention 有用吗？** 单独 −0.2，弱 bias −0.5，几何 bias +1.0；无 bias-only 对照。
5. **邻域和 attention 重复吗？** 邻域是手工六个数，attention 是学出来的加权；各 +1.0。
6. **输出结构为什么没结果？** 上一轮三个实现 bug（派发、M>K 匹配、选模），已修并重跑。
