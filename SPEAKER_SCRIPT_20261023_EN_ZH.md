# DFEI: Bug Fixes and Pruning Optimization — Bilingual Speaker Script
# DFEI：Bug 修复与 Pruning 优化 —— 中英对照讲稿

**Group meeting · 2026-10-23 · 25 slides · about 30 minutes**
**组会 · 2026-10-23 · 25 页 · 约 30 分钟**

> One line of English, one line of Chinese.
> 一行英文，一行中文。

---

## 1 · Cover (30 s) / 封面（30 秒）

Good morning. Today I will cover two things: the bugs we fixed, and what we changed in the pruning stage.
早上好。今天讲两件事：我们修了哪些 bug，以及在 pruning 阶段改了什么。

All 0904 reconstruction numbers in this talk use the total number of signal chains, 13,255, as the denominator; page 6 uses the paper convention (surviving chains), and page 7 uses the 0702 line with N = 17,561. The three are not directly comparable.
0904 的重建数字都以总信号链数 13,255 为分母；第 6 页用论文口径（幸存链），第 7 页的 0702 线用 N = 17,561。三者不可直接比较。

The paper uses a different denominator, so the two sets of numbers are not directly comparable; I will come back to this at the overview on page 8.
论文用的是另一个分母，所以两组数字不能直接比较；第 8 页看总效果时我会再说明。

---

---

---

## 2 · Outline (30 s) / 提纲（30 秒）

The talk follows this outline: Part 1, PV association and the minIP baseline; Part 2, the DOCA bug; Part 3, the pruning optimization; and a short summary.
报告按这个提纲：第 1 部分 PV 关联与 minIP 基线；第 2 部分 DOCA bug；第 3 部分是 pruning 优化；最后是总结。

Parts 1 and 2 are fixes; Part 3 is the main content.
前两部分是修复，第 3 部分是主体。

---

---

---

## 3 · Part 1: PV association, old models are worse than minIP (1.5 min) / 第 1 部分：PV 关联，旧模型比 minIP 还差（1.5 分钟）

This is the PV misassociation rate — lower is better.
这是 PV 关联错误率，越低越好。

The grey bars are four models trained on the leaked sample and evaluated on the corrected one; they average 13.6 percent.
灰柱是四个在泄漏样本上训练、在修正样本上评估的旧模型，平均 13.6%。

The red dashed line is the traditional minIP baseline at 10.15 percent; all four old models are worse than minIP.
红虚线是传统 minIP 基线 10.15%；四个旧模型全部比 minIP 差。

After retraining on the corrected sample, we reach 8.6 percent.
在修正样本上重训后，我们做到 8.6%。

So the leak did exaggerate the PV advantage; the advantage is real, but much smaller.
所以泄漏确实夸大了 PV 优势；优势是真的，但小得多。

---

---

---

## 4 · Part 1: the PV-aware head (1 min) / 第 1 部分：PV 辅助头（1 分钟）

How did we get it back? We added an auxiliary head that predicts, for every surviving track, which PV it belongs to. In the picture the track gets 0.85 / 0.10 / 0.05; the target is the true PV, trained with cross-entropy.
怎么修回来的？我们加了一个辅助头，让每个幸存径迹预测它属于哪个 PV。图里这条径迹被分成 0.85 / 0.10 / 0.05 三个概率，训练目标就是让真 PV 那一项接近 1，用交叉熵。

PV assignment is an event-level property, so learning it forces the shared representation to encode detector geometry.
PV 归属是事件级信息，学它迫使共享表示编码探测器几何。

Those geometric quantities are also what we need to decide whether two tracks belong to the same chain.
这些几何量恰好也是判断两条径迹是否同链所需的。

The head is used only in training and removed at inference, so the reconstruction pipeline is unchanged.
这个头只在训练时使用，推理时去掉，所以重建流程没有改动。

One caveat: retraining and the head were introduced together, so we cannot separate their contributions on this sample.
需要说明：重训和这个头是同时引入的，在这个样本上无法分离两者贡献。

---

---

---

## 5 · Part 2: the DOCA bug is one wrong sign (1.5 min) / 第 2 部分：DOCA bug 错在一行（1.5 分钟）

The bug is not the name, it is one line of the upstream implementation.
问题不在名字，而在上游实现的一行。

Left: the correct closest distance. Treat the tracks as lines through the production vertices A1, A2 with unit momenta u1, u2, and minimise |w + t1u1 − t2u2|², where w = A1 − A2, c = u1·u2, d = w·u1 and e = w·u2. The solution is t1=(c·e−d)/(1−c²), t2=(e−c·d)/(1−c²), and DOCA is the norm of w + t1u1 − t2u2 at that solution.
左边是正确的最近距离：把径迹看成过生产顶点 A1、A2、方向为单位动量 u1、u2 的直线，最小化 |w + t1u1 − t2u2|²，其中 w = A1 − A2、c = u1·u2、d = w·u1、e = w·u2；解是 t1=(c·e−d)/(1−c²)、t2=(e−c·d)/(1−c²)。

Right: what the code did. Only the t1 line differs — it wrote −(d+c·e)/(1−c²). That is not the stationary point of the minimisation.
右边是代码实际做的：只有 t1 这一行不同，写成了 −(d+c·e)/(1−c²)，它不是这个最小化问题的驻点。

The inputs A, u are identical, so this is not a different definition: it is a sign error.
输入 A、u 完全一样，所以这不是定义不同，就是一行符号错。

---

---

---

## 6 · Part 3: 0702 line, GNN saturation (1.5 min) / 第 3 部分：0702 线，GNN 饱和（1.5 分钟）

Last round we optimized the GNN, not the pruning.
上一轮我们优化的是 GNN，不是 pruning。

We added differentiable pruning, a class-0 fix, a hinge term and chain cross-entropy; then we tried nine physics heads on one backbone, and only the mass head paid off.
我们加了可微剪枝、class-0 修正、hinge 项和 chain-CE；然后在同一主干上试了九种物理头，只有质量头有收益。

On the 0702 line with the fixed denominator (N=17,561), reconstruction went from 22.5 to 36.7 percent and then saturated. This line is not directly comparable to the 0904 numbers later; the 0904 starting point is 14.5 percent.
在 0702 线、固定分母 N=17561 下，重建从 22.5% 涨到 36.7%，然后就饱和了。这条线和后面的 0904 数值不可直接比较；0904 的起点是 14.5%。

That saturation is why this round moved upstream, to the pruning inputs.
正是这个饱和促使这一轮转向上游的 pruning 输入。

---

---

---

## 7 · Part 3: 0904 sample, cumulative gain (1 min) / 第 3 部分：0904 样本，累计增益（1 分钟）

Start with the overall effect.
先看整体效果。

Three bands, bottom to top: exact, contaminated, pruned away.
三条带从下往上：精确、带污染、被剪掉。

Exact goes from 14.5 to 28.0 percent; contamination falls from 42.5 to 18.0; pruned-away rises from 43 to 54.
精确从 14.5% 到 28.0%；污染从 42.5% 降到 18.0%；被剪掉从 43% 升到 54%。

The stages are baseline 14.5 → pre-geometry 22.6 → vertex geometry 26.0 → neighbourhood 27.0 → geometry-aware attention 28.0. The pre-geometry step bundles the working point, ranking, cleanup and selection; the next page unpacks it.
图上四个阶段：baseline 14.5% → pre-geometry 22.6% → vertex geometry 26.0% → neighbourhood 27.0% → geometry-aware attention 28.0%。pre-geometry 这一步包含工作点、排序项、清理和选模，下一页拆开讲。

Stricter pruning removes more truth tracks — that is expected. What matters is the size: contamination drops by 24 points and exact reconstruction doubles.
剪枝变严会多剪掉真径迹，这在意料之中；重要的是幅度——污染少了 24 个点，精确重建翻倍。

Assembly failure is exactly zero, so each column sums to 100 percent.
组装失败严格为 0，所以每一列加起来是 100%。

---

---

---

## 8 · Part 3: pre-geometry, six validated gains (1.5 min) / 第 3 部分：pre-geometry，六个有效改动（1.5 分钟）

Six changes kept; each measured on its own at the fixed denominator:
只有六个改动是正的，每个都单独测过，固定分母：

- working point 0.90 → 0.95: **+3.45** (same v601 weights)
- ranking loss: **+1.84** at 0.90; +2.55 at 0.95
- minIP targeting: +0.21
- event bias: +0.54
- derived / triangle features: +0.79
- checkpoint selection on pruning AP: +0.77

They are not additive: these are single-variable effects, while the shipped chain bundles several of them. **The largest single entry is the working point, not a model change.**
这些数不能直接相加——它们是单变量效果，最终版本是打包的链条。最大的单项是工作点，不是模型改动。

---

---

## 9 · Part 3: pre-geometry, one line per change (1 min) / 第 3 部分：pre-geometry，每个改动一句话（1 分钟）

This page only says what each change is.
这一页只回答一个问题：每个改动到底是什么。

The working point is on the **evaluation side**: same weights, only the node / edge threshold moves. The ranking loss is a **loss**: L = L_BCE + 10 L_rank. The other four are **input and training side**: minIP targeting (a canonicalised dz·IP input), event bias (one learned scalar per event added to the pruning logits), derived and triangle features, and checkpoint selection on pruning AP.
工作点是评测侧的：同样的权重，只移动 node / edge 阈值。排序项是损失：L = L_BCE + 10 L_rank。其余四个是输入与训练侧：minIP 定向（canonicalised dz·IP 输入）、event bias（每个事件一个学到标量加在剪枝 logits 上）、派生与三角特征、按 pruning AP 选 checkpoint。

Derivations are in appendix A1–A5; the next five pages only report results.
原理和推导放在附录 A1–A5；接下来五页只给结果。

---

---

## 10 · Part 3: pre-geometry, working point result (1 min) / 第 3 部分：pre-geometry，工作点 0.90 → 0.95 的结果（1 分钟）

Same v601 weights, only the node / edge threshold: 0.90 → 14.5 / 9.5%, 0.95 → 17.9 / 11.7%, 0.97 → 18.5 / 11.9%.
同一个 v601 模型，只动 node / edge 阈值：0.90 → 14.5 / 9.5%，0.95 → 17.9 / 11.7%，0.97 → 18.5 / 11.9%。

No retraining: this is where we choose to cut. It is worth **+3.4 pp** and must be declared; otherwise an evaluation-side change is sold as a model gain. All later numbers use 0.95.
不重训，纯粹是“在哪里切”。它值 +3.4 pp，必须先声明，否则会把评测侧的变化当成模型收益。后面所有数字都用 0.95。

---

---

---

## 11 · Part 3: pre-geometry, ranking loss result (1 min) / 第 3 部分：pre-geometry，排序项的单独效果（1 分钟）

Ranking only, same threshold: at 0.90, 14.5 → 16.3% (+1.8); at 0.95, 17.9 → 20.5% (+2.6). PerfectReco 9.5 → 11.1%, 11.7 → 13.6%.
只加排序项、阈值不变：0.90 下 14.5 → 16.3%（+1.8）；0.95 下 17.9 → 20.5%（+2.6）。PerfectReco 9.5 → 11.1%，11.7 → 13.6%。

This is the honest size of the ranking term. The old "14.5 → 22.6%" step was a bundle: the working point (0.90 → 0.95), removing two harmful modules, derived features, and checkpoint selection — none of those is this loss.
这才是排序项的真实大小。之前说的 14.5→22.6 是打包：工作点 0.9→0.95、删两个负项、派生特征、选 checkpoint，都不是这个 loss。

---

---

---

## 12 · Part 3: vertex geometry, result (1 min) / 第 3 部分：vertex geometry，结果（1 分钟）

This step adds three edge columns: DOCA/100, log(DOCA), |Δstart|/100.
这一步加了三列边特征：DOCA/100、log(DOCA)、|Δstart|/100。

DOCA and log(DOCA) are recomputed: the stored column has the sign bug and correlates with the correct value at only 0.019 — it is noise. |Δstart| is a genuinely new absolute feature; the old derived_vertex only had d_perp/(|Δz|+1), which divided the scale out.
DOCA 和 log(DOCA) 是重算的：上游存的那一列有符号 bug，和正确值相关只有 0.019，是噪声。|Δstart| 是新加的绝对量；旧的 derived_vertex 只给了 d_perp/(|Δz|+1)，把尺度除掉了。

Together: AllParticles 22.6 → 26.0% (+3.4), PerfectReco 14.8 → 17.8%.
三列一起进：AllParticles 22.6 → 26.0%（+3.4），PerfectReco 14.8 → 17.8%。

Do not attribute the +3.4 to the sign fix: no archived run replaces that column alone; it enters together with the other two features, and the block cannot be split further. Inside the block we cannot separate them — there was no run with |Δstart| alone. That is why this page is vertex geometry, not DOCA fix.
不要把 +3.4 归给修符号：归档里没有单独替换该列的运行，它和另外两列一起进入，块内部无法再拆。块内部没法再拆——没有只加 |Δstart| 的单独跑。所以这页叫 vertex geometry，不叫 DOCA fix。

---

---

Two facts belong here. The stored column correlates with the correct value at 0.019, and 99.5\% of reversed pairs differ. No archived run replaces the column alone, so the effect of the fix cannot be separated from the other two inputs; this is why there is no separate DOCA-fix page.
这里补两点事实：上游那一列与正确值的相关只有 0.019，99.5% 的反向对不相等；归档里没有单独替换该列的运行，所以修复本身的效果无法与另外两列分开——因此不再单列 “DOCA fix” 一页。

---

## 13 · Part 3: neighbourhood features, result (1 min) / 第 3 部分：neighbourhood features，结果（1 分钟）

At 0.90: background 4965 → 4177 (−16%), signal edges 649 → 629 (−3%), purity 11.6% → 13.1%, AllParticles 26.0 → 27.0% (+1.0 pp).
0.90：背景 4965 → 4177（−16%），信号边 649 → 629（−3%），purity 11.6% → 13.1%，AllParticles 26.0 → 27.0%（+1.0 pp）。

At 0.97: background 517 → 420, signal edges 471 → 448, purity 47.7% → 51.6%. At 0.99: 135 → 121, 274 → 241, purity 67.0% → 66.6%.
0.97：背景 517 → 420，信号边 471 → 448，purity 47.7% → 51.6%。0.99：135 → 121，274 → 241，purity 67.0% → 66.6%。

Global average precision (AP) falls 0.325 → 0.314: the gain is on look-alike edges, not in the global ranking.
全局平均精度 AP 0.325 → 0.314：增益在长得很像的候选上，不在全局排序。

---

---

## 14 · Part 3: geometry-aware attention, result (1 min) / 第 3 部分：geometry-aware attention，结果（1 分钟）

Same 200 events and the same 0.90 working point as the two earlier benchmark pages (12 and 14).
同样 200 事件、同一个 0.90 工作点，和前两个 benchmark 页（第 12、14 页）用同一张表。

Background 4177 → 3607 (−14%); signal edges 629 → 641 (unchanged); purity 13.1% → 15.1%. At 0.99, purity 66.6% → 74.8%.
保留背景 4177 → 3607（−14%），信号边 629 → 641（基本不变），purity 13.1% → 15.1%。0.99 处 purity 66.6% → 74.8%。

AllParticles 27.0 → 28.0% (+1.0 pp). Note: attention alone (v637) is 21.8 → 21.6, −0.2; with a weak geometric bias (dR / ΔIP / dzp) it is worse, 21.3, −0.5. The +1.0 appears only when vertex geometry (DOCA / logDOCA / |Δstart|) enters the attention score. There is no bias-only control, so the accurate statement is: geometry entering the score helps; attention is the carrier, not the source.
AllParticles 27.0 → 28.0%（+1.0 pp）。注意：attention 单独上（v637）是 21.8 → 21.6，−0.2；加一个弱几何 bias（dR / ΔIP / dzp）更差，21.3，−0.5。这 +1.0 只有喂顶点几何（DOCA / logDOCA / |Δstart|）进 attention 分数时才出现；而且没有“只加 bias、不加 attention”的对照，所以准确说法是：几何进入分数有用，attention 只是载体，不能说 attention 本身有用。

---

---

## 15 · Summary: results and next steps (1 min) / 总结：结果与下一步（1 分钟）

Baseline 14.5%. The largest single step is the working point, 0.90 → 0.95: +3.4 pp with no retraining. Ranking loss +1.8; corrected vertex geometry +3.4; neighbourhood features +1.0; geometry-aware attention +1.0. Two modules we tried were harmful: event-count head −2.8, PV-overlap injection −0.4.
Baseline 14.5%。最大的单项是工作点 0.90→0.95：+3.4 pp，零训练。排序项 +1.8；修正后的顶点几何 +3.4；邻域特征 +1.0；geometry-aware attention +1.0。两个试过的模块是负的：事件计数头 −2.8，PV 重叠 −0.4。

Edge purity at 0.90 rises from 5.8% to 15.1%.
0.90 这个切能用的原因：该切处 edge purity 从 5.8% 升到 15.1%。

Next: object-level assignment with exclusivity and a dustbin; judge on channel / topology, not one global edge AUC.
下一步：逐径迹归属 + 排他 + 允许拒绝背景；按 channel / topology 评价，而不是一个全局 edge AUC。

Overall: AllParticles 14.5 → 28.0%, PerfectReco 9.5 → 19.0%.
总体：AllParticles 14.5 → 28.0%，PerfectReco 9.5 → 19.0%。

---

---

## 16 · Appendix A1: ranking loss, principle (not presented) / 附录 A1：排序项的原理（不讲）

A signal edge is a track pair that exists in the MC-truth decay chain and carries label 1/2/3; background pairs carry label 0. This is a label, not a prediction — nothing truth-based enters the features.
signal edge 指 MC 真值衰变链里真实存在、标签为 1/2/3 的径迹对；背景对标签为 0。它是标签，不是预测——特征里不含任何真值信息。

Only about 0.14 percent of candidates are signal edges, so we used a positive weight of about 700.
候选边里只有约 0.14% 是 signal edge，所以我们用了约 700 的正类权重。

The consequence is that BCE spends almost all gradient lifting signal edges, and never constrains their ordering relative to a look-alike background edge.
后果是 BCE 几乎把所有梯度都花在抬高 signal edge 的绝对分数上，从不约束它相对一条“长得很像”的背景边的次序。

This is one extra term in the same loss — not an extra training stage, and the dataset is unchanged. Concretely, L = L_BCE + λ L_rank with λ = 10; network, data and schedule are untouched.
这是同一个损失里多加的一项——不是额外的训练阶段，数据集也没有变。写出来就是 L = L_BCE + λ L_rank（λ = 10）；网络、数据、训练流程都没动。

At inference we threshold, so what matters near the cut is exactly that ordering. So we added a pairwise term: compare a signal edge with a fake edge that shares one track.
但推理时我们是卡阈值的，阈值附近真正重要的正是这个次序。所以我们加了逐对项：拿一条 signal edge 和一条共享一条径迹的 fake edge 比较。

The fake is built in three steps: take a signal edge (i,j); keep i and replace j with another candidate k from the same event; require (i,k) to be absent from the signal chain.
fake 的构造分三步：取一条 signal edge (i,j)；保留 i，把 j 换成同一事件里的另一条候选 k；要求 (i,k) 不在信号链里。

The two edges share track i, so the model cannot decide from that track alone; it must compare j with k.
两条边共享径迹 i，所以模型不能只靠 i 判断，必须比较 j 和 k。

In the formula, s is the logit the edge-pruning head gives that edge (higher = more likely kept); s_truth is the score of the signal edge, s_fake that of the constructed fake, and τ is a temperature.
公式里的 s 就是剪枝头给这条边输出的 logit（越大越可能保留）；s_truth 是信号边的分数，s_fake 是构造出的假边的分数，τ 是温度。

---

---

---

## 17 · Appendix A2: neighbourhood features, the idea (not presented) / 附录 A2：邻域特征的动机（不讲）

The starting point: an edge should see not only itself, but also the other edges attached to either track.
逻辑起点：一条 edge 不仅看到自己，还要看到两端 track 相连的其余 edge。

The original 12 edge features describe only the relation between i and j (ΔR, Δz, DOCA, pruning score, and so on); they do not say what else i or j is connected to.
原来的 12 维边特征只描述 i 和 j 之间的关系（ΔR、Δz、DOCA、剪枝分等），不知道 i、j 各自还连着哪些候选。

An edge has two endpoints, so each quantity is evaluated at i and at j, and we keep the larger and the smaller end. The input length is fixed, and everything is computed from the inputs alone.
一条 edge 有两个端点，所以每个量在 i 和 j 各算一次，再保留较大端和较小端。输入长度固定，全部从输入算出。

---

---

## 18 · Appendix A3: neighbourhood features, the two quantities (not presented) / 附录 A3：邻域特征的两个量（不讲）

Fix e=(i,j). Each endpoint t has a candidate set C(t), the edges incident to t; e belongs to both C(i) and C(j).
先固定 e=(i,j)。每个端点 t 有候选集合 C(t)，即所有与 t 相连的候选边；e 同时属于 C(i) 和 C(j)。

A = exp(−d3/50): A_f is the value for candidate f, A_e for e itself; A = 1 at d3 = 0, A = 0.5 at d3 ≈ 35 mm.
A = exp(−d3/50)：A_f 是候选 f 的值，A_e 是 e 自己的值；d3 = 0 时 A = 1，d3 ≈ 35 mm 时 A = 0.5。

r_t = A_e / max_{f∈C(t)} A_f: best partner (1 = best; 0.5 when another candidate has A twice as large, i.e. d₃ smaller by about 35 mm).
r_t = A_e / max A_f：最优伙伴（1 = 最优；0.5 = 另一个候选的 A 是它的两倍，即 d₃ 小约 35 mm）。

d_t = count{f∈C(t) : A_f > 0.5} / 10: number of near-degenerate candidates (absolute threshold; e is in C(t) but only counts when A_e > 0.5).
d_t = count{f∈C(t) : A_f > 0.5} / 10：近简并候选数（绝对阈值；e 属于 C(t)，但只有 A_e > 0.5 才计入）。

Two quantities × two endpoints = four numbers; keeping the larger and the smaller end makes them symmetric under i ↔ j.
两个量 × 两个端点 = 四个数；取较大端和较小端，保证交换 i、j 时对称。

12 + 4 + 2 = 18. The last two slots are the candidate count at each endpoint; on a fully connected graph they are constant and carry no information — kept as the interface for a future pruned graph. New first-layer columns start at zero, so adding the neighbourhood features starts from the previous model's function.
12 + 4 + 2 = 18。最后两维是两端候选数；全连通图上恒定、不提供信息，只是给未来剪枝图留接口。新增第一层列初始为 0，所以加上邻域特征时是从上一个模型的函数出发。

---

---

## 19 · Appendix A4: line graph (not presented) / 附录 A4：line graph（不讲）

The previous step gave each edge a hand-made summary of its neighbours — a maximum and a minimum. But we would rather let edges sense each other systematically and learn whom to listen to. That is attention.
上一步给每条边配了一份手工的邻居摘要（最大值、最小值）。但更理想的是让 edge 之间系统性地互相感知、自己学该听谁的——这就是注意力。

Attention needs its objects to be nodes. Today the nodes are tracks, not edges, so the first step is to turn the track graph into a graph of pairs.
注意力要求被注意的对象是节点。现在节点是 track 而不是 edge，所以第一步是把以 track 为节点的图转换成以 edge 为节点的图。

The event graph has 4 tracks and 5 candidate pairs; t1 alone is incident to e1, e2 and e3.
左图是事件图：4 条径迹、5 个候选对；t1 一条径迹就接着 e1、e2、e3。

In the line graph every pair is a node, and two pairs are connected if they share a track. Colour = the shared track.
line graph 里每个候选对是节点；共享一条径迹就连边。颜色表示共享的径迹。

So e1's neighbours are the pairs it competes with; e5 shares no track and is masked.
所以 e1 的邻居就是它的竞争对手；e5 不共享径迹，会被 mask。

---

---

## 20 · Appendix A5: what changes in the network (not presented) / 附录 A5：结构上改了什么（不讲）

Left: h_e → pruning MLP → score. Right: h_e → line-graph attention (4 heads × 8-d, one round) → residual h_e + Δ → same pruning MLP.
左边：h_e → 剪枝 MLP → 分数。右边：h_e → line-graph attention（4 头 × 8 维、1 轮）→ 残差 h_e + Δ → 同一个剪枝 MLP。

Dimension chain: input features 18 → 16-d hidden h_e in the backbone → 8 per head → 4 heads = 32 → W_o back to 16 → residual.
维度链条：输入特征 18 → backbone 的 16 维隐藏态 h_e → 每头 8 → 4 头 32 → W_o 回 16 → 残差。

Only the attention block is new.
只有 attention 这一块是新的。

---

---

## 21 · Appendix A6: standard attention and our slots (not presented) / 附录 A6：标准 attention 与我们填的槽（不讲）

This is the classic picture of scaled dot-product attention: Vaswani et al., "Attention Is All You Need", NeurIPS 2017, Fig. 2 (left). Read it from the bottom up: Q·K, divide by √d, optional mask, softmax, then multiply by V.
这张图是 scaled dot-product attention 的经典画法：Vaswani et al., "Attention Is All You Need", NeurIPS 2017, Fig. 2 左。从下往上看：Q·K、除以 √d、可选 mask、softmax、再乘 V。

We only fill a few slots: Q = h_e (receiver) and K = h_f (neighbour), 16-d → 8-d per head; divide by √8, plus a geometric bias b_ef (expanded on slide 20); mask everything outside N(e); softmax over N(e); V = h_f, four heads → 32-d, W_o back to 16-d, added to h_e.
我们只填几个槽：Q = h_e（接收边）、K = h_f（邻居），16 维 → 每头 8 维；除以 √8，再加几何 bias b_ef（第 20 页展开）；把 N(e) 之外的挡掉；只在 N(e) 上 softmax；V = h_f，四个头拼 32 维，W_o 回 16 维，加回 h_e。

So only three choices are ours: which vectors to project, who may attend to whom, and one geometric term. The rest is standard.
所以真正属于我们的只有三件事：投影什么、谁能注意谁、多一个几何项。其余都是标准件。

---

---

## 22 · Appendix A7: per-head computation and the geometric bias (not presented) / 附录 A7：逐头计算与几何项（不讲）

Our sizes are small: 4 heads × 8 dims, one round, applied after the last GN block and before the pruning MLP.
我们的规模很小：4 头 × 8 维、只有 1 轮，放在最后一个 GN block 之后、边剪枝 MLP 之前。

Per head: h_e is 16-d; q_e, k_f, v_f are 8-d. Score s_ef = q_e·k_f/√8 + b_ef; softmax only over N(e); then c_e = Σ α v, four heads → 32, W_o back to 16, add residual.
单个头：h_e 是 16 维；q_e、k_f、v_f 都是 8 维。打分 s_ef = q_e·k_f/√8 + b_ef；只在 N(e) 上 softmax；然后 c_e = Σ α v，四个头拼 32 维，W_o 回 16 维，加残差。

b_ef is the geometric term: every edge has b_e = (ΔR, Δz, DOCA); for a neighbour pair (e, f) we concatenate [b_e, b_f, |b_e−b_f|, b_e·b_f] and pass it through a small MLP to get one bias per head. It is added to s_ef before the softmax — a dot product knows nothing about geometry, and inside the score it directly changes the weights.
b_ef 是几何项：每条边有 b_e = (ΔR, Δz, DOCA)；对邻居对 (e, f) 拼 [b_e, b_f, |b_e−b_f|, b_e·b_f]，过一个小 MLP，得到每头一个 bias。它加在 s_ef 上、softmax 之前——点积本身不懂几何，放进分数里才能直接改变权重分配。

N(e) = pairs sharing one of e's two tracks, in the same event, excluding e itself.
N(e) = 与 e 共享一条径迹、同一事件里的其它候选对，不含 e 自己。

In the example, e1 receives only e2, e3, e4 (gold); e5 shares no track and is masked (red dashed).
例子里 e1 只接收 e2、e3、e4（金色）；e5 不共享径迹，红色虚线、被 mask。

Numbers for e1, one head: s = 2.0, 1.1, 0.3; exp gives 7.39, 3.00, 1.35; sum 11.74; α = (0.63, 0.26, 0.11), exactly 1.00. e5 gets exactly 0, not a small weight.
e1 的单头数值：s = 2.0、1.1、0.3；exp 后 7.39、3.00、1.35；和 11.74；α = (0.63, 0.26, 0.11)，正好 1.00。e5 的权重是严格的 0，不是一个小数。

Then c1 = 0.63 v2 + 0.26 v3 + 0.11 v4, h1' = h1 + W_o(c1). One round updates every edge once.
于是 c1 = 0.63 v2 + 0.26 v3 + 0.11 v4，h1' = h1 + W_o(c1)。一轮就是每条边更新一次。

---

---

## 23 · Appendix A8: edge ROC (not presented) / 附录 A8：edge ROC（不讲）

This is the same set of versions, now as an ROC of the raw edge score, on the same 200 events.
同一批版本，画成原始边分数的 ROC，同样 200 个事件。

The baseline has the best global AUC, 0.970, and the lowest chain yield, 14.5%; M1 + M2 + M3 reaches 28.0% with AUC 0.953.
baseline 的全局 AUC 最好（0.970），但链产出最低（14.5%）；M1 + M2 + M3 到 28.0%，AUC 是 0.953。

ROC averages over all thresholds, while reconstruction uses one cut. From M1 on, what improves is the decision and the assembly, not the ranking.
ROC 对所有阈值做平均，而重建只用一个阈值；从 M1 开始，真正改善的是决策和组装，不是排序。

And ROC is a global judgement. It averages over every edge and every threshold: one hard, near-degenerate pair is a single point, diluted by thousands of easy background pairs. So a high AUC says nothing about the few decisions that actually break a chain — which is exactly what reconstruction depends on.
而且 ROC 是全局判断。它把所有边、所有阈值一起平均：一条极难的简并候选对只是一个点，会被成千上万条"一眼假"的背景边稀释掉。所以高 AUC 说明不了模型在少数真正切断链的决定上好不好——而重建恰恰只依赖这些决定。

---

---

---

## 24 · Appendix A9: metric definitions (not presented) / 附录 A9：指标定义（不讲）

This is the backup page with metric definitions, including the two denominators.
这是备份页，指标定义，包括两套分母。

---

# Q&A / 问答准备

**Q1: Why are your numbers different from the paper?**
**问 1：为什么你的数字和论文不一样？**

The denominators are different. The paper uses the chains that survive pruning; we use all 13,255 signal chains.
分母不同。论文用剪枝后幸存的链；我们用全部 13,255 条信号链。

For the final model (M1 + M2 + M3), that is 60.9 percent versus 28.0 percent.
对最终模型（M1 + M2 + M3），就是 60.9% 对 28.0%。

**Q2: Is the PV gain from retraining or from the head?**
**问 2：PV 提升来自重训还是来自那个头？**

They were introduced together, so we cannot separate them on this sample.
两者同时引入，在这个样本上无法分离。

We do not attribute all five points to the head.
我们不把 5 个点全部归给头。

**Q3: AUC went down — is the model worse?**
**问 3：AUC 降了，模型是不是变差了？**

AUC is a global average: it averages over all edges and all thresholds, so one hard, near-degenerate pair is a single point diluted by thousands of easy background pairs. A high AUC says nothing about the few decisions that actually break a chain.
AUC 是全局平均：它把所有边、所有阈值一起平均，一条极难的简并候选对只是一个点，被成千上万条"一眼假"的背景边稀释。所以 AUC 高说明不了模型在少数真正切断链的决定上好不好。

The baseline has the highest AUC (0.970) but the worst reconstruction (14.5%), so the two can diverge.
baseline 的 AUC 最高但重建最差，说明两者可以背离。

**Q4: Why not keep adding GNN capacity?**
**问 4：为什么不继续加 GNN？**

The previous round already saturated at 22.5 to 36.7 percent on the 0702 fixed denominator, and this round shows that score-only changes without a new decoder stay in the 18 to 28 percent range.
上一轮在 0702 固定分母下已经在 22.5% 到 36.7% 饱和；这一轮也显示，只改分数、不换解码器，重建落在 18% 到 28%。

**Q5: What is next?**
**问 5：下一步做什么？**

Change the output format: per-track assignment with exclusivity and a dustbin, and judge it on the channel and topology we care about instead of one global edge AUC.
换输出形式：逐径迹归属、排他分配、允许拒绝背景；并按我们关心的 channel / topology 评价，而不是一个全局 edge AUC。
