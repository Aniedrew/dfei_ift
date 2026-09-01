# DFEI Concepts Explained (English)

> A complete, beginner-friendly explanation of every concept in the DFEI optimization talk.
> Written so that someone who knows **no machine learning** and **little about LHCb** can follow.
> Each entry: plain definition → everyday analogy → how it appears in DFEI.
> Read in order the first time; use the glossary at the end for quick lookup.

---

## Part 0 — The one-sentence story

DFEI is a computer program that looks at all the tracks (curved lines) left by particles in a
single LHC collision, and tries to figure out which particles came from the same "mother"
particle, and in what order — i.e. it tries to *rebuild the family tree* of each collision.
We spent months making that program work better, and this document explains every technical
word we used while doing so.

---

## Part 1 — Machine learning basics

### 1.1 Machine learning / a "model"
A **model** is a mathematical function with a huge number of adjustable numbers (called
**weights** or **parameters**). We feed it data, it produces an answer, and during
**training** we slowly adjust the numbers so that the answers get closer to the correct
ones. After training, the model has "learned" the pattern.
- Analogy: a recipe with many dials. You cook, taste, and turn the dials until the dish is right.

### 1.2 Training, epoch, batch
- **Training**: the process of adjusting the weights using example data.
- **Epoch**: one full pass over all the training data.
- **Batch**: training is done in small groups of examples at a time (due to memory limits); one pass over a group is a batch step.
- In DFEI: one epoch over ~200 files of collisions, each file containing ~800 events, takes 40–60 minutes.

### 1.3 Loss function
The **loss** is a number that says "how wrong is the current answer?" Small loss = good.
Training is just repeatedly reducing the loss.
- DFEI uses several losses at once (classification loss, pruning loss, physics losses, ...);
  the **combined loss** is their weighted sum.

### 1.4 Gradient and gradient descent
The **gradient** is the direction in which the weights should move to reduce the loss the most.
**Gradient descent** = repeatedly take small steps in that direction. If two different losses
pull the same weights in opposite directions, we get **gradient conflict/competition**
(this is exactly what happened in our "combined heads" failure — see §7.2).

### 1.5 Representation / latent space / embedding
Inside the network, each object (a track, an edge between two tracks) is stored as a list of
numbers called its **representation** (also called **embedding** or **latent vector**).
The set of all possible such lists is the **latent space**. The claim of this project is that
we can force these numbers to *carry physical meaning* (like mass or position in the tree).
- Analogy: a "representation" of a person might be a few numbers like height, age, mood.
  If we train it to include "sibling relationship", the representation becomes more useful.

### 1.6 Supervised learning, label, data-intrinsic target
- **Supervised learning**: training with examples that have known correct answers (**labels**).
- A **data-intrinsic target** is a label that can be computed from the data itself without any
  human annotation. Example in DFEI: the invariant mass of two tracks is computed from the
  tracks' momenta — no extra work needed, so we use it as a free supervision signal.

### 1.7 Classification vs regression
- **Classification**: the answer is one of a few categories (e.g. "is this pair of tracks:
  background / parent-child / sister / grandparent?").
- **Regression**: the answer is a continuous number (e.g. "what is the log10 of the pair mass?").
- DFEI does both: classification for edges, regression for the physics heads.

### 1.8 Cross-entropy (CE), binary cross-entropy (BCE), hinge
Ways to measure error for classification:
- **Cross-entropy (CE)**: compares the model's probability distribution over classes with the
  true class. Lower = more confident and correct.
- **Binary cross-entropy (BCE)**: the same but for "yes/no" questions (e.g. "is this node the chain root?").
- **Hinge loss**: a "margin" style loss — if the model is confident enough on the correct side of a
  boundary, loss is zero; otherwise it pays proportional to how far off it is.
  Used in DFEI to keep chain edges confident (see §4.5).

### 1.9 Smooth L1 loss
A regression loss that is forgiving of small errors (like L1, absolute difference) but smooth
(like L2, squared difference) near zero. Robust against outliers. Used for the mass, structure,
and momentum heads.

### 1.10 Class weight / class imbalance
When one category has far more examples than another (e.g. 99.9% of edges are background), the
model tends to just predict the big category. **Class weights** multiply the loss of each class
so that rare but important classes count more.
- DFEI: the whole project started because a typo (double underscore) made the class weights
  silently disappear, and class 1 (parent–child) accuracy collapsed to ~0%.

### 1.11 Overfitting / underfitting / generalization
- **Overfitting**: memorizing the training data, failing on new data.
- **Underfitting**: too weak to capture the pattern.
- **Generalization**: performance on data never seen during training (measured on the
  **validation** and **test** sets).

### 1.12 Linear probe (PhyIP-style)
A **linear probe** is a verification trick: freeze the trained model, take its representations,
and train only a *simple linear model* to read out some quantity (e.g. the mass). If a linear
model can read it, the information is *linearly accessible* — the network really stored it.
- DFEI result: before mass supervision, reading mass from the edge representation gives R² = 0.003
  (nothing); after supervision R² = 0.93 (excellent). That is our proof that "physics is in the
  representation".
- **R² (R-squared)**: a number from 0 to 1 measuring how well a regression explains the data.
  0 = useless, 1 = perfect.

### 1.13 Multi-task learning, head
A single network with one **backbone** (shared part) and several output branches called
**heads**, each solving its own task. The heads share the backbone but have their own final layers.
- DFEI has 9 heads (see §5).

---

## Part 2 — Graphs and graph neural networks

### 2.1 Graph, node, edge
A **graph** is a set of **nodes** (points) connected by **edges** (lines).
- In DFEI, nodes = tracks (and also vertices), edges = pairs of tracks.

### 2.2 Message passing
The core operation of a GNN: each node gathers information from its neighbors, mixes it with its
own, and updates its representation. Repeating this a few times lets information travel across the
graph. DFEI uses 4 such blocks.

### 2.3 Heterogeneous graph
A graph with *different types* of nodes/edges (tracks vs vertices; track-track edges vs
track-vertex edges). DFEI is a **heterogeneous GNN**.

### 2.4 Edge classification
Predicting a class for each edge. In DFEI each track pair gets one of 4 **LCAG classes** (see §3.6).

### 2.5 Node / edge "weight"
Each node and edge has a number (0–1) predicting "is this signal?" These are used for pruning.
- Don't confuse with training **weights** (the parameters) — in DFEI docs "weights" can mean
  either; context decides. The pruning scores are also called **confidences**.

---

## Part 3 — LHCb / physics background

### 3.1 LHC and the LHCb detector
At the LHC, protons collide; **LHCb** is one of the detectors, specialized in particles
containing a **b (bottom/beauty) quark** — particles called **B hadrons** that decay quickly
into lighter particles.

### 3.2 Event
One proton-proton collision. Each event has ~150 charged tracks (visible in the detector).

### 3.3 Track
The curved path a charged particle leaves in the detector. From its curvature we infer its
**momentum** (three numbers: px, py, pz). DFEI only sees tracks — it must reconstruct the physics
from them.

### 3.4 Primary vertex (PV), secondary vertex
- **Primary vertex (PV)**: the point where the proton beams actually collide. An event can have
  several PVs (pile-up).
- **Secondary vertex**: where a B hadron decays, displaced from the PV by ~1 cm.
- **PV association**: deciding which PV each track belongs to.

### 3.5 Decay chain / mother / daughter / sister
A B hadron decays into daughters, which may themselves decay — a **tree/family structure**.
- **Mother**: the particle that decays.
- **Daughter**: a product of a decay.
- **Sister (same mother)**: two particles from the same decay.
- **Grandparent**: two steps up the tree.
- DFEI's job: rebuild these trees from tracks alone.

### 3.6 LCAG classification (the 4 classes)
**LCAG = Lineage of Common Ancestor Graph** — a way of labeling the *relationship* between each
pair of tracks:
- **class 0**: background pair (no relation — the vast majority).
- **class 1**: parent–child (one track is the direct decay product of the other).
- **class 2**: sister (both from the same mother).
- **class 3**: grandparent–grandchild.
Class 2 (sister) is the *structural bottleneck* — chains are glued together by class-2 edges.

### 3.7 Invariant mass, resonance
The **invariant mass** of a set of particles is the total mass of the hypothetical particle that
produced them, computed from their momenta. When a real particle (like a B or a ρ) decays, the
invariant mass of its daughters sits near the particle's mass — a **resonance peak**.
- DFEI uses the **ππ (pion-pion) invariant mass** of each track pair: if two tracks are sisters
  from the same B, their pair mass clusters at resonance peaks. That is why supervising the pair
  mass teaches the network about sisterhood (class 2).

### 3.8 PID (particle identification)
Information about *what kind* of particle a track is (pion, kaon, muon...). The CERN MC has PID;
the published dataset does not — one reason the datasets differ.

### 3.9 Trigger, HLT2, Upgrade II
- **Trigger**: the fast hardware+software system that decides in real time which collisions to keep.
- **HLT2**: the second, software stage of the LHCb trigger where a full reconstruction is attempted.
- **Upgrade II**: the planned future detector upgrade.
The question "can a GNN run inside HLT2" is really "is it fast enough (per event) and good enough
to assist a real-time decision". This motivates our trigger-related work (chain scoring, §8.4).

### 3.10 Flavor tagging
Determining whether a B meson contains a quark or an anti-quark. Useful for CP-violation
measurements. One open question is whether DFEI's "same-source clusters" could feed a tagger.

---

## Part 4 — The DFEI task, metrics, and the first optimization line

### 4.1 Pruning
**Pruning** = removing nodes/edges the model considers background, before rebuilding chains.
It is the biggest word in the talk, so let's be precise:
- The network assigns a **confidence** (0–1) to every node and every edge.
- **Hard pruning**: keep only nodes/edges with confidence above a threshold (e.g. 0.9), delete the rest.
- **Soft pruning**: instead of deleting, scale the contribution by a smooth function — so training
  can still learn (it is differentiable).
- Why prune at all? The full event has ~150 tracks and thousands of track-pair edges; most are
  background. Rebuilding chains on the pruned graph is much cleaner and faster.
- Why is pruning a *problem*? Training ran on the full graph, but inference prunes first — the
  model never saw pruned graphs during training. That mismatch is the **train–inference gap**.

### 4.2 Train–inference gap
Training conditions differ from inference conditions (e.g. full graph vs pruned graph), so the
model behaves worse at inference than expected. The whole first optimization is about removing this gap.

### 4.3 Differentiable pruning with temperature annealing
Our fix for the gap:
- During training we multiply every weight by a **soft mask**:
  **w_eff = w · σ((w − cut)/τ)**,
  where **σ** is the sigmoid function (an S-shaped curve mapping anything to 0–1),
  **cut** is a threshold, and **τ (tau)** is a **temperature**.
- **Temperature annealing**: start with a large τ (smooth mask, stable gradients) and slowly
  decrease it to 0.1, so the mask gradually becomes hard-like — i.e. the model is trained on
  graphs that increasingly resemble the pruned graphs of inference.
- The cut was tightened across versions: 0.5 → 0.7 → 0.85, converging to the 0.9 inference threshold.

### 4.4 In-chain LCA consistency
A family of auxiliary losses applied **only to edges that lie on true chains** (we know the truth
from Monte Carlo):
- **Hinge**: penalize a chain edge whose confidence is too low (below a margin 0.3).
- **Cross-entropy on chain-edge classes**: directly supervise the class (1/2/3) of structural edges.
  Why needed: structural edges are ~0.1% of all edges; without direct supervision the overwhelming
  class-0 background dilutes their training signal.

### 4.5 Source head and Rumor Centrality
- **Rumor Centrality (RC)**: a graph-theoretic score measuring how likely each node is to be the
  "center/source" of the structure. DFEI uses it at inference to pick the root of a chain.
- **Source head**: a training-time head that predicts "is this node the RC-argmax root of its
  chain?" — so training and inference agree on where the root is.
- Caveat: the root of the *track graph* is the graph centroid, which is not necessarily the B
  meson itself (most tracks are final-state particles, not parents). This motivates the richer
  structure supervision (§6.3).

### 4.6 Metrics: PerfectReco, AllParticles, NoneIso
- **PerfectReco**: fraction of events where *every* chain is recovered exactly. Strict metric.
- **AllParticles**: fraction of truth particles recovered across all events. Softer metric.
- **NoneIso**: fraction of events where no B chain was isolated (so PerfectReco + NoneIso = 100%).
- Threshold 0.9 = the pruning confidence threshold used in all evaluations.

---

## Part 5 — The "head zoo" (what each head does)

| Head | Level | Target | Status |
|---|---|---|---|
| LCAG classification | edge | 4-class relationship | core, from the start |
| node pruning | node | signal vs background | core, from the start |
| edge pruning | edge | signal vs background | core, from the start |
| PV association | node-edge | which PV a track belongs to | core, from the start |
| chain scorer | chain | candidate-chain likelihood | implemented, not trained (needs GPU) |
| source head | node | is this the chain root? | trained (v36+) |
| mass head | edge | log10(m_ππ) | trained → main result |
| struct head | node | depth + RC value | trained, best single head in ablation |
| mom head | node | normalized momentum | trained |

---

## Part 6 — Physics supervision (the core idea of Part 2)

### 6.1 The idea in one paragraph
Instead of computing physics features by hand and feeding them in as inputs, we *force the
network's internal representation to carry physical quantities* by adding small regression heads
supervised with those quantities. The network stays end-to-end: it must learn to derive the
physics from momenta itself. Then we verify with linear probes that the representation really
became physically readable.

### 6.2 Mass head (edge level)
- Target: **log10(m_ππ)** — log scale because masses span orders of magnitude.
- Why it works: sister pairs sit at resonance masses, so the edge representation is pushed to
  encode "sister or not" — which is class 2 — the structural glue.
- **Sentinel edges**: pairs whose momentum is a filler value (px≈py≈pz≈−1); they are masked out of
  the loss so they don't poison training.
- Result: class2 44.7% → 51.1%, PerfectReco +3.4pp.

### 6.3 Structure head (node level)
- Targets: **depth** (BFS distance from the node to the chain centroid — BFS = breadth-first
  search, counting how many steps along the chain graph) and the **RC value**.
- Together they describe *where a node sits in the tree*, which the 1-bit source head could not.
- Ablation: best single head (AllParticles +3.1), also improves class3 (grandparent).

### 6.4 Momentum head (node level)
- Targets: normalized momentum components (px,py,pz).
- Motivation: a probe showed node representations were linearly unreadable for momentum (R²≈0),
  because normalization + ReLU layers in the network scramble that information. We supervise it to
  force the network to keep momentum around.

### 6.5 Ablation study
Change exactly one thing at a time and measure. The combination experiment (all heads together)
degraded, so we ablated to find which single heads help:
- baseline (mass only): All 51.4 / Perfect 29.7
- +struct: All 54.5 / Perfect 30.5
- +mom: All 53.8 / Perfect 30.5

### 6.6 ReLU, graph normalization (mentioned in the talk)
- **ReLU**: a simple activation function — keeps positive numbers, sets negatives to zero. Cheap
  and effective, but it "kills" negative contributions and can scramble continuous physical values.
- **Graph normalization**: re-centering/re-scaling features across a graph. Helps training
  stability but also redistributes information, which is why momentum became hard to read.

---

## Part 7 — The two controlled failures

### 7.1 PV subgraph training
Idea: split each event into per-PV subgraphs (20–30 nodes instead of 91–139) to reduce
cross-chain interference. Failure: the backbone was trained only on subgraphs, but inference runs
the GNN on the full graph first → full-graph ability degraded (class1 76.8% → 56.4%).
Lesson: the training graph and the inference graph must match, or the model breaks at inference.

### 7.2 Combined heads and gradient competition
Adding mass+struct+mom at the same time: auxiliary losses summed to 0.877 vs main task 0.559.
The aux gradients dominated the shared backbone, and reconstruction dropped 5pp — even though
LCAG classification did not degrade (loss vs downstream metric are not monotonically linked).
Fix: lower auxiliary weights (mom 0.2, struct 0.3).
Lesson: in multi-task learning, gradient balance must be explicit.

---

## Part 8 — Ongoing work (Part 4 of the talk)

### 8.1 Latent space dimension and degrees of freedom (DOF)
A representation of dimension d holds at most d independent pieces of information. Physically, a
track has ~12–14 relevant numbers (position, momentum×3, PID, mass-related, ...), and the node
representation must serve 9 heads — so 16 dimensions is near the lower bound. We widened tracks
nodes to 32 and tt edges to 24. (Status: interrupted at epoch 74/150 — inconclusive.)

### 8.2 Monte Carlo (MC) simulation
**MC** = simulated data based on the known physics + a detailed detector model. Used for training
and validation because truth (the correct answer) is known by construction.

### 8.3 Deterministic annealing (DA) clustering
A clustering algorithm with a temperature knob:
- Start "hot": everything is one soft cluster.
- Gradually cool: clusters split automatically when the data really has sub-structure.
- Deterministic: no random sampling (unlike Gumbel-based methods) — reproducible.
- **Learned DA**: the affinity (how likely a track belongs to a vertex) comes from a trained MLP
  instead of hand-crafted geometry. We apply it at inference only (no subgraph training), which
  avoids the v42 failure mode.

### 8.4 Chain scoring and AUC
- **Candidate chain**: a set of nodes proposed as a decay chain.
- **Chain scoring**: assign each candidate a probability of being a true signal chain.
- **AUC (Area Under the ROC Curve)**: a single number (0–1) measuring how well a score separates
  positives from negatives across all thresholds. 0.5 = random, 1 = perfect. Our chain criteria:
  AUC 0.90 (vs easy negatives), 0.78 (vs realistic pruned components).

### 8.5 Public dataset differences
The published dataset has no PID and 4–14× different class-2/3 edge counts vs the CERN MC — so
results are not directly comparable until the pipeline is adapted.

---

## Part 9 — Quick glossary (alphabetical)

- **Ablation** — removing/changing one component to measure its effect.
- **AllParticles** — fraction of truth particles recovered.
- **Annealing (temperature)** — gradually changing a temperature knob to go from smooth to sharp.
- **AUC** — separation quality of a score, 0.5 random → 1 perfect.
- **Backbone** — the shared part of the network before the heads.
- **BCE / CE** — binary / categorical cross-entropy losses for classification.
- **BFS (breadth-first search)** — counting steps outward from a node along a graph.
- **Batch / epoch** — training units: a group of examples / one full pass over the data.
- **Class weight** — weighting a rare class more so the model learns it.
- **Combined loss** — weighted sum of all head losses.
- **Confidence** — the model's 0–1 "is signal" score for a node/edge.
- **Data-intrinsic target** — a label computable from the data itself.
- **DOF (degrees of freedom)** — independent numbers needed to describe a quantity.
- **Edge / node** — a connection / a point in a graph.
- **Embedding / latent / representation** — the internal number-vector describing an object.
- **GNN (graph neural network)** — a network operating on graphs via message passing.
- **Gradient** — the direction to adjust weights to reduce loss.
- **Hard vs soft pruning** — hard delete vs smooth re-scaling.
- **Head** — an output branch of the network solving one task.
- **Heterogeneous graph** — graph with several node/edge types.
- **Hinge loss** — margin-based loss: zero once confident enough.
- **HLT2 / Upgrade II** — the software trigger stage / future detector upgrade at LHCb.
- **Invariant mass / resonance** — mass of a hypothetical parent; peaks indicate real particles.
- **LCAG** — Lineage of Common Ancestor Graph; the 4-class edge labeling.
- **Linear probe** — linear read-out test of whether info is in a representation.
- **Loss** — how wrong the current answer is.
- **Message passing** — GNN's way of propagating information along edges.
- **MC (Monte Carlo)** — simulated data with known truth.
- **Model / weights / parameters** — the function and its adjustable numbers.
- **Mother / daughter / sister / grandparent** — family relations in a decay chain.
- **Multi-task** — training several related tasks jointly.
- **NoneIso** — fraction of events with no isolated B chain.
- **Overfitting / generalization** — memorizing vs performing on new data.
- **PerfectReco** — fraction of events with every chain fully recovered.
- **PID** — particle identification.
- **Primary vertex (PV)** — the collision point.
- **Probe** — see Linear probe.
- **Pruning** — removing low-confidence nodes/edges before reconstruction.
- **R² (R-squared)** — 0–1 regression quality measure.
- **RC (Rumor Centrality)** — graph score of "how central is this node".
- **ReLU / graph normalization** — activation function / feature re-centering.
- **Regression** — predicting a continuous number.
- **Resonance** — see Invariant mass.
- **Sigmoid / softmax** — functions turning numbers into probabilities (0–1).
- **Smooth L1** — robust regression loss.
- **Sister edge (class 2)** — edge between two tracks of the same mother.
- **Source head** — predicts the chain root.
- **Supervised learning / label** — training with known answers.
- **Temperature** — a knob controlling softness (in softmax, annealing, DA).
- **Track** — the path a charged particle leaves in the detector.
- **Train–inference gap** — mismatch between training and inference conditions.
- **Trigger** — the real-time system deciding which collisions to keep.
- **Validation / test set** — held-out data to measure generalization.
- **Vertex** — a decay/collision point.
