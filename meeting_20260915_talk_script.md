# DFEI Group Meeting — Speech Script (2026-09-15 talk)

> 22 slides · ~24 min · written to be spoken, not read · [brackets] = action or pause cue · **bold** = the sentence you must not forget
>
> This script goes with `DFEI_meeting_20260915_EN.pptx`. Part 0 is the pipeline walkthrough (including the four edge classes, S6); Parts 1–3 each carry a strip on the right showing which pipeline block the change acts on; Part 4 is deliberately short — directions, not versions.

---

## S1 — Title (10 s)

"Hi everyone. Today I want to give you a clean version of the story — how the model works, what we changed in it, and what we learned when the changes stopped working. I've cut the version-by-version detail; if you want it, the long deck exists and I'm happy to go through it offline."

## S2 — Data and the overall effect (1 min)

"Everything here is the official CERN Monte Carlo production, DFEI_IFT_20260702, evaluated at pruning threshold 0.9 on twenty test files.

The headline: over the line from v31 to v47, PerfectReco went from 12.30 to 23.15 percent, and AllParticles from 22.49 to 39.56. Both are fractions of the same 17 561 truth B candidates in the sample — I'll define both properly in a minute.

One number to keep in mind: the run-to-run spread of this protocol is plus or minus 0.43 percent. **Anything below about half a point is not a result.** I'll use that as the bar for the rest of the talk."

## S3 — Outline (30 s)

"The plan. Part 0 is new and it's the one I'd ask you to pay attention to: how the model actually works, stage by stage. Then Part 1 and Part 2 are the two changes that built the main line — differentiable pruning, and rewarding the chain. Part 3 is the physics heads, which is where the last real gain came from. Part 4 is short: the five directions we tried after that, and why none of them worked.

One thing to notice as we go: every slide in Parts 1 to 3 has a small strip on the right like the one below, with the pipeline block that the change acts on highlighted. So you always know where we are in the model."

## S4 — Part 0: how the model works (1.5 min)

"Right — the model. DFEI is a two-stage algorithm, and almost everything in this talk is easier to follow if you hold these two stages in your head.

One event becomes one graph. The nodes are tracks and primary vertices; the edges are track pairs. Every edge carries a truth class: class zero is background, classes one, two and three are structural — the edges that belong to a decay chain.

Stage one prunes that graph with learned scores: two binary heads, one for the tracks and one for the track pairs. Stage two takes whatever survived and rebuilds the decay chain from it — a four-class edge classification, and then chain assembly.

And the thing we score is whole chains, not single edges. That sounds like a detail; it isn't. **The metric rewards whole chains, while the loss is computed edge by edge — and that mismatch is the source of most of the difficulty in this talk.**

Keep this picture in mind. The rest of the talk is just about where in it each change acts."

## S5 — Part 0: from one event to a graph (1 min)

"A bit more on the input. Only tracks become nodes — the B and the J/ψ decay before the tracker, so they are never tracks themselves; the primary vertices are nodes as well. Every track pair becomes an edge, and each node and edge carries a feature vector — kinematics, impact parameters, detector information.

The truth chain from the Monte Carlo marks which tracks and which track pairs belong to the B decay we want. On the right you can see it: the three blue tracks are the chain, the grey ones are background. The blue lines are their pairs.

And the chain is a handful of tracks inside a very dense background graph. That density is the whole reason the problem is hard — remember the class distribution I'll show you in Part 2."

## S6 — Part 0: the four edge classes (1 min)

"Before we go on, the four classes — because everything downstream is phrased in terms of them.

For every pair of tracks in the event, the network has to answer one question: what is the family relationship between these two particles? It answers with four numbers — p0 to p3, from a softmax over the classes — and the largest one is the answer. The four panels on the right are the four possible relations.

Class two is the one to remember: two tracks are *sisters* when they come from the same mother — the two muons from the J/ψ in the example. A sister pair is the signature of a decay vertex, and those are the edges that glue a chain together.

Class three is *grandparent–grandchild*: the same chain, two steps apart — a muon and the kaon, whose common ancestor is the B.

Class one is *parent–child*, where one track is the direct mother of the other. It sounds important, but it is almost empty in practice: the mother usually decays before it leaves a track of its own. That's why we say classes one, two and three are 'structural', but only two and three really carry the chain.

And class zero is everything else — no common ancestor in this event. **That's 99.9 percent of all edges, and that number is the whole problem in Part 2.**"

## S7 — Part 0: stage 1, pruning (1.5 min)

"Stage one is pruning, and it is done by two binary heads. A point-prune head scores every track — keep it or drop it. An edge-prune head does the same for every track pair. Both scores are a weight w between zero and one, learned against the Monte Carlo truth, not hand-tuned.

At inference we apply a hard cut: only nodes and edges with w above 0.9 survive.

Let me be precise about what is *not* here, because it is easy to misread the diagram: the four-class LCAG classification is not part of pruning. That head exists, it reads the same representation — but it acts one step later, on the edges that survived. **Everything downstream sees only what survived this cut, so a mistake here cannot be undone later in the pipeline.**"

## S8 — Part 0: stage 2, reconstruction (1.5 min)

"Stage two, reconstruction. On the edges that survived, the LCAG head assigns one of the four classes from the previous slide to every pair — background, or one of the structural relations. Those classes are what chain assembly uses — it joins tracks while the lowest common ancestor stays consistent, and an edge classified as background cannot glue two tracks together.

The important detail is that both stages read the same sixteen-dimensional node and edge representation, produced by the same network blocks, and every message carries the learned weight w. So pruning is inside the network, not just in front of it.

And the candidates that come out of the assembly are compared with the truth chains, and that is where the metrics come from.

So: **stage two can only work with what stage one let through.** Hold on to that sentence — it comes back in Part 4, when we measure what the pruning is actually worth."

## S9 — Part 0: how we score it (1 min)

"Two flags, both counted per truth B candidate, and both expressed as a fraction of the same 17 561 candidates.

AllParticles asks a content question: are all the truth particles of this chain inside one recovered chain — and is that chain free of anything else? The particle list has to match exactly, one for one.

PerfectReco asks the stricter question: the content matches, *and* the connections match — the same tree, rebuilt exactly. So every perfect candidate is also an all-particles candidate, and PerfectReco is always the smaller number.

A candidate that is neither — a partial chain, or a chain that also contains background tracks — counts as a failure for both.

And the noise floor: four functionally identical runs spread over plus or minus 0.43 percent, or about 75 candidates. That's the resolution of the measurement."

## S10 — Part 1: differentiable pruning with annealing (2 min)

"Part one. This is where the line starts moving.

The problem: during training the model sees the full graph, but at inference we prune with a hard cut first and only then reconstruct. So training and inference see different graphs.

The fix is a soft mask. The effective message weight becomes the weight times a sigmoid of weight-minus-cut over a temperature tau. Tau is annealed from 1.0 down to 0.1. At high tau the mask is smooth and gradients are stable; as tau goes down, the mask converges to the hard cut. Look at the left panel: at tau one, the transition is gradual; at tau 0.1 it's essentially a step. The right panel shows what a message actually carries.

And the cut itself was tightened along the line: 0.5, then 0.7, then 0.85 — walking towards the 0.9 we use at inference.

The result: v31 to v36 gives PerfectReco 12.30 to 18.77 and AllParticles 22.49 to 35.17 — the largest single step before the mass head.

Note what kind of change this is: **training only. Nothing about the architecture changes, and inference still applies the same hard cut.**"

## S11 — Part 2: why chains die, one wrong edge (1 min)

"Now part two, and this is the part I find most interesting.

First, the mechanism. At inference, an edge that the classifier calls class zero gets pruned. So if a single structural edge of a chain is misclassified, the chain cannot be assembled any more — it's gone, even if every other edge was right.

The two trees on the right show it. Above: one edge called background, the chain dies. Below: with chain supervision, that same edge is kept and the chain survives.

So the accuracy that matters is the accuracy on the edges that carry a chain. And those are exactly the rarest edges in the event."

## S12 — Part 2: the class imbalance (1 min)

"Which brings us to the distribution. Class zero, background, is 99.9 percent of all edges. Each structural class is about 0.04 percent. That's roughly a thousand background edges for every structural edge.

And the main LCAG loss is a global cross-entropy over every edge — so almost all of its gradient is saying 'predict background'.

The bottom plot is per-class accuracy, and it collapses exactly where it hurts: the worst class is class two, the sister edges, at about 49 percent. **The thing we need the most is the thing the model learns the worst.**

So the structural classes have to be supervised directly. That's what the two rewards do."

## S13 — Part 2: reward 1, the hinge (v37) (1.5 min)

"Reward one, the hinge.

The idea: a training-only term that behaves like a bonus. Be confident on a chain edge, and you pay nothing. Be unsure, and you pay.

Where does confidence come from? The LCAG head gives a four-class softmax per edge — p0 to p3 — and confidence is the maximum of those: how sure the model is about the class it picked. Careful, this is not the pruning weight w. Different quantity, easy to mix up.

The penalty is max of zero and margin minus confidence, with margin 0.3 — so above 0.3 confidence nothing is paid at all. And it's paid only on truth-chain edges, which we know from the Monte Carlo during training. That's legitimate: it's a training-time target, not an input.

Result: v36 to v37 gives 18.77 to 19.50. Modest. And class-two accuracy actually went DOWN — 47.7 to 44.2 — because we had also pushed the class-two weight to 3.0 and over-corrected. So v38 puts that back."

## S14 — Part 2: reward 2, chain cross-entropy (v38) (1.5 min)

"Reward two, and this is the one that works properly.

Cross-entropy, minus log p_true: it pays for being the RIGHT class, not just for being confident. The hinge only asked for confidence; this asks for correctness.

It's paid only on truth-chain edges — classes one, two and three — so its gradient is not diluted by the thousand background edges. Same function as the global cross-entropy; the difference is where it is paid. That's literally the line under the diagram.

v38 also brings the class-two weight back to 2.0 and tightens the cut to 0.85.

Result: 19.50 to 21.10. And per-class accuracy finally moves: class one from 67.8 to 76.8, class two from 41.3 to 47.9."

## S15 — Part 2: the strategy works (1 min)

"Here's the whole line in one picture. v31 to v38: PerfectReco 12.30 to 21.10, AllParticles 22.49 to 37.59 — about nine points on the strict metric.

And the important part is *which* classes moved: the two structural classes the global loss was ignoring. That's the mechanism, not a coincidence.

Still a training-only change. Inference is untouched."

## S16 — Part 3: first, a probe (1.5 min)

"Part three: the physics heads. And I want to start with the check that told us they were needed.

The method is a linear probe: freeze the backbone, fit a single linear layer on top, and try to read a physical quantity out of the representation. If the quantity is genuinely encoded, a linear map should find it. If it isn't there, the probe fails and R-squared is about zero.

Before any physics head: the pi-pi mass probe gives R-squared 0.003, and momentum is essentially zero. The backbone does not know the physics.

After the mass head: edge mass R-squared is 0.930.

**So physics is not a free by-product of reconstruction — it has to be supervised into the representation.** That's the case for the heads."

## S17 — Part 3: the head zoo (1.5 min)

"Which heads. One backbone, nine supervised targets.

Five of them are the original DFEI machinery — LCAG, node prune, edge prune, PV association, chain scorer.

Four are ours, and they're all label-free from the Monte Carlo: the source head from v36, which predicts the chain root; the mass head; the struct head; and the momentum head.

They all read the same sixteen-dimensional representation, and they're parallel — every arrow starts at the backbone. **Adding a head never changes the network; it only adds a supervised target.**"

## S18 — Part 3: the mass head (1.5 min)

"This is the one head that paid.

The target is the invariant mass of the track pair on each track-track edge, taken from the Monte Carlo truth.

The first attempt, v46, regressed the raw mass and landed at 21.28 — inside the noise. Not a failure of the idea, a failure of the parameterisation.

v47 adds a log-normalisation and masks the sentinel edges, and that's where it works: v38 to v47 gives PerfectReco 21.10 to 23.15 and AllParticles 37.59 to 39.56.

And the honest control: the identical recipe on the source, struct and momentum heads is a wash. So it's not 'more supervision is better' — it's this target, on this representation."

## S19 — Part 3: the struct head (1 min)

"The struct head is the interesting negative result.

Two targets per node, both from the truth tree: the depth — how far the node sits from the chain root, counted in BFS layers — and the normalised Rumor Centrality, which is the 'is this the source of the chain?' score. Both are computed from the truth tree, so no manual labelling.

On the weak ablation base this was the best single addition. On the main line it's a wash — and when we stacked three heads at full weight, v48 lost about five points of AllParticles.

**That's the observation that leads into Part 4: on this representation, more supervision stops adding information.**"

## S20 — Part 4: what came after v47 (1.5 min)

"So, briefly — what we tried after v47. Five directions, in parallel. I'll give you one line each, no versions.

Capacity: widen the latent space, more dimensions, bigger hidden layers. Result: 24 to 27 percent AllParticles, against 37.59. Much worse.

Context: track-level self-attention and edge-feature biases. It helps — but only when you graft it onto a weak base; on the converged model it does nothing.

Restructuring the event: train on subgraphs, split by primary vertex. It made the full-graph pass worse.

Retraining the recipe: we re-ran the whole optimisation chain from a common base, one change per run. Only chain-CE clears the noise floor; everything else is flat.

And another dataset: the public sample. Copying the CERN recipe over gave negative transfer, and rebuilding it layer by layer recovered, but it's a different problem.

**The pattern is the same in all five: adding to the network does not help, and combining the few things that do help also does not help.**"

## S21 — Part 4: the diagnosis, and what runs now (1.5 min)

"So if the network isn't the bottleneck, what is? We measured it.

The oracle test: take v47, freeze it completely, and rewrite the pruning decisions with the Monte Carlo truth. Same weights, same network — only the pruning changes. PerfectReco jumps to 57.8, and AllParticles to 99.75 percent, against 39.56 baseline.

Read the ladder from left to right: fixing point recall alone gets you to 54; point precision alone, 54.7; both point axes, 72; the edges alone, 71; everything, 99.75. And the reverse control — adding fake tracks instead of removing wrong ones — collapses to 16.9, so the effect is real and it has the right sign.

**So the score is decided at the pruning step. Everything after it is downstream of a decision that was already made.**

Two concrete causes: the point head was carrying one thirty-third of the loss weight, and the fine-tuning step was ten times too small.

What runs now: the pruning losses rebalanced, a focal loss for precision, and a chain-level recall term — all stacked at a learning rate of 3e-4."

## S22 — Summary and next steps (1 min)

"To wrap up.

What we know: the line from v31 to v47 is real — 12.30 to 23.15 percent PerfectReco, 22.49 to 39.56 AllParticles. Nothing added after v47 beats it. And the reconstruction is not the bottleneck; the pruning classification is.

Next: finish the pruning-reweighting runs and re-test the old verdicts at the higher learning rate. If the point head responds, the next target is precision — hard negatives and chain-level losses. And continue the public line to see whether these levers transfer.

Two questions where I'd genuinely like your input. One: class two, same-source clustering — is that the real problem DFEI should be solving? Two: what time budget would HLT2 realistically leave for a lightweight GNN per event?

Thanks — happy to take questions."

---

## Likely questions and prepared answers

**Q: Why are your numbers different from the ones you showed last time?**
A: The flags themselves are unchanged — the same code decides whether a candidate counts as "all particles" or "perfect". What changed is the denominator. The old number divided by the truth chains that survived pruning in that particular run, and that count moves with the model and with the threshold: the harder you cut, the more signal tracks are deleted, the smaller the denominator, the higher the percentage. We now divide by the number of truth chains on the *unpruned* graph — 17 561, a constant that doesn't depend on the model. So the absolute values are lower, and they're comparable across runs, thresholds and datasets.

**Q: What exactly is the oracle test — isn't that cheating?**
A: It's an upper bound, it's deliberately cheating. We freeze the weights and replace the pruning decisions with Monte Carlo truth. It doesn't produce a usable model; it tells us how much of the score is decided at the pruning step. The answer is: most of it.

**Q: Why does the mass head help reconstruction and not just classification?**
A: Because it forces the edge representation to encode sister relations — class two — and those edges are the glue of a chain. The class-two accuracy gain (41.3 to 47.9) is the mechanism.

**Q: Why not just feed the mass in as an input feature?**
A: Two reasons. It would break the end-to-end claim — the model is supposed to derive it from the momenta. And our verification method is probing: if a quantity matters, the model should hold it internally.

**Q: How can adding heads hurt, if each head helps on its own?**
A: Gradient competition inside a fixed sixteen-dimensional space. The auxiliary losses together exceeded the main reconstruction loss, and the space is saturated. That's exactly what the capacity experiments later confirmed — widening it didn't fix it either, so the representation is not where the loss is.

**Q: Is the pruning cut of 0.9 special?**
A: It's the operating point we chose, and it's fixed for every number in this deck. The training-time cut walks up to 0.85 so that the model sees something close to it, and the annealing makes the mask converge to a hard step.

**Q: Where does the remaining headroom actually sit?**
A: Almost entirely in the point-pruning head. Fixing point recall is worth about fifteen points of AllParticles; fixing precision is worth about fifteen more; the edges are worth about the same on top.

**Q: What is the single most valuable thing you learned?**
A: That a saturated network is not the same thing as a saturated problem. We spent a long time adding capacity and supervision to the reconstruction, and the measurement that changed our direction was simply looking at how much of the score was already decided before the network ran.
