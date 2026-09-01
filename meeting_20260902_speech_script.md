# DFEI Group Meeting — Speech Script (spoken English, 2026-09-02)

> 18 slides · ~20 min · natural spoken style, read it as you'd talk · [brackets] = action/pause cues · **bold** = the sentence you must not forget

---

## S1 — Title (10 s)

"Hi everyone. Thanks for having me. Today I'll walk you through what I've been doing with DFEI on the CERN Monte Carlo — a bunch of small optimizations. I've organized them by theme, not by timeline, so it should be easier to follow."

## S2 — Data and overall effect (1 min)

"So, first — the data and the headline numbers. Everything here is on the official CERN Monte Carlo production, DFEI_IFT_20260702, evaluated at pruning threshold 0.9 on twenty test files.

And the short version: after we fixed a silent class-weight bug back in v31, the numbers have been climbing. PerfectReco went from 23.9 to 32.7 percent — almost nine points. AllParticles from 43.4 to 55.9 — twelve points.

This plot shows four metrics across versions — PerfectReco, AllParticles, and the class-one and class-two accuracies. Everything basically goes up. The work I'm about to describe is what sits behind those curves."

## S3 — Outline (30 s)

"So the plan. Part one: differentiable pruning with annealing — aligning training with what actually happens at inference. Part two: rewarding the chain — the class-imbalance problem, and the hinge and cross-entropy fixes. Part three: supervising the representations with physics — I'll start with a probe that shows why it's needed, then the head zoo, then what depth and Rumor Centrality actually mean. Part four: why I think we should widen the latent space. And part five: the other lines still running."

## S4 — Part 1: Differentiable pruning with annealing (2 min)

"Okay, part one. Differentiable pruning.

The problem is basically this. During training, the model sees the full graph. At inference, we prune first — we cut edges with hard thresholds — and only then reconstruct. So training and inference see different graphs. That gap is exactly what this fixes.

The fix is a soft mask on the weights. The effective weight is the weight times a sigmoid of weight-minus-cut over a temperature. The temperature is annealed from 1.0 down to 0.1 during training. At high temperature the mask is smooth, so gradients are stable; as the temperature drops, the mask starts to behave like the hard cut you use at inference.

And the cut itself tightened over versions: 0.5 in v36, 0.7 in v37, 0.85 in v38 — heading toward the 0.9 at inference. So the model is forced to learn to separate signal from background under the same conditions it faces when it actually runs.

One thing to keep in mind for later: the flow chart on the right — that's where the soft mask sits in the pipeline, between the weight MLP and the message passing."

## S5 — Part 2: Why chains die (1/2) — class imbalance (1 min)

"Now part two — and this is the part I want to spend the most time on, because I think the story here is the most interesting.

First — why do chains die? Look at the class distribution. Class zero, background, is 99.9% of all edges. Each structural class — one, two, three — is about 0.04%. That's roughly a thousand background edges for every structural edge.

And here's the kicker: the model is weakest exactly on the rarest classes. The right plot is per-class accuracy — class two, the sister class, sits at about 49%. **The thing we need the most is the thing the model learns the worst.**"

## S6 — Part 2: Why chains die (2/2) — one misclassified edge (45 s)

"And the second reason chains die is even simpler. At inference, if the classifier calls an edge class zero, it gets pruned. So if even one structural edge of a chain is misclassified, the whole chain dies. One wrong edge, and the entire B decay chain is gone. The trees on the right show it — the red dashed edge breaks the chain; with chain-CE, the same edge is kept and the chain survives.

So the stakes are clear: we need every structural edge to be right. That's exactly what the rewards are built for."

## S7 — Part 2: The root cause — class-0 dilution (45 s)

"So why is the model so bad at structural edges? The root cause is dilution. The main LCA loss is a global cross-entropy over every edge. So its gradient is roughly 99.9% 'predict background'. The rare structural classes get almost no signal from the main loss. They're undertrained — that's why they get misclassified and chains die.

So the fix for this whole part is, basically: reward the chain edges directly. Concretely three things — weight class two a bit more, then a hinge reward, then a chain cross-entropy reward."

## S8 — Part 2: Reward 1 (v37) — hinge (1.5 min)

"So, reward number one — the hinge. It's a training-only extra loss that pays out when a chain edge is confidently classified. Think of it as a bonus: behave well on chain edges, you pay nothing; misbehave, you get penalized. Strictly speaking it's a loss term — 'reward' is just the framing, the audience can relax.

Where does the confidence come from? For every edge, the LCA head outputs a four-class softmax — p0, p1, p2, p3. The confidence is just the max of those — how sure the model is about the class it picked. Note: this is not the pruning weight w. Different thing, easy to mix up, so I put it on the slide.

The penalty is max of zero and margin minus confidence, with margin 0.3. Once the model is thirty percent confident, the penalty is zero. And it's paid only on truth-chain edges — we know them from Monte Carlo truth during training.

Why? Because those edges give the global loss almost no gradient. The hinge is the only signal pushing the model to be confident about them. But — and this is the catch — the hinge only asks for confidence, not for the right class. That's what the next slide fixes."

## S9 — Part 2: Reward 2 (v38) — chain cross-entropy (1.5 min)

"Reward two: chain cross-entropy. The point is to reward being right, not just being confident.

CE is minus log of p_true, where p_true is the probability the model assigns to the true class of the edge. Confident and right — p_true close to one — the penalty is basically zero. Unsure or wrong — the penalty is big.

The key detail: we apply this only on truth-chain edges, and only on the structural classes. So classes one, two and three finally get a direct, undiluted gradient. The right plot shows the difference — the global CE spreads its signal over everything and gets swamped by background; chain-CE concentrates on the edges that matter.

And the numbers: v37 with the hinge gave 27.3. v38 — chain-CE on top, class weight rebalanced — gave 29.3."

## S10 — Part 2: The strategy works (1 min)

"So — did it actually work? Yes. This is the payoff plot for part two. PerfectReco: v36 after pruning 26.3, v37 with the hinge 27.3, v38 with chain-CE 29.3. Plus one, plus two, on top of what pruning already gave us.

And the per-class accuracy is the more direct proof: class one from 67.8 to 76.8, class two from 41.3 to 47.9 — v31 baseline against v38. **The rewards improved exactly the classes the global loss was ignoring.** That's why chains survive more often."

## S11 — Part 3: First, check — physics is invisible (1.5 min)

"Part three. I'm changing topic a bit — from the loss to the representation itself.

Here's the thing. We had a feeling the backbone wasn't really encoding the physics. So before adding any new heads, we checked with linear probes. A probe is simple: freeze the backbone, train one linear layer on top, and see whether a physical quantity can be read off the representation. If the quantity is in there, R-squared is high; if it's not, R-squared is basically zero. The top-right figure is exactly that recipe.

And the answer was: physics is basically invisible. Reading the pion-pair mass from an edge representation — R-squared 0.003. Momentum from a node — zero. The backbone was not holding the physics.

So we added heads to supervise it. And it worked — after mass supervision, the edge-mass probe jumps to 0.93, and PerfectReco goes from 29.3 to 32.7, class two from 44.7 to 51.1. That's masshead2 — the current best version."

## S12 — Part 3: The head zoo (1 min)

"So — the head zoo. One shared backbone, nine heads. The blue ones — LCA classification, node pruning, edge pruning, PV association, chain scoring — are the original reconstruction machinery. The green ones are the new physics heads: source, mass, structure, momentum.

The idea is simple: same backbone, different targets. Each green head tells the backbone 'you have to represent this physical quantity'. And it all trains end-to-end, so the supervision flows back into the shared representation."

## S13 — Part 3: Structure head — depth (1 min)

"Now the two targets of the structure head — I want to be precise about what they mean.

Depth is just how far a node sits from the chain root, the B candidate. It's computed by BFS on the truth chain: the root is depth zero, its children one, the grandchildren two. The left figure shows it — nodes colored by their layer.

And it matters because depth is basically a label for 'where in the tree is this node'. The struct head regresses it, so the node representation is forced to encode tree position. In the small ablation it was the best single head — AllParticles up about three points, and class three accuracy up from 60.8 to 63.2."

## S14 — Part 3: Structure head — Rumor Centrality (1.5 min)

"And the second target: Rumor Centrality. This answers a slightly deeper question — which node is the source of the chain?

The computation is neat. For each candidate root, you root the tree at that node and measure the subtree sizes. Then log of the rumor centrality is minus the sum of the log of the subtree sizes. The root is the node that maximizes it. In the example on the right, rooting at B gives minus 1.39; rooting at the J/psi leaf gives minus 2.48 — so B wins, which is the physical answer.

We use this in two places. The source head — which came first, in v36 — predicts the argmax root directly. The structure head regresses the normalized value. Both targets come from truth, so no manual labels needed."

## S15 — Part 4: Why extend the latent space (1/2) — more heads made things worse (1.5 min)

"Part four — this is the argument for widening the latent space.

It starts with a failure, and I think it's an instructive one. We tried stacking the mass, structure and momentum heads all at once — version 48. Each head alone helps. Together, it backfired. The auxiliary losses summed to 0.877 — well above the main task at 0.559. Reconstruction dropped five points — AllParticles from 55.9 to 50.6 — even though the classification itself didn't degrade.

My interpretation: **the 16-dimensional representation is saturated.** Adding more supervisors to the same small space doesn't add information — they just compete. So the bottleneck isn't the heads. It's the space."

## S16 — Part 4: Size the latent space from physics (2/2) (1.5 min)

"And here's the nice part — we can actually estimate how big the latent space should be, from physics. A d-dimensional representation holds at most d independent numbers. A track carries something like twelve to fourteen physical degrees of freedom — position, momentum, PID, mass-related quantities — and the node representation has to serve nine heads. So 16 dimensions sits right at the lower bound. It's under-parameterized.

So we widened it asymmetrically — version 53: track nodes to 32 dimensions, track-track edges to 24, everything else stays 16. Retraining from scratch with the v38 stack plus the mass head.

Honest status: the job got interrupted at epoch 74 of 150. It hasn't converged, so I can't claim it works — the early numbers aren't meaningful. But the physics argument says this is the right direction, and a full run is planned."

## S17 — Part 5: Other ongoing lines (1 min)

"Quick run through the other lines still going.

Public-data verification — the published dataset has no PID, and the class-two and class-three counts differ from ours by a factor of four to fourteen, so it's a real distribution shift. After the resume it reached best validation 33.4 at epoch 73.

Learned deterministic annealing — inference-side PV clustering with a learned affinity. The point is to avoid the subgraph-training failure we had before: no training on subgraphs, it runs on the full graph. The core module is implemented and CPU-verified.

And chain scoring for trigger assistance — the chain criteria reach AUC 0.90, and 0.78 against realistic pruned components. The scorer training is ready, just waiting for GPU time."

## S18 — Next steps and open questions (1 min)

"To wrap up — next steps: resume the wider-latent training to 150 epochs, then stack structure and momentum at reduced weights; run the queued evaluations; train the chain scorer; plug the affinity into the learned annealing.

And three questions I'd genuinely like your input on. First — should the target be trigger assistance, scoring candidates, or full-event reconstruction, and which metric would the collaboration actually trust? Second — class two, same-source clustering: is that the real problem DFEI should be solving, or are we optimizing the wrong thing? Third — what's a realistic per-event time budget for a lightweight GNN at HLT2 or Upgrade II?

Thanks — happy to take questions."

---

## Likely questions and prepared answers

**Q: Why does the mass head help reconstruction, not just classification?**
A: Because it forces the edge representation to encode sister relations — class two — and class-two edges are the glue of the chain. The six-point class-two gain is the mechanism.

**Q: Why not feed m_ππ as an input feature instead of supervising with it?**
A: Two reasons. Feeding it in breaks the end-to-end claim — the model wouldn't learn to derive it from the momenta. And our verification method is representation probing — if a quantity matters, the model should hold it internally.

**Q: What exactly is the annealing?**
A: A temperature inside the sigmoid mask, w·σ((w−cut)/τ), annealed from 1.0 to 0.1 over training. High τ gives a smooth mask and stable gradients; low τ makes the mask behave like the hard threshold at inference. The cut itself is fixed per run.

**Q: Is the 74-epoch wider-latent run a failure?**
A: No — it's incomplete, interrupted by the job limit with validation still decreasing. Whether wider latents help is still an open question.

**Q: Why did the three heads together fail when each helps alone?**
A: Gradient competition — the auxiliary losses together exceeded the main task. The ablation shows each head is individually positive; the combination just needs lower weights. And my read is the latent space is saturated — that's the widening argument.

**Q: Is the twelve-to-fourteen degrees-of-freedom count reliable?**
A: It's an estimate — position, three momentum components, PID, mass-related quantities, plus a few more. The point isn't the exact number; it's that 16 dimensions is barely enough before you even count the nine heads sharing the space.
