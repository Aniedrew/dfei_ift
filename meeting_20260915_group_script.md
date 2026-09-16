# DFEI Group Meeting — Short Script (2026-09-15, group version)

> 17 slides · ~15 min · written to be spoken · [brackets] = action cue · **bold** = the sentence you must not forget
>
> Goes with `DFEI_meeting_20260915_group_EN.pptx`. The long version stays where it is (`meeting_20260915_talk_script.md` + `DFEI_meeting_20260915_EN.pptx`).

---

## S1 — Title (15 s)

"Hi everyone. Short version today. I'll show you the whole model in one page, then the changes that built the line, and then what we learned when the changes stopped working. The version-by-version detail is in the long deck — I'm happy to go through it offline."

## S2 — Data and the overall effect (45 s)

"CERN official Monte Carlo, threshold 0.9, twenty test files.

Over the line v31 to v47, PerfectReco goes from 12.30 to 23.15 %, and AllParticles from 22.49 to 39.56 %. Both are fractions of the same 17 561 truth B candidates.

One calibration number: **±0.43 % due to random seed.** Anything smaller than that is not a result — that's the bar for the rest of the talk."

## S3 — Outline (20 s)

"The plan. First the whole model in one page — that's the page worth your attention. Part 1: differentiable pruning. Part 2: rewarding the chain, which is where the line really moved. Part 3: the physics heads. Part 4 is short — what we tried after v47, and the measurement that told us why none of it worked.

Every change after that is labelled with the block of the pipeline it touches, so you always know where we are."

## S4 — The model in one page (2 min)

"This is DFEI. One event becomes one graph — nodes are tracks and primary vertices, edges are track pairs. Stage one prunes that graph with two learned scores, one for tracks and one for pairs; at inference we keep everything above 0.9. Stage two takes what survived, sorts every pair into four classes, and assembles chains out of them. The metrics compare those chains with the truth.

Under the picture I've listed what we changed, block by block. On the shared representation: four label-free targets — source, mass, depth and RC, momentum. On the pruning heads: annealing, and the loss put back in balance. On the LCAG head: class weighting, the hinge, then chain cross-entropy. And chain assembly itself we never touched.

One sentence to hold on to: **stage two can only work with what stage one let through.** That comes back at the end."

## S5 — Part 1: differentiable pruning with annealing (1.5 min)

"Part one. The problem is a mismatch: training sees the full graph, inference prunes before it reconstructs, so the two see different graphs.

The fix is a soft mask — the message weight becomes w times a sigmoid of (w − cut) over a temperature τ — and τ is annealed from 1.0 down to 0.1, so the smooth mask converges to the hard cut we actually use.

It's a training-only change; inference is untouched. v31 to v36: PerfectReco 12.30 to 18.77, AllParticles 22.49 to 35.17. **The biggest single step before the mass head.**"

## S6 — Part 2: why chains die (1 min)

"Part two, and this is the part I find most interesting.

At inference, an edge the classifier calls background gets pruned. If one structural edge of a chain is misclassified, the chain can no longer be assembled — it's gone, even when every other edge was right. The two trees on the right show it.

So the accuracy that matters is the accuracy on the edges that carry a chain. And those are the rarest edges in the event."

## S7 — Part 2: the class imbalance (1 min)

"Background is 99.9 % of all edges. Each structural class is about 0.04 % — roughly a thousand background edges for every structural one.

The main LCAG loss is a global cross-entropy over every edge, so almost all of its gradient is saying 'background'. Per-class accuracy collapses exactly where it hurts: the worst class is class 2, the sister edges.

**The thing we need most is the thing the model learns worst.** So we supervise those classes directly — that's what the two rewards do."

## S8 — Part 2: reward 1, the hinge (v37) (1 min)

"Reward one. Confidence is the largest of the four class probabilities from the LCAG head. The penalty is max of zero and 0.3 minus confidence — be confident and you pay nothing, be unsure and you pay. It's paid only on truth-chain edges, which we know from the Monte Carlo during training.

Result: 18.77 to 19.50. Modest. And class-2 accuracy actually went down — 47.7 to 44.2 — because we had pushed its weight to 3.0. So v38 puts that back."

## S9 — Part 2: reward 2, chain cross-entropy (v38) (1 min)

"Reward two, and this is the one that works properly.

Cross-entropy, minus log p_true, pays for being the right class — not just for being confident. It's paid only on truth-chain edges, so a thousand background edges don't dilute the gradient. It's the same loss function as the global one; the difference is where it's paid.

Result: 19.50 to 21.10, and per-class accuracy finally moves — class 1 from 67.8 to 76.8, class 2 from 41.3 to 47.9."

## S10 — Part 2: the strategy works (45 s)

"The whole line in one picture: PerfectReco 12.30 to 21.10, AllParticles 22.49 to 37.59.

And what moved is exactly the two classes the global loss was ignoring. That's the mechanism, not a coincidence. Still a training-only change."

## S11 — Part 3: first, a probe (1 min)

"Part three: the physics heads. We started with a check.

Freeze the backbone, fit a single linear layer, and try to read a physical quantity out of the representation. Before any physics head, the ππ mass probe gives R² of 0.003 and momentum is flat zero — the physics isn't in there. After the mass head, edge mass R² is 0.930.

**So physics isn't a by-product of reconstruction. It has to be supervised in.**"

## S12 — Part 3: the head zoo (45 s)

"One backbone, nine supervised targets. Five are the original DFEI machinery — LCAG, node prune, edge prune, PV association and the chain scorer. Four are ours, and all four are label-free from the Monte Carlo: source, mass, struct and momentum.

They all read the same representation, and they're parallel. **Adding a head never changes the network — it only adds a target.**"

## S13 — Part 3: the mass head (1 min)

"This is the one that paid.

The target is the invariant mass of the track pair on each track-track edge. The first attempt regressed the raw mass and landed at 21.28 — inside the noise. That was a parameterisation problem, not a problem with the idea. v47 adds a log10 normalisation and masks the sentinel edges: 21.10 to 23.15.

And the honest control — the same recipe on source, struct and momentum is a wash. It's this target, on this representation."

## S14 — Part 3: the struct head (1 min)

"The struct head is the interesting negative result.

Two targets per node, both read off the truth tree: the depth, in BFS layers from the root, and the Rumor Centrality, which says how likely a node is the source of the chain. No manual labels anywhere.

On the weak base this was the best single addition. On the main line it's a wash. **And that leads into Part 4: on this representation, more supervision stopped adding information.**"

## S15 — Part 4: what came after v47 (1.5 min)

"After v47 we pushed five directions in parallel: capacity, attention, restructuring the event, retraining the recipe, and another dataset. One line each.

Capacity: widen the representation — much worse, 24 to 27 % AllParticles against 37.6. Attention: it helps on a weak base, nothing on the converged model. Splitting the event by primary vertex: worse. Retraining the recipe one change at a time: only chain-CE clears the noise. Public data: the CERN recipe doesn't transfer as it stands.

**Same pattern in all five: adding to the network doesn't help — and combining the few things that do help doesn't help either.**"

## S16 — Part 4: the diagnosis (1.5 min)

"So if the network isn't the bottleneck, what is? We measured it.

Take v47, freeze it completely, and rewrite the pruning decisions with Monte Carlo truth. Same weights, same network — only the pruning changes. AllParticles goes from 39.56 % to 99.75 %.

**The score is decided at the pruning step, before the network's output matters.** Two concrete causes: the point head was carrying one thirty-third of the loss weight, and the fine-tuning step was ten times too small. What runs now is the pruning loss rebalanced, at a learning rate of 3e-4."

## S17 — Summary (1 min)

"To wrap up.

The line from v31 to v47 is real. Nothing we added to the network afterwards beats it, and the bottleneck is the pruning classification, not the reconstruction. The rebalanced pruning loss is now the first thing above v47: 42.01 % AllParticles, 24.34 % PerfectReco.

Two questions where I'd like your input. One: class-2 same-source clustering — is that the problem DFEI should really be solving? Two: what time budget would HLT2 leave for a lightweight GNN per event?

Thanks — happy to take questions."

---

## Likely questions

**Q: These numbers are lower than the ones from last time.**
A: The flags are unchanged — the same code decides what counts. The denominator changed: the old one was the truth chains that survived pruning in that run, which shrinks as the cut gets harder, so the percentage inflates. Now we divide by the truth chains on the unpruned graph, 17 561 — a constant. Comparable across runs, thresholds and datasets.

**Q: Isn't the oracle test cheating?**
A: Deliberately. We freeze the weights and replace the pruning decisions with Monte Carlo truth. It doesn't give a usable model; it tells us how much of the score is already decided at the pruning step. The answer is: most of it.

**Q: Why does the mass head help reconstruction, not just classification?**
A: It forces the edge representation to encode sister relations — class 2 — and those edges are the glue of a chain. The class-2 accuracy gain, 41.3 to 47.9, is the mechanism.

**Q: Why not just feed the mass in as an input?**
A: It would break the end-to-end claim — the model is supposed to derive it from the momenta — and our check for "is it in there" is the probe, not an input.

**Q: Where is the remaining headroom?**
A: Almost entirely in the point-pruning head. Fixing point recall is worth about fifteen points of AllParticles, and precision about fifteen more.
