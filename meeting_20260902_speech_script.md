# DFEI Group Meeting — Speech Script (2026-09-02)

> 19 slides · target ~15-18 min speaking · bracketed [ ] = optional, skip if short on time

---

## S1 — Title (5 s)
"Good morning everyone. This is the progress report I've been postponing since July — the full journey of optimizing DFEI, from a silent bug to physics-supervised representations. I'll go through every attempt, including the ones that failed, because I think the failures teach us the most."

## S2 — Roadmap (1 min)
"This is the complete map. Green is what we kept, red is what we closed, orange is what's still running. The story in one line: we went from 23.9% PerfectReco to 32.7% — and every step was verified, including two important negative results: PV clustering, and the combined multi-head training. I'll walk through each row."

## S3 — Starting point & bug (1.5 min)
"Let me start with what I found. DFEI is a full-event GNN: it classifies LCAG edges into four classes, prunes nodes and edges, associates tracks to primary vertices, and rebuilds every heavy-hadron decay chain in the event. The prototype worked, but there was a bug: the config key for the class weights had a double underscore — L-C-A underscore underscore weights. The weights silently never reached the loss. So the model only learned the dominant background class, and class-one, the parent-child edges, sat at zero percent accuracy.

There was also a structural gap: we trained on full graphs, but at inference we first prune with hard thresholds. Those two things — the silent bug and the train-inference gap — frame everything that follows. The baseline after fixing the bug is 23.9% PerfectReco."

## S4 — v31 → v38 optimization line (2 min)
"Then the optimization line, each step verified. First, B2: differentiable pruning. Instead of training on the full graph, we apply a soft mask — the edge weight times a sigmoid of weight minus cut over temperature — and anneal the temperature down. High temperature gives smooth gradients; low temperature approximates the hard threshold. We aligned the cut to inference, 0.5 then 0.7 then 0.85. So the model is trained on graphs that look like what inference prunes.

Second, class-two weighting. Class two, the sister edges — two tracks from the same B — were the structural bottleneck. We tried weight 3.0, it hurt class one, so we settled on 2.0.

Third, in-chain consistency losses. These only supervise edges that lie on truth chains. A hinge loss keeps chain edges confident, and — this is important — a cross-entropy directly on the chain-edge classes. Structural edges are only about a tenth of a percent of all edges, so without this, the class-zero background completely dilutes the signal.

And fourth, the source head: we teach the backbone to predict the chain root. I'll come back to this on the structure slide. Net result: 23.9 to 29.3 percent."

## S5 — PV clustering (closed) (1.5 min)
"Now the first negative result. High-multiplicity events have 91 to 139 tracks; we thought cross-chain interference was hurting us. So we split the event into per-PV subgraphs, rebuild chains inside each, and merge. Training on subgraphs — with a trainable cluster head, Gumbel routing, and a curriculum — was the failure. The reason is subtle: the backbone only ever saw subgraphs during training, but at inference the GNN first runs on the full graph, then we cluster. So full-graph forward ability degraded — class-one accuracy collapsed from 76.8 to 56.4 percent. We tried val-side alignment; it doesn't fix the fundamental train-infer graph mismatch. We closed the line. The lesson is written in red on this map."

## S6 — Physics supervision concept (1 min)
"Next direction: output-side physics supervision. The idea is simple — instead of hand-crafting physics features into the input, we supervise the representations with physical targets, and keep everything end-to-end. Three auxiliary heads on the shared backbone: mass at the edge level, structure at the node level, momentum at the node level. The targets are data-intrinsic — they're computed from the tracks themselves, no new labels. And we verify with linear probes: can a frozen backbone read physics out of its own representations?"

## S7 — Mass head (2 min)
"The mass head is the big win. Every edge connects two tracks; the pair mass under the pion hypothesis is cheap and physics-loaded — same-mother pairs sit near resonance masses, so supervising the mass forces the edge representation to encode sister relations. We regress log10 of the mass with SmoothL1 and mask the sentinel edges. Result, on the same 20 test files: AllParticles 52.1 to 55.9, PerfectReco 29.3 to 32.7, and — the most interesting — class-two sister accuracy from 44.7 to 51.1. And the linear probe confirms why: edge representation readability for the mass went from R-squared 0.003 to 0.93. The physics literally moved into the representation."

## S8 — Structure head (1.5 min)
"The structure head answers a limitation of the source head. The source head predicts the root with one bit — but the root is the centroid of the track graph, not necessarily the B, because most tracks are final-state particles. What's richer is the node's position in the tree: depth, and the Rumor-Centrality value. We regress both. In a controlled ablation from the masshead2 checkpoint — ten more epochs, small five-file eval — the structure head alone is the best single head: AllParticles plus 3.1, Perfect plus 0.8. It also improved class-three, the grandparent edges, from 60.8 to 63.2 percent."

## S9 — Momentum head (1.5 min)
"The momentum head fixes a silent deficit. We probed the node representations and found them linearly unreadable for momenta — R-squared essentially zero. The graph normalization and ReLU activations in the blocks scramble momentum information. So we added a node-level regression of normalized momentum. Ablation: plus 2.4 AllParticles, plus 0.8 Perfect. The general lesson: probe first, find what's missing, then supervise it."

## S10 — Combination failure (2 min)
"Here's the second negative result, and it's the most instructive. We put mass plus struct plus mom on together — and reconstruction dropped five points, even though the loss was lower. The table shows why. At epoch 114 of the combined run, the three auxiliary losses sum to 0.88, which is one-point-six times the main LCA loss of 0.56. The auxiliary gradients dominate the shared backbone. And notice: LCAG accuracy did not drop — class-one even rose. It's the pruning and reconstruction behavior that got pulled away. So the fix for the next version is simple: lower the auxiliary weights — mom to 0.2, struct to 0.3 — so the supervision guides without dominating."

## S11 — PhyIP evidence (45 s)
"Just to make the verification concrete: these are linear probes on the frozen backbone, Ridge regression. Before physics supervision, the edge representation reads the mass at R-squared 0.003. After masshead2, 0.93. Nodes are still unreadable for momentum — which is exactly the deficit we just fixed with the mom head, and we will re-probe."

## S12 — Asymmetric latent dims (1 min)
"The next experiment: we did a physics degrees-of-freedom count. A node representation of sixteen dimensions must carry roughly twelve to fourteen physical degrees of freedom — position, momentum, PID, mass — and serve nine heads. Sixteen sits at the lower bound. So we widened tracks nodes to 32 and tt-edges to 24, and retrained from scratch with the v38 stack plus the mass head. Status: the job was interrupted at epoch 74 of 150 — from-scratch training was nowhere near converged, best validation 39.9 versus the 35.6 family. The preliminary evaluation is therefore not representative. Plan: resume to 150 epochs, then layer struct and mom with the low weights from slide 10."

## S13 — Public data verification (1 min)
"We also verified the v38 stack on the published dataset. Two important caveats: it has no PID features, and the class-two and class-three edge counts differ from our CERN MC by four to fourteen times. The first run was interrupted at epoch 66; we resumed and the best validation improved to 33.4 at epoch 73. The evaluation is queued — we're competing for GPU time on a very loaded cluster. A fair head-to-head with the paper needs adaptation first."

## S14 — Results summary (30 s)
"The summary table. Everything on CERN MC, same threshold. The takeaway: masshead2 at 32.7 percent PerfectReco is the best model, physics supervision works and is reproducible. The ablation rows use a small five-file eval — directionally valid, not comparable to the full rows."

## S15 — Chain criteria AUC (1 min)
"Now the trigger perspective. Full-event reconstruction is the wrong metric for a trigger line — what matters is candidate-chain classification. We measured chain criteria from model outputs on 20 events: against realistic model-pruned components, AUC is about 0.78; against easy random combinations, 0.90, and the strongest single feature — the in-chain non-background probability — reaches 0.956. The chain scorer is implemented and the training script is ready; we need GPU time to train it and measure properly."

## S16 — Learned DA (45 s)
"And a new direction. We closed PV clustering because training on subgraphs hurt the model — not because clustering is useless. Learned deterministic annealing is inference-side clustering: a trained MLP provides the affinity, and deterministic annealing does soft assignment with an adaptive cluster count. No Gumbel sampling, no training-side subgraph split. The core module is implemented and CPU-verified — annealing schedule, adaptive splitting, merge post-processing. Next is plugging the GNN latent representations in as the affinity source."

## S17 — Open questions (1 min)
"Finally, the open questions where I would really value the group's input. One: is trigger assistance — candidate scoring — the right application form, and which metric would the collaboration trust? Two: since parent-child track relations are physically rare, is same-source clustering — class two — the real problem DFEI should solve? Three: the invariant mass is analytic in HLT2, so what's the genuine added value of the GNN — structure and context, not kinematics? Four: could our same-source clusters feed a transformer-based flavor tagger? Five: what's a realistic HLT2 or Upgrade-II time budget for a lightweight GNN per event? And six: is this physics-supervision direction, with linear-probe verification, worth pursuing — and are there better physical targets?"

## S18 — Next steps (30 s)
"Next: resume the asymmetric training to 150 epochs, then layer struct and mom with low weights. Run the queued evaluations. Train the chain scorer and measure candidate-chain AUC properly. Plug GNN affinity into the learned DA. And re-probe the node representations after the mom head to check the momentum deficit is actually fixed."

## S19 — Backup (optional)
"This is the data and hardware setup — everything reproducible from config files, with a version log maintained."

---

## Q&A cheat-sheet (anticipated questions)

**Q: Why does the mass head help reconstruction, not just classification?**
A: The mass forces the edge representation to encode sister relations, which directly strengthens class-2 detection — and class-2 edges are what glue a chain together. The +6.4pp class-2 gain is the mechanism.

**Q: Why not just feed m_ππ as an input feature?**
A: Two reasons. It would break the end-to-end claim — the network wouldn't learn to compute it from momenta. And our verification philosophy: we probe what the representation carries. If a physical quantity is useful, the model should learn to hold it internally, not be handed it.

**Q: Is 74 epochs of the asymmetric run a failure?**
A: No — it's an unfinished run. From-scratch training at 55 minutes per epoch; the validation was still dropping when the job was interrupted. The question of whether wider latents help is still open.

**Q: Why did the combined heads fail while each alone helps?**
A: Gradient competition — the aux losses together exceeded the main task. The ablation showed each head is individually positive; the combination just needs lower weights. This is a well-known multi-task trade-off, and we now have the numbers to set it properly.
