# DFEI Group Meeting — Speech Script (spoken English, 2026-09-02)

> 12 slides · ~18 min · natural spoken style, read directly · [brackets] = action/pause cues

---

## S1 — Title (10 s)

"Hi everyone. Today I'll walk you through the DFEI optimization work I've been doing on CERN Monte Carlo. There's a lot to cover, so I've organized it by theme rather than by timeline."

## S2 — Data and overall effect (1 min)

"First, the data. Everything here uses the official CERN Monte Carlo production — the DFEI_IFT_20260702 sample in yukaiz's folder. Evaluation is at pruning threshold 0.9 on twenty test files.

Straight to the results. After fixing a silent class-weight bug, PerfectReco went from 23.9 up to 32.7 — almost nine points. AllParticles went from 43.4 to 55.9 — twelve points.

This figure shows four metrics across versions: PerfectReco, AllParticles, and the LCAG class-one and class-two accuracies, from v31 to v47. Overall, everything keeps climbing."

## S3 — Outline (30 s)

"The talk has four parts. Part one is about the supervision signal itself and aligning training with inference. Part two is the core idea — supervising the representations with physics — and several heads fall under it. Part three covers two controlled failures that I think are instructive. Part four is what's in progress, mainly the wider latent space."

## S4 — Differentiable pruning with annealing (2 min)

"Part one, first step: differentiable pruning.

The problem is simple. During training the model sees the full graph, but at inference we prune first, with hard thresholds, and then reconstruct. Training and inference see different graphs — that's the gap.

Our fix is a soft mask on the weights: effective weight equals weight times a sigmoid of weight-minus-cut over a temperature. The temperature is annealed from 1.0 down to 0.1 during training. When the temperature is high, the mask is smooth and gradients are stable; as it drops, the mask behaves more and more like the hard threshold used at inference.

The cut itself also tightens across versions: 0.5 in v36, 0.7 in v37, 0.85 in v38, converging toward the 0.9 used at inference. So the model learns to separate signal edges from background edges under exactly the conditions it faces at inference."

## S5 — LCA supervision adjustment (2 min)

"Second step of part one: adjusting the LCA supervision. This one had some trial and error.

Class two — two tracks from the same mother — is the structural bottleneck, so we weighted it explicitly. In v37 we tried 3.0, but that was too aggressive and pushed class one down, so v38 settled on 2.0.

The other idea is in-chain consistency losses, supervised only on edges that lie on truth chains. v37 added a hinge loss that keeps chain edges confident. v38 added a direct cross-entropy on the chain-edge classes. Why the extra cross-entropy? Because structural edges are only about a tenth of a percent of all edges. Without direct supervision, the dominant background class completely dilutes their signal.

Net effect of part one: PerfectReco from 23.9 to 29.3."

## S6 — Physics supervision: one common idea (1.5 min)

"Part two — the part I find most interesting.

The core idea in one sentence: physical quantities are data-intrinsic — you compute them straight from the tracks, no extra labels — so we use them to supervise the representation, while keeping the model end-to-end.

Several heads share this theme. The source head came first, back in v36, supervising the chain root. The mass head works on edges, regressing the log10 of the two-pion invariant mass. The structure head works on nodes, regressing depth and the Rumor-Centrality value. And the momentum head, also on nodes, regresses the normalized momentum. All of them are verified with linear probes — we freeze the backbone and check whether the physical quantity is linearly readable from the representation."

## S7 — Mass head: main result (1.5 min)

"The mass head is the headline result.

The physics intuition: two tracks from the same mother sit near a resonance in the pion-pair mass. So if we supervise that mass, we're effectively forcing the edge representation to encode 'are these two tracks sisters?' — which is exactly class two.

Implementation: SmoothL1 regression of log10 mass, sentinel edges masked, weight one.

Results, same twenty test files: PerfectReco 29.3 to 32.7; AllParticles 52.1 to 55.9; and the key one — class two, from 44.7 to 51.1, up six and a half points. Class two is the glue of the chain, so when it improves, reconstruction follows. The ROC on the right is class-two discrimination of the final model."

## S8 — Structure and momentum heads (1.5 min)

"Then the two node-level heads.

The structure head supervises where a node sits in the tree: depth — the BFS distance to the chain centroid — and the Rumor-Centrality value. The momentum head supervises the normalized momentum. Actually, the momentum head came from a probe finding: node representations were linearly unreadable for momentum, R-squared basically zero — the information was getting scrambled on the way through.

The table is a controlled ablation, resumed from the masshead2 checkpoint and evaluated on a small set. The structure head is the best single head — AllParticles up 3.1 points. The momentum head adds 2.4. The methodological takeaway: probe first, find what's missing, then supervise it."

## S9 — Verification by linear probes (1 min)

"The probe result is quite clean. Freeze the backbone, regress the physical quantity linearly. Before the mass supervision, reading the mass from the edge representation gives R-squared 0.003 — essentially nothing. After masshead2: 0.93. So the mass information is genuinely inside the representation. Nodes are still unreadable for momentum, which is exactly what the momentum head targets — we'll re-probe once it's trained."

## S10 — Controlled failures (2 min)

"Part three: two controlled failures, both instructive.

First, PV subgraph training. The idea made sense: split the event into per-PV subgraphs of twenty to thirty nodes to reduce cross-chain interference. But the failure was structural — the backbone was only ever trained on subgraphs, while inference runs the GNN on the full graph first. Full-graph ability degraded; class one dropped from 76.8 to 56.4. We closed that line.

Second, adding the mass, structure, and momentum heads all at once. The auxiliary losses summed to 0.877, exceeding the main task at 0.559. Reconstruction dropped five points — even though the LCAG classification didn't degrade. The shared backbone was being pulled toward the auxiliary tasks. The fix: lower the auxiliary weights — momentum to 0.2, structure to 0.3.

Two lessons: a train–inference mismatch in the graph structure is fatal, and the gradient balance between heads has to be explicit."

## S11 — Ongoing attempts (1.5 min)

"Part four — what's in progress.

First, the wider latent space. The node representation is 16-dimensional, but the physical degrees of freedom are roughly twelve to fourteen, and it has to serve nine heads — 16 sits at the lower bound. So we widened tracks nodes to 32 dimensions and track-track edges to 24, retraining from scratch. But the job got interrupted at epoch 74 of 150, so it hasn't converged — we can't draw conclusions yet. The plan is to resume.

Second, public-data verification. The published dataset has no PID, and the class-two and class-three edge counts differ from ours by a factor of four to fourteen — big differences. After the resume, the best validation reached 33.4 at epoch 73.

Third, learned deterministic annealing — inference-side PV clustering with a learned affinity, no subgraph training. The core module is implemented and CPU-verified.

Fourth, chain scoring for trigger assistance. Chain criteria reach an AUC of 0.90, and 0.78 against realistic pruned components. The scorer training is ready and waiting for GPU time."

## S12 — Next steps and open questions (1 min)

"Finally, next steps and a couple of questions for you.

Next: resume the wider-latent training to 150 epochs, then stack the structure and momentum heads at reduced weights; run the queued evaluations; train the chain scorer; and plug the GNN affinity into the learned deterministic annealing.

Questions we'd value your input on: first, should the target be trigger assistance through candidate scoring, or full-event reconstruction — and which metric would the collaboration trust? Second, since parent–child track relations are physically rare, is same-source clustering — class two — the real problem DFEI should solve? Third, what's a realistic per-event time budget for a lightweight GNN at HLT2 or Upgrade II?"

---

## Likely questions and prepared answers

**Q: Why does the mass head help reconstruction, not just classification?**
A: Because it forces the edge representation to encode sister relations — class two — and class-two edges are the glue of the chain. The 6.4-point class-two gain is the mechanism.

**Q: Why not feed m_ππ as an input feature instead of supervising with it?**
A: Two reasons. Feeding it in breaks the end-to-end claim — the model wouldn't learn to derive it from momenta. And our verification method is representation probing — if a physical quantity matters, the model should learn to hold it internally.

**Q: What exactly is the annealing?**
A: A temperature inside the sigmoid mask, w·σ((w−cut)/τ), annealed from 1.0 to 0.1 over training. High τ gives a smooth mask and stable gradients; low τ makes the mask behave like the hard threshold used at inference.

**Q: Is the 74-epoch wider-latent run a failure?**
A: No — it's an incomplete run, interrupted by the job limit with validation still decreasing. Whether wider latents help is still an open question.

**Q: Why did the three heads together fail when each helps alone?**
A: Gradient competition — the auxiliary losses together exceeded the main task. The ablation shows each head is individually positive; the combination just needs lower weights.
