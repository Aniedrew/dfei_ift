# DFEI Group Meeting — Speech Script (2026-09-02, v2, formal)

> 25 slides · target ~18-20 min · formal register · bracketed [ ] = optional

---

## S1 — Title (10 s)
"Good morning. This presentation gives a complete account of the DFEI optimization work carried out since June: the full line from a silent configuration bug to physics-supervised representations, including the rationale, the quantitative results, and two controlled failures."

## S2 — Outline (30 s)
"The structure is as follows. I first define the task, the data, and the metrics. I then describe the starting point — a class-weight bug and a train–inference gap. Next, the four verified optimization steps from v31 to v38. After that, the redirect toward output-side physics supervision, and two controlled failures that we quantified rather than hand-waved. I close with the ongoing work, new directions, and open questions."

## S3 — Task (1 min)
"The task, in one sentence: from the charged tracks of a single collision — about one hundred and fifty of them — reconstruct every heavy-hadron decay chain, in hierarchy, while rejecting background. The network solves four supervised sub-tasks jointly: LCAG edge classification with four classes — background, parent–child, sister, and grandparent; node and edge pruning; track-to-primary-vertex association; and, offline, the reconstruction of chains from the classified edges, using Rumor Centrality to find chain roots."

## S4 — Data and metrics (45 s)
"We train on Run-3 Upgrade-I Monte Carlo, roughly one hundred and fifty tracks per event, normalized, with two hundred training, twenty validation, and twenty test files. We also use the published reference dataset for independent verification. Three metrics matter: PerfectReco, the fraction of events in which every chain is recovered; AllParticles, the fraction of truth particles recovered; and per-class LCAG accuracy. Class two, the same-mother edges, is the structural bottleneck."

## S5 — The bug (1.5 min)
"The starting point was not a model problem but a configuration problem. The class weights were defined in the config under a key with a double underscore — L-C-A, underscore underscore, weights — while the code looked for the single-underscore key. The weights silently never reached the loss. The consequence is visible in the figure: class one, the parent–child edges, sat at essentially zero accuracy, because the model only learned the dominant background class. The fix is a single underscore, with inverse-frequency weights. After the fix, the v31 baseline is 23.9 percent PerfectReco and 43.4 percent AllParticles."

## S6 — Train–inference gap (1 min)
"The second issue is structural. Training optimizes the full graph globally, while inference first prunes nodes and edges with hard thresholds before any reconstruction. We measured, on CERN data, that chain death at the edge-pruning step dominates, and that edge deaths overlap strongly with node deaths. The event on the right illustrates a failure mode at inference. This gap directly motivates the first optimization."

## S7 — Overview of the line (30 s)
"The full line is summarized here. Each row is a controlled experiment: baseline at 23.9, differentiable pruning at 26.3, class-two weighting and in-chain consistency at 27.3, and the v38 combination at 29.3. After v38 the line splits into physics supervision, which reached 32.7, and representation widening, which is still in progress. Two lines were closed: PV subgraph training and the naive combination of auxiliary heads."

## S8 — Differentiable pruning (1.5 min)
"The first optimization addresses the train–inference gap directly. Instead of training on the full graph, we apply a soft mask to the node and edge weights — the weight times a sigmoid of weight minus cut, over temperature. The temperature is annealed from one down to one tenth over training. At high temperature the mask is smooth and gradients flow; at low temperature the mask approximates the hard threshold. The cut is aligned to the inference threshold, increasing from 0.5 to 0.85 as the pipeline matured. The schematic on the right shows the mechanism. The effect is that the model explicitly learns to separate signal from background weights under the same conditions it meets at inference."

## S9 — Class-two weighting and in-chain consistency (1.5 min)
"The second and third optimizations target the structure of the chains. Class-two edges — two tracks from the same B — were the bottleneck; we weighted them explicitly, settling at 2.0 after 3.0 over-weighted them and hurt class one. The third optimization is a set of in-chain consistency losses, supervised only on edges that lie on truth chains. A hinge term penalizes low-confidence chain edges, keeping chains confident. A cross-entropy term supervises the chain-edge classes directly. The reason this matters is quantitative: structural edges are only about one tenth of one percent of all edges, so without direct supervision the background class completely dilutes the signal. The training curves at the bottom show the behavior across versions. The net effect from v36 to v38 is 26.3 to 29.3 percent."

## S10 — Source head (1 min)
"The fourth optimization is the source head. Inference locates chain roots by Rumor Centrality, but training never supervised roots. We added a node head that predicts whether a node is the root of its chain, trained with binary cross-entropy on the Rumor-Centrality argmax node of each truth chain. One limitation should be stated: the Rumor-Centrality argmax is the centroid of the track graph, not necessarily the B itself, because most tracks are final-state particles. This observation motivates the richer structure supervision we discuss later."

## S11 — Results v31 to v38 (45 s)
"The numbers through v38. The baseline recovers 23.9 percent of events perfectly; v38 recovers 29.3. Note the class-two gain at v36 — from 41.3 to 47.7 — which is the structural driver, and the class-one rebalancing at v37. This line is the foundation on which everything after v38 builds."

## S12 — Redirect: physics supervision (1.5 min)
"After v38, we stepped back and asked a design question. At trigger level, physical quantities such as invariant mass are computed analytically — so what is the GNN for? Our answer: the GNN should encode structure and context in its internal representation. The approach we took is output-side physics supervision: we do not hand-craft physics into the input; instead we supervise the representations with data-intrinsic physical targets, keeping the model end-to-end. Three heads were added: a mass head at the edge level, regressing the log10 of the two-pion invariant mass, whose resonances encode sister structure; a structure head at the node level, regressing depth and Rumor-Centrality value; and a momentum head at the node level. The verification method is a linear probe on the frozen backbone."

## S13 — Mass head (1.5 min)
"The mass head is the main result. The pair mass under the pion hypothesis carries resonance information: two tracks from the same mother sit near a resonance peak. Supervising the log10 mass therefore forces the edge representation to encode sister relations. The implementation is a SmoothL1 regression with the sentinel edges masked, weight one. The result on the same twenty test files: AllParticles from 52.1 to 55.9, PerfectReco from 29.3 to 32.7, and — the most telling — class-two accuracy from 44.7 to 51.1. The ROC curve on the right is the class-two discrimination of masshead2."

## S14 — Structure head (1 min)
"The structure head answers the limitation of the source head. One bit for the root is not enough; what matters is where a node sits in the tree. We regress two quantities: depth, the BFS distance to the chain centroid, and the Rumor-Centrality value. In a controlled ablation from the masshead2 checkpoint — ten additional epochs, evaluated on a small five-file set — the structure head is the best single head: AllParticles improves by 3.1 points, PerfectReco by 0.8. It also improves class three, the grandparent edges, from 60.8 to 63.2 percent."

## S15 — Momentum head (45 s)
"The momentum head repairs a silent deficit found by probing: the node representations are linearly unreadable for momentum, with R-squared essentially zero, because the graph normalization and ReLU activations scramble that information. We regress the normalized momentum from the node representation. In the same ablation protocol the gain is 2.4 points in AllParticles and 0.8 in PerfectReco. The general lesson is procedural: probe first, identify the missing quantity, then supervise it."

## S16 — Combined heads: a controlled failure (2 min)
"Now the most instructive negative result. We added the three heads simultaneously — and reconstruction dropped by five points, even though the loss decreased. The table quantifies why. At epoch one hundred fourteen of the combined run, the three auxiliary losses sum to 0.877, which is one point six times the main LCA loss of 0.559. The auxiliary gradients dominate the shared backbone. Notice what did not happen: LCAG accuracy did not drop — class one even improved. The degradation was in the pruning and reconstruction behavior. The loss and the reconstruction metric are therefore not monotonically related. The resolution for the next version is straightforward: reduce the auxiliary weights, momentum to 0.2 and structure to 0.3, so that supervision guides the backbone without dominating it."

## S17 — Linear-probe verification (1 min)
"The verification is made concrete here. We freeze the backbone and regress the physical quantity with a linear model. Before mass supervision, the edge representation reads the mass with R-squared 0.003. After masshead2, R-squared 0.93. The mass information literally moved into the representation. Nodes remain unreadable for momentum — precisely the deficit addressed by the momentum head, which we will re-probe after training."

## S18 — Asymmetric latent dimensions (1 min)
"The next experiment concerns the capacity of the representation itself. A sixteen-dimensional node representation must carry roughly twelve to fourteen physical degrees of freedom — position, momentum, PID, mass — and serve nine heads; it sits at the lower bound. We widened tracks nodes to thirty-two dimensions and track-track edges to twenty-four, retraining from scratch with the v38 stack plus the mass head. The status is honest: the job was interrupted at epoch seventy-four of one hundred fifty by the job limit; the training was not converged, with best validation 39.9 against the 35.6 of the v38 family. The preliminary evaluation is therefore not representative. The plan is to resume to one hundred fifty epochs and then add the structure and momentum heads at reduced weights."

## S19 — Public-data verification (1 min)
"We also verified the v38 stack on the published dataset. Two caveats: the published data has no PID features, and the class-two and class-three edge counts differ from our CERN Monte Carlo by a factor of four to fourteen. The first run was interrupted at epoch sixty-six; the resumed run improved the best validation to 33.4 at epoch seventy-three. The evaluation is queued on the cluster. A fair comparison with the paper requires adapting the stack to the PID-less dataset first."

## S20 — PV subgraph training: a closed line (1.5 min)
"The second closed line is PV subgraph training. The rationale was to split the event into per-PV subgraphs of twenty to thirty nodes, reducing cross-chain interference. The failure mode was structural: the backbone was trained only on subgraphs, while inference runs the GNN on the full graph first. Full-graph forward ability degraded, and class-one accuracy fell from 76.8 to 56.4 percent. Inference-only clustering also showed no gain — 28.7 against the 29.3 baseline. The conclusion is that a train–inference mismatch in the graph structure is not fixable by validation-side alignment, and the line was closed. The clustering idea is nevertheless pursued differently, through learned deterministic annealing, which we describe shortly."

## S21 — Results summary (45 s)
"The consolidated numbers are on this slide. Everything is on CERN Monte Carlo at threshold 0.9, full twenty-file evaluation unless noted. Masshead2 at 32.7 percent PerfectReco is the best model. The ablation rows use a small five-file evaluation; they are directionally valid but not directly comparable to the full-evaluation rows."

## S22 — New directions (1.5 min)
"Two new directions. The first is learned deterministic annealing for inference-side PV clustering: a trained MLP provides the track-to-PV affinity, and deterministic annealing produces a soft assignment with an adaptive cluster count. There is no Gumbel sampling and no training-side subgraph split, which avoids the failure mode of the closed line. The core module is implemented and CPU-verified. The second direction is chain scoring for trigger-line assistance: chain criteria extracted from the model outputs reach an AUC of 0.90 including random combinations, 0.78 against realistic pruned components only, and the strongest single feature reaches 0.956. The scorer training script is ready and requires GPU time."

## S23 — Open questions (1.5 min)
"I close with the questions on which we would value the group's input. First, the application form: trigger assistance through candidate scoring versus full-event reconstruction — which metric would the collaboration trust? Second, given that parent–child track relations are physically rare, is same-source clustering — class two — the real problem DFEI should solve? Third, since invariant mass is analytic at trigger level, is the GNN's added value structure and context rather than kinematics? Fourth, could DFEI same-source clusters feed a transformer-based flavor tagger? Fifth, what is a realistic HLT2 or Upgrade-II time budget for a lightweight GNN per event? Sixth, is the physics-supervision direction with linear-probe verification worth pursuing, and are there better physical targets?"

## S24 — Next steps (30 s)
"The immediate next steps are: resume the asymmetric training to one hundred fifty epochs, then add the structure and momentum heads at reduced weights; run the queued evaluations once GPUs are free; train the chain scorer and measure candidate-chain AUC on a full evaluation; plug the GNN latent affinity into the learned deterministic annealing; and re-probe the node representations after the momentum-head training."

## S25 — Backup (optional)
"The hardware and reproducibility setup: a single GPU with a ten-gigabyte memory cap, batch eight, roughly forty to sixty minutes per epoch, with long queues under heavy cluster load. All changes are reproducible from the configuration files, and a version log is maintained in the repository."

---

## Q&A cheat-sheet

**Q: Why does the mass head improve reconstruction and not only classification?**
A: The mass supervision forces the edge representation to encode sister relations, which is exactly class two — the edges that glue a chain together. The 6.4-point class-two gain is the mechanism, and class-two edges directly determine chain survival.

**Q: Why not feed m_ππ as an input feature instead of supervising with it?**
A: Two reasons. First, it would break the end-to-end claim: the network would not learn to derive the quantity from momenta. Second, our verification methodology is representation probing — if a physical quantity matters, the model should learn to hold it internally.

**Q: Is the 74-epoch asymmetric run a failure?**
A: No; it is an incomplete run. From-scratch training at roughly 55 minutes per epoch, interrupted by the job limit with validation still decreasing. Whether wider latents help is still an open question.

**Q: Why did the combined heads fail while each individually helps?**
A: Gradient competition: the auxiliary losses together exceeded the main task. The ablation shows each head is individually positive; the combination only needs lower weights. This is the standard multi-task trade-off, and we now have the numbers to set it correctly.

**Q: Why was the PV subgraph line closed?**
A: Because the backbone was trained on subgraphs but must run on full graphs at inference. That mismatch degraded full-graph ability and is not fixable by validation-side alignment. The clustering idea is being pursued as inference-side deterministic annealing instead.
