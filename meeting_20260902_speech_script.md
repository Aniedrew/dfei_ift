# DFEI Group Meeting — Speech Script (2026-09-02, v4, formal)

> 12 slides · target ~18 min · formal register

---

## S1 — Title (10 s)
"Good morning. This presentation gives an account of the DFEI optimization work carried out on the CERN Monte Carlo production, organized by theme rather than by chronology."

## S2 — Data and overall effect (1 min)
"Everything in this report uses the CERN official Monte Carlo production, DFEI_IFT_20260702, evaluated at threshold 0.9 on twenty test files. The headline result: since fixing a silent class-weight bug, the optimization line has raised PerfectReco from 23.9 to 32.7 percent, and AllParticles from 43.4 to 55.9 percent. The four panels show the trajectory of PerfectReco, AllParticles, and the two LCAG class accuracies across the main versions, from v31 to v47."

## S3 — Outline (30 s)
"The report is organized in four parts. Part one concerns supervision and the alignment of training with inference. Part two is about supervising the representations with physical and structural quantities — several heads share this idea. Part three covers two controlled failures and their lessons. Part four lists the ongoing attempts, most prominently the wider latent space."

## S4 — Differentiable pruning with annealing (2 min)
"Part one, first step: the training–inference mismatch. Training optimizes the full graph, while inference first prunes nodes and edges with hard thresholds. To remove this gap we apply a soft mask to the weights during training: the effective weight is the weight times a sigmoid of weight minus cut over a temperature. The temperature is annealed from one down to one tenth over the run. At high temperature the mask is smooth and the gradients are stable; at low temperature the mask approximates the hard threshold that inference applies. The pruning cut itself is aligned to inference and tightened version by version — 0.5 in v36, 0.7 in v37, 0.85 in v38 — converging toward the 0.9 threshold used at inference. The model therefore learns to separate signal and background weights under the same conditions it encounters at inference."

## S5 — LCA supervision adjustment (2 min)
"The second step of part one concerns the LCA supervision. Class two — two tracks from the same mother — is the structural bottleneck, so we weighted it explicitly. Weight 3.0 in v37 over-weighted the class and hurt class one, so we settled at 2.0 in v38. In parallel we introduced in-chain consistency losses, supervised only on edges that lie on truth chains. Version 37 added a hinge loss that keeps chain edges confident. Version 38 added a direct cross-entropy on the chain-edge classes. The reason is quantitative: structural edges are only about one tenth of one percent of all edges, so without direct cross-entropy the dominant background class dilutes their signal entirely. The net effect of part one is PerfectReco from 23.9 to 29.3 percent."

## S6 — Physics supervision: one common idea (1.5 min)
"Part two. The organizing idea is simple: physical and structural quantities are data-intrinsic — they are computed from the tracks themselves — so we supervise the representation with them, while keeping the model end-to-end. Several heads share this theme. The source head, added early in v36, supervises the Rumor-Centrality root. The mass head regresses the log10 of the two-pion invariant mass at the edge level. The structure head regresses node depth and Rumor-Centrality value. The momentum head regresses the normalized node momentum. All of them are verified with linear probes on the frozen backbone."

## S7 — Mass head: main result (1.5 min)
"The mass head is the main result. Two tracks from the same mother sit near a resonance mass, so supervising the pair mass forces the edge representation to encode sister relations. The implementation is a SmoothL1 regression of the log10 mass, with the sentinel edges masked and weight one. On the same twenty test files, PerfectReco improves from 29.3 to 32.7, AllParticles from 52.1 to 55.9, and class-two accuracy from 44.7 to 51.1 — the class that glues chains together. The ROC curve on the right is the class-two discrimination of the resulting model."

## S8 — Structure and momentum heads (1.5 min)
"The structure and momentum heads supervise the nodes. The structure head regresses depth — the BFS distance to the chain centroid — and the Rumor-Centrality value, i.e. the position of a node in the tree. The momentum head regresses the normalized momentum. The table reports a controlled ablation from the masshead2 checkpoint on a small five-file evaluation: the structure head is the best single head, with AllParticles up 3.1 points; the momentum head adds 2.4. The momentum head was motivated by a probe: node representations were linearly unreadable for momentum, with R-squared essentially zero. The general lesson is procedural — probe first, identify the missing quantity, then supervise it."

## S9 — Verification by linear probes (1 min)
"The verification is quantitative. We freeze the backbone and regress the physical quantity with a linear model. Before mass supervision, the edge representation reads the mass with R-squared 0.003; after masshead2, 0.93. The mass information is demonstrably inside the representation. Nodes remain unreadable for momentum, which is precisely the deficit addressed by the momentum head, to be re-probed after training."

## S10 — Controlled failures (2 min)
"Part three: two controlled failures, both instructive. The first is PV subgraph training. The rationale was to reduce cross-chain interference by training on per-PV subgraphs. The failure was structural: the backbone was trained only on subgraphs, while inference runs the GNN on the full graph first; full-graph ability degraded, and class-one accuracy fell from 76.8 to 56.4 percent. The line was closed. The second failure is the simultaneous combination of mass, structure, and momentum heads: the auxiliary losses summed to 0.877, exceeding the main task at 0.559, and reconstruction dropped by five points even though the LCAG classification did not degrade. The shared backbone was pulled toward the auxiliary tasks. The resolution is to reduce the auxiliary weights — momentum to 0.2 and structure to 0.3. Two lessons: a train–inference mismatch in the graph structure is fatal, and the gradient balance between heads must be explicit."

## S11 — Ongoing attempts (1.5 min)
"Part four lists what is in progress. First, a wider latent space: tracks nodes are widened to 32 dimensions and track-track edges to 24, on the argument that the current 16 dimensions sit at the lower bound of the physical degrees of freedom. The from-scratch training was interrupted at epoch 74 of 150 and has not converged; a resume is planned. Second, public-data verification: the published dataset has no PID and its class-two and class-three edge counts differ from ours by a factor of four to fourteen; the resumed run reached a best validation of 33.4 at epoch 73. Third, learned deterministic annealing for inference-side PV clustering with a learned affinity, which avoids subgraph training; the core module is CPU-verified. Fourth, chain scoring for trigger-line assistance: the criteria reach an AUC of 0.90, and 0.78 against realistic pruned components; the scorer training is ready and awaits GPU time."

## S12 — Next steps and open questions (1 min)
"Finally, the next steps and open questions. We will resume the wider-latent training to 150 epochs and layer the structure and momentum heads at reduced weights; run the queued evaluations; train the chain scorer; and plug the GNN affinity into the learned deterministic annealing. The questions on which we would value the group's input: should the target be trigger assistance through candidate scoring, or full-event reconstruction, and which metric would the collaboration trust? Given that parent–child track relations are physically rare, is same-source clustering — class two — the real problem to solve? And what is a realistic HLT2 or Upgrade-II time budget for a lightweight GNN per event?"

---

## Q&A cheat-sheet

**Q: Why does the mass head improve reconstruction and not only classification?**
A: The mass supervision forces the edge representation to encode sister relations — class two — and class-two edges determine chain survival. The 6.4-point class-two gain is the mechanism.

**Q: Why not feed m_ππ as an input feature instead of supervising with it?**
A: Two reasons: it would break the end-to-end claim, and our verification method is representation probing — if a physical quantity matters, the model should learn to hold it internally.

**Q: What exactly is the annealing in the pruning?**
A: A temperature in the sigmoid mask, w·σ((w−cut)/τ), annealed from 1.0 to 0.1. High τ gives a smooth mask and stable gradients; low τ makes the mask behave like the hard threshold used at inference.

**Q: Is the 74-epoch wider-latent run a failure?**
A: No; it is an incomplete run, interrupted by the job limit with validation still decreasing. Whether wider latents help remains open.

**Q: Why did the combined heads fail while each individually helps?**
A: Gradient competition — the auxiliary losses together exceeded the main task. The ablation shows each head is individually positive; the combination only needs lower weights.

**Q: Why was the PV subgraph line closed?**
A: The backbone was trained on subgraphs but must run on full graphs at inference; that mismatch degraded full-graph ability and is not fixable by validation-side alignment. The clustering idea is being pursued as inference-side deterministic annealing instead.
