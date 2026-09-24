# Replies to the questions on the 2026-09-15 DFEI talk

> Slide numbering refers to the short group deck (`DFEI_meeting_20260915_group_EN.pptx`, 17 slides).
> All numbers below are on the CERN official MC sample, 20 test files, threshold 0.90, and use the
> **fixed denominator** of 17 561 truth B candidates (see S2 below). Quoted uncertainties are
> binomial; the run-to-run spread from the random seed is **±0.43 pp**, so anything smaller is not
> a result.
> Anything marked **[to do]** is not measured yet and is listed again at the end.

---

## S2 — threshold, datasets, ROC and score distributions

> *"Why did u set the threshold to 0.9, which dataset has been used and which one for the evalulation.
> In principal it generates a roc curve and the NN weights distribution between signal and background
> which u can take a look at."*

**Datasets.** CERN official Monte Carlo, single sample `inclusive_00342442`
(`…/DFEI_IFT_20260702/data/MC_normed/inclusive_00342442/`). The directory is pre-split into three
disjoint file pools, and the loader picks by filename prefix, not by index
(`wmpgnn/data_loader/chunk_loader.py:242-290`):

| pool | files | events | used for |
|---|---|---|---|
| `trn_data_*` | 200 (of 200) | 0 – 19 999 | training |
| `val_data_*` | first 5 of 20 | 20 000 – 21 999 | validation / checkpoint selection |
| `tst_data_*` | 20 (of 20) | 22 000 – 23 999 | **all reported numbers** |

Training and evaluation do not share a single event — the split is by event range, and there is no
offset parameter to get wrong. The evaluation sample is inclusive: every B flavour in the event
counts, no decay channel is selected.

**Threshold.** 0.90 is not a free parameter tuned at the end; it is fixed for training-time evaluation
and for every comparison in the talk. We did scan it once, on v47, to check we are not sitting on a
cliff:

| threshold | surviving chains N | AllParticles [%] | PerfectReco [%] |
|---|---|---|---|
| 0.80 | 14 015 | 30.07 | 16.80 |
| 0.85 | 13 361 | 33.82 | 19.09 |
| 0.88 | 12 879 | 36.28 | 20.53 |
| **0.90** | **12 436** | **39.56** | **23.15** |
| 0.95 | 11 106 | 40.11 | 22.90 |
| 0.99 | 7 480 | 34.24 | 19.10 |

So 0.90–0.95 is a plateau and both sides fall off a cliff. One honest caveat: at 0.95 AllParticles is
nominally 0.55 pp higher while PerfectReco is 0.25 pp lower — 0.55 pp is 1.3× the seed band, so this
is right at the edge of being a real difference rather than noise. We keep 0.90 because it maximises
PerfectReco and keeps the evaluation self-consistent with the training recipe, but if the plateau
matters we should quote both points.

**ROC and score distributions.** These already exist and are produced by every evaluation —
per graph-network block, for the node-pruning head (`NN_nodes_{0..3}_roc/_decision`) and for the
edge-pruning head (`NN_edges_{0..3}_roc/_decision`); signal = truth signal tracks (`ft != 1`) and
truth structural edges (`y > 0`), background = the rest
(`wmpgnn/lightning_module/dfei_lightning_module.py:1200-1215`). For example edge block 3 has
AUC = 0.99889.

Two things are missing and are worth fixing before quoting them:

1. The 0.90 working point is not drawn on the plots.
2. There is no equivalent plot for the 4-class LCAG head, whose **class-0 output** turns out to be the
   dominant structural effect (see S6/S7).

And a warning about reading them: at this imbalance (background is 99.9 % of edges) the AUC is
almost uninformative — an AUC of 0.999 coexists with a precision of 0.56 at the working point
(S6/S7). The score distribution at the cut is the plot that means something.

**Why the denominator changed.** Every number in the talk divides by the truth chains on the
*unpruned* graph, 17 561, a constant. The previous convention divided by the truth chains that
survived pruning *in that run*, which shrinks as the cut gets harder, so the percentage inflates with
the threshold. That is why the numbers are lower than the ones circulated earlier — the flags did not
change, the denominator did.

---

## S5 — "training and inference see different graphs"

> *"What do you mean by 'training and inference' see different graph? During the inference per default
> the full graph is used and only during reconstruction pruning is applied. The pruning weights should
> per default passed to as an additional edge or node feature."*

**You are right, and our phrasing was wrong.** The network does not see a different graph. Concretely:

- The GNN forward runs on the **full, unpruned graph in both phases**
  (`dfei_lightning_module.py:710`, `outputs = self.model(batch)`). The hard cut happens *after* the
  forward, inside the reconstruction stage and before the LCAG decoding / chain assembly
  (`wmpgnn/reconstruction/reconstruction.py:228-276`).
- The pruning scores are passed on as node/edge features and weights, exactly as you say
  (`use_node_weights`, `use_edge_weights`, `weighted_pass`).

What we meant to say is narrower: **the graph the decoder sees differs between the two phases.**
During training every loss — including the chain-level cross-entropy paid on truth-chain edges — is
computed on the unpruned graph (`dfei_lightning_module.py:1088-1123`). At inference the decoder only
ever sees the subgraph above the cut. So the model is never asked, during training, to be right on the
graph the decoder actually receives. That is the gap B2 closed from the training side: the message
weight becomes `w · σ((w − cut)/τ)` with τ annealed 1.0 → 0.1, so the training-time weights converge to
the hard cut used in reconstruction (`wmpgnn/model/gnn/hetero_graph_network.py:111-133,174-178`;
`dfei_lightning_module.py:1125-1136`). It is a training-only change and inference is untouched —
as you say it should be.

**Two things we found while checking this, which you may want to know about:**

1. `weighted_pass` only takes effect in the **global** block. `HeteroNodeBlock` is constructed without
   `weighted_mp` (`hetero_graph_network.py:46-50`), and the aggregator multiplies by the weight only
   when `_weighted` is set (`wmpgnn/model/blocks/hetero_aggregators.py:70-73`). So the edge weights
   modulate the aggregation into globals but **not** the node-to-node message passing. If the intent
   is that the pruning weights also modulate node-level messages, this is a bug.
2. In our configs `node_prune_weights` / `edge_prune_weights` are **dead keys** — nothing reads them.
   The live flags are `use_node_weights` / `use_edge_weights` / `weighted_pass`. Our configs set both
   sets, so the behaviour is the intended one, but the naming is misleading and it cost us time.
   Related: `HeteroGraphNetwork` carries internal `self.edge_prune` / `self.node_prune` hard-pruning
   branches that are never enabled (dead code), and one of them calls `faster_node_pruning`, which is
   not defined anywhere in the repo.

---

## S6 / S7 — how chains die, and the class-imbalance argument

> *"Did u look at the percentage of cases where this happens. From my experience the edge pruning has
> a smaller impact than an LCAG classification of 0. So the LCAG class 0 prediction removes more non
> insolated nodes to the graph. Also how is the node pruning included, as it would remove all
> associated edges. But in principal a recovery mechanism makes sense for improvment."*

**You are right, and we had conflated the two mechanisms on the slide.** We have now separated them.
All numbers below are v47, CERN, threshold 0.90, 20 test files.

| head | quantity | precision | recall |
|---|---|---|---|
| node pruning, on truth signal tracks (`ft != 1`) | kept 66 505 · truth 65 650 · both 46 447 | **0.698** | **0.708** |
| edge pruning, on truth structural edges (`y > 0`, node-pruned graph) | kept 217 746 · truth 121 770 · both 121 708 | **0.559** | **0.9995** |

So:

- **The edge-pruning head destroys essentially no structural edges — 0.05 % of them.** Whatever the
  edge pruning is doing, it is not what breaks chains. This matches your experience.
- **The LCAG head calls a truth structural edge "class 0" in 0.65 % (class 1, n = 87 666), 1.00 %
  (class 2, n = 75 404) and 1.23 % (class 3, n = 4 156) of cases** — about 1 375 truth structural
  edges per test sample. An edge predicted class 0 is dropped before chain assembly
  (`wmpgnn/reconstruction/reco_helper.py:172`), so the LCAG class-0 output removes roughly **20× more
  structural links than the edge-pruning head does** (1 375 vs the 62 edges above). Your statement
  holds quantitatively in our setup.
- The node-pruning head is the other large lever: it keeps only 71 % of signal tracks, and 30 % of
  what it keeps is background.

Our slide said "an edge the classifier calls background gets pruned" and showed a chain dying from one
misclassified edge. That is true but it is the *LCAG* classifier doing it, not the pruning head, and
we should have labelled it that way. **[to do]** we will fix the slide.

**Chain-level cascade.** The reason a ~1 % per-edge failure is expensive: a chain survives only if
*all* its nodes **and** *all* its structural edges survive — an AND condition. From our chain-survival
study (paper sample, ~1000 events, threshold 0.2): node pruning kept 98.3 % of signal tracks and edge
pruning kept 93.6 % of truth structural edges, yet only ~82 % of chains survived intact. So small
per-edge losses are multiplied up at the chain level. That is the mechanism worth putting on the
slide, and it is the quantitative version of your point.

**Oracle decomposition (v47, fixed denominator 17 561).** We freeze the weights and replace only the
pruning decisions with truth:

| intervention | AllParticles [%] | PerfectReco [%] |
|---|---|---|
| baseline (predicted) | 39.56 | 23.15 |
| add back the missed signal tracks only (node recall perfect) | 54.09 | 32.06 |
| drop the kept background tracks only (node precision perfect) | 54.68 | 30.60 |
| perfect node pruning | 71.99 | 41.67 |
| perfect edge pruning | 71.10 | 40.08 |
| both | **99.75** | **57.82** |
| node pruning switched off, edge pruning still on | 12.64 | 6.69 |
| both switched off | 0.96 | 0.48 |

(the two right-hand columns are the fixed-denominator versions, i.e. counts divided by 17 561)

So with the same network and only the pruning decisions replaced by truth, AllParticles goes from
39.56 % to 99.75 %. Note what this table does **not** contain: it replaces the *pruning* decisions,
never the *LCAG classification*. **[to do]** The measurement your comment asks for is the missing
one — replace the predicted edge class with the truth class on the surviving graph and see how much
chain yield comes back. That isolates the class-0 effect directly, and we will queue it.

**How node pruning is implemented.** `true_node_pruning` (`wmpgnn/util/pruners.py:51-99`) removes the
node and all its per-node attributes, keeps only the tt edges whose **both** endpoints survive, and
re-indexes those edges; it also updates the track count in the global feature. So yes — it removes all
associated edges, as you say. One caveat we found: at the call site (`reconstruction.py:246`) only the
tt edge type is passed to it, so the track–PV edge index is not re-indexed. That is not currently read
after this point (we checked) and the PV quantities are masked by the same node mask downstream, so the
results are unaffected — but the graph object is left internally inconsistent and should be cleaned up.

**Recovery mechanism.** Agreed in principle, and we have tried three, all of which failed — but on the
models we had at the time, and all of them targeted *edges*:

| mechanism | what it does | result |
|---|---|---|
| scheme F "seed-expand" (`node_expand`) | keep a seed cut, then expand along top-k edges so the kept edge set is connected by construction | tested on v31: AllParticles 1.76 % vs 22.49 % baseline — much worse, abandoned |
| chain-level recall loss | push the weakest node/edge score inside each truth chain above threshold | v555: 35.19 % vs 35.17 % for its own control — a wash |
| chain-scoring MLP re-ranking candidate chains | select among candidate chains after pruning | v530: 32.58 % — worse |

Given the table above, the loss is at the **track** level (node recall 0.71), not at the edge level, so
a recovery mechanism aimed at the node decision is the right target. **[to do]** we will re-test it on
the current model rather than on the weak base where scheme F was abandoned.

---

## S10 — the "not found" reconstruction class

> *"I assume you are evaluating on an inclusive sample. If yes it would be of intrests to monitor the
> behavouir of the not found reconstruction class as well. Since looking at flavour tagging they are
> the only class where no information is passed forward."*

Yes, inclusive sample. **And you have found a real gap: we do not monitor that class.**

The four outcome classes as implemented (`wmpgnn/reconstruction/reconstruction.py:663-705`):

| class | rule |
|---|---|
| `AllParticles` | reconstructed chain has exactly the same track set as the truth chain |
| `PerfectReco` | same track set **and** the same tree (per-edge LCA values) |
| `NoneIso` | all truth tracks present, but the chain is contaminated by extra background tracks |
| `PartReco` | 20 % – 100 % of the truth tracks recovered |
| `NotFound` | none of the above — less than 20 % recovered, i.e. effectively nothing found |

`NotFound` **is** computed in the per-event dataframe but the reporting function writes only four
lines (`wmpgnn/performance/reco_accuracy.py:54-62`), so it has never appeared in any of our reports.
**[to do]** we will add it to the standard evaluation output and to the per-event breakdown by number
of B's in the event.

One related fact: the flavour-tagging head (`ft`: b̄ / background / b) exists in the model but is not
wired into this reconstruction path — no ft description is passed to `reconstruct_heavyhadrons`
(`dfei_lightning_module.py:866`). So the "no information passed forward" case is not exercised yet.
When we look at tagging, `NotFound` is exactly the class to watch, and we will make sure it is
available before then.

---

## S12 — the added heads, and the local-minima concern

> *"I would be careful with the added heads. I dont know the details of the implementations. But
> something to take into account is that the different loss functions do not neccessairly have the same
> local minima. So if even one has a different minima it could worsen the performance. Are the
> additional heads used jsut to timprove the trainign stability/performance or only applied for
> additional selections. If it is only for additional selection a frozen GNN backbone might make more
> sense for the heads."*

**They are the first thing, not the second** — auxiliary training targets, not selection heads — so a
frozen backbone would defeat their purpose. Concretely:

- The nine heads split in two. Consumed at inference: LCAG (4-class), node pruning, edge pruning,
  PV association, chain scorer. **Training-only**: source, mass, struct (depth + RC), momentum. The
  four training-only ones are only attached in the training branch and their outputs are never put on
  the outputs the reconstruction reads (`dfei_lightning_module.py:775-841` vs `847-866`).
- They are trained jointly with the backbone *on purpose*, because the probe showed the physics is not
  in the representation otherwise: freeze the backbone, fit a single linear layer, and the ππ mass
  probe gives R² = 0.003 with momentum flat at zero; after training with the mass head the edge mass
  R² is 0.930. They act on the shared representation — freeze it and there is nothing to act on.

**On the different-minima concern: agreed, and we have two cases in our own line.** (i) Regressing the
raw mass (v46) landed at 21.28 % against 21.10 % for the control — inside the noise; re-parameterising
to log10 m with the sentinel edges masked (v47) gave 23.15 %. Same head, same loss weight — so the
*geometry* of the objective, not just its presence, decides whether it helps. (ii) An auxiliary
objective that actively hurt: focal BCE on the pruning loss collapsed the reconstruction (v558 =
14.09 % AllParticles against 40.04 % for its own control) while the per-edge metric looked good.

Because of exactly this, our protocol is: one change at a time, a control arm for every addition, and
nothing is accepted below the ±0.43 pp seed band. That is how we know the mass target is what paid —
the identical recipe applied to source, struct and momentum is a wash.

One more thing you should know, because it is more important than the number of heads: the loss was
badly imbalanced by construction. With the tt-edge term at a fixed factor of 33, the measured
breakdown was 81–91 % of the combined loss on edge pruning, 6–18 % on the node-pruning head and
~1.5 % on LCAG. The node-pruning head — which we later found to be the recall bottleneck — was
receiving almost no gradient. Rebalancing that (and the learning rate, which was 10× too small for the
fine-tuning step) is what finally moved the score from 39.56 to 42.01 %.

---

## S15 — motivation for each direction

> *"Just a short remark. It is nice to test different approaches/ideas but it is always great to have
> some motivation why one test it."*

Fair, and each of the five had a stated hypothesis. What makes them worth reporting together is that
all five failed the same way, and that is what produced the diagnosis in S16.

| direction | hypothesis | outcome |
|---|---|---|
| capacity (widen the representation) | the shared representation underfits | 24–27 % AllParticles against 37.6 % — rejected, so not capacity |
| attention / cross-graph context | an LCAG decision needs information beyond the 4 message-passing hops | helps on a weak base, nothing on the converged model |
| split the event by primary vertex | cross-PV contamination is polluting the node/edge features | worse (37.59 → 36.91 / 35.46 / 33.51 for the three variants), and PV association degraded from ~1.9 % to 11–14 % mis-association |
| retrain the recipe (lr / steps / loss weights) | the fine-tuning was undertrained rather than the architecture being wrong | **this is the one that paid** — but only once the oracle said *where* to spend it |
| public dataset | does the CERN recipe transfer | it does not transfer as it stands |

The common motivation for Part 4 was deliberately narrow: every added module made the network more
expressive and none of them moved the score, so either the bottleneck is outside the network or we
cannot see it from this side. That is what motivated the oracle experiment — freeze the weights,
replace pruning with truth, same network: AllParticles 39.56 % → 99.75 %. After that, the pruning-loss
rebalance was motivated by a number rather than by a preference.

---

## Summary of what we will do

| # | action | slide |
|---|---|---|
| 1 | Add the 0.90 working point to the existing ROC / score-distribution plots, and add one-vs-rest ROC for the LCAG head; state the train/val/test split explicitly. Quote the 0.95 point as a caveat. | S2 |
| 2 | Re-word the mismatch as *decoder vs. training*, not GNN vs. inference. Investigate whether `weighted_pass` is meant to affect node aggregation as well. | S5 |
| 3 | **Run the LCAG oracle** (replace the predicted edge class with truth on the surviving graph) to isolate the class-0 effect you describe. Re-label the chain-death slide to credit the LCAG head, not the pruning head. | S6/S7 |
| 4 | Add `NotFound` to the standard report and to the per-event-class breakdown; make it available before we look at flavour tagging. | S10 |
| 5 | Keep the one-change-at-a-time + control protocol; treat the loss-weight imbalance as the thing to watch, not the number of heads. | S12 |
| 6 | Re-test a node-level recovery mechanism on the current model (scheme F was abandoned on a weak base, and the loss is at the track level, not the edge level). | S6/S7 |
