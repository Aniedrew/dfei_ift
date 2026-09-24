# Replies to the questions on the 2026-09-15 DFEI talk (short version)

---

**S2 — threshold, datasets, ROC and score distributions**

Both training and evaluation use the sample generated at `/eos/lhcb/user/y/yukaiz/DFEI_IFT_20260702`
(inclusive); it is pre-split into training / validation / test pools (200 / 5 / 20 files, disjoint
event ranges), so nothing is shared between training and evaluation. The threshold 0.90 is fixed for
all reported numbers; we scanned it once on v47 and 0.90–0.95 is a plateau, with both sides falling off
quickly below 0.90 and a 5 pp drop in AllParticles at 0.99, so 0.90 is a safe operating point rather
than a tuned optimum. The ROC curves and the signal/background score distributions already exist for
the node- and edge-pruning heads at every block (edge block 3: AUC 0.99889) — the only thing missing is
that the 0.90 working point is not marked on them. Worth keeping in mind when reading them: at this
imbalance the AUC is not informative, the score distribution at the cut is.

**S5 — "training and inference see different graphs"**

You are right, and the word "inference" in our slide was the problem — it conflated the eval-mode
forward with the reconstruction step. The GNN forward runs on the full unpruned graph in every phase,
training and test alike, and the hard cut is applied only inside the reconstruction stage, before the
LCAG decoding, exactly as you say, with the pruning scores passed on as additional node/edge features.
So the network sees the same graph; what differs is the *decoder's* graph — training computes the
chain-level loss on the unpruned graph, while the decoder only ever receives the subgraph that survived
the cut. We will reword the slide to say "reconstruction stage" instead of "inference" (one aside on
the weights: they are passed as features, yes, but in the current code they only enter the global
aggregation, not the node-to-node messages).

**S6 / S7 — how chains die, and the class-0 argument**

Yes, we looked — and you are right. Measured on v47: the edge-pruning head has recall 0.9995 on truth
structural edges, i.e. it removes 0.05 % of them, while the LCAG head labels a truth structural edge
class 0 in 0.65 % / 1.00 % / 1.23 % of the class-1 / class-2 / class-3 cases — about 1 375 edges
against 62, so the class-0 output removes roughly 20× more structural links than edge pruning does.
Node pruning removes the node together with all its track–track edges, and its recall is 0.708, so it
keeps only 71 % of signal tracks. A recovery mechanism makes sense to us too — we tried three
(seed-expand connectivity, a chain-recall loss, and a chain-scoring MLP) and all failed, but all three
were aimed at edges, so we will re-test it targeted at the track decision.

**S10 — the "not found" class**

Yes, the evaluation is on an inclusive sample. And you are right that we are not watching that class —
it is computed but never printed, so we will add it to the standard output and to the per-event
breakdown.

**S12 — the added heads**

They are auxiliary training targets, not selection heads: the four we added (source, mass, struct/RC,
momentum) are used only in the training branch and never consumed during reconstruction, so a frozen
backbone would defeat their purpose — freezing the backbone and fitting a single linear layer gives
R² 0.003 for the ππ mass, and after training with the mass head the edge mass R² is 0.930. Your concern
about different minima is real and we have seen it: the same mass head was inside the noise with a
raw-mass parameterisation and only paid after re-parameterising to log10, and focal BCE on the pruning
loss collapsed the reconstruction (14.09 % vs 40.04 % for its control) while its per-edge metric looked
fine. That is why we only accept a change that clears the ±0.43 pp seed band with a control arm.

**S15 — motivation for each direction**

Each one had a hypothesis. Capacity: the shared representation underfits — rejected, widening made it
worse (24–27 % vs 37.6 %). Attention: an LCAG decision needs information beyond the four
message-passing hops — helps on a weak base, nothing on the converged model. PV splitting: cross-PV
contamination is polluting the features — worse, and PV association degraded from ~1.9 % to 11–14 %.
Retraining the recipe: the fine-tuning was undertrained — this is the one that paid. Public data: does
the CERN recipe transfer — it does not as it stands. The shared motivation was that adding
expressiveness had stopped moving the score, which is what led to the oracle test (same network, only
the pruning replaced by truth: 39.56 % → 99.75 %).
