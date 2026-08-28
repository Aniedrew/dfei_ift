# DFEI Progress Report — Slide Content (English)
**Date:** 2026-08-11 | **Meeting:** DFEI group meeting
**Figures:** in this folder (`meeting_20260811_figs/`)

---

## Slide 1 — Title
**Title:** Inclusive Beauty-Hadron Reconstruction with HGNNs — Progress Report
**Subtitle:** official CERN MC & public dataset · RTX 2080 Ti (10 GB)
**Footer:** Guoqing Xiang · DFEI group meeting · Aug 11, 2026

---

## Slide 2 — What This Report Covers
Bullets:
- Reconstruction results on the **official CERN MC** and the **public dataset** (per-chain & per-event metrics)
- A concrete **training-speed problem** on the 10 GB GPU
- Why **hard-threshold pruning** loses signal chains (diagnosis)
- **Three targeted changes** to fix it (edges / nodes / candidate selection)

---

## Slide 3 — Setup
Table:
| Item | Detail |
|---|---|
| Code | `scalable_mtl_hgnn` (branch `yukai_IFT`, weight-fix carried locally) |
| Data 1 | Official CERN MC — `DFEI_IFT_20260702/MC_normed` (~91 trk/evt) |
| Data 2 | Public (paper) dataset — `converted_LHCbcollision` (~139 trk/evt) |
| GPU | NVIDIA **RTX 2080 Ti, 10.6 GB VRAM** |
| Model | HGNN, 4 message-passing blocks, ~570 k params |

---

## Slide 4 — Reconstruction Results (thr 0.2)
Bullets:
- **CERN MC (v25):** per-chain **Perfect 27.7%**, AllParticles 55.6%, NoneIso 6.0%, PartReco 15.9%, NotFound 22.5%; **per-event Perfect 21.6%**
- **Public (v23):** per-chain Perfect 19.8%; per-event 15.5%
- Paper WHGNN reference: per-event Perfect 21.5% (harder test set, different metric definition — not directly comparable)
- Status: retraining with corrected class weights is in progress (CERN at ~ep 61, public done)
`fig: fig07_reco_categories.png`

---

## Slide 5 — Problem 1: Training Is Slow
Bullets:
- **~30–50 min/epoch** on the 10 GB RTX 2080 Ti (fully-connected graph: ~150 tracks → ~23k edges/event) → 100 epochs ≈ days
- 10 GB VRAM forces small batches, which is the main slowdown
- Running 2 retrains in parallel; the CERN retrain has taken 2+ days to reach ep 61/100
- Monitoring: per-epoch metrics + finish/fail notifications + auto-resubmit on the farm's faulty GPU
`fig: fig10_epoch_time.png`
`fig: fig02_training_loss.png`

---

## Slide 6 — Problem 2: Hard-Threshold Pruning Loses Chains
Bullets:
- Perfect ≈ (chain **survives pruning**) × (structure correct)
- At threshold 0.2, pruning kills **30.5%** of chains via edges and **26.3%** via nodes (heavily overlapping — edges dominate)
- Only ~40% of surviving chains get the exact LCA structure right
- So there are two bottlenecks: the greedy **hard threshold**, and **classification quality**
`fig: fig06_chain_survival.png`

---

## Slide 7 — Pruning Heads Are Strong (so it's the threshold, not the heads)
Bullets:
- Edge-pruning AUC: **0.999 (CERN) / 1.000 (public)**; node-pruning AUC: **0.974 / 0.997**
- The ranking is nearly perfect — yet the 0.2 cut still deletes true edges whose scores fall below it
- Data note: public data has LCA class-2/3 edges, CERN MC has almost none
`fig: fig04_05_pruning_roc.png`
`fig: fig11_data_compare.png`

---

## Slide 8 — Fix 1 (Edges): Top-k Per Track Instead of a Global Threshold
Bullets:
- Per track, keep its **top-k most confident edges** (k≈10–20) instead of a fixed 0.2 cut
- Hard compute cap: ≤ N×k (~1.5k pairs) vs ~11k fully connected
- Inside top-k, keep low-confidence true edges (no hard delete) — the LCA classifier decides their class
- Why not "keep everything + soft weights": the tree reconstruction is a **discrete** clustering — weights have no entry point, and all background edges would merge the event into one cluster (wrong, not just slow)
- Pure decode change — can be tested on existing models immediately
`fig: fig12_eb2_schematic.png`

---

## Slide 9 — Fix 2 (Nodes): Train With a Smooth Pruning Mask
Bullets:
- Nodes and edges have very different scales (~150 vs ~23k/event), so they need different fixes
- Make node keep/drop a **smooth mask during training** (temperature-annealed), so the model "experiences" pruning and becomes robust to it (closes train/inference gap)
- Fully vectorized tensor ops → GPU-parallel, negligible cost
- Motivation: on CERN data node pruning alone kills ~26% of chains

---

## Slide 10 — Fix 3: Choose the Best Reconstruction Among Candidates
Bullets:
- The greedy reconstruction has no backtracking; softer candidates can yield **several plausible trees** per event
- Add a small **MLP that scores each candidate tree** (size, edge confidence, isolation from background) and selects the most plausible
- Needs one extra supervised training step (labels = the truth-matching tree)
`fig: fig12_eb2_schematic.png`

---

## Slide 11 — Expected Impact
Bullets:
- Current (inactive-weights model, thr 0.2): **27.7%** per-chain
- After retraining with corrected weights: **~42–50%**
- Retrain + the three fixes: **~55–65%** per-chain (≈43–51% per-event)
- Caveats: estimates anchored on paper class accuracies; CERN lacks class-2/3; hyper-parameters not tuned
`fig: fig08_expected_gain.png`

---

## Slide 12 — Next Steps
Bullets:
- Finish & evaluate the CERN retrain (v31); re-evaluate the public retrain (v30) at thr 0.2
- Implement the **edge top-k** change (decode-only) and test on existing models
- Put the **node smooth-mask** training + **selection MLP** into the next retrain
- Clarify with the group the **track-sorting issue** in the current CERN production data
