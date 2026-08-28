# English Speaking Script (讲稿) — DFEI Group Meeting 2026-08-11

Corresponds to `meeting_20260811_slides.md` (same folder). ~1 min per slide.

---

## Slide 1 — Title
"Good afternoon everyone. This is a progress report on inclusive beauty-hadron reconstruction with the HGNN — the DFEI project. I will focus on three things: the reconstruction results, the training-speed problem, and the pruning problem together with the fixes I am planning."

---

## Slide 2 — What this report covers
"I will first show the reconstruction results on the official CERN Monte Carlo and on the public dataset. Then I will talk about a concrete problem: training is slow on our 10-gigabyte GPU. Then I will explain why hard-threshold pruning loses signal chains, and finally the three changes I propose to fix it — for edges, for nodes, and for selecting the final reconstruction."

---

## Slide 3 — Setup
"Briefly, the setup. The code is on the yukai_IFT branch. I use two datasets: the official CERN Monte Carlo, about ninety tracks per event, and the public paper dataset, about a hundred and forty tracks per event. Everything runs on an RTX 2080 Ti with 10.6 gigabytes of memory. The model is the HGNN with four message-passing blocks, about 570 thousand parameters."

---

## Slide 4 — Reconstruction results
"Here are the reconstruction results at threshold zero-point-two, per chain. On the CERN data, 27.7 percent of chains are perfectly reconstructed, 55.6 percent have all particles found, and 22.5 percent are not found at all. Per event, the perfect rate is about 21.6 percent. The public dataset gives about 19.8 percent per chain. As a reference, the paper reports 21.5 percent per event on their own data, but on a harder test set and with a different metric definition, so it is not directly comparable. A retraining with corrected class weights is in progress."

---

## Slide 5 — Problem 1: training is slow
"The first practical problem is speed. One epoch takes thirty to fifty minutes on the 2080 Ti. The reason is the graph: about 150 tracks, fully connected, so 23 thousand edges per event, and the 10-gigabyte memory forces small batches. A hundred epochs therefore take days. The CERN retrain has been running for more than two days and is at epoch sixty-one. To manage this I log metrics every epoch, get notifications when jobs finish or fail, and use a watchdog that resubmits jobs if the farm gives us the faulty GPU."

---

## Slide 6 — Problem 2: pruning loses chains
"The second problem is more fundamental. Perfect reconstruction equals the chain surviving pruning, times the structure being correct. At threshold zero-point-two, pruning kills about 30 percent of chains through edges and 26 percent through nodes, and these are heavily overlapping — edges dominate. And even among chains that survive, only about 40 percent get the exact LCA structure right. So there are two bottlenecks: the greedy hard threshold, and the classification quality."

---

## Slide 7 — Pruning heads are strong
"An important observation: the pruning heads themselves are very good. The edge-pruning AUC is above 0.999 on both datasets, and the node-pruning AUC is 0.97 to 1.0. So the ranking is nearly perfect — the loss comes from the fixed 0.2 cut deleting true edges whose scores fall just below it. Also, the two datasets differ: the public data has class-two and class-three edges, while the CERN data has almost none."

---

## Slide 8 — Fix 1: edges — top-k per track
"The first fix is for edges. Instead of a global threshold, each track keeps only its top-k most confident edges, with k around ten to twenty. This gives a hard cap on the candidate set — about 1,500 pairs instead of 11,000. Inside the top-k we do not hard-delete anything, so low-confidence true edges survive and the LCA classifier decides their class. Let me clarify why I do not just keep everything with soft weights: the tree reconstruction is a discrete clustering, so continuous weights have no entry point, and keeping all edges would flood the clustering with background and merge the whole event into one cluster — it is not just slow, it is wrong. This is a decoding-only change, so I can test it immediately."

---

## Slide 9 — Fix 2: nodes — train with a smooth pruning mask
"The second fix is for nodes, because nodes and edges have very different scales — about 150 nodes versus 23,000 edges per event. The idea is to make node pruning a smooth mask during training, with temperature annealing, so the model literally experiences pruning and becomes robust to it — this closes the gap between training and inference. Technically it is vectorized tensor operations, so it is GPU-parallel and costs almost nothing. And it matters here: node pruning alone kills about a quarter of the chains on the CERN data."

---

## Slide 10 — Fix 3: select the best reconstruction
"Third, the greedy reconstruction has no backtracking, and with softer candidates we can get several plausible decay trees for one event. So I plan to add a small MLP that scores each candidate tree — using features like chain size, edge confidence, and isolation from background — and selects the most plausible one. This needs one extra supervised training step, where the label is the tree that matches the truth."

---

## Slide 11 — Expected impact
"Putting numbers on it: the current model gives 27.7 percent perfect per chain. Retraining with corrected weights should bring it to about 42 to 50 percent. Adding the three fixes pushes it to roughly 55 to 65 percent per chain, about 43 to 51 percent per event. These are estimates anchored on the paper's class accuracies, so there is uncertainty — the CERN data has almost no class-two and class-three, and the hyper-parameters are not tuned yet."

---

## Slide 12 — Next steps
"Finally, the plan. Finish and evaluate the CERN retrain, re-evaluate the public retrain at the paper threshold, implement the edge top-k change, put the node smooth-mask and the selection MLP into the next retrain, and clarify with the group the track-sorting issue in the current CERN production data. Thank you — I am happy to take questions."
