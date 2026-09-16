# DFEI Group Meeting — Speech Script (spoken English, 2026-09-15)

> 57 slides · ~25 min · natural spoken style, read it as you'd talk · [brackets] = action/pause cues · **bold** = the sentence you must not forget
>
> Section headings carry the slide range they cover — in Part 3 one section covers a whole branch, so flick through those pages while you name the attempts. The deck keeps every attempt page; this script deliberately spends the time on Part 4 instead.

---

## S1 — Title (10 s)

"Hi everyone. Today I'll go through what I've been doing with DFEI on the CERN Monte Carlo. I've organised it around the version map rather than the timeline — first the map, then every attempt, and at the end the four tests that explain what all of it actually means."

## S2 — Contents (20 s)

"Short version of the plan: the map, the main line that worked, the five branches we tried after it, the diagnosis, and what's running now. I'll move fast through the branches — there's a page for each attempt but I won't read them out — and slow down for the diagnosis."

## S3 — The complete version map (1.5 min)

"This is the whole campaign on one page. Seventy versions, every one of them its own node. The main line runs across the top: v31 → v36 → v37 → v38 → v47. Everything else hangs below it in groups — capacity, attention, restructuring, retraining, the public dataset, and the work from this month. [point] Colour is PerfectReco, so yellow-green is better; the colour bar on the right is the scale. Red dashed arrows are branches we closed. Grey means we never evaluated it properly. **The only arrow that matters is the top line — everything else is an attempt to get off it, and none of them succeeded.**"

## S4 — Part 1 head (45 s)

"Part one: v31 to v38, four changes, all of them training-time only. Nothing here touches how the model is evaluated — inference always runs at a hard cut of 0.9. And this is the only big clean gain in the campaign: PerfectReco 12.3 to 21.1, AllParticles 22.5 to 37.6."

## S5 — What each step bought (1 min)

"Step by step. v31 is the reference — I fixed a silent class-weight bug, a config key spelled with two underscores while the code read one, so one whole class was trained at zero weight. **v32 and v35 then show that plain continuation does nothing:** lower the learning rate, run 150 more epochs, no new module, and the scores don't move. So from here on every gain has to come from a new optimization. v36 gives +6.5 points, v37 only +0.7, v38 another +1.6."

## S6 — v36: differentiable pruning (1.5 min)

"The problem: training sees the full graph, but at inference we prune with a hard threshold first and only then reconstruct. So the two see different graphs.

The fix is a soft mask on the message weights: the effective weight is the weight times a sigmoid of weight-minus-cut over a temperature, annealed from 1.0 down to 0.1 inside the run. High temperature keeps gradients alive; low temperature behaves like the hard cut. And the cut itself tightened — 0.5, 0.7, 0.85 — heading towards the 0.9 used at inference. **This one change is worth six and a half points — more than everything we tried in the six months after it combined.**"

## S7 — Why chains die (1 min)

"This is the part that explains v37 and v38. Class zero, background, is 99.9% of all edges; each structural class is about 0.04%. The main loss is a global cross-entropy over every edge, so its gradient is essentially 'predict background'. **The model is therefore weakest exactly on the classes we need most**, and at inference pruning one misclassified structural edge kills the whole chain. So the chain edges have to be supervised directly."

## S8 — v37 and v38: the two rewards (slides 8–9) (2 min)

"Two rewards, one version each.

v37 does two things: it weights class-2, the sister edges, three times more, and adds a hinge — a training-only penalty that pays out while a chain edge is less than 0.3 confident, paid only on truth-chain edges. PerfectReco goes 18.8 to 19.5, so it works a little. **But class-2 accuracy goes down, from 47.7 to 44.2 — a weight of three was too much.** That's the kind of thing you only learn by trying it.

v38 then fixes that — class-2 back to two — and adds chain cross-entropy on top of the hinge. CE is minus log p-true, so **it pays for being the right class, not just for being confident**, and again only on truth-chain edges so background can't dilute it. Result: 19.5 to 21.1, and per-class accuracy over the whole part moves class-1 from 67.8 to 76.8 and class-2 from 41.3 to 47.9. The rewards fixed exactly the classes the global loss was ignoring."

## S9 — Part 2 head (30 s)

"Part two — from the loss to the representation. v38 to v47. The heads don't change the architecture at all; they change what the shared representation is forced to contain. And only one of them pays."

## S10 — The probe and the head zoo (slides 11–12) (1.5 min)

"Before adding anything we checked with a linear probe: freeze the backbone, fit one layer, see whether a physical quantity can be read out. **The physics was simply not there** — pion-pair mass gave R² = 0.003, momentum zero. So we supervised it in, and the same probe afterwards gives 0.93.

That's the logic behind the head zoo: one backbone, nine heads — five original DFEI heads and four of ours, source, mass, structure, momentum. All nine read the same 16-dimensional representation, and they're parallel: **every arrow starts at the backbone, no head depends on another.**"

## S11 — The mass head (1 min)

"The one that pays is mass. The first version regressed the raw mass and landed inside the noise; v47 adds log10 normalisation and masks the sentinel edges. **That's 21.1 to 23.2 PerfectReco and 37.6 to 39.6 AllParticles — still the best model of the campaign, and nothing since has beaten it.** The same recipe on the other three heads is a wash."

## S12 — Part 3 head (slide 14) (30 s)

"Part three: everything we tried after v47. Five branches. **From here I'll go quickly** — each attempt has its own page with what it is, how it works and what it gave, so I'll just name them and give the verdict. Two branches were closed outright, three produced one useful fact each, and none of them beat the main line."

## S13 — 3a: capacity (slides 15–20) (1 min)

"First branch: capacity — is the network just too small? Five attempts. v48 stacked three heads at once on v38: a wash, 37.36 against 37.59. v50 to v52 split them up in a small probe and found each head is individually fine, so stacking was the problem. v53 widened the latent to 32/24 and trained from scratch: 26.78, far below v38. v512 did the same widening but inherited the weights: 24.03, worse than from scratch. And v513/v514 only widened the GNN hidden layer: 25.44. **Every capacity path lands eight to fourteen points below v38 — the limit is the supervision, not the size.**"

## S14 — 3b: attention (slides 21–26) (1 min)

"Second branch: context. v510 put self-attention after the last GNN block on v38 — 38.01 and 21.75, the first sign that event context is worth something. v511 added the track-pair features as an attention bias, ParT-style: 38.39 and 22.11, plus 1.6 points, with class-1 recall at 78%. Then we put the same module on v47 — v515 gives 22.95 and v516, sixty epochs more, gives 22.72, both below v47. Two further attempts trained it as a generation from v38 instead: 22.00 and 22.69. **Same module, opposite outcome — the base matters more than the module.**"

## S15 — 3c: restructuring (slides 27–31) (45 s)

"Third branch: change the event rather than the model. The graph is highly connected — ninety to a hundred and forty nodes — and chains interfere across primary vertices. v39 split by truth PV and stopped itself after fifteen epochs. v40 made the split trainable and diverged — validation 35.9, 84.9, 114, 155. v41 fixed all three problems but ran at 3.2 hours per epoch and never converged. v42 asked the same question on fifty files and gave a clean answer: **minus 2.98 points, and class-1 accuracy from 76.8% down to 56.4%. Subgraph training damages the full-graph pass.** The inference-only version of clustering was tested too, and gave nothing. Branch closed."

## S16 — 3d: retraining the recipe (slides 32–44) (1.5 min)

"Fourth branch, and this one is a methodology question. The worry was that the greedy chain — add one ingredient, keep it only if it pays — had cut the synergies, so the recipe was lost. Twelve attempts. The protocol page shows all of them at once: the base is v500, one change per run, twenty epochs each.

Going through them quickly: B2 alone, nothing. The class-2 weight alone, nothing. The hinge alone, nothing — and the reason is nice, its penalty is identically zero because every chain edge on a trained base is already above 0.3 confident. **Chain-CE is the only one that works: plus 216 events, about three sigma.** The source head cancels out; the mass head helps its own classes but drags AllParticles; the structure head is the best single addition. The empty control — same twenty epochs, nothing changed — shows zero drift, which is what makes any of these readable. Then the two 'is it the combination' tests: keeping B2 and source as a group, no gain, so greedy roll-back isn't the explanation; and stacking mass and momentum on the best chain model, no gain either.

**The verdict: the ingredients are worth much less than the version-to-version jumps of Part 1 suggested.** I'll come back to why that comparison is misleading in Part 4."

## S17 — 3e: the public dataset (slides 45–49) (45 s)

"Fifth branch: the same question on public data. It's a genuine distribution shift — no PID, and four to fourteen times more class-2/class-3 edges. The reference is the plain stack, 51.66 and 22.76. v45 transferred the whole CERN recipe: 19.70 and 8.06 — negative transfer. v49 continued it: 14.15 and 5.95, so it's the recipe, not the budget. And then v60 rebuilt it layer by layer from the simple stack: **54.52 and 23.43, above the reference — the pipeline transfers, the copied recipe doesn't.** v61 is the next layer and is still running."

## S18 — Part 4 head (slide 50) (30 s)

"Now Part four, and this is where I'd like to spend the remaining time. **Four tests. And I want to present them properly, because they changed how I read everything I just showed you.** The short version: the reconstruction is not the bottleneck; the pruning classification is."

## S19 — The oracle test: what it is (slide 51) (1.5 min)

"So the first and most important test. The question is: how much of the score is decided before the network even runs?

The method is an intervention, and it's deliberately crude. I take v47 exactly as it is — frozen, never retrained — and at reconstruction time I **replace the point and edge pruning decisions with Monte Carlo truth**. The truth chain is known from the simulation, so I can simply overwrite the decision of which tracks and which edges survive. Nothing else changes. **Because only the input changes, any difference in the score is attributable to pruning alone** — it can't be the weights, because I never touch them.

There are six variants. Fix point recall — put back the truth tracks that were wrongly deleted. Fix point precision — remove the false positives, leaving recall alone. Fix both. Fix only the edges. Fix everything. And a reverse control: instead of removing errors, add fake tracks. That last one matters, because if the reverse control also improved things, the whole test would be meaningless.

And it decomposes cleanly, which is why I trust it: a chain scores only if it survives the point prune and is then recovered. So AllParticles = the probability the chain survives, times the probability it's recovered once it survived. Two factors, and I can measure each."

## S20 — The oracle result (slide 52) (1.5 min)

"Here's what came out. Baseline is 39.56 AllParticles. Fix point recall only: 54.09. Fix point precision only: 54.68. So **the point head is losing us about fifteen points on each side, and they're independent** — repair either one and you gain fifteen. Fix both point axes: 71.99. Fix only the edges: 71.10, so the edge head is worth about thirty-one points. Fix everything and the same pipeline, same weights, same reconstruction, reaches 99.75.

And the reverse control does what it should: adding fake tracks collapses the score to 16.91. So the effect is real, and it's signed in the right direction.

On the right you can see the decomposition. Baseline: chains survive the point prune 70.8% of the time, and of those 55.9% are recovered. With the point decisions replaced by truth: 95.3% survive, and 75.5% are recovered — **the reconstruction itself improves too, because it's now being handed the right graph.** With the edges fixed: 100% of surviving chains are recovered.

**So the ceiling is not the model. The headroom is entirely in the classification that decides which tracks and edges survive.**"

## S21 — The plateau (slide 53) (1.5 min)

"Second test — and this is really a calibration test, because it tells me which of my own past conclusions I'm allowed to believe.

After v47 I ran sixteen more configurations: capacity, attention, long generations, recombination. **Only one of them even ties it.** More compute doesn't help — v520 ran 179 epochs on v47 and still ended half a point below, with a flat validation curve. More model doesn't help — the widening attempts cost seven to thirteen points.

And then the honest part: I measured the noise. Four runs that are functionally identical — same config, different seed — spread over ±0.43 points, which is ±75 events. **So anything under about half a point in this campaign is not a result.** That number is why I went back and re-read the ablation chain rather than trusting it, and it's why I've been careful in Part 3 about which verdicts I call real."

## S22 — Single changes vs combinations (slide 54) (1 min)

"Third test, and it follows from the first two. Four single changes clear the noise floor on their own: chain-CE, the edge-loss weight going from thirty-three to three, attention with the edge bias, and the learning rate at 3e-4. So I tried every combination of them — chain-CE plus structure, mass plus momentum, B2 plus source, attention plus mass, and finally all four winners stacked.

**Every single combination failed.** Combining confirmed winners has never produced a gain here. That's the signature of a plateau rather than of bad ideas — and read together with the oracle, it says the headroom isn't in adding more supervision, it's in the pruning heads."

## S23 — The two concrete causes (slide 55) (1.5 min)

"Fourth test, and this is the one that turns the diagnosis into something I can actually change in the code. Two causes.

First, the model was never frozen — I just had the step size ten times too small. The test is a learning-rate ladder: the same base, the same twenty epochs, at 3e-5, 1e-4 and 3e-4. **At 3e-5 nothing moves; at 3e-4 the same twenty epochs gain 5.9 points and tie the best model of the campaign.** So every 'this doesn't work' verdict from the ablation chain was taken at a step size too small to see anything, and those verdicts are not trustworthy — which is exactly the point I flagged back in Part 3.

Second, the loss was never rebalanced. It weights the edge head thirty-three times the point head, and the config keys I wrote to rebalance it were never actually read by the code — dead keys.

**And both causes point at the same place: the point-pruning head** — the one the oracle says is worth fifteen points on recall and fifteen on precision."

## S24 — What is running now (slide 56) (1 min)

"So this is what's running. v552 raises the point-loss weight from one to five. v553 lowers the edge weight from thirty-three to three — **and it already gives 22.84, above v38, which confirms the edge head was dominating the objective.** v554 adds focal loss to push the hard examples, aimed at precision. v555 is a chain-level recall loss, on the logic that a chain dies at its weakest link rather than on average. v556 stacks every confirmed lever on v551 with the larger learning rate — the target is to break 40% AllParticles. And v557 to v561 are the high-learning-rate controls, including a re-test of the old B2 verdict."

## S25 — Summary (slide 57) (1.5 min)

"To wrap up. The main line is real: PerfectReco 12.3 to 23.2, AllParticles 22.5 to 39.6. Nothing added after v47 beats it — not capacity, not attention, not longer training. **The oracle says the ceiling is 99.75 with the same weights, so the reconstruction is not the bottleneck — the pruning classification is**, and the point head is worth about fifteen points each way while carrying one thirty-third of the loss weight.

Next: finish the pruning-reweighting arms, re-test the old verdicts at the larger step size, and if the point head responds, attack precision with hard negatives and chain-level losses. And continue the public line to see whether the levers transfer.

Thanks — happy to take questions."

---

## Likely questions and prepared answers

**Q: How exactly is the oracle intervention done — isn't it just cheating?**
A: It is cheating, deliberately, and that's the point. Nothing is retrained; I only overwrite the pruning output with truth before reconstruction, so the score I get is an upper bound for the pruning stage rather than a model. The reverse control — adding fake tracks — is there to show the bound moves in the right direction and isn't an artefact of touching the input at all.

**Q: Why do you trust the ablation chain at all, if the step size was wrong?**
A: I trust the structure, not the individual verdicts. The empty control shows the protocol has zero drift, so the comparisons are internally consistent — but they were all taken at lr 3e-5, and v551 shows the same 20 epochs at 3e-4 gain 5.9 points. So I treat the whole chain as "measured at a step size too small to see anything" and I'm re-testing the verdicts rather than relying on them.

**Q: Why does capacity not help, when nine heads share the latent space?**
A: Because widening it costs more than it buys. All three routes — latent 32/24 from scratch, the same inherited, and a wider GNN — land 8 to 14 points below v38 on AllParticles, and the two inherited ones are worse than training from scratch. With a 10 GB card and 200 files, 16 dimensions is what we can train well; the bottleneck is the supervision.

**Q: Why does attention help on v38 but hurt on v47?**
A: Sequential fine-tuning interference. v515 lands at 22.95 and v511 at 22.11 — grafting attention onto v47 gets you to roughly the same place as grafting it onto v38, which means the fine-tuning rewrites the representation it started from rather than adding to it. The module isn't the variable; the base is.

**Q: Why is the mass head the only one that pays?**
A: Because mass is the one quantity the reconstruction objective can't infer. The probe says the backbone held R² = 0.003 for the pion-pair mass, so nothing else in the loss was going to produce it. Root-ness, depth and momentum are partly visible from connectivity alone; mass is not.

**Q: Is chain-CE really the only thing that works, or is that a small-sample effect?**
A: It's plus 216 events against a ±75 event noise floor — about three sigma, the one result in the chain that clearly clears it. Everything else in the chain is inside the noise.

**Q: Does the public line mean the CERN recipe is wrong?**
A: It means the recipe is over-fitted to the CERN background level. The public sample has no PID and four to fourteen times more class-2/class-3 edges, so B2 at cut 0.85 prunes far too aggressively there. Rebuilding layer by layer from the simple base works — v60 beats v27 — so the pipeline transfers; the recipe doesn't.

**Q: What would you do with more compute?**
A: Not width — that's measured. I'd spend it on the point-pruning head: longer runs at a larger learning rate, with the loss rebalanced and a recall-oriented objective. That's where the oracle says the fifteen points are.

**Q: So what is the single most valuable thing in this whole set of experiments?**
A: The oracle test. Everything else told me what didn't work; that one told me where to look. It cost a day and it redirected the project from the network to the pruning heads.
