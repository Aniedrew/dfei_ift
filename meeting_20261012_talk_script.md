# DFEI Group Meeting — Talk Script (2026-10-12)

> 18 slides · ~28 min · written to be spoken, not read · [brackets] = action or pause cue · **bold** = the sentence you must not forget
>
> Goes with `DFEI_principles_20261012_v4_EN.pptx`. This deck covers **only the optimisations that worked** — three ideas plus three training details. The list of things that failed is deliberately left out; it exists and I can go through it offline.
>
> v4 makes the definitions honest about their scope. **PART 0 is a single page (S3)** — the objects, the two evaluation pools and the version codes. Everything else is defined on the slide where it is first used: `p_n` on S5, softmin on S6, the **probe** working point on S7, attention / message passing / permutation invariance on S11, the competition features on S14. The ruler that the whole talk is measured with opens PART 1 (S4). The chain-loss story now ends with the **3000-event main-model result** (S10).

---

## S1 — Title (20 s)

"Hi everyone. Today I want to do something a bit different from the usual progress report.

Over the last few months we tested a long list of ideas — eleven distinct ones, arranged in parallel. Most of them did nothing. What I want to do today is take **only** the ones that actually moved the physics, and explain properly *why* they work — derive each one rather than just show you the number.

There are three ideas and three small training details. That's the whole talk."

## S2 — Outline (25 s)

"The plan. First, **one** setup slide — because everything after it is only confusing if you have to guess what a pool, a chain or a metric name means. I'm deliberately keeping it to one page: from then on, every term I use I define right where it appears. Then the three ideas: the **objective**, the **neighbourhood**, and the **competition**. Then three training details that matter more than they look. Then the summary.

If you want the list of things that failed, I have it — sparse attention, optimal-transport matching, extra features, more capacity. I'm just not spending your time on it today."

## S3 — Setup: the objects, the two pools, the version codes (1.5 min)

"The whole of setup fits on this one slide, so let me take it slowly.

**Objects.** There are **tracks**, and one **primary vertex** — the point where the proton beams collide. A B hadron flies a little and decays at a **secondary vertex**; its decay products are the signal tracks. Every pair of tracks in the event is a **candidate edge** — a complete graph — and only about one in a thousand of those pairs is real. A **truth structural edge** is a candidate edge that actually lies on a truth chain, and a **truth chain** is one connected component of them — roughly one B. The AP numbers are read on the **hard pool**: the truth edges, plus the two kinds of fake whose *both ends are real tracks* — the same-chain-but-wrong-depth ones, `fake_intra`, and the different-chain ones, `fake_inter`.

**Pruning.** Two heads decide what survives: the **node head** asks 'is this track a signal track', the **edge head** asks 'is this edge structural'. Both use the same fixed threshold, nine tenths. Then reconstruction glues the survivors into chains — and here is the key fact: a chain counts only if **every** link survived. It is an AND.

**The two pools — and this warning matters.** Numbers from different pools are **not comparable**. The two benchmarks, five hundred events and three thousand events, are read at the **bench working point** — the fixed score threshold 0.9. The small-model **probe** is different: there the threshold is pushed to wherever edge precision hits ninety percent — the **probe working point**. Same word, 'chain survival', different ruler; I'll always say which one.

**Version codes.** Every v-six-hundred-something is just a **trained configuration of the same model** — a base plus one documented change — and they were all re-scored on the same benchmark. The two you need: **v623** is the AP champion, **v633** is the 'clean stack' base. Then **v643** adds corrected geometry, **v648** adds the competition context and becomes the base for the chain-loss arms, and **v654** is that base plus the chain-loss that worked. The script that generates them records exactly what each one changed, so nothing here is folklore."

## S4 — The ruler: chain survival, not edge AP (1.5 min)

"Before any of it: what are we actually measuring.

Our physics criterion is not edge accuracy. It is: **did this whole truth chain come back.** A chain counts only if every single edge on it is correct — an AND. So **chain survival** is the fraction of truth chains whose every track *and* every truth structural edge is above the working point.

And the working point has two readings, which I keep named apart: the benchmark uses the **fixed** score threshold 0.9, while the probe uses the **data-driven** 90-percent-edge-precision threshold. They are different rulers — the columns are not comparable.

Now look at the two columns of the figure. On the left, the eighteen versions ranked by edge-level AP. On the right, the same versions ranked by chain survival at the fixed 0.9. If the two rankings agreed, those lines would be parallel. They cross everywhere. Two examples: v632 is **last** in AP and **fourth** in chain survival; v623 is **first** in AP and **fifth** in chain survival.

**So the ruler you pick decides which model looks best.** Everything today is measured with chain survival at the 0.9 score threshold, and I'll show AP next to it only as a secondary number."

## S5 — Idea 1a: per-edge BCE optimises the wrong quantity (2 min)

"First idea, and it's the big one: the loss.

A chain survives if every edge on it is right — that's an **AND**, not an average. If the per-edge probabilities are p₁, p₂ and so on, the expected survival is the product. Our loss has always been a per-edge binary cross-entropy — BCE, the ordinary per-edge log-loss. Minimising it maximises each edge's log-probability independently, so it optimises the **average** correctness over edges. The physics wants the product; the loss optimises the mean.

And here is the bridge, which is easy to miss: **training sees the whole graph, but at inference we prune edges at the 0.9 score threshold first.** So the AND between surviving links is decided *after* the loss has already been fit to an average — the training objective and the decision rule are mis-aligned by construction.

The curve on the right is where that bites. It shows pⁿ for chains of two, three and four edges. Let me define the per-link number first: **p-n** is the n-th root of the fraction of n-edge chains that came back — the implied survival of one link, and p-two is the two-edge case. On two-edge chains our base v633 gives p-two of 0.668, survival 44.7 percent. Push p-two to 0.803 — that is v642 — and survival jumps to 64.5. Both are 500-event numbers at the fixed 0.9 threshold.

**A chain is decided by its worst edge, and a per-edge average cannot see that.**"

## S6 — Idea 1b: softmin, a differentiable weakest link (2.5 min)

"Here's how we make the weakest link differentiable.

This is the definition of a **soft minimum**: take each edge probability, scale it by γ, exponentiate, average, take the log, divide by γ, negate. That's softmin-γ. It has the two limits you want. As γ goes to zero it becomes the plain mean — smooth, gradients don't vanish. As γ goes to infinity it becomes exactly the minimum.

On the right, the curve: γ from one to a hundred, with two reference lines — the mean at 0.50 and the true minimum at 0.05. Those three numbers are a worked example, not data. We use γ equal to ten: close to the minimum, still smooth.

The loss is then the negative log of that quantity, averaged over all truth chains in the event. Two implementation details: clamp the probabilities away from zero and one for numerical stability; and group the edges by chain with a **scatter-add** — a grouped sum done in one tensor op rather than a Python loop — keyed on *event id times chain id*, because different events reuse the same chain id and if you forget the event key you silently merge two events' chains."

## S7 — Idea 1c: the fix has to be non-linear (1.5 min)

"One thing this is **not**: it is not 'replace the loss with the product'.

Everything on this slide and the next is from the **696-event multi-B probe** — the small model — not the benchmark, and the probe's own threshold is the ninety-percent edge-precision one.

The product form has a log equal to the mean of log-p — that is the per-edge cross-entropy again, just reorganised. And the measurement agrees: the product form gives 28.0 percent chain survival at that probe threshold. The weakest-link form gives 32.6. The baseline is 24.2.

So the gain does not come from changing the aggregation *target*. It comes from the **non-linearity** — from punishing the worst edge specifically.

**If it had worked as a product, we would have found it years ago, because a product is what we were already approximating.**"

## S8 — Idea 1d: the evidence, seed by seed (1.5 min)

"Here are the three seeds — one dot each. Baseline: 24.6, 23.6, 24.5. Softmin at weight one: 33.5, 28.5, 35.8. All three seeds move. The mean goes from 24.2 to 32.6 — plus eight and a half points, about a third more surviving chains. The sharpness matters too: weight 0.3, a softer minimum, gives 30.1.

And one thing I want to stress: **this is not a trade.** On the probe's own edge pool, edge AP goes up as well, 0.8393 to 0.8533, and AUC from 0.8650 to 0.8732. We are not buying chain survival with edge accuracy — both move together."

## S9 — Idea 1e: the same idea is already inside the main model (1.5 min)

"And it turns out we already had a version of this idea inside the main model. We just didn't recognise it.

Two older arms — v641 and v642 — carry a **chain_recall** loss, which is the same 'punish the weakest link' idea written differently, on the v633 base. On the 500-event benchmark they are **tenth and ninth out of eighteen** by edge AP. That is exactly why we had written them off.

Now look at chain survival, at the fixed 0.9 score threshold: **59.2 and 61.6 percent** — against 47.1 for the AP champion v623, and 42.3 for the base v633 we had been building on. And because the survival ruler uses a **fixed** threshold, this is a genuinely stronger weakest link, not a re-ranking. Per-link p-two: 0.808 and 0.803, against 0.710 for v623 and 0.668 for v633.

**The arm we had sidelined was already the best model we had, on the metric that matters.** That's a lesson about the ruler, not about the loss."

## S10 — Idea 1f: main model, 3000 events — the chain loss reproduces at scale (2 min)

"Now the real test. We took that weakest-link idea onto the main model, on the current base v648, and scored it on a **3000-event benchmark** — hard pool, thirty-three thousand edges, the same fixed score threshold 0.9. This is far beyond noise.

**It worked.** v648 to v654 — that's the chain-recall, min-pooling writing of the loss — moves chain survival from 35.8 to **52.3 percent**. Plus sixteen and a half points, a forty-six percent relative gain. And the cost is tiny: hard-pool AP goes from 0.6684 to 0.6591, minus nine thousandths. Per-link p-two goes from 0.608 to 0.728.

And look at where it helps most: the **long chains**. Three-edge chains go from 35.0 to 52.8 — a bigger jump than the two-edge chains, 36.9 to 53.0. That is exactly what the AND story predicts: the more links a chain has, the more it is decided by its worst one. This also reproduces what v641 and v642 did on v633, 59 and 62 percent, on a much larger pool — so it is the mechanism, not the particular aggregator, that matters.

The small probe — 24.2 to 32.6 — stays as the controlled experiment; this 3000-event run is the main evidence."

## S11 — Idea 2a: attention over the neighbourhood (2 min)

"Second idea: let the edges see each other.

Our baseline here is a small MLP in the probe — it scores every candidate edge from that edge's own features, one at a time. Nothing else in the event enters. The attentive version computes, for each edge, a **weighted sum over its neighbours** — that is the message-passing step — where the weights are a **softmax of a scaled dot product** between a query and a key. The left picture is the MLP; the right is attention, with the line width showing the attention weight.

Three properties matter. **First**, the aggregation is **permutation-invariant** — the neighbourhood is an unordered set, so the output must not depend on the order we happened to store it in. That's also why every one of our edge features has to be symmetric under swapping the two ends. **Second**, the **receptive field** grows with depth: one layer sees one hop — one ring of neighbours — and four layers see four. **Third** — and this is the bridge to the next idea — attention can represent functions of the *neighbourhood distribution*: how many competitors a track has, how spread out they are — whereas a tree model can only see the scalars you feed it explicitly."

## S12 — Idea 2b: it helps most where chains compete (1.5 min)

"Numbers. Same pool, matched features, several seeds. Attention at 30 features — the node, geometry and context groups: AP 0.846 with a spread of 0.012. The MLP at all 47 features — every feature group: AP 0.802 with a spread of 0.015.

That's plus 0.044 AP and plus 0.027 AUC. What makes this a result is the error bar. Let me be precise about which number is which: the MLP's **AUC** run-to-run spread is only **0.0047**, and the AUC gap is about six times that; its **AP** spread is 0.0149, and the AP gap is about three times that.

And here's the bridge to why: the gain is **not** from more input channels — attention keeps its lead at both input sizes, 30-dim and 47-dim. It is from *seeing the neighbourhood*. And the gain is largest exactly where we are weakest: **on events with several B's, where several chains compete for the same tracks.**"

## S13 — Idea 2c: depth is part of the mechanism (1 min)

"Depth is not a separate trick — it's part of the same mechanism, so I'll give it here.

Two layers: 0.7937. Four layers: **0.8018** — and that's positive in three seeds out of three. Six layers: 0.7947, it comes back down. That's the receptive-field story: four hops is enough to see the chain, six starts fitting the event itself.

Head count: two gives 0.7900, four gives 0.8018, eight gives 0.8047. Let me be honest about that table: **four heads is our operating point**. The eight-head score, 0.8047, sits well inside the run-to-run spread of the four-head result — about plus or minus 0.012 over seeds — so we keep the **smaller head count as the cheaper equivalent**, not because it won. The recipe is four layers and four heads — not 'more is better'."

## S14 — Idea 3a: competition context, not just geometry (2.5 min)

"Third idea — and this one comes from the physics rather than the architecture.

The edges we get wrong are mostly **cross-chain fakes**: same primary vertex, different secondary vertex. Geometrically they look almost exactly like a real edge. So geometry alone cannot separate them. What does separate them is the **competition structure**. Ask a different question: on this track, how many other candidate partners are there, and how strong are they? If a track has one dominant partner, the edge is likely real. If it has three comparable partners, it's likely a fake.

So the features are: an affinity from the three-dimensional distance between the two ends — the exponential of minus that distance over fifty, in our units; a **normalised** version of it — that affinity divided by the strongest affinity at that endpoint; the number of partners whose affinity is above a half; and the raw partner count.

One correctness point, and this one has bitten us before. A track pair is **undirected**, so every edge feature must be identical if you swap the two ends. We combine the two endpoints with min and max — never by concatenating them in order. If you concatenate, the feature vector gets permuted whenever the edge happens to be stored the other way round; we measured a maximum change of 0.99 on the same edge, and the model can then read off 'which end was written first'. **That is the same failure mode as the 0702 bug** — a July-2026 production dataset whose features silently depended on the storage order of the two ends."

## S15 — Idea 3b: the evidence (1.5 min)

"Evidence. Adding the competition features on top of the geometry features: edge AP plus 0.015, chain survival 35.95 to **38.67** — that is v643 to v648 on the 500-event benchmark at the fixed 0.9 score threshold.

I need to be clear about the right-hand figure. It is a **GBDT feature-upper-bound probe** — gradient-boosted trees on the raw features, *not* a trained model. It answers a different question: is the separating information even *in* the features? Trees on the node features alone reach AP 0.69 on multi-B; adding geometry takes it to 0.80; all feature groups reach 0.84. Those numbers come from one archived file, `principles_provenance_features.json`, so they are reproducible.

The importance bars are **permutation importances** — the drop in AP when a feature is shuffled — and the top four are the closest-approach distance, a transverse-distance quantity at the two ends, a flight-distance quantity, and the pair invariant mass. Two of those are things a tree model has to be **handed** explicitly and an attention model can compute for itself — which is exactly the point from idea two.

And one number that tells us where the difficulty lives: on multi-B events, the geometry features are worth **3.5 times** what they are worth on single-B events — plus 0.113 against plus 0.032. **Multi-B is a geometry problem before it is anything else.**"

## S16 — Training hygiene: three small things (1.5 min)

"Three small things, quickly — they matter more than they look.

**Learning rate.** Three times ten to the minus four gives the best mean over five seeds — eight-oh-nine test AP — ahead of one-e-four at eight-oh-four and one-e-three at seven-nine-nine. Let me be careful though: the gap between three-e-four and one-e-three is about a hundredth, and the seed-to-seed spread is about the same, so this is a **tie-breaker, not a result**. The exact per-seed numbers are in `principles_provenance_lr.json` — I re-ran the sweep so the slide is backed by a file, not by memory.

**Determinism.** We fixed the numpy and torch seeds in the evaluation loader. Before that, every evaluation sampled a different 200 events — the early-stopping sample — so no two arms were ever compared on the same pool, and some of our 'results' were pure sampling noise. After the fix, two independent runs of the same configuration agree to the last digit: 5964 edges — the size of the 500-event hard pool — ten versions, identical.

**And the bar.** With five hundred events the statistical uncertainty in the hard-pool AP is about 0.006. **Anything smaller than that is not a result** — I've used that bar throughout."

## S17 — Summary and the lesson (1.5 min)

"Three things to take away — and each card names its own pool.

**One: the objective.** Chains are an AND, so the loss has to punish the weakest link rather than the average edge. On the 3000-event benchmark this is worth about half again as many surviving chains — 35.8 to 52.3 percent — for a cost of nine thousandths of AP.

**Two: the neighbourhood.** Letting each edge aggregate its neighbours buys 0.044 AP on the multi-B probe, and it is the only architectural change that survived multiple seeds.

**Three: the competition.** The information that separates a real edge from a cross-chain fake is how many competitors it has — and it has to be built symmetrically.

And the methodological lesson, which I think is the most transferable: **every change that worked changed what the model is optimising, or what it can see. Every change that did nothing changed how it computes.**

Two corollaries I'd keep: fix the ruler before the model — and be exact about which ruler; and assert invariants in code, because a silent mis-framing can make you draw the wrong conclusion for weeks."

## S18 — What runs next (1 min)

"Where this goes next. **v655** — chain-recall at weight fifteen — and a weight sweep around v654 are already running, to see whether the weight is the limiting factor, and we'll rank them head to head by chain survival on the same pool.

And we're moving selection onto the same ruler: the checkpoint that ships should be the one with the best **chain survival at the fixed 0.9 score threshold**, not the best edge AP. Which, after the first slide, is the whole point.

Thanks — happy to take questions."

---

## Timing summary

| Slide | Content | Time |
|---|---|---|
| S1–S2 | Title, outline | 45 s |
| S3 | PART 0 setup (single page: objects, pools, version codes) | 1.5 min |
| S4 | PART 1 the ruler | 1.5 min |
| S5–S9 | Idea 1 (loss): 1a–1e | 9 min |
| S10 | Idea 1f: 3000-event main-model result | 2 min |
| S11–S13 | Idea 2 (attention): 2a–2c | 4.5 min |
| S14–S15 | Idea 3 (competition): 3a–3b | 4 min |
| S16 | Training hygiene | 1.5 min |
| S17–S18 | Summary, next | 2.5 min |
| | **Total** | **~28 min** |

## If someone asks for the negative results

One line, no slides: *"Everything that only changed the machinery did nothing — sparse attention at k = 4 to 32, sparsemax, edge-level Sinkhorn matching, geometric attention biases, extra input features, extra capacity, and a swapped early-stopping metric. All of them landed inside the seed bar. And on the chain loss, the idea was right but our first writing of it — the softmin form at weight one — backfired; the chain-recall writing is the one that works, and it is the one on the slides. The reason is in the summary: none of them changed what the model optimises or what it can see."*
