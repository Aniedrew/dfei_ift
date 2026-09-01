# LCA, explained very slowly (English)

> This note explains ONLY the LCA part of DFEI, from zero, with one concrete example.
> No machine-learning background needed. No physics background needed beyond this page.
> Goal: by the end, "LCAG classification", "chain reconstruction" and "chain-LCA losses"
> should feel obvious.

---

## 1. Start with one real decay chain

A heavy particle (a "B" meson) is produced in a collision. It decays, and its daughters decay too.
The final, stable particles fly into the detector and leave tracks.

Concretely, suppose one B meson decays like this (a simplified, real example):

```
                    B
                 /     \
              J/ψ       K
             /   \
           μ+     μ-
```

- `B` decays into two particles: `J/ψ` and `K`.
- `J/ψ` immediately decays into `μ+` and `μ-`.
- So in the detector we see **three tracks**: the muon, the anti-muon, and the kaon.

In physics language:
- `B` is the **mother** of `J/ψ` and of `K`.
- `J/ψ` is the **mother** of `μ+` and `μ-`.
- `K` is the **daughter** of `B`.
- `μ+` and `μ-` are **sisters** (they share the mother `J/ψ`).
- `μ+` and `K` have the *same grandparent* `B` — `K` is the **grandparent**-generation relative of... no wait.
  Let me be careful: `μ+` is a daughter of `J/ψ`, and `J/ψ` is a daughter of `B`. So from `μ+`'s point of view,
  `B` is its **grandparent**. `K` is `B`'s direct daughter. So `K` and `μ+` are related by one generation
  ("aunt/uncle–niece/nephew" if we use family words) — in DFEI this is the **grandparent–grandchild** relation (class 3).

The entire family tree is called a **decay chain** or **decay tree**.

---

## 2. What the network sees

The network does NOT see this tree. It only sees:

- a list of tracks (the 3 tracks above, plus ~150 others from the same collision),
- and all possible pairs of tracks (edges).

For **every pair of tracks**, the network must answer ONE question:

> "If these two tracks were both produced inside this event, what is their family relationship?"

This is exactly the **LCAG classification**. LCAG = **Lineage of Common Ancestor Graph**.
"Lineage" = family line; "common ancestor" = the nearest particle from which both descend.
So the label of an edge is: **how are these two tracks related by their common ancestor?**

---

## 3. The four classes, one by one

### class 1 — parent–child (母子)
One track's particle is the direct mother of the other.

Example in our tree:
- the edge (μ+, μ−)? No — they are sisters.
- the edge (J/ψ, μ+)... but wait, J/ψ is NOT a track (it decays before reaching the detector).
  We never see J/ψ as a track.
- So do parent–child *track pairs* exist? Almost never! Because intermediate particles decay too
  quickly to leave tracks. **This is a key physics fact**: most tracks are final-state particles,
  so class-1 track pairs are extremely rare.

That is exactly why the original class-weight bug hurt so much: class 1 is rare *and* important.

### class 2 — sister (同母)  ★ the important one
Two tracks whose particles share the same direct mother.

Example: (μ+, μ−) — both come from J/ψ. → **class 2**.
If two tracks are sisters, it means they were emitted together from one decaying particle —
that is *the signature of a decay vertex*. Chains are glued together by class-2 edges.

### class 3 — grandparent–grandchild (祖孙)
The two tracks are two generations apart (one is the daughter of a daughter of the other's parent line).

Example: (μ+, K): μ+ descends from J/ψ descends from B; K descends directly from B.
Common ancestor is B; the relation spans two steps. → **class 3**.

### class 0 — background
Everything else: pairs whose particles have **no common ancestor** within this event
(different B mesons, or one of them is not from any B at all).

In our example: the edge (μ+, K-of-a-different-B) is class 0. Also (μ+, some random track) is class 0.

**Numbers**: class 0 is ~99.9% of all edges. Class 1/2/3 are ~0.1%. That imbalance drives
almost every design decision you will hear about (class weights, chain-CE, etc.).

---

## 4. Why four classes? (what each one is used for)

The classes are not just labels — each one is a *building instruction* for reconstruction:

| class | meaning | what it tells the reconstructor |
|---|---|---|
| 1 | parent–child | who is *above* whom → gives the **depth / order** of the tree |
| 2 | sister | who belongs to the **same decay vertex** → groups tracks into one "bundle" |
| 3 | grandparent–grandchild | confirms the **multi-level structure** (same chain, 2 steps apart) |
| 0 | background | **remove this edge** — do not use it |

So reconstruction works like this:

1. The network classifies every edge.
2. **Prune** class-0 edges (keep only edges the model thinks are structural).
3. The remaining edges (classes 1–3) all mean "these two tracks belong to the same decay tree".
   They connect the tracks into connected components — each component is one chain.
4. Within a component, class-2 edges bundle tracks into decay vertices; class-1 edges give the
   ordering; the Rumor-Centrality score picks the root; and the tree is rebuilt.

---

## 5. The chain-LCA losses (why "in-chain" and why "LCA")

Even though the network is trained to classify edges, the *downstream* object that matters is the
**chain** (the connected component). During training we know the truth chains from Monte Carlo.
The idea of the **chain-LCA losses**: don't just supervise every edge; **supervise the edges that
lie on true chains**, with losses designed around the chain.

Two versions:

1. **chain_lca_loss (hinge, v37)**
   For every edge that lies on a truth chain, demand: "your confidence must be high — at least
   above a margin (0.3)". The loss is zero if the edge is already confident enough, and grows
   linearly if not. Why: at inference, a chain dies if ANY of its edges is pruned. Keeping every
   chain edge confident = keeping the whole chain alive.

2. **chain_lca_ce (cross-entropy, v38)**
   Take the same truth-chain edges, and directly apply cross-entropy on their **class**
   (1/2/3) — not just "be confident", but "be the *right* class".
   Why is this necessary? Because structural edges are ~0.1% of all edges. If the main
   classification loss sees 1000 background edges for every structural edge, the gradient is
   dominated by "predict class 0" and the network never learns the rare classes properly.
   The chain-CE carves out a tiny, focused signal that says: "these specific edges must be
   class 1/2/3 correctly".

The word **"in-chain"** simply means: we restrict these losses to edges that belong to a truth
chain. We do NOT apply them to arbitrary background edges (that would teach nothing useful and
would fight the main loss).

---

## 6. Putting it together — the flow

```
tracks  →  edge classification (4 LCAG classes)  →  prune class 0
        →  connected components (chains)
        →  within each chain: class-2 bundles = vertices, class-1 = order, RC = root
        →  rebuilt decay trees  →  compared with truth (PerfectReco)
```

- The **LCAG classification** is the heart: everything downstream trusts it.
- **Class 2 (sister)** is the most useful signal for structure — which is why:
  - we weighted it explicitly,
  - and why the **mass head** works: the ππ invariant mass of a pair peaks at resonances exactly
    for sister pairs, so supervising the mass teaches the network "this is a sister pair" = class 2.

---

## 7. One-paragraph summary you can say out loud

"In DFEI, every pair of tracks gets one of four labels describing their family relationship in the
decay tree: background, parent–child, sister, or grandparent–grandchild. Class zero means unrelated,
and gets pruned. The other three classes are the glue that lets us group tracks into chains and
rebuild the tree. Because structural edges are only about a tenth of a percent of all edges, we
add special losses — a confidence hinge and a class cross-entropy — applied only to edges that lie
on true chains, so the network actually learns these rare but decisive classes. And the reason the
mass head works is that sister pairs sit at resonance masses, so supervising the pair mass teaches
the network exactly the class-two signal that holds chains together."
