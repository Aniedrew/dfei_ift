# DFEI Evaluation Protocol

> Scope: how every benchmark in this repo is defined, what its numerator/denominator are, where
> each one breaks down, and which columns may legitimately be compared with which.
>
> Written 2026-10-13 alongside `build_eval_matrix.py` -> `report_figs/eval_matrix_all_versions.csv`
> and `report_figs/eval_evolution/E1..E7_*.png`.
>
> Ground truth for the two "self-test" reports is the DFEI code, not this document — every
> definition below carries a `file:line` pointer. Anything I could not pin down is listed in
> **§7 Unconfirmed** rather than guessed.

---

## 0. Data sources (where each number comes from)

| # | Product | File(s) | Produced by |
|---|---|---|---|
| R1 | Old "paper" self-test report | `LHCb_logs/DFEI/version_<N>/info_<signal>_reco.txt` | `wmpgnn/performance/reco_accuracy.py` + `dfei_lightning_module.py` `on_test_epoch_end` |
| R2 | New threshold-free benchmark | `report_figs/bench_auc_v3_seed.csv` | `analyze_prune_loss.py` |
| R3 | Strict end-to-end (offline) | `report_figs/seqrec_offline_v654_thr0.9_*.csv` | `eval_seqrec_offline.py` |

R1 is written by the **training job's own test stage** (no GPU needed for a re-run of just the
test stage, but it does load the checkpoint and run the data pipeline).

---

## 1. R1a — "Reconstruction efficiency" (the paper's headline rulers)

**Code:** `wmpgnn/performance/reco_accuracy.py`
`calculate_accuracy` L33-51; writer L54-62; per-tier split L86-114.

Each row of the input dataframe (`sig_df`) is **one truth B-decay chain** (a truth chain; the loop
in `reconstruction.py` L749 iterates over `tc_dict`, the truth chains). For each truth chain four
0/1 flags are set (`reconstruction.py` L751-828):

| column | = 1 when | meaning |
|---|---|---|
| `all_particles` | a reconstructed cluster contains **exactly** the truth chain's track set | whole chain recovered, no extra tracks |
| `perfect_reco` | `all_particles` **and** the LCA values match | whole chain recovered perfectly |
| `none_iso` | reconstructed cluster contains the truth tracks **plus** background tracks (`true_in_reco == 1` but cluster is larger) | chain recovered but not isolated |
| `part_reco` | `0.2 <= recovered_fraction < 1` | partial reconstruction ("FT decision cannot be trusted") |

**Reported value** = `100 * sum(flag) / N`, where **`N = len(sig_df)`** = the number of truth chains
that are present in the evaluated dataframe. `± err` = binomial `ratio * sqrt(1/k + 1/N)`.

**Tiers** (`reco_accuracy.py` L93-106): the file is partitioned by **how many truth B-chains are in
the same event** ("only 1 B", "only 2 B", "more than 2 B"), then the same four efficiencies are
reported inside each partition. `N` printed next to each header is the number of chains in that
partition.

### ⚠️ Known defects of R1a
1. **Denominator N is not fixed.** `N` is the number of truth chains present on the
   **post-pruning** graph, so it *changes with the model and with the working threshold*
   (team's own measurement: full unpruned graph of `inclusive_00342451` has `N_total = 13255`,
   `events = 20000`). The raw percentage therefore rewards models that simply keep more of the
   graph. The team's correction is `All_fix = all_particles% × N / 13255`; this matrix stores the
   **raw** values plus `events` (= N) so the fix can be re-applied, but the raw columns are
   **not comparable across versions** without it.
2. **"only 1 B / 2 B / more than 2 B" is really "how many truth chains share the event"**, not a
   physics B-count (`SUMMARY_0904_leakfix.md` §6; `B_id` is used only for the sign test).
3. **`part_reco` is almost always 0.00** (the code never does the "prefer exact match first"
   ordering); it is not a usable ruler as printed.
4. The "negative id / positive id" sections use `B_id` sign and are ~0 by construction — ignore.

---

## 2. R1b — PV association performance

**Code:** written by `wmpgnn/performance/reco_accuracy.py` `acc_pv_asso` L117-144; values computed
by `plot_pv_missasso` / `plot_sig_pv_missasso` in `wmpgnn/performance/`.

**Reported value** (L130): `100 - wrong / N_tracks * 100` = **percentage of tracks associated to the
correct primary vertex** (higher = better). The file prints this for two association methods:
`HGNN association` (the model's PV head) and `minIP association` (traditional baseline: associate
each track to the PV giving the smallest impact parameter). Groups: `all_tracks`, plus signal-only
slices `sig_no_selection / sig_perfect_reco / sig_AllParticles / sig_NoneIso / sig_PartReco` and
`sig_b_system`.

### ⚠️ Known defects of R1b
- The **track population** differs between `all_tracks` and the `sig_*` groups, and between the
  0702 and 0904 datasets (0904 removed ghosts and duplicate MC tracks). The *minIP baseline itself*
  moves (12.70 on 0702 -> 10.15 on 0904), so cross-dataset PV numbers are not comparable.
- Not on the leaked channel (PV head reads tr-pv edges, not tt edges), so R1b is the one ruler that
  is **not** directly affected by the `delta_z0` leak — yet its headline result collapses at the
  dataset switch anyway (see §5 finding).

---

## 3. R1c — LCAG classification accuracy (4x4 confusion)

**Code:** `wmpgnn/performance/reco_accuracy.py` `acc_four_class` L8-30; writing appended by
`wmpgnn/lightning_module/dfei_lightning_module.py` L2606-2629.

`pred = argmax` of the 4-class LCA head on tt edges. For each true class `i` the row prints
`pred0..pred3` = the fraction of true-`i` edges predicted as each class. The **diagonal**
(`pred_i` on row `i`) is the per-class recall = what `build_eval_matrix.py` stores as
`lcag_acc_class{i}`; `n` is the number of tt edges of class `i` (batch-weighted, so it is a large
edge count, not a chain count).

### ⚠️ Known defects
- class3 is extremely rare (`n ≈ 1.7e3` for v601), so its diagonal accuracy swings violently
  (0 % .. 100 %) and must not be read as a stable ruler.
- Purely a **per-edge** classifier; says nothing about whether chains come back.

---

## 4. R2 — New threshold-free benchmark (`analyze_prune_loss.py`)

Code: `ap_score` L32-36, `prec_at_recall` L39-43, event-level bootstrap CI `_boot_ci` L50-70,
hard-pool AP L437-443, chain survival L519-552.

| column | definition | population / units |
|---|---|---|
| `ap_hardpool` | average precision of the edge score separating **true structural edges** from the **hard-pool fakes** = `fake_intra` + `fake_inter` (both endpoints are real tracks) | one value per (version, event-count) run; `hardpool_n` = number of edges; bootstrap CI in `ap_hardpool_lo/hi` |
| `p_at_r90_hardpool` | precision at recall 0.90 on the same pool | same |
| `chain_surv` | `100 × mean over truth chains of (all its tracks **and** all its true structural edges pass the working threshold)` — an **AND / all-or-nothing** chain-level metric | `thr` = the fixed edge/node score threshold (= 0.9 in `bench_auc_v3_seed.csv`) |
| `chain_surv_2 / _3 / _4` | `chain_surv` restricted to chains of 2 / 3 / 4 daughters | same |
| `per_link_p_n` | `(chain_surv_n / 100)^(1/n)` — the implied single-link survival of an n-edge chain | same |
| `events` | number of events sampled (200 / 500 / 3000) | **different event counts are different rulers** |

### ⚠️ Known defects of R2
- `chain_surv` is measured **at the pruning layer** (track score / edge score vs threshold). It
  assumes "all nodes + all true edges survive => the chain will be reconstructed". The real
  reconstruction also needs the edge-list to be assembled into a *single* cluster with the right
  boundaries, so `chain_surv` is an **upper bound** on true end-to-end performance, not the same
  thing.
- Event counts (200/500/3000) and thresholds must be held fixed to compare two versions. The CSV
  mixes several event counts and (historically) two thresholds — never merge them into one series.
- Rows written *before* the 2026-10-05 seeding fix used a **different random event subset** for every
  run (see §5 B2); those pre-fix numbers are archived as `bench_auc_v3_INVALID_eventset.csv`.

---

## 5. R3 — Strict end-to-end `chain_full`

**Code:** `eval_seqrec_offline.py` (docstring L11-15).

`chain_full` = `100 × (number of truth chains that are reproduced EXACTLY by some predicted chain) /
(number of truth chains)`, on real reconstructed event graphs, keeping the candidate edges fixed and
**only swapping the final assembly step** (connected components vs. a learned sequential decoder).
This is the closest thing to "did the physics actually come back", and is the **final arbiter**.
Currently only v654 has an offline end-to-end run (600 events, thr 0.9); decoders are
`cc` (connected-component baseline), `learned@*, mix@*, geom@*`.

---

## 6. Comparability matrix — which column may be compared with which

| | old `all_particles` | old tiers 1B/2B/>2B | old `pv_hgnn` vs `pv_minip` | old `lcag_acc` | new `ap_hardpool` | new `chain_surv` | e2e `chain_full` |
|---|---|---|---|---|---|---|---|
| **old `all_particles`** | ⚠️ only within same dataset **and** same `N`-normalisation (use All_fix) **and** same thr/ckpt | ✗ (subsets) | ✗ | ✗ | ✗ (per-edge vs per-chain) | ✗ | ✗ |
| **old tiers** | ✗ | ⚠️ within tier only; >2B has tiny `n` | ✗ | ✗ | ✗ | ✗ | ✗ |
| **`pv_hgnn` vs `pv_minip`** | ✗ | ✗ | ✅ within same dataset; use the **difference** column | ✗ | ✗ | ✗ | ✗ |
| **`lcag_acc`** | ✗ | ✗ | ✗ | ⚠️ within class, same dataset | ✗ | ✗ | ✗ |
| **`ap_hardpool`** | ✗ | ✗ | ✗ | ✗ | ✅ across versions **iff same `events`** | ✅ same run/events | ✗ |
| **`chain_surv`** | ✗ | ✗ | ✗ | ✗ | ✅ same run/events | ✅ across versions iff same `events` **and** same `thr` | ⚠️ same family, but chain_surv is an upper bound |
| **`chain_full`** | ✗ | ✗ | ✗ | ✗ | ✗ | ⚠️ | ✅ across versions iff same events/thr/edge-set |

Legend: ✅ directly comparable · ⚠️ comparable only under the stated condition · ✗ not comparable.

---

## 7. The three evaluation breakpoints (drawn as vertical lines in E1..E3/E5/E7)

| id | what changed | boundary version | date | evidence |
|---|---|---|---|---|
| **B1** | dataset `DFEI_IFT_20260702` -> `DFEI_IFT_20260904`; this **removes the `delta_z0` order leak** (0702 chose the tt-edge sender/receiver by `np.sort(ParticleIndex)`, so the antisymmetric `delta_z0` carried the truth order). Test set changes `00342442` -> `00342451`. | **v601** | ~2026-09-20 | `SUMMARY_0904_leakfix.md` §1-2; `VERSION_LOG.md` L666-688; filename token `00342442` vs `00342451` |
| **B2** | evaluation **event sampling seeded** (before this every benchmark run drew a different event subset -> "A>B" was sampling noise). | **v642** | 2026-10-05 | commit `317ca5b`; archived `bench_auc_v3_INVALID_eventset.csv` -> `bench_auc_v3_seed.csv` |
| **B3** | **local corrected DOCA geometry** as an edge feature (standard least-squares crossing distance, bypassing the upstream `calculate_doca` `t1` sign bug). The bug itself was identified 2026-09-30 and affects *all* data (0702 and 0904). | **v643** | 2026-10-05 | `docs/feature_ceiling_analysis.md` §3; `make_prune_variant_configs.py` L212-230 |

**Sub-switches inside R1 (also shown, thin grey, in the figures):**
- **training inference threshold 0.9 -> 0.95 at v632** (`SUMMARY_0904_leakfix.md` §1: v601-v631 use
  thr 0.9, v632+ use 0.95). The R1 `all_particles` curve mixes two working points across this line.
- **checkpoint口径 at v639**: v639 and later report the **best** checkpoint; earlier versions report
  the **last-epoch in-memory** weights (`VERSION_LOG.md` §8 item 4). R1 values across v639 mix two
  ckpt-selection rules.

---

## 8. Recommended judgement hierarchy

1. **Primary (accept/reject a change): strict end-to-end `chain_full`** (R3) on a fixed event set,
   fixed working point, fixed candidate edges. Only v654 has it today — extending R3 to every
   surviving model is the highest-value next step.
2. **Secondary (cheap proxy of the primary): `chain_surv` and `per_link_p_n`** (R2) at a **fixed
   threshold and fixed event count**. Treat the absolute number as an **upper bound**; only trust
   *differences* between versions run in the same batch.
3. **Secondary (diagnostic): `ap_hardpool` / `p_at_r90_hardpool`** (R2) — is the *ranking* getting
   better, independent of any threshold?
4. **Regression checks (must not get worse, but never the headline): old `all_particles` /
   `perfect_reco` (reported only with the fixed denominator and tier `n`), `pv_hgnn - pv_minip`,
   LCAG diagonal.**
5. **Never** compare two numbers that straddle a breakpoint (§7) without re-deriving them.

---

## 9. Unconfirmed (verify in source before quoting)

1. **Exact construction of `N` (the R1 denominator).** I read `N = len(sig_df)` and each row = one
   truth chain (`reconstruction.py` L749 iterates `tc_dict`), but whether `tc_dict` already excludes
   chains that the pruning removed (which is what makes `N` model/threshold-dependent, per
   `SUMMARY_0904_leakfix.md` §7) is **not verified here** — check where `tc_dict` is built in
   `wmpgnn/reconstruction/reconstruction.py` (`reconstruct_decay` / `collect_results`).
2. **The exact track population behind `all_tracks` vs `sig_*` in R1b** — the counts come from
   `plot_pv_missasso` / `plot_sig_pv_missasso` in `wmpgnn/performance/` (not read in full here).
3. **Whether `chain_surv`'s threshold `thr` is the same variable as the node/edge pruning threshold
   used for the reported N** in R1 — both nominally 0.9/0.95, but the config plumbing was not traced
   end-to-end.
4. **`b2_cut = 0.85` vs inference `thr` mismatch** (`SUMMARY_0904_leakfix.md` §6, "high-severity,
   unfixed"): the B2 branch selection and the pruning threshold are not the same source. Any B2-era
   number may mix two working points.
5. **`trdist` random-axis projection** — numerically rotation-invariant (so deterministic and safe,
   per `docs/feature_ceiling_analysis.md` §3.5), but I did not re-run the check myself.
