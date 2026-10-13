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

---

## 10. APPENDIX — Re-scoring the 0702 line under the new (v3) benchmark (2026-10-13)

> Append-only section; it does not change §0–§9. It records the first time the new threshold-free
> benchmark (R2) was pointed at the **0702-era** models, and the correct reading of the results.
> Produced by `analyze_prune_loss.py --config config_files/eval_bench_v3_n90.yaml --events 3000`
> (same `--version/--tag` plumbing as `submit_bench_v3.sh`); rows appended to
> `report_figs/bench_auc_v3_seed.csv` (subset also in `report_figs/bench_auc_v3_0702_cross.csv`).

### 10.1 What was run, and the cross-dataset label (mandatory)

Every number below is **cross-dataset: model trained on 0702 (leaky), evaluated on 0904 (fixed).**
The checkpoint was trained on `DFEI_IFT_20260702` (test set `00342442`, carrying the `delta_z0` order
leak, §7/B1); the v3 benchmark always runs on the **0904** test split (`inclusive_00342451`,
`data_seed=0`, `nfiles=20`, `thr=0.9`, inputs normalised with the **0904** `normalization_dict.pt`).
So these rows measure "a leaky-trained model applied to the fixed test set", not a within-dataset number.

### 10.2 How to read the R2 columns in the cross-dataset regime

| column | correct reading here | not comparable with |
|---|---|---|
| `ap_hardpool` / `p_at_r90_hardpool` | 0702 model's edge-score ranking quality **on 0904 hard-pool edges**. The absolute value (~0.54–0.55) is depressed by both the leak-trained representation and the 0702→0904 graph-population shift. | any 0904 same-dataset `ap_hardpool` (~0.63–0.67); any 0702 `all_particles` |
| `chain_surv` / `chain_surv_2/3/4` | fraction of 0904 truth chains whose tracks **and** true structural edges all pass `thr=0.9`; here 0.4–4.6 % (vs 32–71 % for 0904 same-dataset). NB `chain_surv` is a pruning-layer **upper bound** (§4). | any 0904 same-dataset `chain_surv`; anything on 0702 |
| `per_link_p_2` | implied single-link survival from `chain_surv_2`; **relative** diagnostic only | across datasets |
| `events` | all 0702 rows at **3000 events** (same tier as the 0904 versions added in §10.4) | 200/500 tiers |

Honest statement: these numbers say *how a 0702-trained model behaves on the fixed 0904 test set*, and
are used only to ask "does the paper-era ranking survive a change of ruler + dataset?".

### 10.3 Compatibility issues encountered (version → error → shim)

Loading path: `adjust_config_evaluation` reads each version's `input_config.yaml` as the architecture
baseline, then overlays only `settings/inference/evaluate`. The model class has since gained
`extra_node_dim`/`extra_edge_dim`/`derived_*`/`line_graph`/`geom_keep`, so each old checkpoint was retested:

| version | result | error (if any) | shim added |
|---|---|---|---|
| **v31, v38, v47, v557, v559, v561, v565** | ✅ ran | — | **none** |
| **v27, v60** | ❌ not evaluable | `RuntimeError: mat1 and mat2 shapes cannot be multiplied (Nx8 and 10x128)` at `hetero_graphcoder.py:45` (tracks node model), first forward | **none — deliberately** |

- **No shims were added anywhere.** The 7 good versions load unchanged (`use_pid:'true'` → the pipeline
  concats the 5-col `tracks.pid` → 13 node features, matching their ckpt; tt edges 5-wide, matching).
- **v27/v60 = genuine FeatureDim/era mismatch, not a missing config default.** Their ckpts expect
  tracks-node input **10** and tt-edge input **4**; the current 0904 pipeline gives **8** / **5**. They
  trained on the older `DFEI_data/converted_LHCbcollision` layout (`use_pid` unset). Fixing needs model
  code changes (out of scope); a "`use_pid='true'`" stop-gap would append the new pid block to an old
  feature vector and give silently wrong numbers. Hence **"not evaluable"**, not back-filled.

### 10.4 Paper champions vs the new ruler (3000 events, cross-dataset)

`all_particles` = old (leaky, 0702-test) ruler from the matrix; `ap_hardpool`/`chain_surv` = new ruler
on 0904. All new-ruler rows: **3000 events, thr 0.9, cross-dataset = yes**.

| version | all_particles [%] (0702) | ap_hardpool (0904) | p@r90_hardpool | chain_surv [%] | per_link_p_2 | events | cross-dataset |
|---|---|---|---|---|---|---|---|
| v27 | 92.30 | — (not evaluable) | — | — | — | — | yes |
| v60 | 92.93 | — (not evaluable) | — | — | — | — | yes |
| v31 | 40.45 | 0.5504 | 0.5224 | 4.57 | 0.2940 | 3000 | yes |
| v38 | 52.13 | 0.5509 | 0.5229 | 2.28 | 0.2079 | 3000 | yes |
| v47 | 55.86 | 0.5500 | 0.5231 | 2.33 | 0.2162 | 3000 | yes |
| v557 | 57.18 | 0.5471 | 0.5219 | 1.27 | 0.1324 | 3000 | yes |
| v559 | 56.86 | 0.5470 | 0.5236 | 2.64 | 0.2162 | 3000 | yes |
| v561 | 51.52 | 0.5546 | 0.5222 | 4.26 | 0.2777 | 3000 | yes |
| v565 | 53.28 | 0.5492 | 0.5235 | 0.41 | 0.0764 | 3000 | yes |

Context: the 0904 same-dataset 3000-event rows (added here) sit at **ap_hardpool 0.636–0.668,
chain_surv 32–71 %** (v601/623/628/631/632/633/637/641/642/643/645/646/648/649/651/652/653/654/655).

### 10.5 Ranking conclusions (E6 / E8)

Common set = versions with both `all_particles` and a new-ruler value (now includes the 7 evaluable
0702 versions). **N = 31; N = 7 cross-dataset.** With the figure's convention (rank 1 = worst):

- **Old ruler:** the 7 cross-dataset versions sit at ranks **16, 22, 23, 26, 27, 29, 30** — v557 is the
  **2nd-best of all 31** by the leaky `all_particles`.
- **New ruler (`ap_hardpool`):** the same 7 occupy ranks **1,2,3,4,5,6,7** — the **entire bottom seven**;
  every 0904 same-dataset version beats every 0702 cross-dataset version.
- **New ruler (`chain_surv`):** again ranks **1,2,3,4,5,6,7** (bottom seven).
- **v557**: 2nd-best (old) → 2nd-worst (new ap AND new chain_surv). **v31**: 16th (old) → 5th (new ap).
  The ranking is essentially **reversed/scrambled** by the ruler change.

**E8** (`E8_paper_champions_under_new_ruler.png`): of the listed paper-era champions only v31/v38/v47/v557
are evaluable (v27/v60 fail, §10.3). Their old-ruler spread (40–57 %) collapses to a flat new-ruler band
(`ap_hardpool` 0.547–0.551, `chain_surv` 1.3–4.6 %), far below the 0904 same-dataset band.

### 10.6 Caveats / conflicts found

1. **Cross-dataset, twice over.** For the 0702 rows the two rulers are on **different test sets**
   (old 0702/`00342442`; new 0904/`00342451`), so part of the gap is genuine dataset shift (the §2 minIP
   baseline itself moves 12.70 → 10.15 across the switch). The *direction* (paper winners → bottom of the
   honest ruler) is robust; the *magnitude* must not be attributed to the leak alone.
2. **Mixed tiers inside E6.** E6 prefers 3000 → 500 → 200 per version; a few versions still only have
   500/200. Tiers are different rulers (§4) — only the large rank moves should be interpreted.
3. **v27/v60 look "best" on the old ruler (92 %)** yet are exactly the versions that cannot be evaluated
   at all on the new ruler. The two highest paper numbers are also the two least verifiable.
4. **Old `all_particles` is leaky and denominator-unfixed** (§1a defect 1) — the 40–57 % band needs
   `All_fix` before it means anything; only the *relative* reshuffle is claimed here.
