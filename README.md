# Evaluating Prediction-Powered Semi-Supervised Learning under Label Scarcity and Non-Random Label Missingness

CS 680 project. It asks whether unlabeled training data improve four-class `Aircraft_Damage_Type` classification when only **50, 100, 200, 500 or 1,000 training labels** are visible. It compares Supervised learning, soft pseudo-label SSL and **PPI-inspired rectified training** under MCAR, MAR and MNAR label missingness. This is a machine-learning classification study, not statistical inference.

## Run order

Use Python 3.13 (`python3.13 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt`), then run from the project directory:

```bash
.venv/bin/python src/01_run_main_study.py      # builds the fixed split if missing; main study: 75 cases × LR and XGBoost Students
.venv/bin/python src/02_run_ablations.py       # A1 Teacher replay / Teacher-only / fixed λ=1, A2 rounds, A3 capacity
.venv/bin/python src/03_run_teacher_study.py   # same-architecture Teachers, label-free preprocessing, TabPFN-3.5 Teacher pilot
.venv/bin/python src/04_run_improvements.py    # exploratory improved pipeline (improved_v2), validation only
.venv/bin/python src/05_run_synthetic_checks.py   # synthetic positive control + PP-SSL anchor (no real data)
.venv/bin/python src/06_make_figures.py        # all figures and tables + key numbers printed to the console
.venv/bin/python src/07_final_test.py          # one-shot Test evaluation, run LAST; locked until the protocol is frozen
.venv/bin/python -m unittest discover -s tests -v
```

Steps 01–05 resume from (or rebuild) their own databases and do nothing if already complete. Step 06 only reads the databases and writes the figures and the LR main-line tables (no retraining). Step 07 refuses to recompute once done. Shared code lives in `src/core/` and is imported by the numbered scripts; it is never run directly. Total compute is about 1.5 CPU-hours, plus about 1 hour on an Apple GPU (MPS) for the TabPFN pilot.

## Data and evaluation

- **Data.** `data/Aircraft_Incident_Dataset_Kaggle.csv` (cp1252). Seven features. The split is fixed (22,586 train / 4,840 validation / 4,841 test, with one representative per exact-duplicate record), stored in `results/split_manifest.csv`.
- **Masks.** Applied to the training set only, with exactly the requested number of visible labels, nested across budgets, seeds 42–46.
  - MCAR is uniform.
  - MAR samples fewer recent records (`Incident_Year ≥ 2023`), with exact known inclusion probabilities. It is used with **oracle IPW**.
  - MNAR samples fewer `destroyed` outcomes. Only the simulator reads these outcomes; it is an **uncorrected stress test**.
- **Information rule.** Hidden training labels never enter training, preprocessing, weighting, CV or selection. SSL weight and PPI λ are chosen by stratified CV on visible labels, with the Teacher nested inside the CV.
- **Validation** (fully labeled) is used only for reporting. Because the protocol was revised after looking at it, treat it as a **development set**.
- **Test has never been evaluated.** Evaluate it once, after the method is frozen.

## Methods

- **Students.** LR (C=1, analytic gradient) and XGBoost (100 rounds, depth 3, η 0.05, L2 5, max_delta_step 1, custom objective). Within a Student, only the training objective differs between methods.
- **Teacher.** CatBoost (500 iterations, depth 6), cross-fitted. Labeled rows get out-of-fold predictions. Unlabeled targets average the same fold Teachers, which keeps the labeled and unlabeled terms consistent because cross-entropy is linear in the target.
- **Risks** (a, b: self-normalized IPW weights under MAR, 1 otherwise):

  ```text
  Supervised: mean_L[a·CE(Y)]
  SSL soft:   (1−w)·mean_L[a·CE(Y)] + w·mean_accepted_U[b·CE(qU)]      (confidence ≥ 0.8)
  PPI:        mean_L[a·(CE(Y) − λ·CE(qL))] + λ·mean_U[b·CE(qU)],  λ ∈ {0, .25, .5, .75, 1}
  ```

- **Why the optimization behaves differently by Student.** Per row, the loss is `−t·z + m·logsumexp(z)` with row mass m ≥ 0 for λ ≤ 1. So the rectified objective is **convex for LR** (bounded by L2). However, each labeled row has slope `−λ(1−q_y) < 0` along its true-class margin, so the **empirical risk is unbounded below for trees**. XGBoost PPI fits are therefore set by the fixed boosting budget; ablation A2 confirms this.
- **PPI-inspired, not PPI++.** λ is chosen for predictive Macro F1, so no coverage or variance guarantee is inherited.

## Studies and outputs (`results/`)

| File | Content |
|---|---|
| `experiments.sqlite` | Main study: every metric, per-class F1, CV candidate score, Teacher score and fallback reason |
| `ablations.sqlite` | A1: Teacher replay, Teacher-only refit, fixed λ=1 (75 cases). A2: XGBoost 100 vs 300 rounds (20 cases). A3: larger XGBoost Student (Fully labeled + 20 cases) |
| `improved.sqlite` | Exploratory improved_v2: Ensemble Teacher, 5-fold, λ by CV and by a gradient-variance heuristic, Macro-F1 threshold rule, XGBoost leaf-refit; 225 cases (15 seeds) |
| `teacher_study.sqlite` | Same-architecture Teachers (LR→LR, XGB→XGB) and label-free preprocessing (fit on labeled + unlabeled features), 6 variants × 75 cases |
| `tabpfn_teacher.sqlite`, `tabpfn_teacher_summary.csv` | Exploratory pilot: pretrained TabPFN-3.5 Teacher → LR (MCAR; 100 and 500 labels; 5 masks each), paired with the CatBoost Teacher on the same masks |
| `learning_curves.png` | Main study: validation Macro F1 for the LR (top) and XGBoost (bottom) Students, with the CatBoost Teacher and Fully labeled references (Accuracy is in the databases) |
| `teacher_quality.png` | Each Teacher (CatBoost cross-fit, CatBoost refit, same-architecture; TabPFN-3.5 cross-fit and alone in the LR/MCAR panel) vs the Supervised Student |
| `ppi_gains.png` | Main result: gain over Supervised on the same mask for SSL soft and PPI (CatBoost or same-architecture Teacher; TabPFN-3.5 pilot dotted in the LR/MCAR panel; PPI with λ fixed at 1 (A1) dash-dot) |
| `improved_gains.png` | Exploratory improved_v2: gain over Supervised for SSL soft, PPI with CV λ and with the λ heuristic (argmax; threshold-rule results are printed by step 06) |
| `lr_paired_uncertainty.csv` | LR main line: PPI−SSL and PPI−Supervised with SE and a stratified paired bootstrap over masks, overall and per class (λ sources and harm rates are printed by step 06) |
| `lr_ablation_questions.csv` | Existing ablations re-read as LR-mainline questions (Teacher, λ, preprocessing, per-mask Teacher choice) |
| `positive_control.sqlite`, `positive_control.png`, `positive_control_checks.csv` | Synthetic positive control (known posterior; controlled Teacher; MCAR/MAR/MNAR) and the verdict on each pre-declared expectation |
| `ppssl_replication.sqlite`, `ppssl_replication.png`, `ppssl_replication_checks.csv` | PP-SSL anchor: two-group synthetic regression, online vs offline λ |
| `xgboost_sensitivity.png` | XGBoost on the pre-declared subset: PPI at 100 vs 300 rounds (top), Supervised capacity (bottom) |

All figures share one layout: a 2 × 3 grid with columns MCAR / MAR / MNAR, visible labels on the x-axis, lines for the mean over 5 masks with ±1 SD bands, and the legend at the bottom. Colors keep one meaning throughout: Supervised blue, SSL soft orange, PPI aqua, Teacher gray. Difference plots draw Supervised as the 0 line.

Each ablation and the Teacher study first reproduce the stored main-study fits **bit-for-bit** before adding anything, and write only to their own databases. Later studies open the main database read-only. CatBoost→CatBoost is not run: a faithful signed objective needs CatBoost's per-row Python callback (hours per case), and CatBoost would compute categorical target statistics from the signed targets.

`archive/source_before_cleanup.zip` holds the exact source that produced `experiments.sqlite`. Full replays with the current code reproduce stored cases exactly. `eda.ipynb` draws the proposal Figures 1–3 (saved to `proposal/`).

## Study roles

- **Main experiment (report's primary evidence):** the original pipeline, with the CatBoost Teacher, strict cross-fitting and λ selected by training CV, on the **LR Student** (`experiments.sqlite`). Its paired comparisons and the Teacher, λ and capacity ablations (`ablations.sqlite`, `teacher_study.sqlite`) are the core results.
- **XGBoost:** reported as an ablation of optimization stability and capacity. The signed rectified loss is unbounded below for boosted trees (A2), and the small main configuration limits capacity (A3).
- **Exploratory extension (`improved.sqlite`, protocol `improved_v2`):** changes several things at once (Teacher, folds, decision rule, Student representation), so its results are descriptive, not attributable to any single component.
- **Test:** `07_final_test.py` stays locked (`FROZEN['status'] = 'draft'`) until the main experiment is settled. Its earlier draft chose the LR Teacher by OOF scores pooled over all 75 masks, which is a protocol-level choice that a single dataset could not make; this must be resolved before freezing. A replacement rule that each case can execute within its own label budget is proposed below (not executed).

## Version protection

Every runner compares the stored fingerprint (config, data and split hashes, package versions, source hashes) with the current one before reusing results:
- **Partially finished study:** refuses to resume under different source, config or data.
- **Finished study:** skipped with a warning that lists exactly what changed.

The stored main, ablation and Teacher-study results predate the later refactors, so they show such warnings. They remain valid because the current code **reproduces them bit-for-bit**:
- full replays of MCAR/200/42, MAR/50/42, MNAR/500/44, MAR/1000/46, MNAR/100/45 and MCAR/500/43;
- the A1 replay of all 75 Teachers and 150 Supervised fits.

## Exploratory improved pipeline (protocol `improved_v2`, validation only)

Declared in `src/core/improved.py` (`PROTOCOL`, stored in `improved.sqlite`):

1. **Ensemble Teacher.** The mean of cross-fitted CatBoost and LR Teachers on the same folds. There is no label-dependent calibration or selection, so every labeled prediction strictly excludes its own label. OOF log losses are recorded as diagnostics only.
2. **5-fold** Teacher cross-fitting and training CV; **15 seeds** (42–56), 225 cases.
3. **λ.** Chosen by training CV, plus a **PPI++-inspired λ heuristic**: tr Cov(g_Y, g_f) / ((1 + n/N)·tr Var(g_f)), using **whole-Student out-of-fold** residuals (the tree structure is re-learned per fold for leaf-refit). It omits the H⁻¹ preconditioning and the exact IPW variance of PPI++ power tuning, so it is not PPI++ power tuning.
4. **Macro-F1 decision rule.** A per-class logit bias fitted on out-of-fold Student probabilities, applied to every method. Argmax results are stored too.
5. **XGBoost leaf-refit = tree features + LR head.** A supervised XGBoost (20 rounds) learns the structure on labeled rows; every method refits leaf weights with the same convex LR objective (C = 0.1). Rounds and C were fixed for optimizer conditioning before any accuracy was seen. This is a separate Student, not the boosted XGBoost Student.

**History.**
- **v1:** used in-sample residuals, which made the λ heuristic 0; aborted.
- **v1.1:** aborted after review, because its calibrated and selected Teacher and its leaf-refit residuals were not strictly out-of-fold.
- **Disclosure:** interim validation numbers for MCAR at 50–100 labels from those runs had been viewed. Both runs were discarded.

The figure `improved_gains.png` is drawn only when all 225 cases are complete.


## LR main line: paired uncertainty and ablations (`06_make_figures.py`; analysis only)

All 75 masks share one train/validation split and one validation set. Seeds vary only the mask and the CV folds. The intervals below therefore describe **mask-to-mask variability on the development set**; they are descriptive, not a generalization guarantee. The bootstrap resamples masks within each (mechanism, budget) cell. Validation Macro F1, CatBoost Teacher → LR:

| Contrast (75 masks) | Mean ± SE | 95% paired bootstrap | better / worse / equal |
|---|---|---|---|
| PPI − Supervised (**improvement**) | +0.0018 ± 0.0011 | [−0.0000, +0.0038] | 33 / 19 / 23 |
| PPI − SSL soft (**robustness** relative to SSL) | +0.0070 ± 0.0020 | [+0.0043, +0.0101] | 43 / 15 / 17 |
| SSL soft − Supervised (SSL harm) | −0.0052 ± 0.0014 | [−0.0074, −0.0033] | 7 / 32 / 36 |

- **Robustness and improvement are separate claims.**
  - *Robustness:* SSL is worse than Supervised on 32 masks (mean −0.0127). On those same masks PPI averages +0.0049, and it is worse than Supervised on only 8 of them. PPI therefore avoids most of SSL's harm.
  - *Improvement:* PPI's improvement over Supervised is small. Its interval touches 0. It is positive at 50–200 labels (+0.003 to +0.004) and under MCAR and MNAR, and negative under MAR (−0.0022).
- **λ = 0 occurs on 23 of 75 masks.**
  - 17 are chosen by CV, because no λ > 0 beat Supervised.
  - 6 are **forced**: a class had fewer than 2 visible labels, so there was no CV. Five of these are at 50 labels.
  - None comes from the feasibility back-off (printed by step 06).
- **Per-class F1** (`lr_paired_uncertainty.csv`, per-class rows):
  - PPI's gain over Supervised is concentrated in the rarest class, *minor*: +0.022 at 50–200 labels, interval [+0.013, +0.032]. PPI loses a little on *substantial* (−0.005) and *damaged beyond repair* (−0.002).
  - SSL's harm falls on the two rare classes: *destroyed* −0.017 and *minor* −0.008. The 0.8 confidence filter keeps mostly majority-class pseudo-labels.
- **Ablations as LR-mainline questions.** Each row is PPI − its own Supervised over 75 masks (`lr_ablation_questions.csv`; the LR row of `ppi_gains.png` draws Q0–Q2):

| Question | PPI − Supervised [95% bootstrap] | Teacher Macro F1 (Teacher − Supervised) |
|---|---|---|
| Q0 main: CatBoost Teacher, CV λ, labeled-only preprocessing | +0.0018 [−0.0000, +0.0038] | 0.477 (−0.038) |
| Q1 LR Teacher instead of CatBoost | +0.0026 [+0.0005, +0.0048] | 0.499 (−0.016) |
| Q2 λ fixed at 1 instead of CV (A1) | +0.0028 [−0.0004, +0.0062] | 0.477 (−0.038) |
| Q3 labeled+unlabeled preprocessing (CatBoost / LR Teacher) | +0.0017 / +0.0018 | unchanged |
| Q3 effect of that preprocessing on Supervised itself | −0.0003 [−0.0006, +0.0001] | — |
| Q4 Teacher chosen per mask by its own training CV (CatBoost vs LR) | +0.0016 [−0.0004, +0.0036] | — |

  - **Q1** changes two things at once: the Teacher's architecture and its quality. The LR Teacher is better than CatBoost at these budgets, though still worse than the Supervised Student. Q1 therefore cannot separate "architecture match" from "a better Teacher".
  - **Q2:** fixed λ = 1 has a larger mean but a larger spread. It wins on 41 masks and loses on 34; CV λ gets 23 exact ties because it allows λ = 0. Under MAR at 500–1,000 labels, fixed λ = 1 is clearly worse.
  - **XGBoost** stays in the capacity/optimization sensitivity analyses (A2/A3) and is not part of these questions.

## Synthetic positive control (`05_run_synthetic_checks.py`, protocol `positive_control_v1`)

**Question.** Is the null-to-small real-data gain an implementation or objective problem, or a consequence of unfavorable real-data conditions?

**Design** (`src/core/positive_control.py`):
- **Data.** Four classes with a known multinomial-logistic posterior p\*(y | x). There are 20 Gaussian features plus a subgroup indicator (30% of rows), and class priors 49/16/8/27% (as in the real data).
- **Calibration.** The signal strength was set **only** by matching the Bayes and Supervised curves to the real data, before any SSL/PPI fit: Bayes Macro F1 0.65 vs fully labeled LR 0.64; Supervised 0.40 → 0.61 vs 0.44 → 0.58.
- **Missingness.** MCAR, MAR and MNAR reuse `masking.make_mask` unchanged. MAR under-observes x₀ > 0.5 with known propensities and IPW. MNAR under-observes class 1 and is not corrected.
- **Replications.** 10,000 training rows and 5,000 validation rows. 10 seeds, each an **independent replication** with a new pool, validation set, Teacher noise and mask. Unlike on the real data, the intervals here therefore are over replications.
- **Teachers.** A fixed, label-free function of x: oracle (q = p\*); p\* with logit noise σ = 1 and σ = 3; uninformative (pure noise); and subgroup-biased (oracle in group A, classes rotated in group B). Because q never sees a label, no cross-fitting is needed.
- **Student and methods.** The LR Student (`ssl_core.fit_student`, C = 1). Supervised; SSL soft; PPI with CV λ (the same rules as `selection.select_settings`); and PPI with λ = 1.
- **Pre-declaration.** The 11 expectations (which settings should and should not benefit) and their decision rules are stored in the database before the first result.

**Results** (`positive_control_checks.csv`; `positive_control.png` draws the oracle and σ=3 Teachers; MCAR + MAR; Macro F1 − Supervised):

| | Expectation | Estimate [95% bootstrap over seeds] | Verdict |
|---|---|---|---|
| E1 | oracle, PPI CV λ, gain at 50–200 | +0.033 [+0.030, +0.036] | confirmed |
| E2 | oracle, PPI λ=1, gain at 50–200 | +0.030 [+0.026, +0.035] | confirmed |
| E3 | oracle, gain shrinks 50 → 1,000 | +0.057 [+0.048, +0.064] | confirmed |
| E4 | noise σ=1, gain at 50–200 | +0.021 [+0.016, +0.027] | confirmed |
| E5/E6 | noise σ=3 (Teacher 0.42 < Student): no material gain, no material harm | +0.002 [+0.000, +0.005] | confirmed |
| E7 | uninformative, no gain | −0.005 [−0.009, −0.002] | confirmed |
| E8 | uninformative, no material harm (> −0.005) | −0.005 [−0.009, −0.002] | **not confirmed** |
| E9 | uninformative, λ fixed at 1 harms | −0.072 [−0.078, −0.064] | confirmed |
| E10 | subgroup-biased, PPI > SSL inside subgroup B | +0.006 [−0.001, +0.012] | **not confirmed** |
| E11 | subgroup-biased, no material harm overall | +0.001 [−0.003, +0.004] | confirmed |

**Reading.**
1. **The objective and implementation can deliver gains.** With an informative, unbiased Teacher, the same `fit_student` objective, CV rule and masks give PPI +0.045 (MAR) to +0.070 (MCAR) Macro F1 at 50 labels, and the gain shrinks with more labels as theory predicts. The real-data null is therefore not explained by a broken implementation.
2. **What matters is how the Teacher is wrong, not its per-row accuracy.**
   - The σ = 3 Teacher has Macro F1 0.42, below the Supervised Student, yet SSL soft gains +0.12 to +0.13 at 50 labels and +0.01 at 1,000. Independent logit noise averages out over 10,000 unlabeled rows, so the Teacher is still an unbiased signal.
   - PPI is more conservative. Its labeled rectifier term keeps O(1/n) variance, which is large when the Bayes error is about 35%. CV then often chooses a small λ.
   - On the real data SSL *hurts*. That points to **systematic** Teacher error: the real Teacher is learned from the same few labels, and its errors fall on the rare classes. That is the regime where PPI's unbiasedness matters, and there PPI is robust relative to SSL but not much better than Supervised.
3. **Selecting λ by training CV with few labels is noisy.** With an uninformative Teacher, CV picked λ > 0 in 70/150 cases and cost −0.005 (E8). Forcing λ = 1 cost −0.07 (E9), so CV still guards against most of the harm.
4. **The subgroup-biased Teacher did not reproduce a clear PPI-over-SSL advantage inside the biased subgroup** (E10, not confirmed). The rotated pseudo-labels are confidently wrong, but the 0.8 confidence filter and CV keep SSL's weight low.
5. **MNAR (declared descriptive).** An informative Teacher partly corrects the label-selection shift, so both SSL and PPI gain more than under MCAR.

## PP-SSL anchor (`05_run_synthetic_checks.py`; Shoham et al., NeurIPS 2025, arXiv 2510.22586)

**What the paper does.**
- Its prediction-powered gradient is g_n + λ(g̃_N,f − g_n,f). This is unbiased for every λ ∈ [0, 1], so λ changes only the variance.
- λ is tuned **online** by 1-D AdaGrad on h_t(λ) = ‖g_t + λ d_t‖², clipped to [0, 1] (Algorithm 1). The variance-optimal λ\* is (1/(1+n/N))·σ²/(σ² + σ_e²).
- Its core synthetic phenomenon (Sec. 4.1): with a Teacher that is an oracle in group A and biased by μ in group B, classic SSL becomes worse than Only Labeled in group B, while PP-SSL stays robust.

**Reproduction** (`src/core/ppssl_replication.py`; paper setup):
- 9 Gaussian features plus a group indicator; n = 20, N = 1,000, 1,000 test rows; μ ∈ {0.1, 1, 3, 5, 7}; 100 runs.
- Linear model, Adam lr 1e-3, 12,000 steps.
- Six targets were declared before the run. Test MSE:

| | Paper claim | Result | Verdict |
|---|---|---|---|
| R1 | Group B: Classic SSL worse than Only Labeled for μ ≥ 3 | +6.1 / +20.9 / +43.1 MSE | reproduced |
| R2 | Group B: PP-SSL better than Classic SSL for μ ≥ 3 | −6.2 / −20.9 / −43.1 | reproduced |
| R3 | PP-SSL no worse than Only Labeled at every μ | −0.105 … −0.027 (lower or equal at every μ) | reproduced |
| R4 | Online λ falls as the bias grows | 0.026 → 0.003 | reproduced |
| R5 | PP-SSL close to the best fixed λ (within 5%) | +0.15 to +0.93 MSE above it | **not reproduced** |
| R6 | PP-SSL beats PPI++ with offline λ | +0.02 (no difference) | **not reproduced** |

**Why R5 and R6 fail here.**
- With n = 20 below the batch size, the labeled gradient g_t is the same full-batch gradient at every step. As w converges, g_t → 0, so the λ minimizing ‖g_t + λ d_t‖² goes to 0.
- In a traced run, λ was about 0.3–0.4 during the first ~50 steps (the range of the paper's Table S9) and 0 by step 1,000. The paper stops early on 200 labeled validation rows; we deliberately do not, because that is 10× the labeled budget.
- The PPI++ plug-in λ (0.007–0.019) is also near 0. Here the benefit of λ is mostly *finite-sample conditioning*: λ·X_Uᵀ X_U/N regularizes a 20-row, 11-parameter fit. Asymptotic variance-based tuning does not target that.
- **This project's offline training-CV rule** (4-fold, λ ∈ {0, .25, .5, .75, 1}, judged by held-out error) does target it. It picks λ 0.56 → 0.21 as μ grows, close to the infeasible best fixed λ (0.6 → 0.1). Its total MSE is 1.80 vs PP-SSL 2.70 at μ = 0.1, but it is worse at μ = 7 (3.22 vs 3.07): CV on 20 labels is noisy.

**Replication vs adaptation boundary.**
- *Replicated:* the unbiased PP gradient; Algorithm 1's λ update; the paper's data, Teacher, n, N and μ grid.
- *Not replicated:* PyTorch; early stopping on extra labeled data; the AdaGrad-norm step for w; the real-data experiments. One smoke run (μ = 5, seed 0) was seen at lr 1e-2 / 3,000 steps before the optimizer was set to the paper's values.
- *This project adapts rather than replicates PP-SSL.* Its risk is cross-entropy for 4-class classification. λ is chosen **offline** by stratified training CV for Macro F1, not online and not by minimizing variance. Its Teacher is learned and cross-fitted on the same few labels, not a fixed oracle. Under MAR it uses IPW, and under MNAR the labels are selection-biased, which the paper's i.i.d. assumption excludes.
- So "PPI-inspired rectified training" here shares PP-SSL's unbiased objective, but none of its online-tuning guarantee.

## Proposed final Test protocol (proposal only; `07_final_test.py` stays locked as a draft)

**Development-set disclosure (kept).** Validation was used as a development set: the protocol was revised after viewing it, interim `improved_v1/v1.1` numbers were viewed, and the analyses above (paired uncertainty, ablations as questions, positive control, PP-SSL anchor) were done after the main results were known. Test has never been evaluated.

**Proposed rules, to be frozen before any Test score exists:**
1. **Primary method = the proposal's design.** LR Student; cross-fitted **CatBoost** Teacher; labeled-only preprocessing; SSL weight and PPI λ by 3-fold training CV on the visible labels (main-study grids); argmax decisions.
   - This needs no Teacher selection, so every case can execute it within its own label budget. It matches the submitted proposal ("CatBoost is used as the Teacher model").
   - Same 75 masks (seeds 42–46) and the same fitted models as on validation. `07_final_test.py` already verifies the Test refit reproduces the validation run's training-only quantities exactly.
2. **Pre-declared primary contrasts (LR).** These are pooled over 75 masks, with mean ± SE and the stratified paired bootstrap 95% interval:
   - (a) PPI − Supervised: **improvement** is claimed only if the interval excludes 0.
   - (b) PPI − SSL soft: **robustness** is claimed only if the interval excludes 0.
   - Both are reported whatever the outcome.
3. **Secondary (not used for any claim).**
   - The per-budget, per-mechanism and per-class tables, and λ sources split into CV-chosen vs forced.
   - A **per-case Teacher-selection rule executable within the case's label budget**: for each mask, fit the cross-fitted CatBoost and LR Teachers on that mask's visible labels only. Keep the one whose selected PPI setting has the higher **training-CV** Macro F1 (`teacher_study.selected_cv_score`; ties and the no-CV small-class fallback keep CatBoost).
   - This replaces the earlier draft rule (an LR Teacher chosen by OOF scores pooled over all 75 masks), which no single dataset could apply. On the development set the per-case rule gives +0.0016 [−0.0004, +0.0036], no better than the fixed CatBoost Teacher, so it stays secondary.
4. **Sensitivity only.** CatBoost → XGBoost (capacity/optimization; A2/A3). `improved_v2`, the threshold rule and the synthetic studies are **not** evaluated on Test.
5. **Freezing.** Write these rules into `FROZEN` in `src/core/final_test.py`, with `status = 'final'` and the date, and change `VARIANTS` to CatBoost→LR (primary), LR→LR (for the secondary rule) and CatBoost→XGBoost (sensitivity). Then run `07_final_test.py` once. That edit and the run are left to the author.
