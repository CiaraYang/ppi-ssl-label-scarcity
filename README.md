# Evaluating Prediction-Powered Semi-Supervised Learning under Label Scarcity and Non-Random Label Missingness

CS 680 project. Does unlabeled data help four-class `Aircraft_Damage_Type` classification when only **50–1,000 training labels** are visible? It compares **Supervised**, **soft pseudo-label SSL** and **PPI-inspired rectified training** under MCAR, MAR and MNAR label missingness.

## Run

Python 3.13: `python3.13 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt`, then:

```bash
.venv/bin/python src/01_main_study.py         # split (if missing) + main study: 75 cases × LR and XGBoost Students
.venv/bin/python src/02_ablations.py          # A1 fixed λ=1 / Teacher replay, A2 XGBoost rounds, A3 XGBoost capacity
.venv/bin/python src/03_teacher_study.py      # same-architecture Teachers, label-free preprocessing, TabPFN-3.5 Teachers
.venv/bin/python src/04_improved.py           # exploratory improved_v2 pipeline
.venv/bin/python src/05_synthetic_checks.py   # synthetic positive control + PP-SSL replication
.venv/bin/python src/06_figures.py            # all figures and tables (reads the databases only)
.venv/bin/python src/07_final_test.py         # one-shot Test evaluation, run last; locked as a draft
.venv/bin/python -m unittest discover -s tests
```

`src/` holds only numbered files: `00_common.py` and `00_methods.py` are shared code; 01–07 are the steps in run order. Each study writes its own database to `results/databases/` and is skipped if already complete. Later studies open earlier databases read-only. The TabPFN pilot needs `tabpfn` and a one-time license login; it takes about 1 hour on an Apple GPU.

## Protocol

- **Data.** 7 features, with a fixed split: 22,586 train / 4,840 validation / 4,841 test (`data/split_manifest.csv`).
- **Masks** (training set only; exact label counts; seeds 42–46):
  - MCAR is uniform.
  - MAR under-samples records from 2023 on, with known probabilities, and uses IPW.
  - MNAR under-samples `destroyed` and is not corrected (a stress test).
- **Information rule.** Hidden labels never enter training, preprocessing, CV or selection. The SSL weight and PPI λ are chosen by stratified training CV on the visible labels, with the Teacher nested inside the CV.
- **Validation** is a **development set**: the protocol was revised after seeing it. **Test has never been evaluated.**
- **Students.** LR (C = 1) and XGBoost (100 rounds, depth 3). Within a Student, only the training objective changes.
- **Teacher.** CatBoost, cross-fitted, so labeled rows get out-of-fold predictions.
- **Risks** (a, b = IPW weights under MAR, 1 otherwise):

```text
Supervised: mean_L[a·CE(Y)]
SSL soft:   (1−w)·mean_L[a·CE(Y)] + w·mean_U,conf≥0.8[b·CE(qU)]
PPI:        mean_L[a·(CE(Y) − λ·CE(qL))] + λ·mean_U[b·CE(qU)],   λ ∈ {0, .25, .5, .75, 1}
```

**Objective shape.** The PPI objective is convex for LR, but unbounded below for boosted trees. XGBoost results are therefore read as an optimization/capacity sensitivity analysis (A2/A3).

## Outputs (`results/`)

| File (databases in `results/databases/`) | Content |
|---|---|
| `experiments.sqlite`, `learning_curves.png` | Main study (Macro F1 vs labels; LR top, XGBoost bottom) |
| `teacher_study.sqlite`, `teacher_quality.png` | Teachers (CatBoost, same-architecture, TabPFN) vs the Supervised Student |
| `ppi_gains.png` | **Main result**: SSL / PPI gain over Supervised on the same mask (incl. λ fixed at 1 and the TabPFN Teacher) |
| `ablations.sqlite`, `xgboost_sensitivity.png` | A1–A3. The figure is 2×2: A2/A3 were declared on MCAR and MAR only (100 and 500 labels) |
| `lr_paired_uncertainty.csv`, `lr_ablation_questions.csv` | LR paired differences with bootstrap intervals (overall and per class); ablations as questions |
| `improved.sqlite`, `improved_gains.png` | Exploratory improved_v2 |
| `positive_control.*`, `ppssl_replication.*` | Synthetic checks (database, figure, and verdicts on the pre-declared checks) |
| `tabpfn_teacher_full.sqlite`, `tabpfn_teacher_summary.csv` | TabPFN-3.5 Teacher on all 75 masks, LR and XGBoost Students (`tabpfn_teacher.sqlite` = the earlier MCAR pilot) |

Figures are 2×3 grids (columns MCAR / MAR / MNAR; lines = mean over masks, no shaded bands; the spread is in the CSV tables), except `xgboost_sensitivity.png`.

Why `xgboost_sensitivity.png` has no MNAR column (reason written down afterwards; the original declaration recorded none):
- A2/A3 test optimization and capacity, which do not depend on the missingness mechanism.
- MNAR would add uncorrected selection bias. Each line is solid, with its own color (colorblind-checked palette) and marker; black and gray lines are references.

## Main result (LR Student, CatBoost Teacher, 75 masks, development set)

| Contrast | Mean | Seeds with a positive mean | Masks better / worse / equal |
|---|---|---|---|
| PPI − Supervised (**improvement**) | +0.0018 | 5 / 5 (per-seed +0.0009 … +0.0027) | 33 / 19 / 23 |
| PPI − SSL soft (**robustness**) | +0.0070 | 5 / 5 | 43 / 15 / 17 |
| SSL soft − Supervised | −0.0052 | 0 / 5 | 7 / 32 / 36 |

**How to read the uncertainty.**
- Masks with the same seed are nested across budgets, so the SE and intervals (`lr_paired_uncertainty.csv`) resample whole seeds.
- With only 5 seeds, and with all masks sharing one split and one validation set, these numbers describe consistency across masks. They do **not** support a claim of a significant improvement.
- The averages themselves are valid.

**Findings.**
- PPI avoids most of SSL's harm. SSL is worse than Supervised on 32 masks, and on those masks PPI averages +0.0049.
- Its gain over Supervised is small and depends on the setting:
  - mostly in the rarest class, *minor* (+0.022 at ≤200 labels);
  - negative under MAR (−0.002).
- λ = 0 on 23 masks: 17 chosen by CV, and 6 forced because a class had fewer than 2 labels.
- **Ablations** (PPI − Supervised; `lr_ablation_questions.csv`):
  - LR Teacher: +0.0026. The Teacher's quality also changes.
  - λ fixed at 1: +0.0028, with a larger spread; worse under MAR at 500–1,000 labels.
  - Label-free preprocessing: no effect.
  - Teacher chosen per mask by CV: +0.0016.
- The learned Teachers are weaker than the Student they teach: CatBoost 0.477 and LR 0.499 vs Supervised 0.514.

## Synthetic checks (pre-declared before running)

- **Positive control.** A known posterior, with difficulty matched to the real data. 9/11 expectations were confirmed.
  - An oracle Teacher gives PPI +0.045 to +0.070 at 50 labels, and the gain shrinks with more labels. This shows the implementation **can produce the expected gain in this synthetic setting**. It is not a proof that the implementation is fully correct.
  - A noisy Teacher (σ = 3; Macro F1 0.42, below the Student) still lets SSL gain a lot, while PPI stays conservative. Its logit noise is independent of y and x. That does not make its probabilities unbiased: softmax distorts zero-mean noise. So per-row Teacher accuracy alone does not predict whether SSL helps or hurts. Whether the real Teacher's errors are systematic is a hypothesis, not established.
  - Not confirmed:
    - E8: CV λ costs −0.005 with an uninformative Teacher, because CV is noisy with few labels.
    - E10: no clear PPI > SSL inside the subgroup where the Teacher is biased.
- **PP-SSL replication** (Shoham et al., NeurIPS 2025; their two-group regression). 4/6 claims reproduced: SSL collapses in the biased group, while PP-SSL stays ≥ Only Labeled and its online λ falls as the bias grows.
  - Not reproduced: closeness to the best fixed λ, and beating PPI++. Without the paper's early stopping, the online λ collapses to 0, because the full-batch labeled gradient goes to 0.
  - This project's offline CV λ tracks the best fixed λ better.
  - **Boundary:** this project adapts, rather than replicates, PP-SSL. It uses classification, offline CV λ chosen for Macro F1, a learned cross-fitted Teacher, IPW, and MNAR.

## TabPFN-3.5 Teacher (exploratory; all 75 masks; LR and XGBoost Students)

**Question.** Does a stronger, pretrained Teacher help the Student?

**Setup.**
- TabPFN-3.5 (pinned checkpoint; synthetic-only pretraining) uses the visible labels as its context. It is cross-fitted and nested in CV like CatBoost.
- Under MAR it is unweighted, because TabPFN takes no sample weights.
- Expectations were declared before the run.
- Supervised fits matched the main study exactly, and the 10 pilot masks reproduced exactly.

| Mean over 75 masks (seed-cluster interval) | LR Student | XGBoost Student |
|---|---|---|
| TabPFN Teacher − CatBoost Teacher (Macro F1) | +0.023 [+0.020, +0.027] | same Teacher |
| PPI − Supervised, TabPFN Teacher | **+0.005** [+0.003, +0.006] | −0.000 [−0.002, +0.001] |
| PPI − Supervised, CatBoost Teacher | +0.002 [+0.001, +0.002] | −0.002 [−0.003, −0.002] |
| SSL − Supervised, TabPFN Teacher | −0.004 [−0.008, −0.002] | −0.005 [−0.010, −0.001] |
| TabPFN alone − Supervised | +0.004 [−0.001, +0.009] | +0.010 [+0.003, +0.017] |

**Reading** (declared expectations E1–E4 all hold):
- **E1:** TabPFN is the better Teacher on every seed.
- **E2:** for LR, PPI gains slightly more with TabPFN, +0.003 over CatBoost; this stays under the declared 0.005 margin. Under MAR, PPI's change versus Supervised turns from negative to positive (−0.002 → +0.003).
- **E3:** XGBoost still gets no reliable PPI gain.
- **E4:** under MCAR, TabPFN alone beats Supervised LR at 500 and 1,000 labels on every mask.
- SSL still hurts with the better Teacher.
- **Bottom line:** a better Teacher helps PPI a little, but the gain stays small, and TabPFN by itself is often as good as or better than the Students.

## Exploratory improved_v2 (`improved.sqlite`, 225 cases)

Changes made together:
- an Ensemble Teacher (CatBoost + LR);
- 5 folds;
- λ chosen by CV and by a gradient-variance heuristic (PPI++-inspired, not PPI++);
- a Macro-F1 threshold rule;
- an XGBoost leaf-refit Student.

**Results:** LR PPI (heuristic λ) − Supervised = +0.0049 at argmax, but the advantage disappears under the threshold rule, which itself lifts Supervised by +0.0085.

**Status:** descriptive only.
- Because so much changed at once, only methods *within* improved_v2 can be compared. No single change can be credited with a gain.
- For the leaf-refit Student, the heuristic λ uses out-of-fold probabilities, but its gradient features come from trees fitted on all visible labels. It is therefore not strictly out-of-fold, and not PPI++.
- Earlier versions v1 and v1.1 were aborted (not strictly out-of-fold), and their interim MCAR 50–100 numbers had been viewed.
- A missing back-off for a non-estimable final PPI fit was added later; none of the 225 stored cases needed it.

## Proposed Test protocol (not executed)

1. **Primary:** the proposal's design (CatBoost Teacher → LR, CV λ). It needs no Teacher selection, so each mask can run it within its own label budget.
2. **Claims:** "improvement" only if the seed-cluster interval of PPI − Supervised stays above 0; "robustness" only if PPI − SSL stays above 0. Both are reported regardless, and are descriptive (5 seeds, one split).
3. **Secondary:** a per-mask Teacher choice (CatBoost vs LR) by that mask's own training CV. XGBoost is a sensitivity analysis only. improved_v2, the synthetic studies and TabPFN are not run on Test.
4. **Freezing:** these rules are already in the draft `FROZEN`/`VARIANTS` in `src/07_final_test.py` (CatBoost→LR, LR→LR, CatBoost→XGBoost), which stays `draft` and refuses to run. To freeze, set `status = 'final'` with the date, then run `07_final_test.py` once.

## Reproducibility

- Each runner stores a fingerprint (config, data, split, package versions, source hashes):
  - It refuses to resume a partial study under changed code.
  - It warns when a finished study predates code changes.
- Stored results were checked to reproduce **bit-for-bit** with the current code (selected full replays plus the A1 replay of all Teachers).
- `archive/source_before_cleanup.zip` holds the source that produced `experiments.sqlite`.
- `eda.ipynb` draws the proposal figures in `proposal/`.

## What the evidence supports

- On LR, PPI gives a **small, setting-dependent gain**. It is often **more stable than plain SSL**, but it does **not consistently beat Supervised**.
- A stronger pretrained Teacher (TabPFN) helps the LR PPI Student only slightly (+0.003 over CatBoost) and does not help XGBoost; TabPFN alone is often as good or better.
- **Reference, not ceiling:** fully labeled scores (LR 0.637) are references for the current model settings, not upper limits.
