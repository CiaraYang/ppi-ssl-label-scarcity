# Evaluating Prediction-Powered Semi-Supervised Learning under Label Scarcity and Non-Random Label Missingness

CS 680 project. Does unlabeled data help four-class `Aircraft_Damage_Type` classification when only **50–1,000 training labels** are visible? It compares **Supervised**, **soft pseudo-label SSL** and **PPI-inspired rectified training** under MCAR, MAR and MNAR label missingness.

## Run

Python 3.13: `python3.13 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt`, then:

```bash
.venv/bin/python src/01_run_main_study.py         # split (if missing) + main study: 75 cases × LR and XGBoost Students
.venv/bin/python src/02_run_ablations.py          # A1 fixed λ=1 / Teacher replay, A2 XGBoost rounds, A3 XGBoost capacity
.venv/bin/python src/03_run_teacher_study.py      # same-architecture Teachers, label-free preprocessing, TabPFN-3.5 Teacher pilot
.venv/bin/python src/04_run_improvements.py       # exploratory improved_v2 pipeline
.venv/bin/python src/05_run_synthetic_checks.py   # synthetic positive control + PP-SSL replication
.venv/bin/python src/06_make_figures.py           # all figures and tables (reads the databases only)
.venv/bin/python src/07_final_test.py             # one-shot Test evaluation, run last; locked as a draft
.venv/bin/python -m unittest discover -s tests
```

Each study resumes from its own database and is skipped if already complete. Later studies open earlier databases read-only. The TabPFN pilot needs `tabpfn` and a one-time license login; it takes about 1 hour on an Apple GPU.

## Protocol

- **Data.** 7 features, with a fixed split: 22,586 train / 4,840 validation / 4,841 test (`results/split_manifest.csv`).
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

| File | Content |
|---|---|
| `experiments.sqlite`, `learning_curves.png` | Main study (Macro F1 vs labels; LR top, XGBoost bottom) |
| `teacher_study.sqlite`, `teacher_quality.png` | Teachers (CatBoost, same-architecture, TabPFN) vs the Supervised Student |
| `ppi_gains.png` | **Main result**: SSL / PPI gain over Supervised on the same mask (incl. λ fixed at 1 and the TabPFN Teacher) |
| `ablations.sqlite`, `xgboost_sensitivity.png` | A1–A3 |
| `lr_paired_uncertainty.csv`, `lr_ablation_questions.csv` | LR paired differences with bootstrap intervals (overall and per class); ablations as questions |
| `improved.sqlite`, `improved_gains.png` | Exploratory improved_v2 |
| `positive_control.*`, `ppssl_replication.*` | Synthetic checks (database, figure, and verdicts on the pre-declared checks) |
| `tabpfn_teacher.sqlite`, `tabpfn_teacher_summary.csv` | TabPFN-3.5 Teacher pilot |

All figures are 2×3 grids: columns MCAR / MAR / MNAR, mean ± SD over masks. Colors: Supervised blue, SSL orange, PPI aqua, Teacher gray.

## Main result (LR Student, CatBoost Teacher, 75 masks, development set)

| Contrast | Mean ± SE | 95% paired bootstrap |
|---|---|---|
| PPI − Supervised (**improvement**) | +0.0018 ± 0.0011 | [−0.0000, +0.0038] |
| PPI − SSL soft (**robustness**) | +0.0070 ± 0.0020 | [+0.0043, +0.0101] |
| SSL soft − Supervised | −0.0052 ± 0.0014 | [−0.0074, −0.0033] |

- **PPI avoids SSL's harm but barely beats Supervised.**
  - SSL is worse than Supervised on 32 masks; on those masks PPI averages +0.0049.
  - PPI's gain sits mostly in the rarest class, *minor* (+0.022 at ≤200 labels). It is negative under MAR (−0.002).
- **λ = 0 on 23 masks.** 17 were chosen by CV; 6 were forced because a class had fewer than 2 labels.
- **The masks share one split,** so the intervals are descriptive.
- **Ablations** (PPI − Supervised):
  - LR Teacher: +0.0026. The Teacher's quality also changes, so this is not a clean test of the Teacher's architecture.
  - λ fixed at 1: +0.0028, with a larger spread; clearly worse under MAR at 500–1,000 labels.
  - Label-free preprocessing: no effect.
  - Teacher chosen per mask by CV: +0.0016.
- **Both learned Teachers are weaker than the Student they teach**: CatBoost 0.477 and LR 0.499 vs Supervised 0.514.

## Synthetic checks (pre-declared before running)

- **Positive control.** A known posterior, with difficulty matched to the real data. 9/11 expectations were confirmed.
  - An oracle Teacher gives PPI +0.045 to +0.070 at 50 labels, and the gain shrinks with more labels. **The implementation and objective work.**
  - A Teacher that is weak but unbiased (σ = 3; Macro F1 0.42, below the Student) still lets SSL gain a lot, while PPI stays conservative. So what matters is whether the Teacher's errors are *systematic*, not its per-row accuracy. Real-data SSL harm points to systematic errors.
  - Not confirmed:
    - E8: CV λ costs −0.005 with an uninformative Teacher, because CV is noisy with few labels.
    - E10: no clear PPI > SSL inside the subgroup where the Teacher is biased.
- **PP-SSL replication** (Shoham et al., NeurIPS 2025; their two-group regression). 4/6 claims reproduced: SSL collapses in the biased group, while PP-SSL stays ≥ Only Labeled and its online λ falls as the bias grows.
  - Not reproduced: closeness to the best fixed λ, and beating PPI++. Without the paper's early stopping, the online λ collapses to 0, because the full-batch labeled gradient goes to 0.
  - This project's offline CV λ tracks the best fixed λ better.
  - **Boundary:** this project adapts, rather than replicates, PP-SSL. It uses classification, offline CV λ chosen for Macro F1, a learned cross-fitted Teacher, IPW, and MNAR.

## TabPFN-3.5 Teacher pilot (exploratory; MCAR; 5 masks per budget)

**Question.** Is the small PPI gain caused by a Teacher that learns from the same few labels?

**Setup.**
- The Teacher is TabPFN-3.5: pretrained only on synthetic data, research license, run on an Apple GPU.
- It uses the visible labels as its in-context examples, and is cross-fitted and nested in CV exactly like CatBoost.
- Everything else matches the main study. The Supervised fits reproduce the main study exactly.

| Validation Macro F1 | 100 labels | 500 labels |
|---|---|---|
| Supervised LR | 0.494 | 0.585 |
| CatBoost Teacher (cross-fit) | 0.442 | 0.536 |
| TabPFN Teacher (cross-fit) | 0.465 | 0.591 |
| **TabPFN alone** (all visible labels) | 0.482 | **0.603** |
| PPI − Supervised: TabPFN / CatBoost Teacher | +0.009 / +0.009 | +0.000 / +0.002 |
| SSL − Supervised: TabPFN / CatBoost Teacher | −0.019 / −0.027 | −0.001 / −0.001 |

**Reading.**
- TabPFN is a much better Teacher, but **the LR Student does not inherit it**: PPI gains no more than with CatBoost, and SSL still hurts (slightly less).
- At 500 labels, TabPFN alone beats every LR Student. The bottleneck is the linear Student, not only the Teacher.
- With 5 masks per budget, this is descriptive.
- One smoke case (MCAR/100/42) was viewed before the run; it reproduced exactly.

## Exploratory improved_v2 (`improved.sqlite`, 225 cases)

Changes made together:
- an Ensemble Teacher (CatBoost + LR);
- 5 folds;
- λ chosen by CV and by a gradient-variance heuristic (PPI++-inspired, not PPI++);
- a Macro-F1 threshold rule;
- an XGBoost leaf-refit Student.

**Results:** LR PPI (heuristic λ) − Supervised = +0.0049 at argmax, but the advantage disappears under the threshold rule, which itself lifts Supervised by +0.0085.

**Status:** descriptive only. Earlier versions v1 and v1.1 were aborted (not strictly out-of-fold), and their interim MCAR 50–100 numbers had been viewed.

## Proposed Test protocol (not executed)

1. **Primary:** the proposal's design (CatBoost Teacher → LR, CV λ). It needs no Teacher selection, so each mask can run it within its own label budget.
2. **Claims:** "improvement" only if the bootstrap interval of PPI − Supervised excludes 0; "robustness" only if PPI − SSL excludes 0. Both are reported regardless.
3. **Secondary:** a per-mask Teacher choice (CatBoost vs LR) by that mask's own training CV. XGBoost is a sensitivity analysis only. improved_v2, the synthetic studies and TabPFN are not run on Test.
4. **Freezing:** write these rules into `FROZEN` in `src/core/final_test.py` (`status = 'final'`, with the date), then run `07_final_test.py` once.

## Reproducibility

- Each runner stores a fingerprint (config, data, split, package versions, source hashes):
  - It refuses to resume a partial study under changed code.
  - It warns when a finished study predates code changes.
- Stored results were checked to reproduce **bit-for-bit** with the current code (selected full replays plus the A1 replay of all Teachers).
- `archive/source_before_cleanup.zip` holds the source that produced `experiments.sqlite`.
- `eda.ipynb` draws the proposal figures in `proposal/`.
