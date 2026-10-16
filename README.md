# Evaluating Prediction-Powered Semi-Supervised Learning under Label Scarcity and Non-Random Label Missingness

CS 680 project. It asks whether unlabeled training data improve four-class `Aircraft_Damage_Type` classification when only **50, 100, 200, 500 or 1,000 training labels** are visible. It compares Supervised learning, soft pseudo-label SSL and **PPI-inspired rectified training** under MCAR, MAR and MNAR label missingness. This is a machine-learning classification study, not statistical inference.

## Run order

Use Python 3.13 (`python3.13 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt`), then run from the project directory:

```bash
.venv/bin/python src/01_prepare_data.py        # fixed train / validation / test split (only if missing)
.venv/bin/python src/02_run_main_study.py      # main study: 75 cases × LR and XGBoost Students
.venv/bin/python src/03_run_ablations.py       # A1 Teacher replay / Teacher-only / fixed λ=1, A2 rounds, A3 capacity
.venv/bin/python src/04_run_teacher_study.py   # same-architecture Teachers and label-free preprocessing
.venv/bin/python src/05_make_figures.py        # all figures + key numbers printed to the console
.venv/bin/python -m unittest discover -s tests -v
```

Steps 02–04 resume from their databases and do nothing if already complete. Shared code lives in `src/core/` and is imported by the numbered scripts; it is never run directly. Total compute is about 1.5 CPU-hours.

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
| `teacher_study.sqlite` | Same-architecture Teachers (LR→LR, XGB→XGB) and label-free preprocessing (fit on labeled + unlabeled features), 6 variants × 75 cases |
| `learning_curves_lr.png`, `learning_curves_xgboost.png` | Main study: Macro F1 (top) and Accuracy (bottom), with the Teacher and Fully labeled references |
| `teacher_quality.png` | Each Teacher (CatBoost cross-fit, CatBoost refit, same-architecture) vs the Supervised Student |
| `ppi_gains.png` | Main result: gain over Supervised on the same mask for SSL soft and PPI (CatBoost or same-architecture Teacher) |
| `fixed_lambda.png` | PPI gain with λ chosen by training CV vs fixed at 1 |
| `xgboost_sensitivity.png` | XGBoost on the pre-declared subset: PPI at 100 vs 300 rounds (top), Supervised capacity (bottom) |

All six figures share one layout: a 2 × 3 grid with columns MCAR / MAR / MNAR, visible labels on the x-axis, lines for the mean over 5 masks with ±1 SD bands, and the legend at the bottom. Colors keep one meaning throughout: Supervised blue, SSL soft orange, PPI aqua, Teacher gray. Difference plots draw Supervised as the 0 line.

Each ablation and the Teacher study first reproduce the stored main-study fits **bit-for-bit** before adding anything, and write only to their own databases. Later studies open the main database read-only. CatBoost→CatBoost is not run: a faithful signed objective needs CatBoost's per-row Python callback (hours per case), and CatBoost would compute categorical target statistics from the signed targets.

`archive/source_before_cleanup.zip` holds the exact source that produced `experiments.sqlite`. Full replays with the current code reproduce stored cases exactly. `eda.ipynb` holds the project proposal (text, Figures 1–3, Table 1; figures saved to `proposal/`) plus read-only data checks.
