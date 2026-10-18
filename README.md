## Prediction-Powered Semi-Supervised Learning under Label Scarcity

### Overview

This project investigates whether prediction-powered semi-supervised learning improves aircraft damage classification when training labels are scarce. Using the [Global Aviation Incident and Accident Dataset](https://www.kaggle.com/datasets/tejal5kunjir/global-aviation-incident-and-accident-dataset) from Kaggle, it compares supervised learning, soft pseudo-label SSL, and PPI-inspired rectified training under MCAR, MAR, and MNAR label missingness.

The current task classifies aircraft damage into three categories: written off, substantial, and minor. It uses 12 aircraft, incident, and route-related features. CatBoost, pretrained TabPFN-3.5 and Ensemble provide Teacher predictions. Logistic Regression is the primary Student, while XGBoost is used for sensitivity analysis.

Experiments examine how label availability and Teacher quality affect classification performance. The original four-class, seven-feature experiments are retained as the earlier development version.

### File Structure

| Folder / File | Description |
|---|---|
| [data/](data/) | Dataset and split assignments |
| [src/](src/) | Models, experiments, and plotting scripts |
| [results/](results/) | Experiment results, figures, and tables |
| [tests/](tests/) | Protocol and implementation tests |
| [requirements.txt](requirements.txt) | Python dependencies |

### Experimental Setup

#### Data

The three classes are:

- Written off: destroyed or damaged beyond repair.
- Substantial: substantial aircraft damage.
- Minor: minor aircraft damage.

The features are incident year, incident month, aircraft age, onboard total, aircraft model, aircraft nature, aircraft phase, registration prefix, departure region, route availability, same-airport indicator, and domestic-route indicator.

Fatalities, investigated causes, and incident category are excluded.

| Split | Samples |
|---|---:|
| Training | 22,586 |
| Validation | 4,840 |
| Test | 4,841 |

Validation is used for development. Final test evaluation remains pending.

#### Label Missingness

Each experiment uses 50, 100, 200, 500, or 1,000 visible training labels.

| Mechanism | Simulation |
|---|---|
| MCAR | Labels are observed uniformly at random |
| MAR | Records from 2023 onward are less likely to be labeled; inverse probability weighting is applied |
| MNAR | Records originally labeled destroyed are less likely to be labeled; selection bias is not corrected |

Five seeds give 75 conditions per Teacher configuration. Hidden training labels are unavailable to the learning methods.

#### Methods

| Method | Training information |
|---|---|
| Supervised | Visible labels |
| SSL soft | Visible labels and confidence-filtered Teacher predictions |
| PPI selected | Visible labels, Teacher predictions, and a correction term |
| Fully labeled reference | All 22,586 training labels |

The SSL weight and PPI λ are selected by training-only cross-validation using Macro F1. Teacher cross-fitting provides out-of-fold predictions for labeled records. Comparisons use matched Student settings and label masks.

### Results

Metrics include Macro F1, per-class F1, Accuracy, Balanced Accuracy, and Log Loss.

| Figures | Comparison |
|---|---|
| [Learning curves](results/v2/figures/01_learning_curves/) | Performance versus visible labels |
| [Performance gains](results/v2/figures/02_gains/) | SSL and PPI versus Supervised |
| [Per-class F1](results/v2/figures/03_class_f1/) | Gains and losses for each class |
| [Teacher quality](results/v2/figures/04_teacher_quality/) | Teacher prediction performance |
| [Ablations](results/v2/figures/05_ablations/) | Correction strength and model settings |
| [Additional metrics](results/v2/figures/07_metrics/) | Paired differences across evaluation metrics |
| [Summary tables](results/v2/tables/) | Numerical results |

Real-data comparisons use matched label masks. Masks sharing a seed are nested across label budgets, and all runs use the same dataset split. Variation across seeds reflects sensitivity to label availability.

### Versions

| Version | Classes | Features | Results |
|---|---:|---:|---|
| v1 | 4 | 7 | [Original](results/) |
| v2 | 3 | 12 | [Current](results/v2/) |

The code defaults to v1. Select v2 with:

```bash
export PPI_VERSION=v2
```

Both classes and features change between versions, so score differences cannot be attributed to feature engineering alone.

### Related Work

- [Prediction-Powered Inference](https://arxiv.org/abs/2301.09633)
- [PPI++](https://arxiv.org/abs/2311.01453)
- [Prediction-Powered Semi-Supervised Learning with Online Power Tuning](https://arxiv.org/abs/2510.22586)
- [TabPFN](https://github.com/PriorLabs/TabPFN)

The project uses PPI-inspired rectified training with CV-selected λ, rather than the variance-optimal PPI++ estimator.