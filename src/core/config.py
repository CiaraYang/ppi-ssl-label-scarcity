# Step 1: Keep All Experiment Choices in One Place

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Config:
    budgets: tuple = (50, 100, 200, 500, 1000)
    mechanisms: tuple = ('MCAR', 'MAR', 'MNAR')
    seeds: tuple = (42, 43, 44, 45, 46)
    student_C: float = 1.0
    student_models: tuple = ('LR', 'XGBoost')
    student_model: str = 'LR'
    tree_iterations: int = 100
    tree_depth: int = 3
    tree_learning_rate: float = 0.05
    tree_reg_lambda: float = 5.0
    tree_min_child_weight: float = 1.0
    tree_max_delta_step: float = 1.0
    teacher_iterations: int = 500
    teacher_model: str = 'CatBoost'
    design_fit: str = 'labeled'
    cv_folds: int = 3
    ssl_weights: tuple = (0.0, 0.1, 0.5)
    ssl_thresholds: tuple = (0.8,)
    ppi_lambdas: tuple = (0.0, 0.25, 0.5, 0.75, 1.0)
    selection_ratio: float = 0.25
    recent_year: int = 2023
    protocol: str = 'matched_students_v5'

    def __post_init__(self):
        if self.student_model not in self.student_models or set(self.student_models) - {'LR', 'XGBoost'}:
            raise ValueError('Supported Students are LR and XGBoost.')
        if self.student_C <= 0 or self.cv_folds < 2:
            raise ValueError('Use a positive C and at least two CV folds.')
        if self.teacher_model not in {'CatBoost', 'CatBoostBalanced', 'LR', 'XGBoost'}:
            raise ValueError('Supported Teachers are CatBoost, CatBoostBalanced, LR and XGBoost.')
        if self.design_fit not in {'labeled', 'labeled+unlabeled'}:
            raise ValueError('Fit preprocessing on labeled features, or on labeled plus unlabeled features.')
        if not 0 < self.selection_ratio < 1:
            raise ValueError('The lower-observation stratum ratio must be in (0,1).')

    def to_dict(self):
        return asdict(self)


DEFAULT = Config()
