# Step 1: Define Paths, Classes and the Original Seven Features

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import DEFAULT

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / 'data/Aircraft_Incident_Dataset_Kaggle.csv'
RESULTS_DIR = PROJECT_ROOT / 'results'
SPLIT_PATH = RESULTS_DIR / 'split_manifest.csv'
DB_PATH = RESULTS_DIR / 'experiments.sqlite'
TARGET = 'Aircraft_Damage_Type'
CLASSES = ['damaged beyond repair', 'destroyed', 'minor', 'substantial']
NUMERIC_FEATURES = ['Incident_Year', 'Incident_Month', 'Aircraft_Age', 'Onboard_Total']
CATEGORICAL_FEATURES = ['Aircraft_Model', 'Aircraft_Nature', 'Aircraft_Phase']
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
RANDOM_STATE = 42
PROTOCOL = DEFAULT.protocol


# Step 2: Preserve Row-Wise Cleaning and Validate the Fixed Split

def read_data():
    df = pd.read_csv(DATA_PATH, encoding='cp1252')
    if df.Incident_ID.isna().any() or not df.Incident_ID.is_unique:
        raise ValueError('Raw Incident_ID values must be nonmissing and unique.')
    df[TARGET] = df[TARGET].astype('string').str.strip().str.lower().replace('', pd.NA)
    for col in ['Incident_Year', 'Incident_Month', 'Aircraft_First_Flight', 'Onboard_Total']:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
    df['Aircraft_Age'] = df.Incident_Year - df.Aircraft_First_Flight
    df.loc[~df.Aircraft_Age.between(0, 100), 'Aircraft_Age'] = np.nan
    for col in CATEGORICAL_FEATURES:
        value = df[col].astype('string').str.strip().replace('', pd.NA)
        df[col] = value.astype(object).where(value.notna(), np.nan)
    return df


def read_split():
    split = pd.read_csv(SPLIT_PATH)
    if split.Incident_ID.isna().any() or not split.Incident_ID.is_unique:
        raise ValueError('Split IDs must be nonmissing and unique.')
    if split['split'].isna().any() or not split['split'].isin(['train', 'validation', 'test']).all():
        raise ValueError('Unknown split value.')
    if set(split['split']) != {'train', 'validation', 'test'}:
        raise ValueError('All three splits must be nonempty.')
    return split


def training_features():
    ids = read_split().loc[lambda x: x['split'] == 'train', 'Incident_ID']
    return read_data().set_index('Incident_ID').loc[ids, FEATURES]


def encode_target(values):
    if not values.isin(CLASSES).all():
        raise ValueError('Unexpected visible target value.')
    return values.map({label: i for i, label in enumerate(CLASSES)}).to_numpy(dtype=int)


# Step 3: Load Visible Labels Only; Missing Targets Never Reach the Learner

@dataclass
class CaseData:
    X: pd.DataFrame
    y: np.ndarray
    U: pd.DataFrame
    V: pd.DataFrame
    vy: np.ndarray
    labeled_weights: np.ndarray
    unlabeled_weights: np.ndarray
    mechanism: str


def load_case(mask, evaluation='validation'):
    raw = read_data().set_index('Incident_ID')
    split = read_split()
    train = split.loc[split['split'] == 'train', 'Incident_ID']
    if evaluation not in {'validation', 'test'}:
        raise ValueError('Evaluate on the validation or the test split.')
    # The evaluation rows are only scored: no fit (Teacher, Student, preprocessing, CV) ever sees them.
    evaluated = split.loc[split['split'] == evaluation, 'Incident_ID']
    if not set(split.Incident_ID).issubset(raw.index):
        raise ValueError('Unknown split ID.')
    if mask.Incident_ID.isna().any() or not mask.Incident_ID.is_unique or set(mask.Incident_ID) != set(train):
        raise ValueError('Mask must cover each training ID exactly once.')
    rows = mask.set_index('Incident_ID').loc[train]
    if not rows.is_labeled.isin([True, False]).all() or rows.is_labeled.isna().any():
        raise ValueError('Mask availability must be boolean.')
    visible = rows.is_labeled.to_numpy(dtype=bool)
    mechanism = mask.attrs['mechanism']
    if mechanism not in DEFAULT.mechanisms:
        raise ValueError('Unknown missingness mechanism.')
    if mechanism == 'MAR':
        probability = rows.propensity.to_numpy(dtype=float)
        if not np.all(np.isfinite(probability) & (probability > 0) & (probability < 1)):
            raise ValueError('MAR requires positive known inclusion probabilities below one.')
        wl, wu = 1 / probability[visible], 1 / (1-probability[~visible])
    else:
        # MNAR oracle propensities are intentionally unavailable to training.
        wl, wu = np.ones(visible.sum()), np.ones((~visible).sum())
    return CaseData(
        raw.loc[train[visible], FEATURES], encode_target(raw.loc[train[visible], TARGET]),
        raw.loc[train[~visible], FEATURES], raw.loc[evaluated, FEATURES],
        encode_target(raw.loc[evaluated, TARGET]), wl, wu, mechanism,
    )


def full_training_data(evaluation='validation'):
    raw = read_data().set_index('Incident_ID')
    split = read_split()
    train = split.loc[split['split'] == 'train', 'Incident_ID']
    if evaluation not in {'validation', 'test'}:
        raise ValueError('Evaluate on the validation or the test split.')
    evaluated = split.loc[split['split'] == evaluation, 'Incident_ID']
    return (raw.loc[train, FEATURES], encode_target(raw.loc[train, TARGET]),
            raw.loc[evaluated, FEATURES], encode_target(raw.loc[evaluated, TARGET]))


# Step 4: Fit Imputation, Scaling and Encoding Inside Each Training Fold

def create_preprocessor():
    return ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)),
            ('scaler', StandardScaler()),
        ]), NUMERIC_FEATURES),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown', keep_empty_features=True)),
            ('encoder', OneHotEncoder(handle_unknown='ignore')),
        ]), CATEGORICAL_FEATURES),
    ])


def prepare_catboost_features(X):
    X = X.copy()
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].fillna('Unknown').astype(str)
    return X
