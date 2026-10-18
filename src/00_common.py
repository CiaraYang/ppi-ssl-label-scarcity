# Shared: Configuration, Data and Split, Label Masks, Result Storage (Imported by Every Step)

import sys
from pathlib import Path

from dataclasses import asdict, dataclass
import json
import hashlib
from contextlib import contextmanager
import importlib.metadata
import sqlite3

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))  # The numbered files import each other by name.


# ===== From config.py — Step 1: Keep All Experiment Choices in One Place =====

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
        if self.teacher_model not in {'CatBoost', 'LR', 'XGBoost', 'TabPFN'}:
            raise ValueError('Supported Teachers are CatBoost, LR, XGBoost and the pretrained TabPFN.')
        if self.design_fit not in {'labeled', 'labeled+unlabeled'}:
            raise ValueError('Fit preprocessing on labeled features, or on labeled plus unlabeled features.')
        if not 0 < self.selection_ratio < 1:
            raise ValueError('The lower-observation stratum ratio must be in (0,1).')

    def to_dict(self):
        return asdict(self)


DEFAULT = Config()


# ===== From model_utils.py — Step 1: Define Paths, Classes and the Original Seven Features =====

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / 'data/Aircraft_Incident_Dataset_Kaggle.csv'
DATA_DIR = PROJECT_ROOT / 'data'
RESULTS_DIR = PROJECT_ROOT / 'results'         # figures and CSV tables
DATABASE_DIR = RESULTS_DIR / 'databases'       # every study's SQLite database
SPLIT_PATH = DATA_DIR / 'split_manifest.csv'
DB_PATH = DATABASE_DIR / 'experiments.sqlite'
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


# ===== From prepare_data.py — Step 1: Import Libraries =====

# Step 2: Reproduce the Original Stratified Assignment

def build_split():
    raw = pd.read_csv(DATA_PATH, encoding='cp1252')
    if raw.Incident_ID.isna().any() or not raw.Incident_ID.is_unique:
        raise ValueError('Dataset IDs must be nonmissing and unique.')
    target = raw[TARGET].astype('string').str.strip().str.lower().replace('', pd.NA)
    if not target.dropna().isin(CLASSES + ['aircraft missing']).all():
        raise ValueError('Unexpected target categories.')
    eligible = raw.loc[target.isin(CLASSES)].copy()
    eligible[TARGET] = target.loc[eligible.index]
    train, temporary = train_test_split(
        eligible, test_size=.3, stratify=eligible[TARGET], random_state=RANDOM_STATE
    )
    validation, test = train_test_split(
        temporary, test_size=.5, stratify=temporary[TARGET], random_state=RANDOM_STATE
    )
    split = pd.concat([
        train[['Incident_ID']].assign(split='train'),
        validation[['Incident_ID']].assign(split='validation'),
        test[['Incident_ID']].assign(split='test'),
    ], ignore_index=True)

    # Step 3: Keep One Representative per Exact-Record Group
    # Prefer the original held-out representative; preserve all other assignments.
    columns = [col for col in raw if col != 'Incident_ID']
    groups = eligible.assign(group=pd.util.hash_pandas_object(eligible[columns], index=False).to_numpy())
    groups = groups[['Incident_ID', 'group']].merge(split, on='Incident_ID', validate='one_to_one')
    groups['priority'] = groups['split'].map({'train': 0, 'validation': 1, 'test': 2})
    representatives = groups.sort_values('priority', ascending=False, kind='stable').drop_duplicates('group')
    kept = set(representatives.Incident_ID)
    split = split.loc[split.Incident_ID.isin(kept)].copy()
    excluded = groups.loc[~groups.Incident_ID.isin(kept), 'Incident_ID'].tolist()
    if len(split) != groups.group.nunique():
        raise RuntimeError('Exact-record group deduplication failed.')

    # Step 4: Save the Versioned Protocol and Split Manifest
    split.to_csv(SPLIT_PATH, index=False)
    metadata = {
        'protocol': PROTOCOL,
        'seed': RANDOM_STATE,
        'duplicate_policy': 'one exact-record representative; prefer original test, then validation, then train',
        'excluded_duplicate_ids': excluded,
        'split_counts': split['split'].value_counts().to_dict(),
        'natural_missing_targets': int(target.isna().sum()),
        'scope': 'synthetic masking on originally labeled four-class records; raw CSV unchanged',
    }
    (DATA_DIR / 'split_metadata.json').write_text(json.dumps(metadata, indent=2))
    print(f"Split saved: {metadata['split_counts']} | Exact-record representatives removed: {len(excluded)}")


# ===== From masking.py — Step 1: Simulate Exact Label Counts with Reproducible Nested Budgets =====

def make_mask(X, n_labeled, seed=42, mechanism='MCAR', *, outcomes=None, config=DEFAULT):
    n = len(X)
    if not isinstance(n_labeled, (int, np.integer)) or not 2 <= n_labeled <= n-2:
        raise ValueError('Visible labels must be an integer between 2 and n_train-2.')
    if mechanism not in config.mechanisms:
        raise ValueError('Unknown mechanism.')
    rng = np.random.default_rng(seed)
    visible = np.zeros(n, dtype=bool)
    propensity = np.full(n, np.nan)
    if mechanism == 'MCAR':
        visible[rng.permutation(n)[:n_labeled]] = True
        propensity[:] = n_labeled / n
    else:
        # MAR depends on observed year; MNAR depends on outcome only in simulation.
        if mechanism == 'MAR':
            low_observation = X.Incident_Year.fillna(-1).to_numpy() >= config.recent_year
        else:
            if outcomes is None or len(outcomes) != n or outcomes.isna().any():
                raise ValueError('MNAR simulation requires aligned complete targets.')
            if not outcomes.index.equals(X.index):
                raise ValueError('MNAR targets must align with the training IDs.')
            low_observation = outcomes.to_numpy() == 'destroyed'
        groups = [np.flatnonzero(~low_observation), np.flatnonzero(low_observation)]
        sizes = np.array([len(ids) for ids in groups])
        if (sizes < 2).any():
            raise ValueError('Both sampling strata must have at least two rows.')
        fraction = sizes[1] * config.selection_ratio / (sizes[0] + sizes[1] * config.selection_ratio)
        low_count = int(np.clip(round(n_labeled*fraction), max(1, n_labeled-sizes[0]+1), min(sizes[1]-1, n_labeled-1)))
        for ids, count in zip(groups, [n_labeled-low_count, low_count]):
            visible[rng.permutation(ids)[:count]] = True
            if mechanism == 'MAR':
                # Fixed-quota SRS within each observed stratum: exact inclusion probability.
                propensity[ids] = count / len(ids)
    mask = pd.DataFrame({'Incident_ID': X.index, 'is_labeled': visible, 'propensity': propensity})
    mask.attrs.update(mechanism=mechanism, n_labeled=n_labeled, seed=seed)
    return mask


# Step 2: Keep Outcome-Dependent Mask Generation Separate from Model Inputs

def generate_mask(n_labeled, seed=42, mechanism='MCAR', config=DEFAULT):
    X = training_features()
    outcomes = None
    if mechanism == 'MNAR':
        outcomes = read_data().set_index('Incident_ID').loc[X.index, TARGET]
    return make_mask(X, n_labeled, seed, mechanism, outcomes=outcomes, config=config)


# ===== From storage.py — Step 1: Store All Metrics, CV Details and Resume State in One SQLite File =====

def dumps(value):
    return json.dumps(value, allow_nan=False, default=lambda x: x.item() if isinstance(x, np.generic) else x.tolist())


def fingerprint(config=DEFAULT):
    return {'config': config.to_dict(),
            'data_sha256': hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
            'split_sha256': hashlib.sha256(SPLIT_PATH.read_bytes()).hexdigest(),
            'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
            'versions': {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'catboost', 'xgboost']},
            'evaluation': 'fixed validation/development set; test performance not evaluated'}


def fingerprint_differences(old, new):
    """What changed between two fingerprints: config fields, data/split hashes, versions, source files."""
    changed = [f'config.{k}' for k in sorted(set(old.get('config', {})) | set(new.get('config', {})))
               if old.get('config', {}).get(k) != new.get('config', {}).get(k)]
    changed += [key for key in ['data_sha256', 'split_sha256', 'versions', 'evaluation'] if old.get(key) != new.get(key)]
    old_src, new_src = old.get('source_sha256', {}), new.get('source_sha256', {})
    changed += sorted(f'src/{name}' for name in set(old_src) | set(new_src) if old_src.get(name) != new_src.get(name))
    return changed


def stored_fingerprint_warning(path, key='fingerprint', config=DEFAULT):
    """Read-only check used when a finished study is skipped: never silent about version differences."""
    db = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    try:
        row = db.execute('SELECT value FROM metadata WHERE key=?', (key,)).fetchone()
    finally:
        db.close()
    if row is None:
        return ' WARNING: no stored fingerprint.'
    changed = fingerprint_differences(json.loads(row[0]), json.loads(dumps(fingerprint(config))))
    return f' WARNING: stored results predate changes to {changed}; see README (replay checks).' if changed else ''


@contextmanager
def connect(path=DB_PATH):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.execute('PRAGMA foreign_keys=ON')
    db.executescript('''
    CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS reference(id INTEGER PRIMARY KEY CHECK(id=1), value TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS cases(mechanism TEXT, budget INTEGER, seed INTEGER, status TEXT,
        audit TEXT, error TEXT, PRIMARY KEY(mechanism,budget,seed));
    CREATE TABLE IF NOT EXISTS metrics(mechanism TEXT, budget INTEGER, seed INTEGER, method TEXT,
        status TEXT, accuracy REAL, macro_f1 REAL, balanced_accuracy REAL, log_loss REAL,
        disagreement REAL, class_f1 TEXT, delta_f1 REAL, weight REAL, effective_weight REAL,
        threshold REAL, accepted INTEGER, used INTEGER, lambda_value REAL,
        PRIMARY KEY(mechanism,budget,seed,method),
        FOREIGN KEY(mechanism,budget,seed) REFERENCES cases(mechanism,budget,seed) ON DELETE CASCADE);
    ''')
    try:
        yield db
        db.commit()
    except BaseException:
        db.rollback()
        raise
    finally:
        db.close()


def set_metadata(db, key, value):
    db.execute('INSERT OR REPLACE INTO metadata VALUES(?,?)', (key, dumps(value)))
    db.commit()


def get_metadata(db, key):
    row = db.execute('SELECT value FROM metadata WHERE key=?', (key,)).fetchone()
    return None if row is None else json.loads(row[0])


def initialize(db, config=DEFAULT):
    current = json.loads(dumps(fingerprint(config)))
    previous = get_metadata(db, 'fingerprint')
    if previous is not None and previous != current:
        raise ValueError('Configuration/source/data changed. Use --fresh to replace the old experiment database.')
    set_metadata(db, 'fingerprint', current)


# Step 2: Commit Each Completed Case Atomically; Preserve Explicit Failures

def save_case(db, key, rows, audit, error=None):
    with db:
        db.execute('DELETE FROM cases WHERE mechanism=? AND budget=? AND seed=?', key)
        db.execute('INSERT INTO cases VALUES(?,?,?,?,?,?)', (*key, 'failed' if error else 'complete', dumps(audit), error))
        columns = ['method', 'status', 'accuracy', 'macro_f1', 'balanced_accuracy', 'log_loss',
                   'disagreement', 'class_f1', 'delta_f1', 'weight', 'effective_weight',
                   'threshold', 'accepted', 'used', 'lambda_value']
        values = []
        for row in rows:
            values.append((*key, *[dumps(row[name]) if name == 'class_f1' and row[name] is not None else row[name] for name in columns]))
        db.executemany('INSERT INTO metrics VALUES('+','.join('?' for _ in range(18))+')', values)


# Step 3: Summarize Valid Runs Without Hiding Failed or Non-Estimable Cases

def read_results(db):
    return pd.read_sql_query('SELECT * FROM metrics', db)


def summary(db):
    rows = read_results(db)
    if rows.empty:
        return pd.DataFrame()
    return rows.groupby(['mechanism', 'budget', 'method'], sort=False).agg(
        runs=('seed', 'count'), valid_runs=('macro_f1', 'count'),
        accuracy_mean=('accuracy', 'mean'), accuracy_std=('accuracy', 'std'),
        macro_f1_mean=('macro_f1', 'mean'), macro_f1_std=('macro_f1', 'std'),
        delta_f1_mean=('delta_f1', 'mean'), delta_f1_std=('delta_f1', 'std'),
        used_mean=('used', 'mean')).reset_index()
