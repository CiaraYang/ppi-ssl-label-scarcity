# Shared: Configuration, Data and Split, Label Masks, Result Storage (Imported by Every Step)

import os
import re
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
        if self.teacher_model not in {'CatBoost', 'LR', 'XGBoost', 'TabPFN', 'Ensemble'}:
            raise ValueError('Supported Teachers are CatBoost, LR, XGBoost, the pretrained TabPFN and the CatBoost+LR Ensemble.')
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
# Task version: v1 = the original task (4 classes, 7 features); v2 = 3 classes + contextual features.
# Select with the environment variable PPI_VERSION=v2; v1 stays the default, and v2 writes only under results/v2/.
VERSION = os.environ.get('PPI_VERSION', 'v1')
if VERSION not in {'v1', 'v2'}:
    raise ValueError('PPI_VERSION must be v1 or v2.')
RESULTS_DIR = PROJECT_ROOT / 'results' / ('' if VERSION == 'v1' else VERSION)   # figures and CSV tables
DATABASE_DIR = RESULTS_DIR / 'databases'       # every study's SQLite database
SPLIT_PATH = DATA_DIR / 'split_manifest.csv'   # the same Train / Validation / Test IDs in every version
DB_PATH = DATABASE_DIR / 'experiments.sqlite'
TARGET = 'Aircraft_Damage_Type'
RAW_CLASSES = ['damaged beyond repair', 'destroyed', 'minor', 'substantial']
# v2 merges the two written-off outcomes (an aircraft destroyed or damaged beyond repair is a hull loss).
CLASS_MAP = {'damaged beyond repair': 'written off', 'destroyed': 'written off', 'substantial': 'substantial', 'minor': 'minor'}
CLASSES = RAW_CLASSES if VERSION == 'v1' else ['written off', 'substantial', 'minor']
NUMERIC_FEATURES = ['Incident_Year', 'Incident_Month', 'Aircraft_Age', 'Onboard_Total']
# Contextual features (v2): parsed from the record by fixed, label-free rules; not all are strictly pre-incident.
CONTEXTUAL_FEATURES = ['Registration_Prefix', 'Departure_Region', 'Has_Route', 'Same_Airport', 'Domestic_Route']
CATEGORICAL_FEATURES = ['Aircraft_Model', 'Aircraft_Nature', 'Aircraft_Phase'] + (CONTEXTUAL_FEATURES if VERSION == 'v2' else [])
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
    for col in ['Aircraft_Model', 'Aircraft_Nature', 'Aircraft_Phase']:
        value = df[col].astype('string').str.strip().replace('', pd.NA)
        df[col] = value.astype(object).where(value.notna(), np.nan)
    if VERSION == 'v2':
        df = df.join(contextual_features(df))
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
    if not values.isin(RAW_CLASSES).all():
        raise ValueError('Unexpected visible target value.')
    if VERSION == 'v2':
        values = values.map(CLASS_MAP)
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


# ===== Contextual Features (v2): Validated Parsing With Fixed, Label-Free Row-Wise Rules =====

REGIONS = {
    'North America': ['united states of america', 'canada', 'greenland', 'bermuda'],
    'Latin America': ['mexico', 'brazil', 'colombia', 'argentina', 'venezuela', 'bolivia', 'peru', 'chile', 'ecuador', 'cuba',
                      'guatemala', 'honduras', 'nicaragua', 'panama', 'costa rica', 'el salvador', 'paraguay', 'uruguay',
                      'haiti', 'dominican republic', 'jamaica', 'bahamas', 'belize', 'guyana', 'suriname', 'puerto rico',
                      'trinidad and tobago', 'french guiana', 'guadeloupe', 'martinique', 'barbados', 'u.s. virgin islands',
                      'british virgin islands', 'turks and caicos islands', 'curaçao', 'sint maarten', 'saint lucia',
                      'antigua and barbuda', 'st. kitts and nevis', 'grenada', 'dominica', 'aruba', 'cayman islands',
                      'saint vincent and the grenadines', 'montserrat', 'anguilla', 'falkland islands'],
    'Europe': ['united kingdom', 'france', 'germany', 'italy', 'spain', 'norway', 'greece', 'portugal', 'netherlands',
               'switzerland', 'belgium', 'sweden', 'denmark', 'finland', 'austria', 'ireland', 'poland', 'czech republic',
               'czechia', 'iceland', 'hungary', 'romania', 'bulgaria', 'yugoslavia', 'serbia', 'croatia', 'slovakia',
               'slovenia', 'luxembourg', 'malta', 'cyprus', 'estonia', 'latvia', 'lithuania', 'albania',
               'bosnia and herzegovina', 'north macedonia', 'montenegro', 'monaco', 'czechoslovakia', 'gibraltar',
               'faroe islands', 'liechtenstein', 'andorra', 'san marino', 'kosovo', 'isle of man', 'jersey', 'guernsey'],
    'Russia & former USSR': ['russia', 'ukraine', 'kazakhstan', 'uzbekistan', 'belarus', 'georgia', 'armenia', 'azerbaijan',
                             'kyrgyzstan', 'tajikistan', 'turkmenistan', 'moldova', 'u.s.s.r.', 'ussr', 'soviet union'],
    'Asia': ['india', 'indonesia', 'china', 'japan', 'vietnam', 'philippines', 'thailand', 'myanmar', 'taiwan', 'malaysia',
             'bangladesh', 'nepal', 'pakistan', 'south korea', 'north korea', 'laos', 'cambodia', 'sri lanka', 'singapore',
             'mongolia', 'hong kong', 'afghanistan', 'bhutan', 'brunei', 'timor-leste', 'east timor', 'macau', 'maldives'],
    'Middle East': ['turkey', 'iran', 'iraq', 'saudi arabia', 'israel', 'syria', 'lebanon', 'jordan', 'yemen', 'oman',
                    'united arab emirates', 'kuwait', 'qatar', 'bahrain', 'palestine'],
    'Africa': ['democratic republic of the congo', 'angola', 'nigeria', 'south africa', 'sudan', 'kenya', 'algeria', 'egypt',
               'south sudan', 'ethiopia', 'libya', 'morocco', 'tanzania', 'uganda', 'somalia', 'chad', 'cameroon', 'zambia',
               'zimbabwe', 'mozambique', 'ghana', 'senegal', 'mali', 'niger', 'tunisia', 'madagascar', 'gabon', 'congo',
               'republic of the congo', 'ivory coast', "côte d'ivoire", 'namibia', 'botswana', 'liberia', 'sierra leone',
               'guinea', 'burkina faso', 'central african republic', 'rwanda', 'burundi', 'mauritania', 'benin', 'togo',
               'malawi', 'eritrea', 'djibouti', 'equatorial guinea', 'guinea-bissau', 'lesotho', 'eswatini', 'swaziland',
               'são tomé and príncipe', 'cape verde', 'comoros', 'gambia', 'st. helena', 'mauritius', 'seychelles',
               'western sahara', 'réunion', 'mayotte'],
    'Oceania': ['australia', 'papua new guinea', 'new zealand', 'fiji', 'solomon islands', 'vanuatu', 'new caledonia',
                'french polynesia', 'guam', 'samoa', 'tonga', 'marshall islands', 'micronesia', 'kiribati', 'american samoa',
                'u.s. minor outlying islands', 'northern mariana islands', 'palau', 'nauru', 'tuvalu', 'cook islands',
                'wallis and futuna', 'norfolk island'],
    'Other': ['antarctica'],
}
COUNTRY_REGION = {country: region for region, countries in REGIONS.items() for country in countries}
COUNTRY_ALIASES = {'usa': 'united states of america', 'u.s.a.': 'united states of america', 'united states': 'united states of america',
                   "cote d'ivoire": "côte d'ivoire", 'uk': 'united kingdom', 'great britain': 'united kingdom',
                   'curacao': 'curaçao', 'sao tome and principe': 'são tomé and príncipe', 'russian federation': 'russia'}
US_STATES = set('AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK '
                'OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC'.split())
AU_STATES = {'NSW', 'QLD', 'VIC', 'SA', 'WA', 'TAS', 'NT', 'ACT'}
ICAO = re.compile(r'\((?:[A-Z0-9]{3}/)?([A-Z]{4})\)')


def icao_region(code):
    """Region from the ICAO airport code's standard regional prefix (first one or two letters)."""
    if code[:2] in {'LL', 'LT'}:
        return 'Middle East'
    if code[:2] in {'OA', 'OP'}:
        return 'Asia'
    if code[:2] in {'PA', 'PH', 'PF', 'PO', 'PP'}:
        return 'North America'
    return {'K': 'North America', 'C': 'North America', 'M': 'Latin America', 'T': 'Latin America', 'S': 'Latin America',
            'E': 'Europe', 'L': 'Europe', 'B': 'Europe', 'U': 'Russia & former USSR', 'O': 'Middle East', 'V': 'Asia',
            'W': 'Asia', 'Z': 'Asia', 'R': 'Asia', 'Y': 'Oceania', 'N': 'Oceania', 'A': 'Oceania', 'P': 'Oceania',
            'D': 'Africa', 'F': 'Africa', 'G': 'Africa', 'H': 'Africa'}.get(code[0], np.nan)


def airport_country(text):
    """Country only when the tail (or the whole string) is a verified country name, alias or state code; else NaN."""
    if not isinstance(text, str) or not text.strip():
        return np.nan
    tail = text.rsplit(',', 1)[1].strip() if ',' in text else text.strip()
    name = COUNTRY_ALIASES.get(tail.lower(), tail.lower())
    if name in COUNTRY_REGION:
        return name
    code = re.match(r'^([A-Z]{2,3})(?:\s*\(|$)', tail)
    if code:
        icao = ICAO.search(text)
        first = icao.group(1)[0] if icao else None
        if code.group(1) in US_STATES and (code.group(1) not in AU_STATES or first == 'K'):
            return 'united states of america'
        if code.group(1) in AU_STATES and (code.group(1) not in US_STATES or first == 'Y'):
            return 'australia'
    return np.nan


def airport_region(text):
    """Verified country → region; otherwise the ICAO regional prefix; otherwise NaN (never a guess)."""
    country = airport_country(text)
    if isinstance(country, str):
        return COUNTRY_REGION[country]
    icao = ICAO.search(text) if isinstance(text, str) else None
    return icao_region(icao.group(1)) if icao else np.nan


def registration_prefix(registration):
    """Nationality prefix of the registration (country of registry); all-digit serials are military-style."""
    if not isinstance(registration, str) or not registration.strip():
        return np.nan
    registration = registration.strip().upper()
    if re.fullmatch(r'[\d\-/ ]+', registration):
        return 'numeric serial'
    if re.match(r'N\d', registration):
        return 'N'
    return registration.split('-')[0][:4] if '-' in registration else registration[:2]


def contextual_features(df):
    departure, destination = df.Departure_Airport, df.Destination_Airport
    has_route = departure.notna() & destination.notna()
    dep_country, dest_country = departure.map(airport_country), destination.map(airport_country)
    both = dep_country.notna() & dest_country.notna() & has_route
    out = pd.DataFrame(index=df.index)
    out['Registration_Prefix'] = df.Aircraft_Registration.map(registration_prefix)
    out['Departure_Region'] = departure.map(airport_region)
    out['Has_Route'] = np.where(has_route, 'yes', 'no')
    out['Same_Airport'] = np.where(has_route & (departure == destination), 'yes', 'no')
    out['Domestic_Route'] = np.where(~both, 'unknown', np.where(dep_country == dest_country, 'yes', 'no'))
    return out.astype(object).where(out.notna(), np.nan)


def contextual_parsing_report(df):
    """Validation of the airport parsing: coverage of each source and agreement where both sources exist."""
    airports = pd.concat([df.Departure_Airport, df.Destination_Airport]).dropna()
    by_country = airports.map(lambda t: COUNTRY_REGION.get(airport_country(t), np.nan) if isinstance(airport_country(t), str) else np.nan)
    by_icao = airports.map(lambda t: icao_region(ICAO.search(t).group(1)) if ICAO.search(t) else np.nan)
    both = by_country.notna() & by_icao.notna()
    final = airports.map(airport_region)
    return {'airport strings': int(len(airports)), 'region from verified country': float(by_country.notna().mean()),
            'region from ICAO only': float((by_country.isna() & by_icao.notna()).mean()),
            'unparsed (left missing)': float(final.isna().mean()),
            'rows with both sources': int(both.sum()), 'country-vs-ICAO region agreement': float((by_country[both] == by_icao[both]).mean())}


# Stored in every v2 database before its first result; v2 is adopted as the final setting whatever PPI shows.
V2_DECLARATION = {
    'name': 'v2_three_classes_contextual_features', 'declared_on': '2026-10-18',
    'why': 'the v1 fully labeled reference was low (LR Macro F1 0.637). On the training split, merging the two written-off '
           'outcomes and adding contextual features raised the fully labeled ceiling (validation: LR 0.709 → 0.725, '
           'CatBoost 0.719 → 0.735, TabPFN 0.746); v1 remains as the development version',
    'adoption rule': 'v2 replaces v1 as the final setting regardless of how Supervised, SSL or PPI compare',
    'classes': 'written off (destroyed + damaged beyond repair) / substantial / minor',
    'contextual features': 'registration prefix; departure region (verified country name or ICAO regional prefix, else '
                           'missing); has route; same airport; domestic route (unknown unless both countries are verified). '
                           'Fixed row-wise rules, no labels, nothing fitted outside the training fold; called contextual, '
                           'not claimed to be strictly pre-incident',
    'excluded': 'Incident_Category, Incident_Causes, Fatalities and any fatality flag (outcome information)',
    'missing data': 'never filled with 0; median imputation with missing indicators and an Unknown category, fitted in '
                    'the training fold',
    'design': 'same Train / Validation / Test IDs; budgets 50–1,000; seeds 42–46 (75 cases); MCAR and MAR unchanged '
              '(Students keep IPW); MNAR still under-samples the recorded destroyed outcome, i.e. a subgroup selection '
              'inside written off; every method in a case uses the same mask',
    'models': 'primary Teachers CatBoost and TabPFN; secondary Teacher Ensemble (CatBoost + LR, only the Teacher '
              'changes); primary Student LR; sensitivity Student XGBoost; Supervised / SSL soft / PPI selected with the '
              'v1 SSL weights, confidence threshold, λ grid, training CV, cross-fitting and nested Teacher fitting',
    'evaluation': 'primary Macro F1 with argmax decisions; secondary Accuracy, Balanced Accuracy, Log Loss and per-class '
                  'F1; validation only (development set); Test untouched',
}


# ===== From prepare_data.py — Step 1: Import Libraries =====

# Step 2: Reproduce the Original Stratified Assignment

def build_split():
    raw = pd.read_csv(DATA_PATH, encoding='cp1252')
    if raw.Incident_ID.isna().any() or not raw.Incident_ID.is_unique:
        raise ValueError('Dataset IDs must be nonmissing and unique.')
    target = raw[TARGET].astype('string').str.strip().str.lower().replace('', pd.NA)
    if not target.dropna().isin(RAW_CLASSES + ['aircraft missing']).all():
        raise ValueError('Unexpected target categories.')
    eligible = raw.loc[target.isin(RAW_CLASSES)].copy()
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
            'evaluation': 'fixed validation/development set; test performance not evaluated',
            **({'task_version': VERSION, 'classes': CLASSES, 'features': FEATURES} if VERSION != 'v1' else {})}


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
