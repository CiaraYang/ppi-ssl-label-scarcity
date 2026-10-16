# Step 1: Store All Metrics, CV Details and Resume State in One SQLite File

import hashlib
from contextlib import contextmanager
import importlib.metadata
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from config import DEFAULT
from model_utils import DATA_PATH, DB_PATH, SPLIT_PATH


def dumps(value):
    return json.dumps(value, allow_nan=False, default=lambda x: x.item() if isinstance(x, np.generic) else x.tolist())


def fingerprint(config=DEFAULT):
    return {'config': config.to_dict(),
            'data_sha256': hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
            'split_sha256': hashlib.sha256(SPLIT_PATH.read_bytes()).hexdigest(),
            'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
            'versions': {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'catboost', 'xgboost']},
            'evaluation': 'fixed validation/development set; test performance not evaluated'}


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
