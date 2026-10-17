# Step 1: Frozen, One-Shot Test Evaluation of the Final Method and Its Pre-Declared Comparisons

import json
import sqlite3
from dataclasses import replace
from itertools import product

import pandas as pd
from joblib import Parallel, delayed, parallel_config

from config import DEFAULT
from experiment import fully_labeled_reference, run_case
from model_utils import DB_PATH, RESULTS_DIR
from storage import dumps, fingerprint, fingerprint_differences
from teacher_study import STUDY_DB, variant_name

FINAL_DB = RESULTS_DIR / 'final_test.sqlite'
# Written to the database before any Test score exists; the run refuses to start if it ever differs.
FROZEN = {
    'status': 'draft',  # Set to 'final' only after the main experiment is settled; run() refuses a draft.
    'frozen_on': None,
    'final_method': 'LR Student with a cross-fitted LR Teacher; labeled-only preprocessing; '
                    'SSL weight and PPI λ chosen by training CV (main-study grids)',
    'teacher_rule': 'Teacher chosen by mean training-only OOF Macro F1 pooled over all 75 simulated masks '
                    '(LR 0.490 > XGBoost 0.487 > CatBoost 0.469). This is a protocol-level choice using every mask, '
                    'not a rule a single dataset with its own n labels could apply; validation not used',
    'variants': ['LR→LR (final)', 'CatBoost→LR (original design)', 'XGBoost→XGBoost', 'CatBoost→XGBoost (original design)'],
    'references': 'Fully labeled LR and XGBoost',
    'models': 'identical to the validation runs (fits never see the evaluation split); only the scored split changes',
}
VARIANTS = [('LR', 'LR'), ('CatBoost', 'LR'), ('XGBoost', 'XGBoost'), ('CatBoost', 'XGBoost')]


def variant_config(teacher, student):
    return replace(DEFAULT, student_models=(student,), student_model=student, teacher_model=teacher)


def validation_audits():
    """Stored validation-run audits, used only to prove the Test run refits the same models."""
    audits = {}
    for path, query in [(DB_PATH, "SELECT 'main',mechanism,budget,seed,audit FROM cases WHERE status='complete'"),
                        (STUDY_DB, "SELECT variant,mechanism,budget,seed,audit FROM cases WHERE status='complete'")]:
        db = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
        try:
            for v, m, b, s, a in db.execute(query):
                audits[(v, m, b, s)] = json.loads(a)
        finally:
            db.close()
    return audits


def expected(audits, teacher, student, key):
    audit = audits[('main', *key)] if teacher == 'CatBoost' else audits[(variant_name(teacher, student, 'labeled'), *key)]
    return audit['teacher_oof'], audit['students'][student]['selected'], audit['students'][student]['ppi_selected_lambda']


def evaluate(variant, key, stored):
    teacher, student = variant
    mechanism, budget, seed = key
    try:
        rows, audit = run_case(budget, mechanism, seed, variant_config(teacher, student), evaluation='test')
        oof, selected, lam = stored
        entry = audit['students'][student]
        # Training-only quantities must match the validation run exactly, so these are the same fitted models.
        if (json.loads(dumps(audit['teacher_oof'])) != oof or json.loads(dumps(entry['selected'])) != selected
                or entry['ppi_selected_lambda'] != lam):
            raise RuntimeError('Test-time refit differs from the stored validation run.')
        return variant, key, rows, audit, None
    except Exception as error:
        return variant, key, [], {}, f'{type(error).__name__}: {error}'


# Step 2: Run Once. A Completed Test Evaluation Is Never Repeated

def connect():
    db = sqlite3.connect(FINAL_DB)
    db.executescript('''
    CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS cases(variant TEXT, mechanism TEXT, budget INTEGER, seed INTEGER, status TEXT,
        audit TEXT, error TEXT, PRIMARY KEY(variant,mechanism,budget,seed));
    CREATE TABLE IF NOT EXISTS metrics(variant TEXT, mechanism TEXT, budget INTEGER, seed INTEGER, method TEXT,
        accuracy REAL, macro_f1 REAL, balanced_accuracy REAL, log_loss REAL, class_f1 TEXT, delta_f1 REAL,
        lambda_value REAL, PRIMARY KEY(variant,mechanism,budget,seed,method));
    ''')
    return db


def run(workers=8):
    if FROZEN['status'] != 'final':
        raise RuntimeError('The Test protocol is still a draft. Settle the main experiment, set FROZEN["status"] to '
                           '"final" with a date, then run once.')
    current = json.loads(dumps(fingerprint()))
    db = connect()
    try:
        found = dict(db.execute('SELECT key, value FROM metadata').fetchall())
        if found.get('status') == '"complete"':
            print('Test evaluation already completed once; it is not repeated. Reporting stored results.', flush=True)
            return
        if 'fingerprint' in found and fingerprint_differences(json.loads(found['fingerprint']), current):
            raise ValueError('Source/data changed since the Test evaluation started; it cannot be resumed.')
        if 'frozen_protocol' in found and json.loads(found['frozen_protocol']) != FROZEN:
            raise ValueError('The frozen protocol changed after the Test evaluation started.')
        db.execute('INSERT OR REPLACE INTO metadata VALUES(?,?)', ('frozen_protocol', dumps(FROZEN)))
        db.execute('INSERT OR REPLACE INTO metadata VALUES(?,?)', ('fingerprint', dumps(current)))
        db.commit()
        if 'fully_labeled_test' not in found:
            reference = fully_labeled_reference(evaluation='test')
            db.execute('INSERT OR REPLACE INTO metadata VALUES(?,?)', ('fully_labeled_test', dumps(reference)))
            db.commit()
        audits = validation_audits()
        done = {(row[0], tuple(row[1:])) for row in
                db.execute("SELECT variant,mechanism,budget,seed FROM cases WHERE status='complete'")}
        tasks = [(v, key) for v in VARIANTS for key in product(DEFAULT.mechanisms, DEFAULT.budgets, DEFAULT.seeds)
                 if (variant_name(*v, 'labeled'), key) not in done]
        print(f'{len(tasks)} Test cases to run | {len(done)} done | {workers} CPU workers', flush=True)
        with parallel_config(backend='loky', inner_max_num_threads=1):
            results = Parallel(n_jobs=workers, return_as='generator_unordered')(
                delayed(evaluate)(v, key, expected(audits, *v, key)) for v, key in tasks)
            for count, (variant, key, rows, audit, error) in enumerate(results, 1):
                name = variant_name(*variant, 'labeled')
                with db:
                    db.execute('DELETE FROM metrics WHERE variant=? AND mechanism=? AND budget=? AND seed=?', (name, *key))
                    db.execute('INSERT OR REPLACE INTO cases VALUES(?,?,?,?,?,?,?)',
                               (name, *key, 'failed' if error else 'complete', dumps(audit), error))
                    db.executemany('INSERT INTO metrics VALUES(?,?,?,?,?,?,?,?,?,?,?,?)', [
                        (name, *key, row['method'].split(' / ')[1], row['accuracy'], row['macro_f1'],
                         row['balanced_accuracy'], row['log_loss'], dumps(row['class_f1']), row['delta_f1'],
                         row['lambda_value']) for row in rows])
                if error or count % 50 == 0 or count == len(tasks):
                    print(f'[{count}/{len(tasks)}] {name} {key} {"FAILED: "+error if error else "saved"}', flush=True)
        failed = db.execute("SELECT COUNT(*) FROM cases WHERE status!='complete'").fetchone()[0]
        if failed:
            raise RuntimeError(f'{failed} Test cases failed; see {FINAL_DB.name}.')
        db.execute('INSERT OR REPLACE INTO metadata VALUES(?,?)', ('status', dumps('complete')))
        db.commit()
    finally:
        db.close()


def load():
    db = sqlite3.connect(f'file:{FINAL_DB}?mode=ro', uri=True)
    try:
        rows = pd.read_sql_query('SELECT * FROM metrics', db)
        reference = json.loads(db.execute("SELECT value FROM metadata WHERE key='fully_labeled_test'").fetchone()[0])
    finally:
        db.close()
    parts = rows.variant.str.extract(r'^(?P<teacher>\w+)→(?P<student>\w+) \[')
    return pd.concat([rows, parts], axis=1), reference
