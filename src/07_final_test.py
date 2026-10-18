# Step 7: Frozen One-Shot Test Evaluation (Locked as a Draft; Run Last and Only Once)

import sys
from importlib import import_module
from pathlib import Path

import json
import sqlite3
from dataclasses import replace
from itertools import product

import pandas as pd
from joblib import Parallel, delayed, parallel_config

sys.path.insert(0, str(Path(__file__).resolve().parent))  # The numbered files import each other by name.

common = import_module('00_common')
DEFAULT, DB_PATH, RESULTS_DIR, dumps, fingerprint, fingerprint_differences = common.DEFAULT, common.DB_PATH, common.RESULTS_DIR, common.dumps, common.fingerprint, common.fingerprint_differences
methods = import_module('00_methods')
fully_labeled_reference, run_case = methods.fully_labeled_reference, methods.run_case
_teacher_study = import_module('03_teacher_study')
STUDY_DB, variant_name, full_db, FULL_TEACHERS = _teacher_study.STUDY_DB, _teacher_study.variant_name, _teacher_study.full_db, _teacher_study.FULL_TEACHERS


# ===== From final_test.py — Step 1: Frozen, One-Shot Test Evaluation of the Final Method and Its Pre-Declared Comparisons =====

FINAL_DB = RESULTS_DIR / 'databases' / 'final_test.sqlite'
# Written to the database before any Test score exists; the run refuses to start if it ever differs.
FROZEN_V1 = {
    'status': 'draft',  # Set to 'final' (with a date) only when the author freezes the protocol; run() refuses a draft.
    'frozen_on': None,
    'final_method': 'Primary (the proposal design): LR Student with a cross-fitted CatBoost Teacher; labeled-only '
                    'preprocessing; SSL weight and PPI λ chosen by training CV on the visible labels (main-study grids); '
                    'argmax decisions. No Teacher selection, so every mask runs it within its own label budget',
    'primary_claims': 'Pooled over 75 masks, mean with a seed-cluster interval (5 seeds; descriptive): "improvement" only if '
                      'PPI − Supervised stays above 0, "robustness" only if PPI − SSL soft stays above 0; both reported '
                      'whatever the outcome',
    'teacher_rule': 'Secondary only: per mask, keep the CatBoost or LR Teacher whose selected PPI setting has the higher '
                    'training-CV Macro F1 on that mask\'s visible labels (teacher_study.selected_cv_score; ties and the '
                    'no-CV small-class fallback keep CatBoost). Replaces the earlier draft rule that pooled OOF scores '
                    'over all 75 masks, which no single dataset could apply',
    'variants': ['CatBoost→LR (primary)', 'LR→LR (for the secondary Teacher rule)', 'CatBoost→XGBoost (sensitivity)'],
    'not_on_test': 'improved_v2, the threshold rule, the synthetic checks and the TabPFN pilot',
    'references': 'Fully labeled LR and XGBoost',
    'disclosure': 'validation was used as a development set; the protocol was revised after viewing it',
    'models': 'identical to the validation runs (fits never see the evaluation split); only the scored split changes',
}
# v2 (PPI_VERSION=v2): same one-shot procedure on the v2 task; CatBoost and TabPFN-3.5 are co-primary Teachers.
FROZEN_V2 = {
    'status': 'draft',  # Set to 'final' (with a date) only when the author freezes the protocol; run() refuses a draft.
    'frozen_on': None,
    'task': 'v2: 3 classes (written off / substantial / minor), 7 original + 5 contextual features',
    'final_method': 'Co-primary: LR Student with a cross-fitted CatBoost Teacher, and LR Student with a cross-fitted '
                    'TabPFN-3.5 Teacher (pinned checkpoint, visible labels as context); labeled-only preprocessing; SSL '
                    'weight and PPI λ chosen by training CV Macro F1 on the visible labels (main-study grids); argmax '
                    'decisions. Each Teacher is reported separately; no Teacher selection',
    'primary_claims': 'For each primary Teacher, pooled over 75 masks, mean with a seed-cluster interval (5 seeds; '
                      'descriptive): "improvement" only if PPI − Supervised stays above 0, "robustness" only if '
                      'PPI − SSL soft stays above 0; Macro F1 is primary, Balanced Accuracy, Accuracy and Log Loss are '
                      'reported whatever the outcome',
    'teacher_rule': 'Secondary only: the Ensemble Teacher (mean of cross-fitted CatBoost and LR) is reported as a '
                    'secondary comparison; there is no per-mask Teacher selection',
    'variants': ['CatBoost→LR (primary)', 'TabPFN→LR (primary)', 'Ensemble→LR (secondary)',
                 'CatBoost→XGBoost (sensitivity)', 'TabPFN→XGBoost (sensitivity)'],
    'not_on_test': 'the v1 task, A1–A3 ablations, improved_v2, the threshold rule, probability calibration and the '
                   'synthetic checks',
    'references': 'Fully labeled LR and XGBoost; each Teacher\'s own Test score (teacher_test in every audit)',
    'disclosure': 'validation was used as a development set; the v2 task (classes, features, Ensemble) was chosen after '
                  'viewing v1 validation results; under MAR the TabPFN Teacher is unweighted (no sample weights) while '
                  'the Students keep IPW',
    'models': 'identical to the validation runs (fits never see the evaluation split); only the scored split changes',
}
FROZEN = FROZEN_V1 if common.VERSION == 'v1' else FROZEN_V2
VARIANTS = ([('CatBoost', 'LR'), ('LR', 'LR'), ('CatBoost', 'XGBoost')] if common.VERSION == 'v1' else
            [('CatBoost', 'LR'), ('TabPFN', 'LR'), ('Ensemble', 'LR'), ('CatBoost', 'XGBoost'), ('TabPFN', 'XGBoost')])


def variant_config(teacher, student):
    return replace(DEFAULT, student_models=(student,), student_model=student, teacher_model=teacher)


def validation_audits():
    """Stored validation-run audits, used only to prove the Test run refits the same models."""
    audits = {}
    sources = [(DB_PATH, "SELECT 'main',mechanism,budget,seed,audit FROM cases WHERE status='complete'")]
    if common.VERSION == 'v1':
        sources.append((STUDY_DB, "SELECT variant,mechanism,budget,seed,audit FROM cases WHERE status='complete'"))
    else:  # v2: the full-Teacher studies (Ensemble, TabPFN) hold the non-CatBoost validation runs
        sources += [(full_db(t), f"SELECT '{t}',mechanism,budget,seed,audit FROM cases WHERE status='complete'")
                    for t in FULL_TEACHERS]
    for path, query in sources:
        db = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
        try:
            for v, m, b, s, a in db.execute(query):
                audits[(v, m, b, s)] = json.loads(a)
        finally:
            db.close()
    return audits


def expected(audits, teacher, student, key):
    if teacher == 'CatBoost':
        audit = audits[('main', *key)]
    elif common.VERSION == 'v1':
        audit = audits[(variant_name(teacher, student, 'labeled'), *key)]
    else:
        audit = audits[(teacher, *key)]
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


if __name__ == '__main__':
    run()
    import_module('06_figures').final_test_outputs()
