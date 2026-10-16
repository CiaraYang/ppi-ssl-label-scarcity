# Step 1: Same-Architecture Teachers and Label-Free Preprocessing, in a Separate Database

import argparse
import json
import sqlite3
from dataclasses import replace
from itertools import product

import pandas as pd
from joblib import Parallel, delayed, parallel_config

from config import DEFAULT
from experiment import run_case
from model_utils import DB_PATH, RESULTS_DIR
from storage import dumps, fingerprint

STUDY_DB = RESULTS_DIR / 'teacher_study.sqlite'
PROTOCOL = 'teacher_study_v1'
# (Teacher, Student, preprocessing fit). CatBoost→Student on labeled-only preprocessing is the main study itself.
VARIANTS = [('LR', 'LR', 'labeled'), ('XGBoost', 'XGBoost', 'labeled'),
            ('CatBoost', 'LR', 'labeled+unlabeled'), ('CatBoost', 'XGBoost', 'labeled+unlabeled'),
            ('LR', 'LR', 'labeled+unlabeled'), ('XGBoost', 'XGBoost', 'labeled+unlabeled')]
MAIN = [('CatBoost', 'LR', 'labeled'), ('CatBoost', 'XGBoost', 'labeled')]
METHODS = ['Supervised', 'SSL soft', 'PPI selected']


def variant_name(teacher, student, design):
    return f'{teacher}→{student} [{design}]'


def variant_config(teacher, student, design):
    return replace(DEFAULT, student_models=(student,), student_model=student, teacher_model=teacher, design_fit=design)


def connect(path=STUDY_DB):
    db = sqlite3.connect(path)
    db.executescript('''
    CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS cases(variant TEXT, mechanism TEXT, budget INTEGER, seed INTEGER, status TEXT,
        audit TEXT, error TEXT, PRIMARY KEY(variant,mechanism,budget,seed));
    CREATE TABLE IF NOT EXISTS metrics(variant TEXT, mechanism TEXT, budget INTEGER, seed INTEGER, method TEXT,
        accuracy REAL, macro_f1 REAL, balanced_accuracy REAL, log_loss REAL, class_f1 TEXT,
        delta_f1 REAL, lambda_value REAL, weight REAL, PRIMARY KEY(variant,mechanism,budget,seed,method));
    ''')
    return db


def evaluate(variant, key):
    mechanism, budget, seed = key
    try:
        rows, audit = run_case(budget, mechanism, seed, variant_config(*variant))
        return variant, key, rows, audit, None
    except Exception as error:
        return variant, key, [], {}, f'{type(error).__name__}: {error}'


# Step 2: Run Every Variant on the Same 75 Masks; Resume by (Variant, Case)

def pending_tasks():
    """(Variant, case) pairs still to run, checked read-only so a finished study is never rewritten."""
    keys = [(v, key) for v in VARIANTS for key in product(DEFAULT.mechanisms, DEFAULT.budgets, DEFAULT.seeds)]
    if not STUDY_DB.exists():
        return keys
    db = sqlite3.connect(f'file:{STUDY_DB}?mode=ro', uri=True)
    try:
        done = {(row[0], tuple(row[1:])) for row in
                db.execute("SELECT variant,mechanism,budget,seed FROM cases WHERE status='complete'")}
    finally:
        db.close()
    return [(v, key) for v, key in keys if (variant_name(*v), key) not in done]


def run_study(workers=8):
    if not pending_tasks():
        print('Teacher study already complete; nothing to run.', flush=True)
        return
    db = connect()
    try:
        previous = db.execute("SELECT value FROM metadata WHERE key='fingerprint'").fetchone()
        current = json.loads(dumps(fingerprint()))
        if previous is not None and json.loads(previous[0]) != current:
            raise ValueError('Source/data changed since this study started; move teacher_study.sqlite aside to rerun.')
        for key, value in [('fingerprint', current), ('protocol', PROTOCOL),
                           ('variants', [variant_name(*v) for v in VARIANTS])]:
            db.execute('INSERT OR REPLACE INTO metadata VALUES(?,?)', (key, dumps(value)))
        db.commit()
        done = {(row[0], tuple(row[1:])) for row in
                db.execute("SELECT variant,mechanism,budget,seed FROM cases WHERE status='complete'")}
        tasks = [(v, key) for v in VARIANTS for key in product(DEFAULT.mechanisms, DEFAULT.budgets, DEFAULT.seeds)
                 if (variant_name(*v), key) not in done]
        print(f'{len(tasks)} variant cases to run | {len(done)} cached | {workers} CPU workers', flush=True)
        with parallel_config(backend='loky', inner_max_num_threads=1):
            results = Parallel(n_jobs=workers, return_as='generator_unordered')(
                delayed(evaluate)(variant, key) for variant, key in tasks)
            for count, (variant, key, rows, audit, error) in enumerate(results, 1):
                name = variant_name(*variant)
                with db:
                    db.execute('DELETE FROM metrics WHERE variant=? AND mechanism=? AND budget=? AND seed=?', (name, *key))
                    db.execute('INSERT OR REPLACE INTO cases VALUES(?,?,?,?,?,?,?)',
                               (name, *key, 'failed' if error else 'complete', dumps(audit), error))
                    db.executemany('INSERT INTO metrics VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)', [
                        (name, *key, row['method'].split(' / ')[1], row['accuracy'], row['macro_f1'],
                         row['balanced_accuracy'], row['log_loss'], dumps(row['class_f1']),
                         row['delta_f1'], row['lambda_value'], row['weight']) for row in rows])
                if error or count % 25 == 0 or count == len(tasks):
                    print(f'[{count}/{len(tasks)}] {name} {key} {"FAILED: "+error if error else "saved"}', flush=True)
        failed = db.execute("SELECT COUNT(*) FROM cases WHERE status!='complete'").fetchone()[0]
    finally:
        db.close()
    if failed:
        raise RuntimeError(f'{failed} variant cases failed; see {STUDY_DB.name}.')


# Step 3: Load Both Databases Read-Only in One Long Table

def load():
    main = sqlite3.connect(f'file:{DB_PATH}?mode=ro', uri=True)
    study = sqlite3.connect(f'file:{STUDY_DB}?mode=ro', uri=True)
    try:
        rows = pd.read_sql_query('SELECT * FROM metrics', study)
        audits = {(v, m, b, s): json.loads(a) for v, m, b, s, a in
                  study.execute("SELECT variant,mechanism,budget,seed,audit FROM cases WHERE status='complete'")}
        stored = pd.read_sql_query('SELECT mechanism,budget,seed,method,accuracy,macro_f1,balanced_accuracy,log_loss,'
                                   'lambda_value FROM metrics', main)
        for (m, b, s, a) in main.execute("SELECT mechanism,budget,seed,audit FROM cases WHERE status='complete'"):
            audit = json.loads(a)
            for teacher, student, design in MAIN:
                entry = dict(audit, students={student: audit['students'][student]})
                audits[(variant_name(teacher, student, design), m, b, s)] = entry
    finally:
        main.close()
        study.close()
    stored[['student', 'method']] = stored.method.str.split(' / ', expand=True)
    for teacher, student, design in MAIN:
        part = stored.loc[stored.student == student].drop(columns='student').assign(variant=variant_name(teacher, student, design))
        rows = pd.concat([rows, part], ignore_index=True)
    parts = rows.variant.str.extract(r'^(?P<teacher>\w+)→(?P<student>\w+) \[(?P<design>[^\]]+)\]$')
    return pd.concat([rows, parts], axis=1), audits


# Step 4: Pre-Declared Training-CV Choice of Teacher (No Validation Labels Involved)

def selected_cv_score(cv):
    """Mean training-CV Macro F1 of the PPI setting that selection.select_settings would choose."""
    table = pd.DataFrame(cv)
    if table.empty:
        return None
    best = table.loc[table['mode'] == 'supervised', 'macro_f1'].mean()
    folds = int((table['mode'] == 'supervised').sum())
    for value, group in table.loc[table['mode'] == 'ppi'].groupby('value'):
        if len(group) == folds and (group.status == 'ok').all() and group.macro_f1.mean() > best+1e-8:
            best = group.macro_f1.mean()
    return float(best)


def cv_selected_teacher(rows, audits, student, design):
    """Per case, keep the Teacher (CatBoost or same-architecture) with the higher selected training-CV score.
    Ties, including the no-CV small-class fallback, keep CatBoost as declared in advance."""
    choices = []
    for (mechanism, budget, seed), group in rows.loc[(rows.student == student) & (rows.design == design)].groupby(
            ['mechanism', 'budget', 'seed']):
        scores = {}
        for teacher in ['CatBoost', student]:
            audit = audits.get((variant_name(teacher, student, design), mechanism, budget, seed))
            if audit is None:
                break
            scores[teacher] = selected_cv_score(audit['students'][student]['cv'])
        if len(scores) < 2:
            continue
        pick = student if None not in scores.values() and scores[student] > scores['CatBoost']+1e-8 else 'CatBoost'
        chosen = group.loc[group.teacher == pick].set_index('method').macro_f1
        choices.append(dict(mechanism=mechanism, budget=budget, seed=seed, teacher=pick,
                            supervised=chosen['Supervised'], ppi=chosen['PPI selected']))
    return pd.DataFrame(choices)


def main():
    parser = argparse.ArgumentParser(description='Same-architecture Teachers and label-free preprocessing.')
    parser.add_argument('--workers', type=int, default=8)
    run_study(parser.parse_args().workers)


if __name__ == '__main__':
    main()
