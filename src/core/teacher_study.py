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
from storage import dumps, fingerprint, fingerprint_differences, stored_fingerprint_warning

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
        print('Teacher study already complete; nothing to run.'+stored_fingerprint_warning(STUDY_DB), flush=True)
        return
    db = connect()
    try:
        previous = db.execute("SELECT value FROM metadata WHERE key='fingerprint'").fetchone()
        current = json.loads(dumps(fingerprint()))
        if previous is not None and json.loads(previous[0]) != current:
            raise ValueError(f'Source/data changed since this study started ({fingerprint_differences(json.loads(previous[0]), current)}); '
                             'move teacher_study.sqlite aside to rerun rather than mixing versions.')
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


# Step 5: Exploratory Pretrained-Teacher Pilot — TabPFN-3.5 → LR (MCAR, 100 and 500 Labels, Validation Only)

TABPFN_DB = RESULTS_DIR / 'tabpfn_teacher.sqlite'
TABPFN_VARIANT = ('TabPFN', 'LR', 'labeled')
TABPFN_GRID = {'mechanisms': ['MCAR'], 'budgets': [100, 500], 'seeds': list(DEFAULT.seeds)}
# Stored before the first result. Added after the main validation results were known, so it is exploratory.
TABPFN_DECLARATION = {
    'name': 'tabpfn_teacher_pilot_v1', 'declared_on': '2026-10-17', 'role': 'exploratory; main study unchanged',
    'question': 'Is the small PPI gain because a Teacher learned from the same few labels carries no extra information?',
    'teacher': 'TabPFN-3.5 (tabpfn-v3.5-20260909.safetensors; pretrained purely on synthetic tabular tasks; research '
               'licence) with the fold\'s visible labels as context; cross-fitted and nested in training CV exactly like '
               'the CatBoost Teacher; unweighted (TabPFN takes no sample weights), hence MCAR only',
    'everything else': 'identical to the main study: same masks, LR Student, preprocessing, SSL/PPI grids and CV rule',
    'expectations': ['TabPFN Teacher Macro F1 > cross-fitted CatBoost Teacher on the same masks',
                     'TabPFN Teacher Macro F1 > Supervised LR Student at 100 labels',
                     'PPI − Supervised larger with TabPFN than with CatBoost (descriptive: 5 masks per budget)',
                     'SSL soft gains more than PPI with this stronger Teacher (as in the synthetic positive control)',
                     'The LR Student stays below TabPFN alone if TabPFN is strong (LR is capped near 0.637 fully labeled)'],
}


def tabpfn_fingerprint():
    import importlib.metadata
    return {**json.loads(dumps(fingerprint(variant_config(*TABPFN_VARIANT)))), 'declaration': TABPFN_DECLARATION,
            'tabpfn_version': importlib.metadata.version('tabpfn'), 'torch_version': importlib.metadata.version('torch')}


def run_tabpfn_pilot():
    keys = list(product(TABPFN_GRID['mechanisms'], TABPFN_GRID['budgets'], TABPFN_GRID['seeds']))
    name = variant_name(*TABPFN_VARIANT)
    db = connect(TABPFN_DB)
    try:
        done = {tuple(r) for r in db.execute("SELECT mechanism,budget,seed FROM cases WHERE status='complete'")}
        if set(keys) <= done:
            print('TabPFN Teacher pilot already complete; nothing to run.', flush=True)
            return
        current = tabpfn_fingerprint()
        previous = db.execute("SELECT value FROM metadata WHERE key='fingerprint'").fetchone()
        if previous is not None and json.loads(previous[0]) != current:
            raise ValueError('Source, data or TabPFN version changed since the pilot started; move tabpfn_teacher.sqlite aside.')
        for key, value in [('fingerprint', current), ('declaration', TABPFN_DECLARATION)]:
            db.execute('INSERT OR REPLACE INTO metadata VALUES(?,?)', (key, dumps(value)))
        db.commit()
        main = read_only_main_supervised()
        tasks = [key for key in keys if key not in done]
        print(f'{len(tasks)} TabPFN Teacher cases to run sequentially (one Apple-GPU/CPU process)', flush=True)
        for count, key in enumerate(tasks, 1):
            _, _, rows, audit, error = evaluate(TABPFN_VARIANT, key)
            if not error:  # TabPFN alone: all visible labels as context (the deployable Teacher-only predictor).
                audit['tabpfn_refit_validation'] = tabpfn_alone(*key)
            if not error:  # The Supervised Student never sees the Teacher: it must equal the main study exactly.
                supervised = next(r for r in rows if r['method'].endswith('/ Supervised'))['macro_f1']
                if supervised != main[key]:
                    error = f'ReplayMismatch: Supervised {supervised} differs from the main study {main[key]}'
            with db:
                db.execute('DELETE FROM metrics WHERE variant=? AND mechanism=? AND budget=? AND seed=?', (name, *key))
                db.execute('INSERT OR REPLACE INTO cases VALUES(?,?,?,?,?,?,?)',
                           (name, *key, 'failed' if error else 'complete', dumps(audit), error))
                if not error:
                    db.executemany('INSERT INTO metrics VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)', [
                        (name, *key, row['method'].split(' / ')[1], row['accuracy'], row['macro_f1'],
                         row['balanced_accuracy'], row['log_loss'], dumps(row['class_f1']),
                         row['delta_f1'], row['lambda_value'], row['weight']) for row in rows])
            print(f'[{count}/{len(tasks)}] {key} {"FAILED: "+error if error else "saved"} '
                  f'({audit.get("elapsed_seconds", 0):.0f} s)', flush=True)
        failed = db.execute("SELECT COUNT(*) FROM cases WHERE status!='complete'").fetchone()[0]
    finally:
        db.close()
    if failed:
        raise RuntimeError(f'{failed} TabPFN Teacher cases failed; see {TABPFN_DB.name}.')


def tabpfn_alone(mechanism, budget, seed):
    from masking import generate_mask
    from model_utils import load_case
    from ssl_core import score, tabpfn_teacher
    data = load_case(generate_mask(budget, seed, mechanism))
    return score(tabpfn_teacher(data.X, data.y, seed)(data.V), data.vy)


def read_only_main_supervised():
    db = sqlite3.connect(f'file:{DB_PATH}?mode=ro', uri=True)
    try:
        return {(m, b, s): f for m, b, s, f in
                db.execute("SELECT mechanism,budget,seed,macro_f1 FROM metrics WHERE method='LR / Supervised'")}
    finally:
        db.close()


def load_tabpfn():
    """TabPFN pilot rows and audits (read-only), or empty results if the pilot has not been run."""
    if not TABPFN_DB.exists():
        return pd.DataFrame(), {}
    db = sqlite3.connect(f'file:{TABPFN_DB}?mode=ro', uri=True)
    try:
        rows = pd.read_sql_query('SELECT * FROM metrics', db)
        audits = {(m, b, s): json.loads(a) for m, b, s, a in
                  db.execute("SELECT mechanism,budget,seed,audit FROM cases WHERE status='complete'")}
    finally:
        db.close()
    return rows, audits


def main():
    parser = argparse.ArgumentParser(description='Same-architecture Teachers, label-free preprocessing and the TabPFN pilot.')
    parser.add_argument('--workers', type=int, default=8)
    run_study(parser.parse_args().workers)
    run_tabpfn_pilot()


if __name__ == '__main__':
    main()
