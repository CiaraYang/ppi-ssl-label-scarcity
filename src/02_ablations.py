# Step 2: Ablations — A1 Teacher Replay / Teacher-Only / Fixed λ=1, A2 XGBoost Rounds, A3 XGBoost Capacity

import sys
from importlib import import_module
from pathlib import Path

import argparse
import hashlib
import json
import sqlite3
import time
from dataclasses import replace
from itertools import product

import numpy as np
from joblib import Parallel, delayed, parallel_config
from threadpoolctl import threadpool_limits

common = import_module('00_common')
DEFAULT, generate_mask, DB_PATH, RESULTS_DIR, load_case, prepare_catboost_features = common.DEFAULT, common.generate_mask, common.DB_PATH, common.RESULTS_DIR, common.load_case, common.prepare_catboost_features
connect, dumps, fingerprint, fingerprint_differences, initialize, read_results = common.connect, common.dumps, common.fingerprint, common.fingerprint_differences, common.initialize, common.read_results
save_case, set_metadata, stored_fingerprint_warning = common.save_case, common.set_metadata, common.stored_fingerprint_warning
methods = import_module('00_methods')
fully_labeled_reference, metric_row, K, NonEstimableCandidate, crossfit_teacher, fit_student = methods.fully_labeled_reference, methods.metric_row, methods.K, methods.NonEstimableCandidate, methods.crossfit_teacher, methods.fit_student
fit_teacher, prepare_design, probabilities, score, teacher_probabilities = methods.fit_teacher, methods.prepare_design, methods.probabilities, methods.score, methods.teacher_probabilities


# ===== From ablations.py — Step 1: Replay the Main Study's Teachers Exactly and Add Pre-Specified Ablations =====

ABLATION_DB = RESULTS_DIR / 'databases' / 'ablations.sqlite'
PROTOCOL = 'ablation_v1'
CHECKED = ['accuracy', 'macro_f1', 'balanced_accuracy', 'log_loss']


class ReplayMismatch(RuntimeError):
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))  # The numbered files import each other by name.


def read_main_study():
    """Read the original results without any possibility of writing to them."""
    db = sqlite3.connect(f'file:{DB_PATH}?mode=ro', uri=True)
    try:
        audits = {tuple(row[:3]): json.loads(row[3]) for row in
                  db.execute("SELECT mechanism,budget,seed,audit FROM cases WHERE status='complete'")}
        metrics = read_results(db)
    finally:
        db.close()
    return audits, metrics


def require_equal(name, new, old):
    differences = {key: abs(new[key]-old[key]) for key in CHECKED if key in old}
    if any(value != 0 for value in differences.values()):
        raise ReplayMismatch(f'{name} does not reproduce the main study: {differences}')


# Step 2: Describe Teacher Probability Quality Without Using Hidden Training Labels

def calibration(probability, y, bins=10):
    confidence, correct = probability.max(axis=1), probability.argmax(axis=1) == y
    edges = np.linspace(0, 1, bins+1)
    which = np.clip(np.digitize(confidence, edges[1:-1]), 0, bins-1)
    ece = sum(abs(correct[which == b].mean()-confidence[which == b].mean())*np.mean(which == b)
              for b in range(bins) if np.any(which == b))
    return {'ece_top_label': float(ece), 'brier': float(np.mean(np.sum((probability-np.eye(K)[y])**2, axis=1))),
            'mean_confidence': float(confidence.mean()), 'top_label_accuracy': float(correct.mean())}


def confidence_profile(probability):
    confidence = probability.max(axis=1)
    return {'quantiles': dict(zip(['10%', '25%', '50%', '75%', '90%'],
                                  np.quantile(confidence, [.1, .25, .5, .75, .9]).tolist())),
            'fraction_at_least_0.8': float(np.mean(confidence >= .8))}


# Step 3: One Case: Verified Teacher Replay, Teacher-Only Refit and Fixed λ=1 Students

def run_ablation(mechanism, budget, seed, stored_audit, stored_metrics, config=DEFAULT):
    started = time.monotonic()
    data = load_case(generate_mask(budget, seed, mechanism, config))
    rows = []
    with threadpool_limits(limits=1):
        # D1: the identical call made by experiment.run_case must give identical Teacher scores.
        qL, qU, qV, policy = crossfit_teacher(data.X, data.y, data.U, data.V,
                                              wl=data.labeled_weights, seed=seed, config=config)
        require_equal('Teacher OOF', score(qL, data.y), stored_audit['teacher_oof'])
        require_equal('Cross-fit Teacher validation', score(qV, data.vy), stored_audit['teacher_validation'])
        # Teacher-only: one CatBoost on all visible labels, same settings and design weights.
        teacher, classes = fit_teacher(prepare_catboost_features(data.X), data.y, data.labeled_weights, seed, config)
        qR = teacher_probabilities(teacher, prepare_catboost_features(data.V), classes)
        rows.append(metric_row('Teacher / Cross-fit ensemble', score(qV, data.vy)))
        rows.append(metric_row('Teacher / Refit on visible labels', score(qR, data.vy, qV)))
        audit = dict(mechanism=mechanism, n_labeled=budget, seed=seed, protocol=PROTOCOL,
                     teacher_fold_policy=policy, replay={'teacher': 'exact'},
                     teacher_calibration={'oof_visible_labels': calibration(qL, data.y),
                                          'cross_fit_validation': calibration(qV, data.vy),
                                          'refit_validation': calibration(qR, data.vy)},
                     teacher_unlabeled_confidence=confidence_profile(qU), students={})

        # A1: fixed λ=1, otherwise exactly the main-study PPI fit (same baseline warm start and weights).
        XL, XU, XV = prepare_design(data.X, data.U, data.V)
        for name in config.student_models:
            settings = replace(config, student_model=name)
            baseline, _ = fit_student(XL, data.y, XU, qL, qU, wl=data.labeled_weights, config=settings)
            bp = probabilities(XV, baseline)
            stored = stored_metrics.set_index('method')
            require_equal(f'{name} Supervised', score(bp, data.vy), stored.loc[f'{name} / Supervised'])
            entry = {'replay_supervised': 'exact'}
            try:
                fitted, diagnostics = fit_student(XL, data.y, XU, qL, qU, 'ppi', 1.,
                                                  wl=data.labeled_weights, wu=data.unlabeled_weights,
                                                  initial=baseline, config=settings)
                values = score(probabilities(XV, fitted), data.vy, bp)
                row = metric_row(f'{name} / PPI fixed λ=1', values, lambda_value=1.)
                row['delta_f1'] = values['macro_f1']-score(bp, data.vy)['macro_f1']
                entry['optimizer'] = diagnostics
                if stored_audit['students'][name]['ppi_selected_lambda'] == 1.:
                    require_equal(f'{name} PPI selected at λ=1', values, stored.loc[f'{name} / PPI selected'])
                    entry['matches_stored_selected_lambda_1'] = True
            except NonEstimableCandidate as error:
                row = metric_row(f'{name} / PPI fixed λ=1', status='non_estimable', lambda_value=1.)
                entry['non_estimable'] = str(error)
            rows.append(row)
            audit['students'][name] = entry
    audit['elapsed_seconds'] = time.monotonic()-started
    return rows, audit


def evaluate(key, stored_audit, stored_metrics):
    try:
        rows, audit = run_ablation(*key, stored_audit, stored_metrics)
        return key, rows, audit, None
    except Exception as error:
        return key, [], {}, f'{type(error).__name__}: {error}'


# Step 4: Run A1 Into a Separate Database; the Main Study Is Only Ever Opened Read-Only

def pending(table, keys):
    """Cases still to run, checked read-only so a finished ablation is never rewritten."""
    if not ABLATION_DB.exists():
        return list(keys)
    db = sqlite3.connect(f'file:{ABLATION_DB}?mode=ro', uri=True)
    try:
        if db.execute('SELECT 1 FROM sqlite_master WHERE name=?', (table,)).fetchone() is None:
            return list(keys)
        done = {tuple(row) for row in db.execute(f"SELECT mechanism,budget,seed FROM {table} WHERE status='complete'")}
    finally:
        db.close()
    return [key for key in keys if key not in done]


def run_ablations(budgets, mechanisms, seeds, workers=2, fresh=False):
    if fresh:
        ABLATION_DB.unlink(missing_ok=True)
    if not pending('cases', product(mechanisms, budgets, seeds)):
        print('A1 (Teacher replay, Teacher-only, fixed λ=1) already complete; nothing to run.'
              +stored_fingerprint_warning(ABLATION_DB), flush=True)
        return
    audits, metrics = read_main_study()
    with connect(ABLATION_DB) as db:
        initialize(db)
        set_metadata(db, 'protocol', PROTOCOL)
        set_metadata(db, 'main_study_sha256', hashlib.sha256(DB_PATH.read_bytes()).hexdigest())
        set_metadata(db, 'grid', {'budgets': list(budgets), 'mechanisms': list(mechanisms), 'seeds': list(seeds)})
        done = {tuple(row) for row in db.execute("SELECT mechanism,budget,seed FROM cases WHERE status='complete'")}
        tasks = [key for key in product(mechanisms, budgets, seeds) if key not in done]
        missing = [key for key in tasks if key not in audits]
        if missing:
            raise ValueError(f'No completed main-study case to replay for {missing}.')
        print(f'{len(tasks)} ablation cases to run | {len(done)} cached | {workers} CPU workers', flush=True)
        def stored(key):
            mechanism, budget, seed = key
            return metrics.loc[(metrics.mechanism == mechanism) & (metrics.budget == budget) & (metrics.seed == seed)]
        with parallel_config(backend='loky', inner_max_num_threads=1):
            results = Parallel(n_jobs=workers, return_as='generator_unordered')(
                delayed(evaluate)(key, audits[key], stored(key)) for key in tasks)
            for count, (key, rows, audit, error) in enumerate(results, 1):
                save_case(db, key, rows, audit, error)
                print(f'[{count}/{len(tasks)}] {key} {"FAILED: "+error if error else "saved"}', flush=True)
        failed = db.execute("SELECT COUNT(*) FROM cases WHERE status!='complete'").fetchone()[0]
        set_metadata(db, 'status', 'complete_with_failures' if failed else 'complete')
    if failed:
        raise RuntimeError(f'{failed} ablation cases failed; see {ABLATION_DB.name}.')


# Step 5: A2 — XGBoost Rounds Sensitivity of PPI λ=1 with the Replayed Teacher

A2_GRID = {'mechanisms': ['MCAR', 'MAR'], 'budgets': [100, 500], 'seeds': list(DEFAULT.seeds)}
A2_ROUNDS = (DEFAULT.tree_iterations, 300)


def run_rounds(mechanism, budget, seed, stored_main, stored_fixed, config=DEFAULT, rounds_grid=A2_ROUNDS):
    started = time.monotonic()
    data = load_case(generate_mask(budget, seed, mechanism, config))
    with threadpool_limits(limits=1):
        # The Teacher is deterministic; the exact-reproduction checks below confirm it is the main-study Teacher.
        qL, qU, _, _ = crossfit_teacher(data.X, data.y, data.U, wl=data.labeled_weights, seed=seed, config=config)
    rows, audit = [], dict(mechanism=mechanism, n_labeled=budget, seed=seed, protocol=PROTOCOL+'/A2', fits={})
    with threadpool_limits(limits=1):
        XL, XU, XV = prepare_design(data.X, data.U, data.V)
        for rounds in rounds_grid:
            settings = replace(config, student_model='XGBoost', tree_iterations=rounds)
            baseline, base_diag = fit_student(XL, data.y, XU, qL, qU, wl=data.labeled_weights, config=settings)
            bp = probabilities(XV, baseline)
            fitted, ppi_diag = fit_student(XL, data.y, XU, qL, qU, 'ppi', 1., wl=data.labeled_weights,
                                           wu=data.unlabeled_weights, initial=baseline, config=settings)
            sup, ppi = score(bp, data.vy), score(probabilities(XV, fitted), data.vy, bp)
            if rounds == config.tree_iterations:
                # Must reproduce both the stored main-study baseline and the stored A1 fit exactly.
                require_equal('XGBoost Supervised (main rounds)', sup, stored_main)
                require_equal('XGBoost PPI fixed λ=1 (main rounds)', ppi, stored_fixed)
            else:
                rows.append(metric_row(f'XGBoost[{rounds} rounds] / Supervised', sup))
                rows.append(metric_row(f'XGBoost[{rounds} rounds] / PPI fixed λ=1', ppi, lambda_value=1.,
                                       delta_f1=ppi['macro_f1']-sup['macro_f1']))
            audit['fits'][rounds] = {'supervised': base_diag, 'ppi_lambda_1': ppi_diag,
                                     'supervised_macro_f1': sup['macro_f1'], 'ppi_macro_f1': ppi['macro_f1']}
    audit['elapsed_seconds'] = time.monotonic()-started
    return rows, audit


def evaluate_rounds(key, stored_main, stored_fixed):
    try:
        rows, audit = run_rounds(*key, stored_main, stored_fixed)
        return key, rows, audit, None
    except Exception as error:
        return key, [], {}, f'{type(error).__name__}: {error}'


def save_extra(db, table, key, rows, audit, error=None):
    """Add rows for a later ablation without deleting the A1 case row (which would cascade to its metrics)."""
    columns = ['method', 'status', 'accuracy', 'macro_f1', 'balanced_accuracy', 'log_loss',
               'disagreement', 'class_f1', 'delta_f1', 'weight', 'effective_weight',
               'threshold', 'accepted', 'used', 'lambda_value']
    with db:
        db.execute(f'INSERT OR REPLACE INTO {table} VALUES(?,?,?,?,?,?)',
                   (*key, 'failed' if error else 'complete', dumps(audit), error))
        db.executemany('INSERT OR REPLACE INTO metrics VALUES('+','.join('?'*18)+')',
                       [(*key, *[dumps(row[c]) if c == 'class_f1' and row[c] is not None else row[c] for c in columns])
                        for row in rows])


def require_same_version(name):
    """A partially finished ablation may only resume under the source/config/data it started with."""
    if not ABLATION_DB.exists():
        return
    db = sqlite3.connect(f'file:{ABLATION_DB}?mode=ro', uri=True)
    try:
        row = db.execute('SELECT value FROM metadata WHERE key=?', (f'{name}_fingerprint',)).fetchone()
    finally:
        db.close()
    if row is not None:
        changed = fingerprint_differences(json.loads(row[0]), json.loads(dumps(fingerprint())))
        if changed:
            raise ValueError(f'{name.upper()} started under different source/data ({changed}); rerun it from scratch.')


def run_extra(name, grid, worker, arguments, workers=2):
    """Run one later ablation into its own case table; A1 rows and the A1 fingerprint stay untouched."""
    table = f'{name}_cases'
    if not pending(table, product(grid['mechanisms'], grid['budgets'], grid['seeds'])):
        print(f'{name.upper()} already complete; nothing to run.'
              +stored_fingerprint_warning(ABLATION_DB, f'{name}_fingerprint'), flush=True)
        return
    require_same_version(name)
    with connect(ABLATION_DB) as db:
        db.execute(f'CREATE TABLE IF NOT EXISTS {table}(mechanism TEXT, budget INTEGER, seed INTEGER, status TEXT, '
                   'audit TEXT, error TEXT, PRIMARY KEY(mechanism,budget,seed))')
        set_metadata(db, f'{name}_fingerprint', fingerprint())
        set_metadata(db, f'{name}_grid', grid)
        done = {tuple(row) for row in db.execute(f"SELECT mechanism,budget,seed FROM {table} WHERE status='complete'")}
        tasks = [key for key in product(grid['mechanisms'], grid['budgets'], grid['seeds']) if key not in done]
        print(f'{len(tasks)} {name.upper()} cases to run | {len(done)} cached | {workers} CPU workers', flush=True)
        with parallel_config(backend='loky', inner_max_num_threads=1):
            results = Parallel(n_jobs=workers, return_as='generator_unordered')(
                delayed(worker)(key, *arguments(db, key)) for key in tasks)
            for count, (key, rows, audit, error) in enumerate(results, 1):
                save_extra(db, table, key, rows, audit, error)
                print(f'[{count}/{len(tasks)}] {key} {"FAILED: "+error if error else "saved"}', flush=True)
        failed = db.execute(f"SELECT COUNT(*) FROM {table} WHERE status!='complete'").fetchone()[0]
        set_metadata(db, f'{name}_status', 'complete_with_failures' if failed else 'complete')
    if failed:
        raise RuntimeError(f'{failed} {name.upper()} cases failed; see {ABLATION_DB.name}.')


def stored_row(table, key, method):
    mechanism, budget, seed = key
    return table.loc[(table.mechanism == mechanism) & (table.budget == budget) & (table.seed == seed)
                     & (table.method == method)].iloc[0]


def run_a2(workers=2):
    _, metrics = read_main_study()
    def arguments(db, key):
        return stored_row(metrics, key, 'XGBoost / Supervised'), stored_row(read_results(db), key, 'XGBoost / PPI fixed λ=1')
    run_extra('a2', {**A2_GRID, 'rounds': list(A2_ROUNDS), 'lambda': 1.0}, evaluate_rounds, arguments, workers)


# Step 6: A3 — Pre-Declared XGBoost Capacity Sensitivity (Supervised and Fully Labeled Only)

A3_SETTINGS = {'tree_iterations': 300, 'tree_depth': 5, 'tree_reg_lambda': 5.0}
A3_LABEL = 'XGBoost[300 rounds, depth 5]'


def xgboost_config(**changes):
    return replace(DEFAULT, student_models=('XGBoost',), student_model='XGBoost', **changes)


def run_capacity(mechanism, budget, seed, stored_main, config=DEFAULT, large=None):
    started = time.monotonic()
    large = xgboost_config(**A3_SETTINGS) if large is None else large
    data = load_case(generate_mask(budget, seed, mechanism, config))
    # Supervised ignores Teacher targets; zero arrays keep the fit_student call identical to the main study.
    qL, qU = np.zeros((len(data.y), K)), np.zeros((len(data.U), K))
    with threadpool_limits(limits=1):
        XL, XU, XV = prepare_design(data.X, data.U, data.V)
        main, main_diag = fit_student(XL, data.y, XU, qL, qU, wl=data.labeled_weights,
                                      config=replace(config, student_model='XGBoost'))
        require_equal('XGBoost Supervised (main configuration)', score(probabilities(XV, main), data.vy), stored_main)
        fitted, diagnostics = fit_student(XL, data.y, XU, qL, qU, wl=data.labeled_weights, config=large)
        values = score(probabilities(XV, fitted), data.vy, probabilities(XV, main))
    row = metric_row(f'{A3_LABEL} / Supervised', values, delta_f1=values['macro_f1']-stored_main['macro_f1'])
    audit = dict(mechanism=mechanism, n_labeled=budget, seed=seed, protocol=PROTOCOL+'/A3', settings=A3_SETTINGS,
                 replay_main='exact', main_optimizer=main_diag, large_optimizer=diagnostics,
                 elapsed_seconds=time.monotonic()-started)
    return [row], audit


def evaluate_capacity(key, stored_main):
    try:
        rows, audit = run_capacity(*key, stored_main)
        return key, rows, audit, None
    except Exception as error:
        return key, [], {}, f'{type(error).__name__}: {error}'


def run_a3(workers=2):
    if not pending('a3_cases', product(A2_GRID['mechanisms'], A2_GRID['budgets'], A2_GRID['seeds'])):
        print('A3 already complete; nothing to run.'+stored_fingerprint_warning(ABLATION_DB, 'a3_fingerprint'), flush=True)
        return
    require_same_version('a3')
    _, metrics = read_main_study()
    db = sqlite3.connect(f'file:{DB_PATH}?mode=ro', uri=True)
    try:
        stored_reference = json.loads(db.execute('SELECT value FROM reference').fetchone()[0])['XGBoost']
    finally:
        db.close()
    # Step 6a: Fully labeled reference first — reproduce the stored main configuration, then fit the larger one.
    with connect(ABLATION_DB) as db:
        reference = json.loads(dumps(fully_labeled_reference(xgboost_config())['XGBoost']))
        require_equal('Fully labeled XGBoost (main configuration)', reference, stored_reference)
        large = json.loads(dumps(fully_labeled_reference(xgboost_config(**A3_SETTINGS))['XGBoost']))
        set_metadata(db, 'a3_reference', {'main': reference, 'large': large, 'settings': A3_SETTINGS, 'replay_main': 'exact'})
    print(f"Fully labeled XGBoost Macro F1: main {reference['macro_f1']:.4f} (exact replay) | large {large['macro_f1']:.4f}", flush=True)
    run_extra('a3', {**A2_GRID, 'settings': A3_SETTINGS}, evaluate_capacity,
              lambda db, key: (stored_row(metrics, key, 'XGBoost / Supervised'),), workers)


def main():
    parser = argparse.ArgumentParser(description='A1 Teacher replay/Teacher-only/fixed λ=1, A2 rounds and A3 capacity ablations.')
    parser.add_argument('--workers', type=int, choices=[1, 2], default=2)
    parser.add_argument('--fresh', action='store_true', help='Delete ablations.sqlite only; the main study is never touched.')
    args = parser.parse_args()
    run_ablations(DEFAULT.budgets, DEFAULT.mechanisms, DEFAULT.seeds, args.workers, args.fresh)
    run_a2(args.workers)
    run_a3(args.workers)


if __name__ == '__main__':
    main()
