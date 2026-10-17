# Step 1: Use Three Training Methods Within Each Fixed Student Architecture

import argparse
from dataclasses import replace
from itertools import product
import json
import sqlite3
import time
import numpy as np
from joblib import Parallel, delayed, parallel_config
from threadpoolctl import threadpool_limits

from config import DEFAULT
from masking import generate_mask
from model_utils import DB_PATH, RESULTS_DIR, SPLIT_PATH, full_training_data, load_case, training_features
from selection import select_settings
from ssl_core import (K, NonEstimableCandidate, crossfit_teacher, fit_student,
                      prepare_design, probabilities, score)
from storage import connect, dumps, initialize, save_case, set_metadata, stored_fingerprint_warning

METHODS = ['Supervised', 'SSL soft', 'PPI selected']


def metric_row(method, values=None, **details):
    row = dict(method=method, status='ok', accuracy=None, macro_f1=None, balanced_accuracy=None,
               log_loss=None, disagreement=None, class_f1=None, delta_f1=None, weight=None,
               effective_weight=None, threshold=None, accepted=None, used=None, lambda_value=None)
    row.update(values or {})
    row.update(details)
    return row


# Step 2: Isolate a Fully Labeled Reference for Each Architecture

def fully_labeled_reference(config=DEFAULT, evaluation='validation'):
    X, y, V, vy = full_training_data(evaluation)
    with threadpool_limits(limits=1):
        XL, XU, XV = prepare_design(X, X.iloc[:0], V, config)
        result = {}
        for name in config.student_models:
            settings = replace(config, student_model=name)
            fitted, _ = fit_student(XL, y, XU, np.eye(K)[y], np.zeros((0, K)), config=settings)
            result[name] = {'method': 'Fully labeled '+name, 'labeled_count': len(y),
                            **score(probabilities(XV, fitted), vy)}
    return result


# Step 3: Reuse Identical Masks, Teacher Predictions and CV Folds Across Students

def run_case(n_labeled, mechanism, seed=42, config=DEFAULT, evaluation='validation'):
    started = time.monotonic()
    data = load_case(generate_mask(n_labeled, seed, mechanism, config), evaluation)
    rows, students, teacher_cache = [], {}, {}
    with threadpool_limits(limits=1):
        qL, qU, qV, teacher_policy = crossfit_teacher(
            data.X, data.y, data.U, data.V, wl=data.labeled_weights, seed=seed, config=config)
        XL, XU, XV = prepare_design(data.X, data.U, data.V, config)
        for name in config.student_models:
            settings = replace(config, student_model=name)
            selected, audit = select_settings(data, seed, settings, teacher_cache)
            baseline, baseline_diagnostics = fit_student(
                XL, data.y, XU, qL, qU, wl=data.labeled_weights, config=settings)
            bp = probabilities(XV, baseline)
            baseline_score = score(bp, data.vy)['macro_f1']
            model_rows = [metric_row(name+' / Supervised', score(bp, data.vy, bp))]
            ssl = selected['soft']
            accepted = int((qU.max(axis=1) >= ssl['threshold']).sum())
            if not ssl['weight'] or not accepted:
                fitted, diag = baseline, dict(**ssl, effective_weight=0., accepted=accepted, used=0)
            else:
                fitted, diag = fit_student(XL, data.y, XU, qL, qU, 'soft',
                    wl=data.labeled_weights, wu=data.unlabeled_weights, initial=baseline, config=settings, **ssl)
            details = {key: diag[key] for key in ['weight', 'effective_weight', 'threshold', 'accepted', 'used']}
            model_rows.append(metric_row(name+' / SSL soft', score(probabilities(XV, fitted), data.vy, bp), **details))
            requested = value = selected['lambda']
            while True:
                try:
                    fitted, diagnostics = (baseline, baseline_diagnostics) if value == 0 else fit_student(
                        XL, data.y, XU, qL, qU, 'ppi', value,
                        wl=data.labeled_weights, wu=data.unlabeled_weights, initial=baseline, config=settings)
                    pp = probabilities(XV, fitted)
                    model_rows.append(metric_row(name+' / PPI selected', score(pp, data.vy, bp), lambda_value=value))
                    audit['ppi_optimizer'] = diagnostics
                    break
                except NonEstimableCandidate as error:
                    audit.setdefault('ppi_feasibility', []).append({'lambda': value, 'error': str(error)})
                    value = max(candidate for candidate in config.ppi_lambdas if candidate < value)
            if requested != value:
                audit['ppi_final_feasibility_backoff'] = {'requested': requested, 'effective': value}
            # Record continuous prediction changes, even when argmax classes stay the same.
            audit['ppi_probability_mean_absolute_change'] = float(np.abs(pp-bp).mean())
            audit['ppi_selected_lambda'] = value
            audit['baseline_optimizer'] = baseline_diagnostics
            for row in model_rows:
                row['delta_f1'] = row['macro_f1']-baseline_score
            rows.extend(model_rows)
            audit['selected'] = selected
            students[name] = audit
        audit = dict(mechanism=mechanism, n_labeled=n_labeled, seed=seed, students=students,
                     unlabeled_count=len(data.U), teacher_fold_policy=teacher_policy,
                     teacher_model=config.teacher_model, design_fit=config.design_fit,
                     teacher_oof=score(qL, data.y), **{f'teacher_{evaluation}': score(qV, data.vy)},
                     reference_for_delta='same-architecture Supervised (IPW under MAR)',
                     weighting='self_normalized_known_design_IPW' if mechanism == 'MAR' else 'none',
                     scope='predictive correction; MNAR is an unadjusted stress test',
                     elapsed_seconds=time.monotonic()-started)
    return rows, audit


# Step 4: Run or Resume the Main Study (75 Cases × Two Students); Build the Fixed Split First If Missing

def prepare_split():
    if not SPLIT_PATH.exists():
        from prepare_data import main as build_split
        build_split()


def evaluate(key):
    mechanism, budget, seed = key
    try:
        rows, audit = run_case(budget, mechanism, seed)
        return key, rows, audit, None
    except Exception as error:
        return key, [], {}, f'{type(error).__name__}: {error}'


# Step 5: Resume Matching Cases and Commit Only in the Parent Process

def all_complete(tasks):
    """Check completion read-only, so a finished study is never rewritten."""
    if not DB_PATH.exists():
        return False
    db = sqlite3.connect(f'file:{DB_PATH}?mode=ro', uri=True)
    try:
        done = {tuple(row) for row in db.execute("SELECT mechanism,budget,seed FROM cases WHERE status='complete'")}
    finally:
        db.close()
    return set(tasks) <= done


def run_study(budgets, mechanisms, seeds, workers=2, fresh=False, retry_failed=False):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    prepare_split()
    if fresh:
        DB_PATH.unlink(missing_ok=True)
    elif all_complete(list(product(mechanisms, budgets, seeds))):
        print(f'Main study already complete in {DB_PATH.name}; nothing to run.'+stored_fingerprint_warning(DB_PATH), flush=True)
        return
    with connect() as db:
        initialize(db)
        if db.execute('SELECT COUNT(*) FROM reference').fetchone()[0] == 0:
            reference = fully_labeled_reference()
            db.execute('INSERT INTO reference VALUES(1,?)', (dumps(reference),))
            db.commit()
        reference = json.loads(db.execute('SELECT value FROM reference').fetchone()[0])
        for name, result in reference.items():
            print(f"Fully labeled {name}: Accuracy={result['accuracy']:.4f}, Macro F1={result['macro_f1']:.4f}", flush=True)
        grid = {'budgets': list(budgets), 'mechanisms': list(mechanisms), 'seeds': list(seeds)}
        set_metadata(db, 'grid', grid)
        set_metadata(db, 'status', 'running')
        tasks = list(product(mechanisms, budgets, seeds))
        existing = {tuple(row[:3]): row[3] for row in db.execute('SELECT mechanism,budget,seed,status FROM cases')}
        pending = [key for key in tasks if key not in existing or (retry_failed and existing[key] == 'failed')]
        completed = len(tasks)-len(pending)
        print(f'{len(tasks)} cases | {completed} cached | {workers} CPU workers', flush=True)
        try:
            with parallel_config(backend='loky', inner_max_num_threads=1):
                results = Parallel(n_jobs=workers, return_as='generator_unordered')(
                    delayed(evaluate)(key) for key in pending)
                for key, rows, audit, error in results:
                    save_case(db, key, rows, audit, error)
                    completed += 1
                    if error:
                        print(f'FAILED {key}: {error}', flush=True)
                    elif completed % 10 == 0 or completed == len(tasks):
                        print(f'[{completed}/{len(tasks)}] saved', flush=True)
        except BaseException:
            set_metadata(db, 'status', 'interrupted')
            raise
        failed = [key for key in tasks if db.execute('SELECT status FROM cases WHERE mechanism=? AND budget=? AND seed=?', key).fetchone()[0] != 'complete']
        set_metadata(db, 'failed_cases', failed)
        set_metadata(db, 'status', 'complete_with_failures' if failed else 'complete')
    if failed:
        raise RuntimeError(f'{len(failed)} cases failed; see the database and use --retry-failed after diagnosis.')


# Step 6: Command-Line Options

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--budgets', type=int, nargs='+', default=list(DEFAULT.budgets))
    parser.add_argument('--mechanisms', choices=DEFAULT.mechanisms, nargs='+', default=list(DEFAULT.mechanisms))
    parser.add_argument('--seeds', type=int, nargs='+', default=list(DEFAULT.seeds))
    parser.add_argument('--workers', type=int, choices=[1, 2], default=2)
    parser.add_argument('--fresh', action='store_true')
    parser.add_argument('--retry-failed', action='store_true')
    args = parser.parse_args()
    for values in [args.budgets, args.mechanisms, args.seeds]:
        if len(set(values)) != len(values):
            parser.error('Duplicate budgets, mechanisms or seeds are not allowed.')
    prepare_split()
    maximum = len(training_features())-2
    if any(budget < 2 or budget > maximum for budget in args.budgets):
        parser.error(f'Budgets must be between 2 and n_train-2 ({maximum}).')
    run_study(args.budgets, args.mechanisms, args.seeds, args.workers, args.fresh, args.retry_failed)
