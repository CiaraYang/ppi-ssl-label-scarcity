# Step 1: Main Study — 3 Mechanisms × 5 Budgets × 5 Seeds, LR and XGBoost Students (Builds the Split If Missing)

import sys
from importlib import import_module
from pathlib import Path

import argparse
from itertools import product
import json
import sqlite3

from joblib import Parallel, delayed, parallel_config

sys.path.insert(0, str(Path(__file__).resolve().parent))  # The numbered files import each other by name.
common = import_module('00_common')
DEFAULT, DB_PATH, RESULTS_DIR, SPLIT_PATH = common.DEFAULT, common.DB_PATH, common.RESULTS_DIR, common.SPLIT_PATH
training_features, connect, dumps, initialize, save_case = common.training_features, common.connect, common.dumps, common.initialize, common.save_case
set_metadata, stored_fingerprint_warning = common.set_metadata, common.stored_fingerprint_warning
methods = import_module('00_methods')
fully_labeled_reference, run_case = methods.fully_labeled_reference, methods.run_case


# Run or Resume the Main Study (75 Cases × Two Students); Build the Fixed Split First If Missing

def prepare_split():
    if not SPLIT_PATH.exists():
        common.build_split()


def evaluate(key):
    mechanism, budget, seed = key
    try:
        rows, audit = run_case(budget, mechanism, seed)
        return key, rows, audit, None
    except Exception as error:
        return key, [], {}, f'{type(error).__name__}: {error}'


# Resume Matching Cases and Commit Only in the Parent Process

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
        if common.VERSION == 'v2':  # Pre-declaration stored before the first v2 result.
            set_metadata(db, 'declaration', common.V2_DECLARATION)
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


# Command-Line Options

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


if __name__ == '__main__':
    main()
