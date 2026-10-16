# Step 1: Pre-Declared Improved Pipeline (Protocol improved_v1), Evaluated on Validation Only

import json
import sqlite3
import time
from dataclasses import replace
from itertools import product

import numpy as np
import xgboost as xgb
from joblib import Parallel, delayed, parallel_config
from scipy.optimize import minimize_scalar
from scipy.sparse import csr_matrix, hstack
from scipy.special import logsumexp, softmax
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from threadpoolctl import threadpool_limits

from config import DEFAULT
from experiment import metric_row
from masking import generate_mask
from model_utils import RESULTS_DIR, load_case
from ssl_core import (K, NonEstimableCandidate, crossfit_teacher, fit_student, macro_f1, normalized,
                      prepare_design, probabilities, score, stratified_folds)
from storage import dumps, fingerprint
from tree_student import tree_parameters

IMPROVED_DB = RESULTS_DIR / 'improved.sqlite'
CONFIG = replace(DEFAULT, cv_folds=5, seeds=tuple(range(42, 57)))
TEACHER_CANDIDATES = ('CatBoost', 'CatBoostBalanced', 'LR')
STUDENTS = ('LR', 'XGBoost leaf-refit')
METHODS = ('Supervised', 'SSL soft', 'PPI CV λ', 'PPI power λ')
BIAS_GRID = np.arange(-2, 2.001, .25)
# Leaf-refit settings fixed for computational conditioning before any accuracy was seen (100 rounds, C=1 took
# ~90 s per fit at 1,000 labels; 20 rounds, C=0.1 takes ~4 s). The trees only define the partition.
LEAF_ROUNDS, LEAF_C = 20, 0.1
# Written to the database before any result exists; a resumed run refuses to continue if it changes.
PROTOCOL = {
    'name': 'improved_v1.1', 'declared_on': '2026-10-16',
    'amendment': 'v1.1: power-tuned λ uses cross-fitted (out-of-fold) Supervised residuals. In v1 the in-sample residuals '
                 'made λ* = 0 in all 26 completed cases (estimator defect); v1 was aborted and discarded. Interim v1 '
                 'validation numbers for MCAR 50–100 had been viewed; nothing else changed.', 'evaluation': 'validation only; Test untouched',
    'grid': '3 mechanisms × budgets 50–1,000 × 15 seeds (42–56) = 225 cases',
    'folds': '5-fold Teacher cross-fitting and 5-fold training CV',
    'teacher': 'per case, among CatBoost, class-balanced CatBoost, LR and Ensemble(CatBoost, LR), each temperature-'
               'calibrated on its OOF predictions; pick the lowest weighted OOF log loss (ties keep CatBoost)',
    'methods': 'Supervised; SSL soft (w ∈ {0.1, 0.5} by CV, confidence 0.8); PPI CV λ (λ ∈ {.25,.5,.75,1} by CV); '
               'PPI power λ (PPI++ trace form with cross-fitted Supervised residuals, clipped to [0,1], no λ search)',
    'decision': 'argmax and a per-class logit bias (grid −2..2 step .25, coordinate ascent) fitted on out-of-fold '
                'Student probabilities to maximize weighted Macro F1; applied to every method',
    'students': 'LR on the main features (C=1); XGBoost leaf-refit = supervised XGBoost (20 rounds, otherwise main '
                'settings) learns tree structure on labeled rows, then every method refits leaf weights with the same '
                'convex LR objective (C=0.1; both fixed for optimizer conditioning before any result)',
}


# Step 2: Teacher Candidates, Temperature Calibration and OOF Log-Loss Selection

def weighted_log_loss(q, y, weights):
    return float(-np.average(np.log(np.clip(q[np.arange(len(y)), y], 1e-12, 1)), weights=normalized(weights)))


def fit_temperature(q, y, weights):
    logq, w = np.log(np.clip(q, 1e-12, 1)), normalized(weights)
    def nll(log_t):
        z = logq/np.exp(log_t)
        return -np.mean(w*(z[np.arange(len(y)), y]-logsumexp(z, axis=1)))
    return float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method='bounded').x))


def temper(q, t):
    if q is None:
        return None
    p = softmax(np.log(np.clip(q, 1e-12, 1))/t, axis=1)
    p[q == 0] = 0.  # Classes a fold Teacher never saw stay impossible.
    return p/p.sum(axis=1, keepdims=True)


def build_teacher(X, y, U, E, wl, seed=42, config=CONFIG):
    candidates = {}
    for name in TEACHER_CANDIDATES:
        qL, qU, qE, _ = crossfit_teacher(X, y, U, E, wl=wl, seed=seed, config=replace(config, teacher_model=name))
        t = fit_temperature(qL, y, wl)
        candidates[name] = (temper(qL, t), temper(qU, t), temper(qE, t), t)
    c, lr = candidates['CatBoost'], candidates['LR']
    candidates['Ensemble'] = tuple(None if a is None else (a+b)/2 for a, b in zip(c[:3], lr[:3]))+(None,)
    losses = {name: weighted_log_loss(v[0], y, wl) for name, v in candidates.items()}
    pick = min(losses, key=lambda name: (losses[name], name != 'CatBoost'))
    qL, qU, qE, t = candidates[pick]
    return qL, qU, qE, {'teacher': pick, 'temperature': t, 'oof_log_loss': losses}


# Step 3: Student Designs — Main Features, or Leaf Embedding of a Supervised XGBoost

def student_design(student, X, y, wl, U, E, config=CONFIG):
    XL, XU, XE = prepare_design(X, U, E, config)
    if student == 'LR':
        return XL, XU, XE
    # Tree structure is learned from labeled rows only; leaf one-hot coding uses no labels.
    params = tree_parameters(replace(config, student_models=('XGBoost',), student_model='XGBoost'), K)
    booster = xgb.train(params, xgb.DMatrix(XL, label=y, weight=normalized(wl), nthread=1),
                        num_boost_round=min(config.tree_iterations, LEAF_ROUNDS))
    leaves = [booster.predict(xgb.DMatrix(M, nthread=1), pred_leaf=True).astype(int) if M.shape[0] else None
              for M in (XL, XU, XE)]
    encoder = OneHotEncoder(handle_unknown='ignore').fit(np.vstack([v for v in leaves[:2] if v is not None]))
    def encode(v):
        if v is None:
            return csr_matrix((0, len(encoder.get_feature_names_out())+1))
        return hstack([encoder.transform(v), np.ones((len(v), 1))], format='csr')
    return tuple(encode(v) for v in leaves)


def student_C(student):
    return 1. if student == 'LR' else LEAF_C


# Step 4: PPI++ Power Tuning (Trace Form) at the Supervised Fit

def power_lambda(XL, y, qL, wl, n_unlabeled, settings):
    """λ* = tr Cov(g_Y, g_f) / ((1 + n/N) tr Var(g_f)), g = x ⊗ (p − target).

    p comes from cross-fitted Supervised Students: in-sample p nearly interpolates the few labels, which makes
    the residuals p − y and p − q anti-correlated and drives λ* to 0 (observed in the aborted improved_v1 run).
    """
    n = len(y)
    folds = stratified_folds(y, 0, settings) or list(KFold(min(5, n), shuffle=True, random_state=0).split(np.zeros(n)))
    P, empty, zeros = np.zeros((n, K)), XL[:0], np.zeros((0, K))
    for fit_ids, held_ids in folds:
        weights, _ = fit_student(XL[fit_ids], y[fit_ids], empty, zeros, zeros, wl=wl[fit_ids], config=settings)
        P[held_ids] = probabilities(XL[held_ids], weights)
    RY, RF = P-np.eye(K)[y], P-qL
    a = normalized(wl)
    norms = np.asarray(XL.multiply(XL).sum(axis=1)).ravel()
    GY, GF = np.asarray(XL.T @ (a[:, None]*RY))/n, np.asarray(XL.T @ (a[:, None]*RF))/n
    cov = np.mean(a*norms*np.sum(RY*RF, axis=1))-np.sum(GY*GF)
    var = np.mean(a*norms*np.sum(RF*RF, axis=1))-np.sum(GF*GF)
    return 0. if var <= 0 else float(np.clip(cov/((1+n/n_unlabeled)*var), 0, 1))


def fit_methods(XL, y, XU, qL, qU, wl, wu, config=CONFIG, ssl_weights=(.1, .5), lambdas=(.25, .5, .75, 1.), C=1.):
    """Every candidate fit, all with the same LR objective, penalty and design."""
    settings = replace(config, student_models=('LR',), student_model='LR', student_C=C)
    supervised, _ = fit_student(XL, y, XU, qL, qU, wl=wl, config=settings)
    fits = {('Supervised', 0.): supervised}
    for w in ssl_weights:
        fits[('SSL soft', w)], _ = fit_student(XL, y, XU, qL, qU, 'soft', wl=wl, wu=wu, initial=supervised,
                                               weight=w, threshold=.8, config=settings)
    for lam in lambdas:
        try:
            fits[('PPI CV λ', lam)], _ = fit_student(XL, y, XU, qL, qU, 'ppi', lam, wl=wl, wu=wu,
                                                     initial=supervised, config=settings)
        except NonEstimableCandidate:
            fits[('PPI CV λ', lam)] = None
    lam = power_lambda(XL, y, qL, wl, XU.shape[0], settings)
    while True:  # Back off only if the corrected class mass is not estimable at this λ.
        try:
            fits[('PPI power λ', lam)] = supervised if lam == 0 else fit_student(
                XL, y, XU, qL, qU, 'ppi', lam, wl=wl, wu=wu, initial=supervised, config=settings)[0]
            break
        except NonEstimableCandidate:
            lam = 0. if lam < .01 else lam/2
    return fits


# Step 5: Out-of-Fold Probabilities, CV Selection and the Macro-F1 Decision Rule

def fit_bias(P, y, weights):
    """Per-class logit bias maximizing weighted Macro F1 of out-of-fold predictions (class 0 fixed at 0)."""
    logp, bias = np.log(np.clip(P, 1e-300, 1)), np.zeros(K)
    best = macro_f1(y, logp.argmax(axis=1), weights)
    for _ in range(3):
        for k in range(1, K):
            for value in BIAS_GRID:
                trial = bias.copy()
                trial[k] = value
                f1 = macro_f1(y, (logp+trial).argmax(axis=1), weights)
                if f1 > best+1e-9:
                    best, bias = f1, trial
    return bias


def biased(P, bias):
    return softmax(np.log(np.clip(P, 1e-300, 1))+bias, axis=1)


def select_and_calibrate(data, folds, teachers, student, seed, config=CONFIG):
    n = len(data.y)
    oof, scores = {}, {}
    for f, (fit_ids, held_ids) in enumerate(folds):
        X, y, wl = data.X.iloc[fit_ids], data.y[fit_ids], data.labeled_weights[fit_ids]
        if f not in teachers:
            teachers[f] = build_teacher(X, y, data.U, None, wl, seed, config)
        qL, qU, _, _ = teachers[f]
        XL, XU, XH = student_design(student, X, y, wl, data.U, data.X.iloc[held_ids], config)
        for (method, value), weights in fit_methods(XL, y, XU, qL, qU, wl, data.unlabeled_weights, config,
                                                    C=student_C(student)).items():
            key = (method, value if method != 'PPI power λ' else 'power')
            oof.setdefault(key, np.zeros((n, K)))
            if weights is None:
                scores.setdefault(key, []).append(None)
                continue
            P = probabilities(XH, weights)
            oof[key][held_ids] = P
            scores.setdefault(key, []).append(macro_f1(data.y[held_ids], P.argmax(axis=1), data.labeled_weights[held_ids]))
    mean = {key: None if None in s else float(np.mean(s)) for key, s in scores.items()}
    chosen = {'Supervised': 0., 'PPI power λ': 'power'}
    for method in ['SSL soft', 'PPI CV λ']:
        best, chosen[method] = mean[('Supervised', 0.)], 0.
        for (m, value), s in sorted(mean.items(), key=lambda item: str(item[0][1])):
            if m == method and s is not None and s > best+1e-8:
                best, chosen[method] = s, value
    bias = {}
    for method, value in chosen.items():
        key = ('Supervised', 0.) if value == 0. and method != 'PPI power λ' else (method, value)
        bias[method] = fit_bias(oof[key], data.y, data.labeled_weights)
    return chosen, bias, mean


# Step 6: One Case — Both Students Share the Fold Teachers and the Final Teacher

def run_improved_case(n_labeled, mechanism, seed, config=CONFIG):
    started = time.monotonic()
    data = load_case(generate_mask(n_labeled, seed, mechanism, config))
    rows, audit = [], dict(mechanism=mechanism, n_labeled=n_labeled, seed=seed, protocol=PROTOCOL['name'], students={})
    with threadpool_limits(limits=1):
        folds = stratified_folds(data.y, seed, config)
        teachers = {}
        qL, qU, qV, info = build_teacher(data.X, data.y, data.U, data.V, data.labeled_weights, seed, config)
        audit.update(teacher=info, teacher_validation=score(qV, data.vy), cv_folds=len(folds))
        rows.append(metric_row('Teacher / selected', score(qV, data.vy)))
        for student in STUDENTS:
            if folds:
                chosen, bias, cv = select_and_calibrate(data, folds, teachers, student, seed, config)
            else:  # Predefined small-class fallback: no CV, no bias; power λ needs no CV.
                chosen, bias, cv = {'Supervised': 0., 'SSL soft': 0., 'PPI CV λ': 0., 'PPI power λ': 'power'}, \
                    {m: np.zeros(K) for m in METHODS}, {}
            XL, XU, XV = student_design(student, data.X, data.y, data.labeled_weights, data.U, data.V, config)
            ssl = (chosen['SSL soft'],) if chosen['SSL soft'] else ()
            lam = (chosen['PPI CV λ'],) if chosen['PPI CV λ'] else ()
            fits = fit_methods(XL, data.y, XU, qL, qU, data.labeled_weights, data.unlabeled_weights, config, ssl, lam,
                               C=student_C(student))
            final = {'Supervised': fits[('Supervised', 0.)],
                     'SSL soft': fits[('SSL soft', chosen['SSL soft'])] if ssl else fits[('Supervised', 0.)],
                     'PPI CV λ': fits[('PPI CV λ', chosen['PPI CV λ'])] if lam else fits[('Supervised', 0.)]}
            power = [(value, w) for (m, value), w in fits.items() if m == 'PPI power λ'][0]
            final['PPI power λ'] = power[1]
            base = probabilities(XV, final['Supervised'])
            for method in METHODS:
                P = probabilities(XV, final[method])
                rows.append(metric_row(f'{student} / {method}', score(P, data.vy, base)))
                rows.append(metric_row(f'{student} / {method} + threshold', score(biased(P, bias[method]), data.vy,
                                                                                   biased(base, bias['Supervised']))))
            audit['students'][student] = {'chosen': {m: v for m, v in chosen.items() if m != 'PPI power λ'},
                                          'power_lambda': power[0], 'bias': {m: b.tolist() for m, b in bias.items()},
                                          'cv_mean_macro_f1': {f'{m}|{v}': s for (m, v), s in cv.items()}}
        audit['fold_teachers'] = [t[3]['teacher'] for _, t in sorted(teachers.items())]
    audit['elapsed_seconds'] = time.monotonic()-started
    return rows, audit


# Step 7: Run All 225 Cases Into improved.sqlite (Resumable; Never Rewrites a Finished Study)

def evaluate(key):
    mechanism, budget, seed = key
    try:
        rows, audit = run_improved_case(budget, mechanism, seed)
        return key, rows, audit, None
    except Exception as error:
        return key, [], {}, f'{type(error).__name__}: {error}'


def connect():
    db = sqlite3.connect(IMPROVED_DB)
    db.executescript('''
    CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS cases(mechanism TEXT, budget INTEGER, seed INTEGER, status TEXT, audit TEXT, error TEXT,
        PRIMARY KEY(mechanism,budget,seed));
    CREATE TABLE IF NOT EXISTS metrics(mechanism TEXT, budget INTEGER, seed INTEGER, method TEXT, accuracy REAL,
        macro_f1 REAL, balanced_accuracy REAL, log_loss REAL, class_f1 TEXT, PRIMARY KEY(mechanism,budget,seed,method));
    ''')
    return db


def run(workers=9):
    keys = list(product(CONFIG.mechanisms, CONFIG.budgets, CONFIG.seeds))
    db = connect()
    try:
        found = dict(db.execute('SELECT key, value FROM metadata').fetchall())
        done = {tuple(r) for r in db.execute("SELECT mechanism,budget,seed FROM cases WHERE status='complete'")}
        tasks = [key for key in keys if key not in done]
        if not tasks:
            print('Improved study already complete; nothing to run.', flush=True)
            return
        if 'protocol' in found and json.loads(found['protocol']) != PROTOCOL:
            raise ValueError('The pre-declared protocol changed after the study started.')
        for key, value in [('protocol', PROTOCOL), ('fingerprint', fingerprint(CONFIG))]:
            db.execute('INSERT OR REPLACE INTO metadata VALUES(?,?)', (key, dumps(value)))
        db.commit()
        print(f'{len(tasks)} improved cases to run | {len(done)} done | {workers} CPU workers', flush=True)
        with parallel_config(backend='loky', inner_max_num_threads=1):
            results = Parallel(n_jobs=workers, return_as='generator_unordered')(delayed(evaluate)(key) for key in tasks)
            for count, (key, rows, audit, error) in enumerate(results, 1):
                with db:
                    db.execute('DELETE FROM metrics WHERE mechanism=? AND budget=? AND seed=?', key)
                    db.execute('INSERT OR REPLACE INTO cases VALUES(?,?,?,?,?,?)',
                               (*key, 'failed' if error else 'complete', dumps(audit), error))
                    db.executemany('INSERT INTO metrics VALUES(?,?,?,?,?,?,?,?,?)', [
                        (*key, r['method'], r['accuracy'], r['macro_f1'], r['balanced_accuracy'], r['log_loss'],
                         dumps(r['class_f1'])) for r in rows])
                if error or count % 25 == 0 or count == len(tasks):
                    print(f'[{count}/{len(tasks)}] {key} {"FAILED: "+error if error else "saved"}', flush=True)
        failed = db.execute("SELECT COUNT(*) FROM cases WHERE status!='complete'").fetchone()[0]
    finally:
        db.close()
    if failed:
        raise RuntimeError(f'{failed} improved cases failed; see {IMPROVED_DB.name}.')
