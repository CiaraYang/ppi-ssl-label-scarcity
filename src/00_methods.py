# Shared: Teachers, Students, the Supervised / SSL / PPI Objectives, Training CV and One Case

import sys
from importlib import import_module
from pathlib import Path

from dataclasses import replace
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.optimize import minimize
from scipy.sparse import csr_matrix, hstack
from scipy.special import logsumexp, softmax
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss
from sklearn.model_selection import KFold, StratifiedKFold
from scipy.sparse import vstack
from threadpoolctl import threadpool_limits

common = import_module('00_common')
DEFAULT, CLASSES, CATEGORICAL_FEATURES, create_preprocessor, prepare_catboost_features, generate_mask = common.DEFAULT, common.CLASSES, common.CATEGORICAL_FEATURES, common.create_preprocessor, common.prepare_catboost_features, common.generate_mask
full_training_data, load_case = common.full_training_data, common.load_case



# ===== From ssl_core.py — Step 1: Share One Weighted Multiclass LR Objective Across All Students =====

K = len(CLASSES)


class NonEstimableCandidate(ValueError):
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))  # The numbered files import each other by name.


def normalized(weights):
    weights = np.asarray(weights, dtype=float)
    if not len(weights):
        return weights
    if not np.all(np.isfinite(weights) & (weights > 0)):
        raise ValueError('Risk weights must be finite and positive.')
    return weights / weights.mean()


def stratified_folds(y, seed, config=DEFAULT):
    minimum = int(np.bincount(y, minlength=K).min())
    if minimum < 2:
        return []
    return list(StratifiedKFold(min(config.cv_folds, minimum), shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def objective(parameters, XL, targetL, XU, targetU, penalty):
    weights = parameters.reshape(XL.shape[1], targetL.shape[1])
    loss, gradient = 0., np.zeros_like(weights)
    for X, target in [(XL, targetL), (XU, targetU)]:
        if not np.any(target):
            continue
        logits = X @ weights
        log_p = logits - logsumexp(logits, axis=1, keepdims=True)
        loss -= np.sum(target * log_p) / len(target)
        residual = np.exp(log_p) * target.sum(axis=1, keepdims=True) - target
        gradient += np.asarray(X.T @ residual) / len(target)
    loss += .5 * penalty * np.sum(weights[:-1]**2)
    gradient[:-1] += penalty * weights[:-1]
    return float(loss), gradient.ravel()


# Step 2: Support Confidence Filtering, MAR Weighting and Missing Classes

def fit_student(XL, y, XU, qL, qU, mode='supervised', strength=0., *,
                wl=None, wu=None, initial=None, weight=0., threshold=0., config=DEFAULT):
    wl = normalized(np.ones(len(y)) if wl is None else wl)
    wu = normalized(np.ones(len(qU)) if wu is None else wu)
    Y = np.eye(K)[y]
    targetL, targetU = wl[:, None]*Y, np.zeros_like(qU)
    diagnostics = {'weight': None, 'effective_weight': None, 'threshold': None, 'accepted': None, 'used': None}
    if mode == 'soft':
        if not 0 <= weight <= 1 or not 0 <= threshold <= 1:
            raise ValueError('SSL weight/threshold must be in [0,1].')
        accepted = qU.max(axis=1) >= threshold
        count = int(accepted.sum())
        effective = weight if count else 0.
        XU, qU, wu = XU[accepted], qU[accepted], normalized(wu[accepted])
        targetL, targetU = (1-effective)*wl[:, None]*Y, effective*wu[:, None]*qU
        diagnostics.update(weight=weight, effective_weight=effective, threshold=threshold,
                           accepted=count, used=count if effective else 0)
    elif mode == 'ppi':
        if not 0 <= strength <= 1:
            raise ValueError('PPI lambda must be in [0,1].')
        targetL, targetU = wl[:, None]*(Y-strength*qL), strength*wu[:, None]*qU
    elif mode != 'supervised':
        raise ValueError('Unknown Student risk.')
    active = np.any(targetL != 0, axis=0) | np.any(targetU != 0, axis=0)
    mass = targetL.mean(axis=0) + (targetU.mean(axis=0) if len(targetU) else 0)
    if (mass[active] <= 0).any():
        raise NonEstimableCandidate('Nonpositive corrected class mass.')
    if config.student_model == 'XGBoost':
        model = fit_tree(XL, targetL, XU, targetU, active, config)
        diagnostics.update(iterations=config.tree_iterations, **model.diagnostics)
        return model, diagnostics
    start = np.zeros((XL.shape[1], active.sum())) if initial is None else initial[:, active]
    result = minimize(objective, start.ravel(),
                      args=(XL, targetL[:, active], XU, targetU[:, active], 1/(len(y)*config.student_C)),
                      method='L-BFGS-B', jac=True,
                      options={'maxiter': 5000, 'ftol': 1e-10, 'gtol': 1e-6, 'maxls': 100})
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f'Student optimizer failed: {result.message}')
    weights = np.zeros((XL.shape[1], K))
    weights[-1, ~active] = -1000.  # Absent classes receive zero probability, not invented labels.
    weights[:, active] = result.x.reshape(XL.shape[1], active.sum())
    diagnostics.update(iterations=int(result.nit), gradient=float(abs(result.jac).max()))
    return weights, diagnostics


# Step 3: Fit All Preprocessing on the Current Labeled Fold

def design_transformer(X, U=None, config=DEFAULT):
    # 'labeled+unlabeled' uses unlabeled training features only (never labels); validation/test never enter.
    fit_on = X if config.design_fit == 'labeled' or U is None or U.empty else pd.concat([X, U])
    preprocessor = create_preprocessor().fit(fit_on)
    def transform(table):
        if table.empty:
            return csr_matrix((0, len(preprocessor.get_feature_names_out())+1))
        return hstack([csr_matrix(preprocessor.transform(table)), np.ones((len(table), 1))], format='csr')
    return transform


def prepare_design(X, U, E, config=DEFAULT):
    transform = design_transformer(X, U, config)
    return tuple(transform(table) for table in [X, U, E])


# Step 4: Cross-Fit Teachers with Explicit Four-Class Probability Alignment

def fit_teacher(X, y, wl, seed=42, config=DEFAULT):
    """Fit one CatBoost Teacher on CatBoost-ready features; a single observed class needs no model."""
    classes = np.unique(y)
    if len(classes) == 1:
        return None, classes
    teacher = CatBoostClassifier(iterations=config.teacher_iterations, depth=6, learning_rate=.05,
                                 loss_function='MultiClass', cat_features=tuple(CATEGORICAL_FEATURES),
                                 random_seed=seed, thread_count=1, verbose=False, allow_writing_files=False)
    teacher.fit(X, y, sample_weight=normalized(wl))
    return teacher, teacher.classes_.astype(int)


def teacher_probabilities(model, table, classes):
    probability = np.zeros((len(table), K))
    if table.empty:
        return probability
    if model is None:
        probability[:, classes[0]] = 1.
    else:
        probability[:, classes] = model.predict_proba(table)
    return probability


TABPFN_CHECKPOINT = 'tabpfn-v3.5-20260909.safetensors'  # TabPFN-3.5, pretrained on synthetic tasks only
TABPFN_BATCH = 4000  # rows per predict call (22,500 rows at once exceeded the 7.5 GB Apple-GPU limit with 12 features)


def tabpfn_teacher(X, y, seed=42):
    """Pretrained TabPFN-3.5 (synthetic-only pretraining) with this fold's visible labels as its context.
    Categories are coded from the fold's own rows; TabPFN takes no sample weights, so it is unweighted (MCAR only)."""
    import sys
    if 'xgboost' in sys.modules and 'torch' not in sys.modules:
        # On macOS, XGBoost's OpenMP runtime loaded before PyTorch's makes TabPFN segfault; load torch first.
        raise RuntimeError('Import torch before xgboost in this process to use the TabPFN Teacher.')
    import torch
    from sklearn.preprocessing import OrdinalEncoder
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    from tabpfn.model_loading import prepend_cache_path
    NUMERIC_FEATURES = common.NUMERIC_FEATURES
    coder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan, encoded_missing_value=np.nan)
    coder.fit(X[CATEGORICAL_FEATURES].astype(object).fillna('Unknown'))
    def encode(table):
        return np.hstack([table[NUMERIC_FEATURES].to_numpy(float),
                          coder.transform(table[CATEGORICAL_FEATURES].astype(object).fillna('Unknown'))])
    classes = np.unique(y)
    if len(classes) == 1:
        return lambda table: teacher_probabilities(None, table, classes)
    # Pinned checkpoint, not the package default, so a tabpfn upgrade cannot silently swap the model.
    model = TabPFNClassifier.create_default_for_version(
        ModelVersion.V3_5, model_path=prepend_cache_path(TABPFN_CHECKPOINT), random_state=seed,
        device='mps' if torch.backends.mps.is_available() else 'cpu',
        categorical_features_indices=list(range(len(NUMERIC_FEATURES), X.shape[1])))
    model.fit(encode(X), y)
    def predict(table):
        probability = np.zeros((len(table), K))
        if len(table):
            # Batches keep GPU memory bounded; each test row attends only to the context, so batching changes nothing.
            probability[:, model.classes_.astype(int)] = np.vstack(
                [model.predict_proba(encode(table.iloc[i:i+TABPFN_BATCH])) for i in range(0, len(table), TABPFN_BATCH)])
            # TabPFN returns float32 rows that can sum to 1 + 1e-7; renormalize so every Teacher row is an exact
            # distribution (otherwise PPI at λ=1 gives a tiny negative row mass that the tree Student rejects).
            probability /= probability.sum(axis=1, keepdims=True)
        return probability
    return predict


def fold_teacher(X, y, wl, U, seed=42, config=DEFAULT):
    """Fit one Teacher on one fold's visible labels; return aligned class probabilities for any table."""
    if config.teacher_model == 'Ensemble':
        # Mean of the CatBoost and LR Teachers fitted on the same fold rows (as improved_v2; no label-based weighting).
        members = [fold_teacher(X, y, wl, U, seed, replace(config, teacher_model=name)) for name in ('CatBoost', 'LR')]
        return lambda table: (members[0](table)+members[1](table))/2
    if config.teacher_model == 'TabPFN':
        return tabpfn_teacher(X, y, seed)
    if config.teacher_model == 'CatBoost':
        teacher, classes = fit_teacher(prepare_catboost_features(X), y, wl, seed, config)
        return lambda table: teacher_probabilities(teacher, prepare_catboost_features(table), classes)
    # Same-architecture Teacher: the Student's own supervised fit and settings, preprocessed per fold.
    transform = design_transformer(X, U, config)
    settings = replace(config, student_models=(config.teacher_model,), student_model=config.teacher_model)
    model, _ = fit_student(transform(X), y, transform(U.iloc[:0]), np.zeros((len(y), K)), np.zeros((0, K)),
                           wl=wl, config=settings)
    return lambda table: np.zeros((0, K)) if table.empty else probabilities(transform(table), model)


def crossfit_teacher(X, y, U, E=None, *, wl=None, seed=42, config=DEFAULT):
    wl = normalized(np.ones(len(y)) if wl is None else wl)
    folds = stratified_folds(y, seed, config)
    policy = 'stratified'
    if not folds:
        folds = list(KFold(min(config.cv_folds, len(y)), shuffle=True, random_state=seed).split(X))
        policy = 'kfold_class_alignment'
    qL, qU = np.zeros((len(y), K)), np.zeros((len(U), K))
    qE = None if E is None else np.zeros((len(E), K))
    for fit_ids, held_ids in folds:
        predict = fold_teacher(X.iloc[fit_ids], y[fit_ids], wl[fit_ids], U, seed, config)
        qL[held_ids] = predict(X.iloc[held_ids])
        fraction = len(held_ids)/len(y)
        qU += fraction*predict(U)
        if E is not None:
            qE += fraction*predict(E)
    return qL, qU, qE, policy


# Step 5: Evaluate All Four Classes on the Same Validation Set

def macro_f1(y, prediction, sample_weight=None):
    return f1_score(y, prediction, labels=np.arange(K), average='macro', zero_division=0, sample_weight=sample_weight)


def score(probability, y, baseline=None):
    prediction = probability.argmax(axis=1)
    result = {'accuracy': accuracy_score(y, prediction), 'macro_f1': macro_f1(y, prediction),
              'balanced_accuracy': balanced_accuracy_score(y, prediction),
              'log_loss': log_loss(y, probability, labels=np.arange(K)),
              'class_f1': f1_score(y, prediction, labels=np.arange(K), average=None, zero_division=0).tolist()}
    result['disagreement'] = 0. if baseline is None else float(np.mean(prediction != baseline.argmax(axis=1)))
    return result


def probabilities(X, weights):
    if hasattr(weights, 'predict_proba'):
        return weights.predict_proba(X)
    return softmax(X @ weights, axis=1)


# ===== From tree_student.py — Step 1: Differentiate the Same Signed Cross-Entropy Risk for a Tree Student =====

def signed_derivatives(logits, target):
    probability = softmax(logits, axis=1)
    mass = target.sum(axis=1, keepdims=True)
    # Signed entries are retained. Only the nonnegative row mass enters curvature.
    if (mass < -1e-10).any():
        raise ValueError('Negative row curvature is unsupported.')
    gradient = probability*mass-target
    hessian = np.maximum(2*np.maximum(mass, 0)*probability*(1-probability), 1e-6)
    return gradient.astype(np.float32), hessian.astype(np.float32)


# Step 2: Align Absent Classes Without Adding Fake Labels

class TreeStudent:
    def __init__(self, booster, active):
        self.booster, self.active = booster, active
        self.diagnostics = {}

    def predict_proba(self, X):
        result = np.zeros((X.shape[0], len(self.active)))
        if self.booster is None:
            result[:, self.active] = 1.
        else:
            import xgboost as xgb
            logits = self.booster.predict(xgb.DMatrix(X, nthread=1), output_margin=True)
            result[:, self.active] = softmax(logits.astype(float), axis=1)
        return result


# Step 3: Hold Tree Structure, Regularization and Rounds Fixed Across Methods

def tree_parameters(config, classes):
    return dict(objective='multi:softprob', num_class=classes, base_score=0.,
                tree_method='hist', nthread=1, seed=42, max_depth=config.tree_depth,
                eta=config.tree_learning_rate, reg_lambda=config.tree_reg_lambda,
                min_child_weight=config.tree_min_child_weight, max_delta_step=config.tree_max_delta_step,
                subsample=1., colsample_bytree=1., disable_default_eval_metric=True)


def fit_tree(XL, targetL, XU, targetU, active, config):
    import xgboost as xgb  # Lazy: see the TabPFN note in tabpfn_teacher.
    if active.sum() == 1:
        return TreeStudent(None, active)
    # Scale mean risks by n_L, so supervised has ordinary per-label gradients.
    # The same scale and tree penalty apply to all three training methods.
    tables, targets = [XL], [targetL[:, active]]
    if len(targetU) and np.any(targetU):
        tables.append(XU)
        targets.append(len(targetL)/len(targetU)*targetU[:, active])
    X, target = vstack(tables, format='csr'), np.vstack(targets)
    keep = np.any(target != 0, axis=1)
    X, target = X[keep], target[keep]
    train = xgb.DMatrix(X, label=np.zeros(X.shape[0]), nthread=1)
    def objective(logits, matrix):
        return signed_derivatives(logits, target)
    booster = xgb.train(tree_parameters(config, int(active.sum())), train,
                        num_boost_round=config.tree_iterations, obj=objective, verbose_eval=False)
    model = TreeStudent(booster, active)
    logits = booster.predict(train, output_margin=True).astype(float)
    log_p = logits-logsumexp(logits, axis=1, keepdims=True)
    model.diagnostics = dict(empirical_risk_initial=float(target.sum()*np.log(active.sum())/len(targetL)),
                             empirical_risk_final=float(-np.sum(target*log_p)/len(targetL)),
                             max_absolute_margin=float(np.abs(logits).max()))
    return model


# ===== From selection.py — Step 1: Use Only Visible Training Labels for Model Selection =====

def select_settings(data, seed=42, config=DEFAULT, teacher_cache=None):
    teacher_cache = {} if teacher_cache is None else teacher_cache
    folds = stratified_folds(data.y, seed, config)
    fallback = {'soft': {'weight': 0., 'threshold': 0.}, 'lambda': 0.}
    audit = {'cv_folds': len(folds), 'visible_class_counts': np.bincount(data.y, minlength=K).tolist(), 'cv': []}
    if not folds:
        audit['selection_policy'] = 'supervised_fallback: fewer than two visible labels in a class'
        return fallback, audit
    audit['selection_policy'] = 'training_stratified_cv_macro_f1'
    candidates = [('supervised', 0., 0.)]
    candidates += [(mode, weight, threshold) for mode in ['soft']
                   for threshold in config.ssl_thresholds for weight in config.ssl_weights if weight > 0]
    candidates += [('ppi', value, 0.) for value in config.ppi_lambdas if value > 0]

    # Step 2: Reuse Nested Teacher Predictions Across Every Candidate
    counts = np.bincount(data.y, minlength=K)
    for fold, (fit_ids, held_ids) in enumerate(folds, 1):
        X, y = data.X.iloc[fit_ids], data.y[fit_ids]
        # Correct small class-specific fold fractions as well as MAR sampling.
        fit_fraction = np.bincount(y, minlength=K)/counts
        held_fraction = np.bincount(data.y[held_ids], minlength=K)/counts
        wl = data.labeled_weights[fit_ids]/fit_fraction[y]
        validation_weight = data.labeled_weights[held_ids]/held_fraction[data.y[held_ids]]
        if fold not in teacher_cache:
            teacher_cache[fold] = crossfit_teacher(X, y, data.U, wl=wl, seed=seed, config=config)
        qL, qU, _, teacher_policy = teacher_cache[fold]
        XL, XU, XE = prepare_design(X, data.U, data.X.iloc[held_ids], config)
        supervised, _ = fit_student(XL, y, XU, qL, qU, wl=wl, config=config)
        warm = {}
        for mode, value, threshold in candidates:
            row = {'fold': fold, 'mode': mode, 'value': value, 'threshold': threshold,
                   'status': 'ok', 'teacher_policy': teacher_policy}
            if mode == 'soft' and not np.any(qU.max(axis=1) >= threshold):
                row.update(status='empty_selection', macro_f1=None)
            else:
                try:
                    if mode == 'supervised':
                        weights = supervised
                    else:
                        weights, _ = fit_student(
                            XL, y, XU, qL, qU, mode, value if mode == 'ppi' else 0.,
                            wl=wl, wu=data.unlabeled_weights,
                            initial=warm.get((mode, threshold), supervised), weight=value, threshold=threshold, config=config)
                        warm[mode, threshold] = weights
                    row['macro_f1'] = macro_f1(data.y[held_ids], probabilities(XE, weights).argmax(axis=1), validation_weight)
                except NonEstimableCandidate as error:
                    row.update(status='non_estimable', macro_f1=None, error=str(error))
            audit['cv'].append(row)

    # Step 3: Require Valid Scores in Every Fold; Prefer Zero on Ties
    baseline = [row['macro_f1'] for row in audit['cv'] if row['mode'] == 'supervised']
    selected = fallback.copy()
    for mode in ['soft', 'ppi']:
        best, best_value, best_threshold = float(np.mean(baseline)), 0., 0.
        options = sorted((value, threshold) for name, value, threshold in candidates if name == mode)
        for value, threshold in options:
            rows = [row for row in audit['cv'] if (row['mode'], row['value'], row['threshold']) == (mode, value, threshold)]
            if len(rows) != len(folds) or any(row['status'] != 'ok' for row in rows):
                continue
            mean = float(np.mean([row['macro_f1'] for row in rows]))
            if mean > best+1e-8:
                best, best_value, best_threshold = mean, value, threshold
        if mode == 'ppi':
            selected['lambda'] = best_value
        else:
            selected[mode] = {'weight': best_value, 'threshold': best_threshold}
    return selected, audit


# ===== From experiment.py — Step 1: Use Three Training Methods Within Each Fixed Student Architecture =====

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
