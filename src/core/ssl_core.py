# Step 1: Share One Weighted Multiclass LR Objective Across All Students

from dataclasses import replace

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.optimize import minimize
from scipy.sparse import csr_matrix, hstack
from scipy.special import logsumexp, softmax
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss
from sklearn.model_selection import KFold, StratifiedKFold

from config import DEFAULT
from model_utils import CLASSES, CATEGORICAL_FEATURES, create_preprocessor, prepare_catboost_features

K = len(CLASSES)


class NonEstimableCandidate(ValueError):
    pass


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
        from tree_student import fit_tree
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
    if config.teacher_model == 'CatBoostBalanced':
        # Equal total weight per observed class, computed from the visible labels of this fit only.
        counts = np.bincount(y, minlength=K)
        wl = np.asarray(wl, dtype=float)*len(y)/(len(classes)*counts[y])
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


def fold_teacher(X, y, wl, U, seed=42, config=DEFAULT):
    """Fit one Teacher on one fold's visible labels; return aligned four-class probabilities for any table."""
    if config.teacher_model in ('CatBoost', 'CatBoostBalanced'):
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
