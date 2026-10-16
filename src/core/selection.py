# Step 1: Use Only Visible Training Labels for Model Selection

import numpy as np

from config import DEFAULT
from ssl_core import (K, NonEstimableCandidate, crossfit_teacher, fit_student,
                      macro_f1, prepare_design, probabilities, stratified_folds)


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
