# Step 1: Use Three Training Methods Within Each Fixed Student Architecture

from dataclasses import replace
import time
import numpy as np
from threadpoolctl import threadpool_limits

from config import DEFAULT
from masking import generate_mask
from model_utils import full_training_data, load_case
from selection import select_settings
from ssl_core import (K, NonEstimableCandidate, crossfit_teacher, fit_student,
                      prepare_design, probabilities, score)

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
