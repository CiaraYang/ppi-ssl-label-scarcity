# Step 1: Differentiate the Same Signed Cross-Entropy Risk for a Tree Student

import numpy as np
from scipy.sparse import vstack
from scipy.special import softmax, logsumexp
import xgboost as xgb


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
