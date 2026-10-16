# Step 1: Simulate Exact Label Counts with Reproducible Nested Budgets

import numpy as np
import pandas as pd

from config import DEFAULT
from model_utils import TARGET, read_data, training_features


def make_mask(X, n_labeled, seed=42, mechanism='MCAR', *, outcomes=None, config=DEFAULT):
    n = len(X)
    if not isinstance(n_labeled, (int, np.integer)) or not 2 <= n_labeled <= n-2:
        raise ValueError('Visible labels must be an integer between 2 and n_train-2.')
    if mechanism not in config.mechanisms:
        raise ValueError('Unknown mechanism.')
    rng = np.random.default_rng(seed)
    visible = np.zeros(n, dtype=bool)
    propensity = np.full(n, np.nan)
    if mechanism == 'MCAR':
        visible[rng.permutation(n)[:n_labeled]] = True
        propensity[:] = n_labeled / n
    else:
        # MAR depends on observed year; MNAR depends on outcome only in simulation.
        if mechanism == 'MAR':
            low_observation = X.Incident_Year.fillna(-1).to_numpy() >= config.recent_year
        else:
            if outcomes is None or len(outcomes) != n or outcomes.isna().any():
                raise ValueError('MNAR simulation requires aligned complete targets.')
            if not outcomes.index.equals(X.index):
                raise ValueError('MNAR targets must align with the training IDs.')
            low_observation = outcomes.to_numpy() == 'destroyed'
        groups = [np.flatnonzero(~low_observation), np.flatnonzero(low_observation)]
        sizes = np.array([len(ids) for ids in groups])
        if (sizes < 2).any():
            raise ValueError('Both sampling strata must have at least two rows.')
        fraction = sizes[1] * config.selection_ratio / (sizes[0] + sizes[1] * config.selection_ratio)
        low_count = int(np.clip(round(n_labeled*fraction), max(1, n_labeled-sizes[0]+1), min(sizes[1]-1, n_labeled-1)))
        for ids, count in zip(groups, [n_labeled-low_count, low_count]):
            visible[rng.permutation(ids)[:count]] = True
            if mechanism == 'MAR':
                # Fixed-quota SRS within each observed stratum: exact inclusion probability.
                propensity[ids] = count / len(ids)
    mask = pd.DataFrame({'Incident_ID': X.index, 'is_labeled': visible, 'propensity': propensity})
    mask.attrs.update(mechanism=mechanism, n_labeled=n_labeled, seed=seed)
    return mask


# Step 2: Keep Outcome-Dependent Mask Generation Separate from Model Inputs

def generate_mask(n_labeled, seed=42, mechanism='MCAR', config=DEFAULT):
    X = training_features()
    outcomes = None
    if mechanism == 'MNAR':
        outcomes = read_data().set_index('Incident_ID').loc[X.index, TARGET]
    return make_mask(X, n_labeled, seed, mechanism, outcomes=outcomes, config=config)
