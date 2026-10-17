# Step 1: Anchor to PP-SSL (Shoham et al., NeurIPS 2025): Reproduce the Two-Group Synthetic Phenomenon

import hashlib
import importlib.metadata
import json
import sqlite3
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from model_utils import RESULTS_DIR
from storage import dumps

PPSSL_DB = RESULTS_DIR / 'ppssl_replication.sqlite'
# Paper setup (arXiv 2510.22586v1, Sec. 4.1 and App. D): m = 10 inputs (9 standard normal + a group indicator),
# y = wᵀx + b with b ~ N(0, 1); group A = the 80% of samples with the smallest clean label wᵀx, group B adds ε ~ N(μ, 1);
# the Teacher is wᵀx with the true weights (oracle on A, biased by −μ on B); n = 20, N = 1,000, n_test = 1,000;
# μ ∈ {0.1, 1, 3, 5, 7}; 100 runs. (App. D states the threshold as the 20th percentile; Sec. 4.1 says A is the
# smallest 80%; we follow Sec. 4.1, so B is the top 20%.)
SETUP = {'features': 9, 'share_b': .2, 'n': 20, 'N': 1000, 'n_test': 1000, 'mus': [.1, 1., 3., 5., 7.],
         'seeds': list(range(100)), 'steps': 12000, 'adam_lr': .001, 'batch_unlabeled': 256, 'lambda_init': 1.,
         'fixed_lambdas': [round(x, 1) for x in np.linspace(0, 1, 11)], 'cv_lambdas': [0., .25, .5, .75, 1.], 'cv_folds': 4}
# Pre-declared reproduction targets (the paper's claims, checked on group-wise test MSE; paired over seeds).
TARGETS = [
    dict(id='R1', claim='In group B, Classic SSL is worse than Only Labeled once the Teacher is badly biased',
         left='Classic SSL', right='Only Labeled', metric='mse_b', mus=[3., 5., 7.], rule='left_worse'),
    dict(id='R2', claim='In group B, PP-SSL is better than Classic SSL under strong bias',
         left='PP-SSL (online λ)', right='Classic SSL', metric='mse_b', mus=[3., 5., 7.], rule='left_better'),
    dict(id='R3', claim='Overall, PP-SSL is no worse than Only Labeled at any bias',
         left='PP-SSL (online λ)', right='Only Labeled', metric='mse', mus=[.1, 1., 3., 5., 7.], rule='left_not_worse'),  # Mean MSE at most 2% above.
    dict(id='R4', claim='The online λ decreases as the Teacher bias grows (λ at μ=7 below λ at μ=0.1)',
         left='PP-SSL (online λ)', right=None, metric='lambda', mus=[.1, 7.], rule='decreases'),
    dict(id='R5', claim='PP-SSL is close to the best fixed λ (total MSE within 5%)',
         left='PP-SSL (online λ)', right='best fixed λ', metric='mse', mus=[.1, 1., 3., 5., 7.], rule='within_5pct'),
    dict(id='R6', claim='PP-SSL beats PPI++ with offline-tuned λ (total MSE, strong bias)',
         left='PP-SSL (online λ)', right='PPI++ (offline λ)', metric='mse', mus=[3., 5., 7.], rule='left_better'),
]
BOUNDARY = {
    'replicated': 'the unbiased prediction-powered gradient g_n + λ(g̃_N,f − g_n,f); Algorithm 1 λ update (1-D AdaGrad '
                  'on h_t(λ) = ‖g_t + λ d_t‖², clipped to [0, 1], λ₀ = 1); the two-group data, Teacher, n, N, μ grid',
    'not replicated': 'early stopping on 200 labeled validation rows (200 extra labels would be 10× the labeled budget, '
                      'so none is used); PyTorch (we use numpy Adam, lr 1e-3 as in Table S1, 3,000 epochs read as 3,000 '
                      'passes of 4 unlabeled batches of 256 = 12,000 steps; the full labeled set is the labeled batch); '
                      'the AdaGrad-norm step for w used in the theory; real-data experiments. One smoke run (μ=5, seed 0) '
                      'was viewed under lr 1e-2 / 3,000 steps before these values were set to the paper\'s',
    'adaptation in this project': 'classification with a cross-entropy risk, offline λ by stratified training CV over '
                                  '{0,.25,.5,.75,1} chosen for Macro F1 (not online, not variance-minimizing), IPW under MAR, '
                                  'a cross-fitted learned Teacher (not a fixed oracle), and selection-biased labels (MNAR)',
}


# Step 2: Data, Teacher and the Prediction-Powered Gradient (Squared Loss, Linear Model With Intercept)

def make_data(mu, seed, setup=SETUP):
    rng = np.random.default_rng([seed, 2510])
    w = np.append(rng.normal(size=setup['features']), 0.)  # The group-indicator weight is zero.
    total = setup['n']+setup['N']+setup['n_test']
    Z = rng.normal(size=(total, setup['features']))
    clean = Z @ w[:-1]
    group_b = clean > np.quantile(clean, 1-setup['share_b'])
    X = np.hstack([Z, group_b[:, None], np.ones((total, 1))]).astype(float)
    y = clean+rng.normal(size=total)+np.where(group_b, rng.normal(mu, 1., total), 0.)
    f = clean  # Teacher: the true weights, so oracle on A and biased by −μ on B.
    ids = rng.permutation(total)
    parts = np.split(ids, [setup['n'], setup['n']+setup['N']])
    return [(X[p], y[p], f[p], group_b[p]) for p in parts], rng


def gradient(X, target, w):
    return X.T @ (X @ w-target)/len(X)


def train(labeled, unlabeled, mode, rng, lam=0., setup=SETUP):
    """mode: 'labeled' | 'ssl' | 'fixed' (PPI gradient with fixed λ) | 'online' (Algorithm 1 λ update)."""
    (XL, yL, fL), (XU, fU) = labeled, unlabeled
    w, m, v = np.zeros(XL.shape[1]), np.zeros(XL.shape[1]), np.zeros(XL.shape[1])
    lam, grad_sq = (setup['lambda_init'] if mode == 'online' else lam), 0.
    n, N = len(XL), len(XU)
    for t in range(1, setup['steps']+1):
        batch = rng.integers(0, N, setup['batch_unlabeled'])
        g = gradient(XL, yL, w)
        if mode == 'ssl':  # Classic SSL: mean loss over the n labeled and N pseudo-labeled rows.
            g = (n*g+N*gradient(XU[batch], fU[batch], w))/(n+N)
        elif mode in ('fixed', 'online'):
            d = gradient(XU[batch], fU[batch], w)-gradient(XL, fL, w)
            if mode == 'online':  # λ_{t+1} = clip(λ_t − γ_t ∇h_t(λ_t)), γ_t = (2 Σ_s ‖∇h_s‖²)^(−1/2).
                h_grad = 2*float(d @ (g+lam*d))
                grad_sq += h_grad**2
                step_lambda = lam
                if grad_sq > 0:
                    lam = float(np.clip(lam-h_grad/np.sqrt(2*grad_sq), 0, 1))
                g = g+step_lambda*d
            else:
                g = g+lam*d
        m = .9*m+.1*g
        v = .999*v+.001*g**2
        w -= setup['adam_lr']*(m/(1-.9**t))/(np.sqrt(v/(1-.999**t))+1e-8)
    return w, lam


# Step 3: Offline λ Choices — PPI++ Power Tuning and This Project's Training-CV Rule

def exact_fit(XL, yL, fL, XU, fU, lam):
    """Exact minimizer of mean_L(y−Xw)² + λ[mean_U(f−Xw)² − mean_L(f−Xw)²] (a small ridge keeps it defined)."""
    H = (1-lam)*XL.T @ XL/len(XL)+lam*XU.T @ XU/len(XU)+1e-6*np.eye(XL.shape[1])
    return np.linalg.solve(H, XL.T @ (yL-lam*fL)/len(XL)+lam*XU.T @ fU/len(XU))


def ppi_plus_plus_lambda(XL, yL, fL, XU, fU):
    """PPI++ (Angelopoulos et al., 2023b) plug-in power tuning at the labeled-only estimate:
    λ̂ = tr(H⁻¹[Cov(∇ℓ_Y, ∇ℓ_f) + Cov(∇ℓ_f, ∇ℓ_Y)]H⁻¹) / (2(1 + n/N) tr(H⁻¹ Cov(∇ℓ_f) H⁻¹)), clipped to [0, 1]."""
    w = exact_fit(XL, yL, fL, XU, fU, 0.)
    X = np.vstack([XL, XU])
    H_inv = np.linalg.inv(X.T @ X/len(X)+1e-6*np.eye(X.shape[1]))
    gY, gF = XL*(XL @ w-yL)[:, None], XL*(XL @ w-fL)[:, None]
    gF_all = X*(X @ w-np.concatenate([fL, fU]))[:, None]
    cross = (gY-gY.mean(0)).T @ (gF-gF.mean(0))/(len(XL)-1)
    numerator = np.trace(H_inv @ (cross+cross.T) @ H_inv)
    denominator = 2*(1+len(XL)/len(XU))*np.trace(H_inv @ np.cov(gF_all, rowvar=False) @ H_inv)
    return float(np.clip(numerator/denominator, 0, 1))


def cv_lambda(XL, yL, fL, XU, fU, seed, setup=SETUP):
    """This project's rule transplanted to regression: K-fold training CV on visible labels, grid, ties → smaller λ."""
    folds = np.array_split(np.random.default_rng(seed).permutation(len(XL)), setup['cv_folds'])
    best, best_lambda = np.inf, 0.
    for lam in setup['cv_lambdas']:
        errors = []
        for held in folds:
            fit = np.setdiff1d(np.arange(len(XL)), held)
            w = exact_fit(XL[fit], yL[fit], fL[fit], XU, fU, lam)
            errors.append(np.mean((XL[held] @ w-yL[held])**2))
        if np.mean(errors) < best-1e-12:
            best, best_lambda = float(np.mean(errors)), lam
    return best_lambda


# Step 4: One Run (One μ, One Seed): Every Method on the Same Data

def run_one(mu, seed, setup=SETUP):
    ((XL, yL, fL, _), (XU, _, fU, _), (XT, yT, fT, bT)), rng = make_data(mu, seed, setup)
    labeled, unlabeled = (XL, yL, fL), (XU, fU)
    def errors(prediction):
        e = (prediction-yT)**2
        return {'mse': float(e.mean()), 'mse_a': float(e[~bT].mean()), 'mse_b': float(e[bT].mean())}
    rows = [dict(method='Teacher', **errors(fT), lam=None)]
    lam_pp = ppi_plus_plus_lambda(XL, yL, fL, XU, fU)
    lam_cv = cv_lambda(XL, yL, fL, XU, fU, seed, setup)
    for method, mode, lam in [('Only Labeled', 'labeled', 0.), ('Classic SSL', 'ssl', 0.),
                              ('PPI++ (offline λ)', 'fixed', lam_pp), ('PP-SSL (online λ)', 'online', None),
                              ('PPI, λ by training CV (this project)', 'fixed', lam_cv)] + \
                             [(f'fixed λ={x}', 'fixed', x) for x in setup['fixed_lambdas']]:
        w, final = train(labeled, unlabeled, mode, np.random.default_rng([seed, 7]), lam or 0., setup)
        rows.append(dict(method=method, **errors(XT @ w), lam=None if mode in ('labeled', 'ssl') else final))
    return [dict(mu=mu, seed=seed, **row) for row in rows]


def fingerprint():
    core = Path(__file__).parent
    return {'setup': SETUP, 'targets': TARGETS, 'boundary': BOUNDARY,
            'source_sha256': hashlib.sha256((core/'ppssl_replication.py').read_bytes()).hexdigest(),
            'versions': {n: importlib.metadata.version(n) for n in ['numpy', 'pandas']}}


def is_complete(path=PPSSL_DB):
    if not Path(path).exists():
        return False
    db = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    try:
        row = db.execute("SELECT value FROM metadata WHERE key='status'").fetchone()
    finally:
        db.close()
    return row is not None and json.loads(row[0]) == 'complete'


def run(workers=6, path=PPSSL_DB):
    if is_complete(path):
        print('PP-SSL replication already complete; nothing to run.', flush=True)
        return
    Path(path).unlink(missing_ok=True)  # Runs take minutes; an unfinished database is rebuilt, never mixed.
    db = sqlite3.connect(path)
    try:
        db.executescript('''CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE runs(mu REAL, seed INTEGER, method TEXT, mse REAL, mse_a REAL, mse_b REAL, lam REAL,
                              PRIMARY KEY(mu, seed, method));''')
        db.execute('INSERT INTO metadata VALUES(?,?)', ('fingerprint', dumps(fingerprint())))  # Before any result.
        db.commit()
        tasks = list(product(SETUP['mus'], SETUP['seeds']))
        print(f'{len(tasks)} PP-SSL replication runs | {workers} CPU workers', flush=True)
        results = Parallel(n_jobs=workers)(delayed(run_one)(mu, seed) for mu, seed in tasks)
        with db:
            db.executemany('INSERT INTO runs VALUES(?,?,?,?,?,?,?)',
                           [(r['mu'], r['seed'], r['method'], r['mse'], r['mse_a'], r['mse_b'], r['lam'])
                            for rows in results for r in rows])
            db.execute('INSERT INTO metadata VALUES(?,?)', ('status', dumps('complete')))
    finally:
        db.close()


# Step 5: Check the Pre-Declared Reproduction Targets (Paired Over Seeds, Bootstrap 95% Interval)

def load(path=PPSSL_DB):
    db = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    try:
        return pd.read_sql_query('SELECT * FROM runs', db)
    finally:
        db.close()


def with_best_fixed(runs):
    """Infeasible oracle: per μ, the fixed λ with the lowest mean total test MSE (as in the paper's App. H.3)."""
    fixed = runs.loc[runs.method.str.startswith('fixed λ=')]
    best = fixed.groupby(['mu', 'method']).mse.mean().groupby(level='mu').idxmin().map(lambda k: k[1])
    chosen = fixed.loc[fixed.method == fixed.mu.map(best)].assign(method='best fixed λ')
    return pd.concat([runs, chosen], ignore_index=True), best


def bootstrap(values, draws=4000, seed=0):
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    return np.percentile(values[rng.integers(0, len(values), (draws, len(values)))].mean(axis=1), [2.5, 97.5])


def check_targets(runs):
    runs, _ = with_best_fixed(runs)
    wide = {metric: runs.pivot_table(index=['mu', 'seed'], columns='method', values=metric)
            for metric in ['mse', 'mse_a', 'mse_b', 'lam']}
    out = []
    for t in TARGETS:
        details = []
        for mu in t['mus']:
            if t['rule'] == 'decreases':
                continue
            w = wide[t['metric']].xs(mu, level='mu')
            d = w[t['left']]-w[t['right']]
            lo, hi = bootstrap(d)
            ok = {'left_worse': lo > 0, 'left_better': hi < 0, 'left_not_worse': d.mean() <= .02*w[t['right']].mean(),
                  'within_5pct': d.mean() <= .05*w[t['right']].mean()}[t['rule']]
            details.append(dict(mu=mu, diff=d.mean(), ci_low=lo, ci_high=hi, ok=ok))
        if t['rule'] == 'decreases':
            lam = wide['lam'][t['left']]
            d = lam.xs(t['mus'][-1], level='mu')-lam.xs(t['mus'][0], level='mu')
            lo, hi = bootstrap(d)
            details.append(dict(mu=f"{t['mus'][-1]} vs {t['mus'][0]}", diff=d.mean(), ci_low=lo, ci_high=hi, ok=hi < 0))
        for row in details:
            out.append(dict(id=t['id'], claim=t['claim'], contrast=f"{t['left']} − {t['right'] or 'λ at μ=0.1'}",
                            metric=t['metric'], **row))
    table = pd.DataFrame(out)
    verdicts = table.groupby('id', sort=False).ok.all().map({True: 'reproduced', False: 'NOT reproduced'})
    return table.assign(verdict=table.id.map(verdicts))
