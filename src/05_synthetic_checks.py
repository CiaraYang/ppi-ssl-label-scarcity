# Step 5: Synthetic Checks — Positive Control and PP-SSL Replication (No Real Data Is Read)

import sys
from importlib import import_module
from pathlib import Path

import hashlib
import importlib.metadata
import json
import sqlite3
import time
from itertools import product

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from scipy.special import softmax
from sklearn.metrics import accuracy_score, f1_score, log_loss
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))  # The numbered files import each other by name.

common = import_module('00_common')
DEFAULT, make_mask, RESULTS_DIR, dumps = common.DEFAULT, common.make_mask, common.RESULTS_DIR, common.dumps
methods = import_module('00_methods')
K, NonEstimableCandidate, fit_student, macro_f1, probabilities, stratified_folds = methods.K, methods.NonEstimableCandidate, methods.fit_student, methods.macro_f1, methods.probabilities, methods.stratified_folds


# ===== From positive_control.py — Step 1: Synthetic Positive Control — Known Posterior, Controlled Teacher, Known Missingness =====

PC_DB = RESULTS_DIR / 'databases' / 'positive_control.sqlite'
# The generating mechanism. Signal and feature count were set by matching only the Bayes Macro F1 (0.656) and the
# Supervised LR learning curve (0.41 at 50 → 0.63 at 1,000 labels) to the real data (fully labeled LR 0.637;
# Supervised 0.44 → 0.58), before any SSL or PPI fit was run on synthetic data.
WORLD = {'features': 20, 'signal': .5, 'subgroup_share': .3, 'priors': (.49, .16, .08, .27), 'weight_seed': 2026,
         'n_train': 10_000, 'n_validation': 5_000, 'mar_threshold': .5, 'mnar_class': 1}
GRID = {'mechanisms': list(DEFAULT.mechanisms), 'budgets': list(DEFAULT.budgets), 'seeds': list(range(42, 52))}
# Teacher q is a fixed function of x (and of noise drawn independently of y), so it never sees a label and needs no
# cross-fitting: any difference between methods is due to the objective, not to Teacher leakage.
TEACHERS = {
    'oracle': 'q = true posterior p*(y|x)',
    'noise σ=1': 'q = softmax(log p* + 1·ε), ε ~ N(0, I) per row',
    'noise σ=3': 'q = softmax(log p* + 3·ε)',
    'uninformative': 'q = softmax(3·ε): no information about y',
    'subgroup-biased': 'q = p* in group A (70%); in group B (30%) q is p* with classes rotated (k−1 → k), '
                       'so it is confidently wrong there',
}
METHODS = ('SSL soft', 'PPI CV λ', 'PPI λ=1')
MARGIN = .005  # Macro F1 difference treated as material.

# Step 2: Pre-Declared Expectations (Written to the Database Before Any Result Exists)

EXPECTATIONS = [
    dict(id='E1', teacher='oracle', contrast='PPI CV λ − Supervised', mechanisms=['MCAR', 'MAR'], budgets=[50, 100, 200],
         rule='gain', why='Informative, unbiased Teacher: if the objective and its implementation work, PPI must help '
                          'where labels are scarce.'),
    dict(id='E2', teacher='oracle', contrast='PPI λ=1 − Supervised', mechanisms=['MCAR', 'MAR'], budgets=[50, 100, 200],
         rule='gain', why='Same, without λ selection.'),
    dict(id='E3', teacher='oracle', contrast='PPI CV λ − Supervised', mechanisms=['MCAR', 'MAR'], budgets=[50, 1000],
         rule='shrinks', why='The gain should shrink as labels become plentiful (50 vs 1,000).'),
    dict(id='E4', teacher='noise σ=1', contrast='PPI CV λ − Supervised', mechanisms=['MCAR', 'MAR'], budgets=[50, 100, 200],
         rule='gain', why='A noisier but still informative Teacher should still help at small budgets.'),
    dict(id='E5', teacher='noise σ=3', contrast='PPI CV λ − Supervised', mechanisms=['MCAR', 'MAR'], budgets=DEFAULT.budgets,
         rule='no_gain', why='A Teacher worse than the Supervised Student (as on the real data) should give no material gain.'),
    dict(id='E6', teacher='noise σ=3', contrast='PPI CV λ − Supervised', mechanisms=['MCAR', 'MAR'], budgets=DEFAULT.budgets,
         rule='no_harm', why='Unbiased objective + CV λ (λ=0 allowed): no material harm.'),
    dict(id='E7', teacher='uninformative', contrast='PPI CV λ − Supervised', mechanisms=['MCAR', 'MAR'], budgets=DEFAULT.budgets,
         rule='no_gain', why='Negative control: no information, so no gain.'),
    dict(id='E8', teacher='uninformative', contrast='PPI CV λ − Supervised', mechanisms=['MCAR', 'MAR'], budgets=DEFAULT.budgets,
         rule='no_harm', why='CV should pick λ≈0 and avoid harm.'),
    dict(id='E9', teacher='uninformative', contrast='PPI λ=1 − Supervised', mechanisms=['MCAR', 'MAR'], budgets=[50, 100, 200],
         rule='harm', why='Forcing λ=1 with pure-noise targets only adds variance: it should hurt, showing what CV guards against.'),
    dict(id='E10', teacher='subgroup-biased', contrast='PPI CV λ − SSL soft', metric='macro_f1_g1', mechanisms=['MCAR', 'MAR'],
         budgets=DEFAULT.budgets, rule='gain', why='In the subgroup where the Teacher is wrong, the rectifier should protect PPI '
                                                   'relative to SSL (the PP-SSL group-B phenomenon).'),
    dict(id='E11', teacher='subgroup-biased', contrast='PPI CV λ − Supervised', mechanisms=['MCAR', 'MAR'],
         budgets=DEFAULT.budgets, rule='no_harm', why='Overall, no material harm; a gain is not predicted (the Teacher is '
                                                       'informative in 70% and adversarial in 30%).'),
]
DECLARATION = {
    'name': 'positive_control_v1', 'declared_on': '2026-10-16', 'evaluation': 'synthetic validation sets only',
    'world': WORLD, 'grid': GRID, 'teachers': TEACHERS, 'margin': MARGIN, 'expectations': EXPECTATIONS,
    'rules': {'gain': '95% CI lower bound > 0', 'harm': '95% CI upper bound < 0', 'no_harm': f'95% CI lower bound > −{MARGIN}',
              'no_gain': f'mean < +{MARGIN}', 'shrinks': 'mean gain at the smallest budget − at the largest: 95% CI lower bound > 0'},
    'interval': 'percentile bootstrap (2,000 draws) over seeds; each seed is an independent synthetic training pool, '
                'validation set, Teacher noise and mask, so unlike the real data these intervals are over replications',
    'mnar': 'MNAR (class 1 under-observed, uncorrected) is reported descriptively: the labeled-term bias and the shifted '
            'unlabeled pool make the direction depend on the Teacher, so no direction is declared',
    'methods': 'LR Student (ssl_core.fit_student, C=1) for Supervised, SSL soft (w ∈ {0.1, 0.5} by CV, confidence 0.8), '
               'PPI with λ ∈ {0,.25,.5,.75,1} by 3-fold training CV (same rules as selection.select_settings) and PPI λ=1',
}


# Step 3: The Synthetic World (Fixed Coefficients; Each Seed Draws a New Pool, Validation Set, Teacher Noise and Mask)

def coefficients(world=WORLD):
    beta = world['signal']*np.random.default_rng(world['weight_seed']).normal(size=(world['features']+1, K))
    features = sample_features(200_000, np.random.default_rng(world['weight_seed']+1), world)
    logits, intercept = features @ beta, np.log(world['priors'])
    for _ in range(200):  # Intercepts that reproduce the declared class priors.
        intercept += np.log(world['priors'])-np.log(softmax(logits+intercept, axis=1).mean(axis=0))
    return beta, intercept


def sample_features(n, rng, world=WORLD):
    return np.hstack([rng.normal(size=(n, world['features'])), rng.random((n, 1)) < world['subgroup_share']]).astype(float)


def sample(n, rng, beta, intercept, world=WORLD):
    X = sample_features(n, rng, world)
    posterior = softmax(X @ beta+intercept, axis=1)
    y = (posterior.cumsum(axis=1) > rng.random((n, 1))).argmax(axis=1)
    return X, y, posterior


def teacher_targets(name, posterior, group, noise):
    log_p = np.log(np.clip(posterior, 1e-300, 1))
    if name == 'oracle':
        return posterior
    if name.startswith('noise σ='):
        return softmax(log_p+float(name.split('=')[1])*noise, axis=1)
    if name == 'uninformative':
        return softmax(3*noise, axis=1)
    if name == 'subgroup-biased':
        return np.where(group[:, None] == 1, np.roll(posterior, 1, axis=1), posterior)
    raise ValueError(f'Unknown Teacher {name}.')


def design(X):
    return np.hstack([X, np.ones((len(X), 1))])  # Intercept last: fit_student leaves it unpenalized.


def synthetic_mask(X, y, budget, seed, mechanism, world=WORLD):
    """Reuse masking.make_mask unchanged: MAR under-observes rows with x₀ > threshold (known propensities, IPW);
    MNAR under-observes class 1 (uncorrected)."""
    frame = pd.DataFrame({'Incident_Year': np.where(X[:, 0] > world['mar_threshold'], DEFAULT.recent_year, 2000)})
    outcomes = pd.Series(np.where(y == world['mnar_class'], 'destroyed', 'other'), index=frame.index)
    mask = make_mask(frame, budget, seed, mechanism, outcomes=outcomes)
    visible = mask.is_labeled.to_numpy()
    if mechanism == 'MAR':
        propensity = mask.propensity.to_numpy()
        return visible, 1/propensity[visible], 1/(1-propensity[~visible])
    return visible, np.ones(visible.sum()), np.ones((~visible).sum())


# Step 4: Training-CV Selection With the Main Study's Rules (Fixed Teacher, So No Nested Re-Fitting Is Needed)

def select(XL, y, XU, qL, qU, wl, wu, seed, config=DEFAULT, cache=None):
    folds = stratified_folds(y, seed, config)
    selected = {'soft': {'weight': 0., 'threshold': 0.}, 'lambda': 0.}
    if not folds:
        return selected, 'supervised_fallback: fewer than two visible labels in a class'
    candidates = [('soft', w, t) for t in config.ssl_thresholds for w in config.ssl_weights if w > 0]
    candidates += [('ppi', value, 0.) for value in config.ppi_lambdas if value > 0]
    counts = np.bincount(y, minlength=K)
    cache = {} if cache is None else cache
    scores = {('supervised', 0., 0.): []}
    for fold, (fit_ids, held_ids) in enumerate(folds):
        fit_fraction = np.bincount(y[fit_ids], minlength=K)/counts
        held_fraction = np.bincount(y[held_ids], minlength=K)/counts
        w_fit = wl[fit_ids]/fit_fraction[y[fit_ids]]
        w_held = wl[held_ids]/held_fraction[y[held_ids]]
        if fold not in cache:  # The Supervised fold fit does not depend on the Teacher.
            cache[fold] = fit_student(XL[fit_ids], y[fit_ids], XU, qL[fit_ids], qU, wl=w_fit, config=config)[0]
        supervised = cache[fold]
        scores[('supervised', 0., 0.)].append(macro_f1(y[held_ids], probabilities(XL[held_ids], supervised).argmax(1), w_held))
        warm = {}
        for mode, value, threshold in candidates:
            key = (mode, value, threshold)
            if mode == 'soft' and not np.any(qU.max(axis=1) >= threshold):
                scores.setdefault(key, []).append(None)
                continue
            try:
                weights, _ = fit_student(XL[fit_ids], y[fit_ids], XU, qL[fit_ids], qU, mode,
                                         value if mode == 'ppi' else 0., wl=w_fit, wu=wu,
                                         initial=warm.get((mode, threshold), supervised),
                                         weight=value, threshold=threshold, config=config)
                warm[mode, threshold] = weights
                scores.setdefault(key, []).append(
                    macro_f1(y[held_ids], probabilities(XL[held_ids], weights).argmax(1), w_held))
            except NonEstimableCandidate:
                scores.setdefault(key, []).append(None)
    baseline = float(np.mean(scores[('supervised', 0., 0.)]))
    for mode in ['soft', 'ppi']:
        best, best_value, best_threshold = baseline, 0., 0.
        for (name, value, threshold) in sorted(k for k in scores if k[0] == mode):
            values = scores[(name, value, threshold)]
            if len(values) != len(folds) or None in values:
                continue
            if np.mean(values) > best+1e-8:  # Ties keep the smaller value, ultimately Supervised.
                best, best_value, best_threshold = float(np.mean(values)), value, threshold
        if mode == 'ppi':
            selected['lambda'] = best_value
        else:
            selected['soft'] = {'weight': best_value, 'threshold': best_threshold}
    return selected, 'training_stratified_cv_macro_f1'


# Step 5: One Case — All Teachers Share the Pool, Mask, Supervised Fit and CV Folds

def evaluate_scores(probability, y, group):
    prediction = probability.argmax(axis=1)
    return {'macro_f1': macro_f1(y, prediction), 'accuracy': accuracy_score(y, prediction),
            'log_loss': log_loss(y, np.clip(probability, 1e-15, 1), labels=np.arange(K)),
            'class_f1': f1_score(y, prediction, labels=np.arange(K), average=None, zero_division=0).tolist(),
            'macro_f1_g0': macro_f1(y[group == 0], prediction[group == 0]),
            'macro_f1_g1': macro_f1(y[group == 1], prediction[group == 1])}


def run_case(mechanism, budget, seed, world=WORLD, teachers=tuple(TEACHERS), config=DEFAULT):
    started = time.monotonic()
    beta, intercept = coefficients(world)
    rng = np.random.default_rng([seed, 680])
    X, y, posterior = sample(world['n_train'], rng, beta, intercept, world)
    XV_raw, yv, pv = sample(world['n_validation'], rng, beta, intercept, world)
    noise, noise_v = rng.normal(size=(len(X), K)), rng.normal(size=(len(XV_raw), K))
    visible, wl, wu = synthetic_mask(X, y, budget, seed, mechanism, world)
    XL, XU, XV = design(X[visible]), design(X[~visible]), design(XV_raw)
    yl, group, gv = y[visible], X[:, -1], XV_raw[:, -1]
    rows, audit = [], {'mechanism': mechanism, 'n_labeled': budget, 'seed': seed, 'teachers': {},
                       'visible_class_counts': np.bincount(yl, minlength=K).tolist()}
    with threadpool_limits(limits=1):
        baseline, _ = fit_student(XL, yl, XU, np.zeros((len(yl), K)), np.zeros((len(XU), K)), wl=wl, config=config)
        bp = probabilities(XV, baseline)
        rows.append(dict(teacher='none', method='Supervised', **evaluate_scores(bp, yv, gv)))
        rows.append(dict(teacher='none', method='Bayes', **evaluate_scores(pv, yv, gv)))
        cache = {}
        for name in teachers:
            q = teacher_targets(name, posterior, group, noise)
            qL, qU = q[visible], q[~visible]
            rows.append(dict(teacher=name, method='Teacher', **evaluate_scores(teacher_targets(name, pv, gv, noise_v), yv, gv)))
            selected, policy = select(XL, yl, XU, qL, qU, wl, wu, seed, config, cache)
            ssl = selected['soft']
            if ssl['weight']:
                fitted, _ = fit_student(XL, yl, XU, qL, qU, 'soft', wl=wl, wu=wu, initial=baseline, config=config, **ssl)
            else:
                fitted = baseline
            rows.append(dict(teacher=name, method='SSL soft', weight=ssl['weight'], **evaluate_scores(probabilities(XV, fitted), yv, gv)))
            entry = {'selection_policy': policy, 'selected': selected}
            for method, requested in [('PPI CV λ', selected['lambda']), ('PPI λ=1', 1.)]:
                value = requested
                while True:  # Same feasibility back-off as experiment.run_case.
                    try:
                        fitted = baseline if value == 0 else fit_student(
                            XL, yl, XU, qL, qU, 'ppi', value, wl=wl, wu=wu, initial=baseline, config=config)[0]
                        break
                    except NonEstimableCandidate:
                        value = max(c for c in config.ppi_lambdas if c < value)
                if value != requested:
                    entry[f'{method} backoff'] = {'requested': requested, 'effective': value}
                rows.append(dict(teacher=name, method=method, lambda_value=value,
                                 **evaluate_scores(probabilities(XV, fitted), yv, gv)))
            audit['teachers'][name] = entry
    audit['elapsed_seconds'] = time.monotonic()-started
    return rows, audit


# Step 6: Run Into Its Own Database; Resume Only Under the Same Declaration and Source

def study_fingerprint():
    core = Path(__file__).parent
    return {'declaration': DECLARATION,
            'source_sha256': {n: hashlib.sha256((core/n).read_bytes()).hexdigest()
                              for n in ['05_synthetic_checks.py', '00_methods.py', '00_common.py']},
            'versions': {n: importlib.metadata.version(n) for n in ['numpy', 'scipy', 'scikit-learn', 'pandas']}}


def connect(path=PC_DB):
    db = sqlite3.connect(path)
    db.executescript('''
    CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS cases(mechanism TEXT, budget INTEGER, seed INTEGER, status TEXT, audit TEXT, error TEXT,
        PRIMARY KEY(mechanism,budget,seed));
    CREATE TABLE IF NOT EXISTS metrics(mechanism TEXT, budget INTEGER, seed INTEGER, teacher TEXT, method TEXT,
        macro_f1 REAL, accuracy REAL, log_loss REAL, class_f1 TEXT, macro_f1_g0 REAL, macro_f1_g1 REAL,
        lambda_value REAL, weight REAL, PRIMARY KEY(mechanism,budget,seed,teacher,method));
    ''')
    return db


def evaluate(key):
    try:
        rows, audit = run_case(*key)
        return key, rows, audit, None
    except Exception as error:
        return key, [], {}, f'{type(error).__name__}: {error}'


def tasks():
    return list(product(GRID['mechanisms'], GRID['budgets'], GRID['seeds']))


def is_complete(path=PC_DB):
    if not Path(path).exists():
        return False
    db = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    try:
        done = {tuple(r) for r in db.execute("SELECT mechanism,budget,seed FROM cases WHERE status='complete'")}
    finally:
        db.close()
    return set(tasks()) <= done


def run(workers=4, path=PC_DB):
    current = json.loads(dumps(study_fingerprint()))
    if is_complete(path):
        db = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
        try:
            stored = json.loads(db.execute("SELECT value FROM metadata WHERE key='fingerprint'").fetchone()[0])
        finally:
            db.close()
        changed = sorted(k for k in current['source_sha256'] if stored['source_sha256'].get(k) != current['source_sha256'][k])
        print('Positive control already complete; nothing to run.'
              +(f' WARNING: source changed since the run: {changed}.' if changed else ''), flush=True)
        return
    db = connect(path)
    try:
        found = dict(db.execute('SELECT key, value FROM metadata').fetchall())
        if 'fingerprint' in found and json.loads(found['fingerprint']) != current:
            raise ValueError('The declaration or source changed since this study started; move the database aside to rerun.')
        # The declaration (with every expectation) is stored before the first result.
        for key, value in [('fingerprint', current), ('declaration', DECLARATION)]:
            db.execute('INSERT OR REPLACE INTO metadata VALUES(?,?)', (key, dumps(value)))
        db.commit()
        done = {tuple(r) for r in db.execute("SELECT mechanism,budget,seed FROM cases WHERE status='complete'")}
        pending = [key for key in tasks() if key not in done]
        print(f'{len(pending)} positive-control cases to run | {len(done)} cached | {workers} CPU workers', flush=True)
        columns = ['macro_f1', 'accuracy', 'log_loss', 'class_f1', 'macro_f1_g0', 'macro_f1_g1', 'lambda_value', 'weight']
        with parallel_config(backend='loky', inner_max_num_threads=1):
            results = Parallel(n_jobs=workers, return_as='generator_unordered')(delayed(evaluate)(key) for key in pending)
            for count, (key, rows, audit, error) in enumerate(results, 1):
                with db:
                    db.execute('DELETE FROM metrics WHERE mechanism=? AND budget=? AND seed=?', key)
                    db.execute('INSERT OR REPLACE INTO cases VALUES(?,?,?,?,?,?)',
                               (*key, 'failed' if error else 'complete', dumps(audit), error))
                    db.executemany('INSERT INTO metrics VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)', [
                        (*key, row['teacher'], row['method'],
                         *[dumps(row[c]) if c == 'class_f1' else row.get(c) for c in columns]) for row in rows])
                if error or count % 25 == 0 or count == len(pending):
                    print(f'[{count}/{len(pending)}] {key} {"FAILED: "+error if error else "saved"}', flush=True)
        failed = db.execute("SELECT COUNT(*) FROM cases WHERE status!='complete'").fetchone()[0]
        db.execute('INSERT OR REPLACE INTO metadata VALUES(?,?)', ('status', dumps('complete_with_failures' if failed else 'complete')))
        db.commit()
    finally:
        db.close()
    if failed:
        raise RuntimeError(f'{failed} positive-control cases failed; see {Path(path).name}.')


# Step 7: Check Each Pre-Declared Expectation (Bootstrap Over Independent Seeds)

def load(path=PC_DB):
    db = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    try:
        rows = pd.read_sql_query('SELECT * FROM metrics', db)
        declaration = json.loads(db.execute("SELECT value FROM metadata WHERE key='declaration'").fetchone()[0])
    finally:
        db.close()
    return rows, declaration


def paired(rows, teacher, contrast, metric='macro_f1'):
    """Paired differences (indexed by mechanism, budget, seed) for 'A − B' within one Teacher condition."""
    left, right = contrast.split(' − ')
    def column(method):
        source = 'none' if method == 'Supervised' else teacher
        part = rows.loc[(rows.teacher == source) & (rows.method == method)]
        return part.set_index(['mechanism', 'budget', 'seed'])[metric]
    return (column(left)-column(right)).dropna()


def seed_bootstrap(values, statistic, draws=2000, seed=0):
    """Percentile interval resampling whole seeds (each an independent synthetic replication)."""
    seeds = values.index.get_level_values('seed').unique().to_numpy()
    rng = np.random.default_rng(seed)
    groups = {s: values.xs(s, level='seed', drop_level=False) for s in seeds}
    stats = [statistic(pd.concat([groups[s] for s in rng.choice(seeds, len(seeds))])) for _ in range(draws)]
    return np.percentile(stats, [2.5, 97.5])


def check(rows, expectation, margin=MARGIN):
    d = paired(rows, expectation['teacher'], expectation['contrast'], expectation.get('metric', 'macro_f1'))
    d = d.loc[d.index.get_level_values('mechanism').isin(expectation['mechanisms'])
              & d.index.get_level_values('budget').isin(expectation['budgets'])]
    if expectation['rule'] == 'shrinks':
        low, high = min(expectation['budgets']), max(expectation['budgets'])
        def statistic(v):
            b = v.index.get_level_values('budget')
            return v[b == low].mean()-v[b == high].mean()
        mean = statistic(d)
    else:
        statistic, mean = np.mean, d.mean()
    lo, hi = seed_bootstrap(d, statistic)
    passed = {'gain': lo > 0, 'harm': hi < 0, 'no_harm': lo > -margin, 'no_gain': mean < margin, 'shrinks': lo > 0}[expectation['rule']]
    return {'id': expectation['id'], 'teacher': expectation['teacher'], 'contrast': expectation['contrast'],
            'metric': expectation.get('metric', 'macro_f1'), 'scope': f"{'+'.join(expectation['mechanisms'])} @ "
            f"{'/'.join(map(str, expectation['budgets']))}", 'rule': expectation['rule'], 'n_pairs': len(d),
            'estimate': float(mean), 'ci_low': float(lo), 'ci_high': float(hi),
            'verdict': 'confirmed' if passed else 'NOT confirmed'}


def checks(rows, declaration=DECLARATION):
    return pd.DataFrame([check(rows, e, declaration['margin']) for e in declaration['expectations']])


# ===== From ppssl_replication.py — Step 1: Anchor to PP-SSL (Shoham et al., NeurIPS 2025): Reproduce the Two-Group Synthetic Phenomenon =====

PPSSL_DB = RESULTS_DIR / 'databases' / 'ppssl_replication.sqlite'
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
            'source_sha256': hashlib.sha256((core/'05_synthetic_checks.py').read_bytes()).hexdigest(),
            'versions': {n: importlib.metadata.version(n) for n in ['numpy', 'pandas']}}


def ppssl_complete(path=PPSSL_DB):
    if not Path(path).exists():
        return False
    db = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    try:
        row = db.execute("SELECT value FROM metadata WHERE key='status'").fetchone()
    finally:
        db.close()
    return row is not None and json.loads(row[0]) == 'complete'


def run_ppssl(workers=6, path=PPSSL_DB):
    if ppssl_complete(path):
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

def load_ppssl(path=PPSSL_DB):
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


if __name__ == '__main__':
    run()           # positive control → results/databases/positive_control.sqlite
    run_ppssl()     # PP-SSL replication → results/databases/ppssl_replication.sqlite
    figures = import_module('06_figures')
    figures.positive_control_outputs()
    figures.ppssl_outputs()
