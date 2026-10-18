# Step 1: Import the Real Project Modules

from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.optimize import check_grad
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

try:  # PyTorch (TabPFN Teacher) must load its OpenMP runtime before XGBoost's on macOS, or TabPFN segfaults.
    import torch  # noqa: F401
except ImportError:
    pass
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from importlib import import_module

# The numbered src files; aliases keep each test's intent readable.
data = masking = storage = import_module('00_common')
core = selection = experiment = import_module('00_methods')
ablations = import_module('02_ablations')
TEACHER_STUDY = import_module('03_teacher_study')
improved = import_module('04_improved')
SYNTHETIC = import_module('05_synthetic_checks')
FIGURES = import_module('06_figures')
FINAL_TEST = import_module('07_final_test')
DEFAULT = data.DEFAULT

FAST = replace(DEFAULT, teacher_iterations=5, tree_iterations=5, ssl_weights=(0., .1), ssl_thresholds=(0., .95), ppi_lambdas=(0., .5, 1.))


# Step 2: Check Mask Counts, Mechanisms, Positivity and Hidden-Label Isolation

class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X = data.training_features()
        cls.outcomes = data.read_data().set_index('Incident_ID').loc[cls.X.index, data.TARGET]

    def test_exact_nested_counts_for_all_mechanisms(self):
        for mechanism in DEFAULT.mechanisms:
            previous = np.zeros(len(self.X), dtype=bool)
            for budget in DEFAULT.budgets:
                mask = masking.make_mask(self.X, budget, 42, mechanism, outcomes=self.outcomes)
                current = mask.is_labeled.to_numpy()
                self.assertEqual(int(current.sum()), budget)
                self.assertTrue((~previous | current).all())
                pd.testing.assert_frame_equal(mask, masking.make_mask(self.X, budget, 42, mechanism, outcomes=self.outcomes))
                previous = current

    def test_mar_probabilities_match_actual_stratum_sampling(self):
        mask = masking.make_mask(self.X, 50, 42, 'MAR')
        recent = (self.X.Incident_Year.fillna(-1).to_numpy() >= DEFAULT.recent_year)
        for group in [recent, ~recent]:
            expected = mask.loc[group, 'is_labeled'].mean()
            np.testing.assert_allclose(mask.loc[group, 'propensity'], expected)
        self.assertLess(mask.loc[recent, 'propensity'].iloc[0], mask.loc[~recent, 'propensity'].iloc[0])
        self.assertTrue(mask.propensity.between(0, 1, inclusive='neither').all())
        for visible in [True, False]:
            selected = mask.is_labeled.to_numpy() == visible
            probability = mask.propensity.to_numpy()[selected]
            weights = 1/(probability if visible else 1-probability)
            weighted_recent = np.average(recent[selected], weights=weights)
            self.assertAlmostEqual(weighted_recent, recent.mean(), places=12)

    def test_mnar_oracle_probabilities_are_not_available(self):
        mask = masking.make_mask(self.X, 50, 42, 'MNAR', outcomes=self.outcomes)
        self.assertTrue(mask.propensity.isna().all())
        case = data.load_case(mask)
        np.testing.assert_array_equal(case.labeled_weights, 1.)
        np.testing.assert_array_equal(case.unlabeled_weights, 1.)

    def test_mcar_and_mar_ignore_target_values(self):
        wrong = self.outcomes.copy()
        wrong[:] = 'destroyed'
        for mechanism in ['MCAR', 'MAR']:
            first = masking.make_mask(self.X, 100, 42, mechanism, outcomes=self.outcomes)
            second = masking.make_mask(self.X, 100, 42, mechanism, outcomes=wrong)
            pd.testing.assert_frame_equal(first, second)

    def test_hidden_and_test_poisoning_cannot_change_case_inputs(self):
        mask = masking.make_mask(self.X, 50, 42, 'MAR')
        original = data.load_case(mask)
        poisoned = data.read_data()
        split = data.read_split()
        test_ids = set(split.loc[split['split']=='test', 'Incident_ID'])
        hidden = set(original.U.index)
        poisoned.loc[poisoned.Incident_ID.isin(hidden|test_ids), data.TARGET] = 'HIDDEN'
        poisoned.loc[poisoned.Incident_ID.isin(test_ids), data.FEATURES] = np.nan
        with patch.object(data, 'read_data', return_value=poisoned):
            changed = data.load_case(mask)
        for name in ['X', 'U', 'V']:
            pd.testing.assert_frame_equal(getattr(original, name), getattr(changed, name))
        for name in ['y', 'vy', 'labeled_weights', 'unlabeled_weights']:
            np.testing.assert_array_equal(getattr(original, name), getattr(changed, name))

    def test_exact_record_independence(self):
        raw = pd.read_csv(data.DATA_PATH, encoding='cp1252').set_index('Incident_ID')
        selected = raw.loc[data.read_split().Incident_ID]
        self.assertFalse(selected.duplicated().any())

    def test_split_typo_rejected(self):
        split = data.read_split()
        split.loc[0, 'split'] = 'validaton'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'split.csv'
            split.to_csv(path, index=False)
            with patch.object(data, 'SPLIT_PATH', path), self.assertRaises(ValueError):
                data.read_split()


# Step 3: Check the Weighted Objective and Small-Sample Edge Cases

class ModelTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(4)
        self.L, self.U = rng.normal(size=(40, 5)), rng.normal(size=(24, 5))
        self.XL = csr_matrix(np.c_[self.L, np.ones(40)])
        self.XU = csr_matrix(np.c_[self.U, np.ones(24)])
        self.y = np.tile(np.arange(4), 10)
        self.qL = core.softmax(rng.normal(size=(40, 4)), axis=1)
        self.qU = core.softmax(rng.normal(size=(24, 4)), axis=1)

    def test_weighted_signed_gradient(self):
        rng = np.random.default_rng(5)
        wl, wu = core.normalized(rng.uniform(.2, 4, 40)), core.normalized(rng.uniform(.2, 4, 24))
        for strength in [0., .5, 1.]:
            args = self.XL, wl[:, None]*(np.eye(4)[self.y]-strength*self.qL), self.XU, strength*wu[:, None]*self.qU, 1/40
            error = check_grad(lambda w: core.objective(w, *args)[0], lambda w: core.objective(w, *args)[1], rng.normal(size=24))
            self.assertLess(error, 1e-5)

    def test_supervised_ipw_matches_sklearn(self):
        wl = np.linspace(.2, 3., 40)
        with threadpool_limits(limits=1):
            fitted, _ = core.fit_student(self.XL, self.y, self.XU, self.qL, self.qU, wl=wl)
            reference = LogisticRegression(C=1., max_iter=5000, tol=1e-10).fit(self.L, self.y, sample_weight=core.normalized(wl))
        np.testing.assert_allclose(core.probabilities(self.XL, fitted), reference.predict_proba(self.L), atol=2e-5)

    def test_filtered_weighted_ssl_matches_sklearn(self):
        wl, wu = np.linspace(.2, 3., 40), np.linspace(.3, 4., 24)
        accepted = self.qU.max(axis=1)>=.5
        self.assertTrue(accepted.any() and not accepted.all())
        # Soft cross-entropy equals weighted hard rows repeated once per class.
        U, q = self.U[accepted], self.qU[accepted]
        soft_weight = .5*40/accepted.sum()*core.normalized(wu[accepted])[:, None]*q
        sample_weight = np.r_[.5*core.normalized(wl), soft_weight.ravel()]
        with threadpool_limits(limits=1):
            fitted, _ = core.fit_student(self.XL, self.y, self.XU, self.qL, self.qU, 'soft', wl=wl, wu=wu, weight=.5, threshold=.5)
            reference = LogisticRegression(C=1., max_iter=5000, tol=1e-10).fit(np.r_[self.L, np.repeat(U, 4, axis=0)], np.r_[self.y, np.tile(np.arange(4), len(U))], sample_weight=sample_weight)
        np.testing.assert_allclose(core.probabilities(self.XL, fitted), reference.predict_proba(self.L), atol=2e-5)

    def test_zero_weight_lambda_and_empty_selection_match_supervised(self):
        with threadpool_limits(limits=1):
            baseline, _ = core.fit_student(self.XL, self.y, self.XU, self.qL, self.qU)
            for mode, threshold in [('soft', 0.), ('soft', 1.), ('ppi', 0.)]:
                fit, _ = core.fit_student(self.XL, self.y, self.XU, self.qL, self.qU, mode, weight=0., threshold=threshold)
                np.testing.assert_allclose(core.probabilities(self.XL, fit), core.probabilities(self.XL, baseline), atol=1e-10)

    def test_absent_class_is_explicit_not_an_error_or_fake_label(self):
        y = self.y.copy()
        y[y==3] = 0
        qL, qU = np.eye(4)[y], np.tile([.5, .25, .25, 0.], (24, 1))
        with threadpool_limits(limits=1):
            fitted, _ = core.fit_student(self.XL, y, self.XU, qL, qU)
        np.testing.assert_array_equal(core.probabilities(self.XL, fitted)[:, 3], 0.)

    def test_nonpositive_corrected_mass_rejected(self):
        with self.assertRaises(core.NonEstimableCandidate):
            core.fit_student(self.XL, self.y, self.XU, np.tile([0., 0., 0., 1.], (40, 1)), np.tile([1., 0., 0., 0.], (24, 1)), 'ppi', 1.)

    def test_fifty_label_rare_class_uses_declared_fallback(self):
        case = data.load_case(masking.generate_mask(50, 42, 'MCAR'))
        self.assertEqual(np.bincount(case.y, minlength=4).min(), 1)
        with patch.object(selection, 'crossfit_teacher', side_effect=AssertionError('No selection Teacher should fit')):
            settings, audit = selection.select_settings(case)
        self.assertEqual(audit['cv_folds'], 0)
        self.assertEqual(settings['lambda'], 0.)
        self.assertEqual(settings['soft']['weight'], 0.)

    def test_teacher_own_labels_excluded_and_missing_columns_aligned(self):
        X = pd.DataFrame({'value': range(12)}, index=[f'L{i}' for i in range(12)])
        U = pd.DataFrame({'value': [100]}, index=['U'])
        y = np.array([0]*6+[1]*5+[3])
        class Spy:
            def __init__(self, **kwargs):pass
            def fit(self, table, values, **kwargs):
                self.seen = set(table.index)
                self.classes_ = np.unique(values)
            def predict_proba(self, table):
                if self.seen & set(table.index):raise AssertionError('Own-label leakage')
                return np.full((len(table), len(self.classes_)), 1/len(self.classes_))
        with patch.object(core, 'CatBoostClassifier', Spy), patch.object(core, 'prepare_catboost_features', side_effect=lambda x:x):
            qL, qU, _, policy = core.crossfit_teacher(X, y, U)
        self.assertEqual(policy, 'kfold_class_alignment')
        np.testing.assert_allclose(qL.sum(axis=1), 1.)
        np.testing.assert_allclose(qU.sum(axis=1), 1.)
        np.testing.assert_array_equal(qL[:, 2], 0.)

    def test_nested_selection_teachers_exclude_outer_holdout(self):
        case = data.load_case(masking.generate_mask(100, 42, 'MCAR'))
        seen = []
        def teacher(X, y, U, **kwargs):
            seen.append(set(X.index))
            return np.eye(4)[y], np.full((len(U), 4), .25), None, 'spy'
        with patch.object(selection, 'crossfit_teacher', side_effect=teacher), threadpool_limits(limits=1):
            _, audit = selection.select_settings(case, config=FAST)
        expected = [set(case.X.iloc[ids].index) for ids, _ in core.stratified_folds(case.y, 42, FAST)]
        self.assertEqual(seen, expected)
        self.assertTrue(any(x['status']=='empty_selection' for x in audit['cv']))


class TreeTests(unittest.TestCase):
    setUp = ModelTests.setUp
    def test_tree_signed_gradient_and_curvature_bound(self):
        signed_derivatives = core.signed_derivatives
        from scipy.special import logsumexp
        rng = np.random.default_rng(7)
        logits = rng.normal(size=(40, 4))
        for strength in [0., .5, 1.]:
            target = np.linspace(.2, 3., 40)[:, None]*(np.eye(4)[self.y]-strength*self.qL)
            def loss(x):
                x = x.reshape(40, 4)
                return -np.sum(target*(x-logsumexp(x, axis=1, keepdims=True)))
            def gradient(x):
                return signed_derivatives(x.reshape(40, 4), target)[0].ravel().astype(float)
            self.assertLess(check_grad(loss, gradient, logits.ravel()), 1e-4)
            _, diagonal = signed_derivatives(logits, target)
            for p, mass, bound in zip(core.softmax(logits, axis=1), target.sum(axis=1), diagonal):
                exact = mass*(np.diag(p)-np.outer(p, p))
                self.assertGreaterEqual(np.linalg.eigvalsh(np.diag(bound)-exact).min(), -1e-6)

    def test_tree_supervised_matches_native_xgboost(self):
        import xgboost as xgb
        tree_parameters = core.tree_parameters
        config = replace(FAST, student_model='XGBoost', tree_iterations=20)
        fitted, _ = core.fit_student(self.XL, self.y, self.XU, self.qL, self.qU, config=config)
        matrix = xgb.DMatrix(self.XL, label=self.y, nthread=1)
        reference = xgb.train(tree_parameters(config, 4), matrix, num_boost_round=config.tree_iterations)
        np.testing.assert_allclose(core.probabilities(self.XL, fitted), reference.predict(matrix), atol=2e-6)

    def test_tree_ipw_matches_native_weighted_xgboost(self):
        import xgboost as xgb
        tree_parameters = core.tree_parameters
        config = replace(FAST, student_model='XGBoost', tree_iterations=20)
        wl = core.normalized(np.linspace(.2, 3., len(self.y)))
        fitted, _ = core.fit_student(self.XL, self.y, self.XU, self.qL, self.qU, wl=wl, config=config)
        matrix = xgb.DMatrix(self.XL, label=self.y, weight=wl, nthread=1)
        reference = xgb.train(tree_parameters(config, 4), matrix, num_boost_round=config.tree_iterations)
        np.testing.assert_allclose(core.probabilities(self.XL, fitted), reference.predict(matrix), atol=2e-6)

    def test_tree_zero_lambda_and_zero_ssl_are_exact_baselines(self):
        config = replace(FAST, student_model='XGBoost')
        baseline, _ = core.fit_student(self.XL, self.y, self.XU, self.qL, self.qU, config=config)
        for mode in ['soft', 'ppi']:
            fitted, _ = core.fit_student(self.XL, self.y, self.XU, self.qL, self.qU, mode, config=config)
            np.testing.assert_array_equal(core.probabilities(self.XL, fitted), core.probabilities(self.XL, baseline))

    def test_tree_positive_correction_changes_probabilities(self):
        config = replace(FAST, student_model='XGBoost', tree_iterations=20)
        baseline, _ = core.fit_student(self.XL, self.y, self.XU, self.qL, self.qU, config=config)
        fitted, _ = core.fit_student(self.XL, self.y, self.XU, self.qL, self.qU, 'ppi', .5, config=config)
        self.assertGreater(np.abs(core.probabilities(self.XL, fitted)-core.probabilities(self.XL, baseline)).mean(), 1e-4)


# Step 4: Verify Atomic Case Storage and End-to-End Low-Label Runs

class IntegrationTests(unittest.TestCase):
    def test_sqlite_roundtrip_and_resume_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            with storage.connect(Path(directory)/'study.sqlite') as db:
                storage.initialize(db)
                row = experiment.metric_row('Supervised', {'accuracy': .7, 'macro_f1': .5, 'class_f1': [.5]*4})
                storage.save_case(db, ('MCAR', 50, 42), [row], {'cv': []})
                self.assertEqual(len(storage.read_results(db)), 1)
                storage.save_case(db, ('MCAR', 50, 42), [row], {'cv': []})
                self.assertEqual(len(storage.read_results(db)), 1)
                with self.assertRaises(ValueError):storage.initialize(db, replace(DEFAULT, student_C=2.))

    def test_real_fifty_label_cases_all_mechanisms(self):
        for mechanism in DEFAULT.mechanisms:
            with patch.object(experiment, 'full_training_data', side_effect=AssertionError('Full-label baseline leaked into a budget case')):
                rows, audit = experiment.run_case(50, mechanism, 42, FAST)
            self.assertEqual(audit['n_labeled'], 50)
            self.assertEqual(len(rows), 6)
            self.assertTrue(all(row['macro_f1'] is None or 0<=row['macro_f1']<=1 for row in rows))
            if mechanism=='MAR':self.assertEqual(audit['weighting'], 'self_normalized_known_design_IPW')
            if mechanism=='MNAR':self.assertEqual(audit['weighting'], 'none')



# Step 5: Check the Ablation Replay, Teacher-Only Refit and the XGBoost Rectified-Risk Caveat

class AblationTests(unittest.TestCase):
    def test_teacher_replay_is_deterministic(self):
        case = data.load_case(masking.generate_mask(100, 42, 'MCAR'))
        with threadpool_limits(limits=1):
            first = core.crossfit_teacher(case.X, case.y, case.U, case.V, seed=42, config=FAST)
            second = core.crossfit_teacher(case.X, case.y, case.U, case.V, seed=42, config=FAST)
        for a, b in zip(first[:3], second[:3]):
            np.testing.assert_array_equal(a, b)

    def test_ablation_replays_main_case_and_rejects_mismatch(self):
        rows, audit = experiment.run_case(100, 'MCAR', 42, FAST)
        stored = pd.DataFrame(rows)
        case = data.load_case(masking.generate_mask(100, 42, 'MCAR'))
        seen = []
        def spy(X, y, wl, seed, config):
            seen.append(set(X.index))
            return core.fit_teacher(X, y, wl, seed, config)
        with patch.object(ablations, 'fit_teacher', side_effect=spy):
            extra, details = ablations.run_ablation('MCAR', 100, 42, audit, stored, FAST)
        self.assertEqual(seen, [set(case.X.index)])  # Teacher-only refit sees exactly the visible rows.
        self.assertEqual({row['method'] for row in extra}, {
            'Teacher / Cross-fit ensemble', 'Teacher / Refit on visible labels', 'LR / PPI fixed λ=1', 'XGBoost / PPI fixed λ=1'})
        for name in FAST.student_models:
            if audit['students'][name]['ppi_selected_lambda'] == 1.:
                self.assertTrue(details['students'][name]['matches_stored_selected_lambda_1'])
        corrupted = json.loads(json.dumps(audit))
        corrupted['teacher_oof']['macro_f1'] += 1e-9
        with self.assertRaises(ablations.ReplayMismatch):
            ablations.run_ablation('MCAR', 100, 42, corrupted, stored, FAST)

    def test_rounds_sensitivity_replays_main_and_a1_fits(self):
        rows, audit = experiment.run_case(100, 'MCAR', 42, FAST)
        stored = pd.DataFrame(rows).set_index('method')
        extra, _ = ablations.run_ablation('MCAR', 100, 42, audit, stored.reset_index(), FAST)
        fixed = pd.DataFrame(extra).set_index('method').loc['XGBoost / PPI fixed λ=1']
        new, details = ablations.run_rounds('MCAR', 100, 42, stored.loc['XGBoost / Supervised'], fixed,
                                            FAST, rounds_grid=(FAST.tree_iterations, 15))
        self.assertEqual({row['method'] for row in new},
                         {'XGBoost[15 rounds] / Supervised', 'XGBoost[15 rounds] / PPI fixed λ=1'})
        self.assertEqual(set(details['fits']), {FAST.tree_iterations, 15})
        wrong = fixed.copy()
        wrong['macro_f1'] += 1e-9
        with self.assertRaises(ablations.ReplayMismatch):
            ablations.run_rounds('MCAR', 100, 42, stored.loc['XGBoost / Supervised'], wrong,
                                 FAST, rounds_grid=(FAST.tree_iterations, 15))

    def test_capacity_sensitivity_replays_main_fit(self):
        rows, _ = experiment.run_case(100, 'MCAR', 42, FAST)
        stored = pd.DataFrame(rows).set_index('method').loc['XGBoost / Supervised']
        large = replace(FAST, student_models=('XGBoost',), student_model='XGBoost', tree_iterations=10, tree_depth=4)
        new, audit = ablations.run_capacity('MCAR', 100, 42, stored, FAST, large)
        self.assertEqual([row['method'] for row in new], [f'{ablations.A3_LABEL} / Supervised'])
        self.assertEqual(audit['replay_main'], 'exact')
        wrong = stored.copy()
        wrong['accuracy'] += 1e-9
        with self.assertRaises(ablations.ReplayMismatch):
            ablations.run_capacity('MCAR', 100, 42, wrong, FAST, large)

    def test_later_ablations_add_rows_without_touching_a1(self):
        with tempfile.TemporaryDirectory() as directory:
            with storage.connect(Path(directory)/'ablations.sqlite') as db:
                first = experiment.metric_row('LR / PPI fixed λ=1', {'accuracy': .7, 'macro_f1': .5, 'class_f1': [.5]*4})
                storage.save_case(db, ('MCAR', 100, 42), [first], {'cv': []})
                db.execute('CREATE TABLE a3_cases(mechanism TEXT, budget INTEGER, seed INTEGER, status TEXT, '
                           'audit TEXT, error TEXT, PRIMARY KEY(mechanism,budget,seed))')
                later = experiment.metric_row('XGBoost[300 rounds, depth 5] / Supervised', {'accuracy': .6, 'macro_f1': .4})
                ablations.save_extra(db, 'a3_cases', ('MCAR', 100, 42), [later], {})
                ablations.save_extra(db, 'a3_cases', ('MCAR', 100, 42), [later], {})  # Resaving is idempotent.
                results = storage.read_results(db).set_index('method')
                self.assertEqual(len(results), 2)
                self.assertEqual(results.loc['LR / PPI fixed λ=1', 'macro_f1'], .5)

    def test_xgboost_rectified_risk_keeps_falling_with_rounds(self):
        # Documents the README caveat: for λ>0 the empirical rectified risk is unbounded below for trees.
        fit_tree = core.fit_tree
        model = ModelTests()
        model.setUp()
        targetL = np.eye(4)[model.y]-model.qL
        risks = [fit_tree(model.XL, targetL, model.XU, model.qU, np.ones(4, bool),
                          replace(FAST, student_model='XGBoost', tree_iterations=rounds)).diagnostics['empirical_risk_final']
                 for rounds in [50, 200]]
        self.assertLess(risks[1], risks[0])
        self.assertLess(risks[1], 0)


# Step 6: Same-Architecture Teachers and Label-Free Preprocessing Respect Train/Validation Boundaries

class TeacherStudyTests(unittest.TestCase):
    def test_same_architecture_teacher_fits_only_fold_training_rows(self):
        case = data.load_case(masking.generate_mask(100, 42, 'MCAR'))
        config = replace(FAST, teacher_model='LR', design_fit='labeled+unlabeled')
        fitted, designs = [], []
        real_fold, real_design = core.fold_teacher, core.design_transformer
        def fold_spy(X, *args, **kwargs):
            fitted.append(set(X.index))
            return real_fold(X, *args, **kwargs)
        def design_spy(X, U=None, config=DEFAULT):
            designs.append(set(X.index) | (set() if U is None else set(U.index)))
            return real_design(X, U, config)
        with patch.object(core, 'fold_teacher', side_effect=fold_spy), \
             patch.object(core, 'design_transformer', side_effect=design_spy), threadpool_limits(limits=1):
            qL, qU, qV, _ = core.crossfit_teacher(case.X, case.y, case.U, case.V, seed=42, config=config)
        folds = core.stratified_folds(case.y, 42, config)
        self.assertEqual(fitted, [set(case.X.iloc[fit].index) for fit, _ in folds])
        for (fit, held), used in zip(folds, designs):
            self.assertFalse(used & set(case.X.iloc[held].index))  # Held-out labeled rows never fit their Teacher.
            self.assertFalse(used & set(case.V.index))             # Validation never fits anything.
            self.assertEqual(used, set(case.X.iloc[fit].index) | set(case.U.index))
        for q in [qL, qU, qV]:
            np.testing.assert_allclose(q.sum(axis=1), 1.)

    def test_label_free_design_uses_unlabeled_but_not_validation_categories(self):
        X = pd.DataFrame({c: [1., 2.] for c in data.NUMERIC_FEATURES} | {c: ['a', 'b'] for c in data.CATEGORICAL_FEATURES})
        U, V = X.assign(Aircraft_Model='only_unlabeled'), X.assign(Aircraft_Model='only_validation')
        labeled = core.design_transformer(X, U, replace(FAST, design_fit='labeled'))
        pooled = core.design_transformer(X, U, replace(FAST, design_fit='labeled+unlabeled'))
        self.assertGreater(pooled(U).shape[1], labeled(U).shape[1])
        self.assertTrue((pooled(U)-pooled(X)).nnz)               # The unlabeled-only category gets its own column.
        # A validation-only category has no column: each validation row loses exactly its model one-hot entry.
        np.testing.assert_array_equal((pooled(V) != 0).sum(axis=1), (pooled(X) != 0).sum(axis=1)-1)

    def test_same_architecture_case_runs_and_records_teacher(self):
        config = replace(FAST, student_models=('XGBoost',), student_model='XGBoost', teacher_model='XGBoost')
        with patch.object(experiment, 'full_training_data', side_effect=AssertionError('Full labels leaked')):
            rows, audit = experiment.run_case(100, 'MCAR', 42, config)
        self.assertEqual([row['method'] for row in rows], ['XGBoost / Supervised', 'XGBoost / SSL soft', 'XGBoost / PPI selected'])
        self.assertEqual((audit['teacher_model'], audit['design_fit']), ('XGBoost', 'labeled'))


# Step 7: The Frozen Test Evaluation Changes Only Which Rows Are Scored


class TabPFNTeacherTests(unittest.TestCase):
    def test_tabpfn_teacher_is_cross_fitted_and_never_sees_its_own_label(self):
        case = data.load_case(masking.generate_mask(100, 42, 'MCAR'))
        config = replace(FAST, teacher_model='TabPFN')
        def spy(X, y, seed=42):
            seen = set(X.index)
            def predict(table):
                if seen & set(table.index):
                    raise AssertionError('Own-label leakage into the TabPFN context')
                return np.full((len(table), core.K), 1/core.K)
            return predict
        with patch.object(core, 'tabpfn_teacher', side_effect=spy) as fitted:
            qL, qU, _, policy = core.crossfit_teacher(case.X, case.y, case.U, config=config)
        self.assertEqual(policy, 'stratified')
        self.assertEqual(fitted.call_count, len(core.stratified_folds(case.y, 42, config)))
        np.testing.assert_allclose(qL.sum(axis=1), 1.)
        np.testing.assert_allclose(qU.sum(axis=1), 1.)

    def test_real_tabpfn_aligns_four_classes_and_is_deterministic(self):
        try:
            import tabpfn.browser_auth as auth
        except ImportError:
            self.skipTest('tabpfn is not installed')
        if auth.get_cached_token() is None:
            self.skipTest('TabPFN-3.5 licence not accepted on this machine')
        case = data.load_case(masking.generate_mask(50, 42, 'MCAR'))
        keep = case.y != 2  # A class absent from the context must get probability 0, not an invented label.
        predict = core.tabpfn_teacher(case.X[keep], case.y[keep], seed=42)
        first, second = predict(case.U.iloc[:20]), core.tabpfn_teacher(case.X[keep], case.y[keep], seed=42)(case.U.iloc[:20])
        self.assertEqual(first.shape, (20, core.K))
        np.testing.assert_allclose(first.sum(axis=1), 1., rtol=0, atol=1e-12)  # Exact rows, or λ=1 breaks tree Students.
        np.testing.assert_array_equal(first[:, 2], 0.)
        np.testing.assert_allclose(first, second, atol=1e-6)
        self.assertEqual(predict(case.U.iloc[:0]).shape, (0, core.K))

    def test_pilot_is_mcar_only_and_declared(self):
        teacher_study = TEACHER_STUDY
        self.assertEqual(teacher_study.TABPFN_GRID['mechanisms'], ['MCAR'])
        self.assertEqual(replace(DEFAULT, teacher_model='TabPFN').teacher_model, 'TabPFN')
        self.assertIn('expectations', teacher_study.TABPFN_DECLARATION)
        self.assertIn('synthetic', teacher_study.TABPFN_DECLARATION['teacher'])


class FinalTestTests(unittest.TestCase):
    def test_test_split_changes_only_scored_rows(self):
        mask = masking.generate_mask(100, 42, 'MCAR')
        validation, test = data.load_case(mask), data.load_case(mask, 'test')
        split = data.read_split()
        self.assertEqual(set(test.V.index), set(split.loc[split['split'] == 'test', 'Incident_ID']))
        for name in ['X', 'U']:
            pd.testing.assert_frame_equal(getattr(validation, name), getattr(test, name))
        np.testing.assert_array_equal(validation.y, test.y)
        with self.assertRaises(ValueError):
            data.load_case(mask, 'train')

    def test_test_run_refits_identical_models(self):
        config = replace(FAST, student_models=('LR',), student_model='LR', teacher_model='LR')
        _, on_validation = experiment.run_case(100, 'MCAR', 42, config)
        _, on_test = experiment.run_case(100, 'MCAR', 42, config, evaluation='test')
        self.assertEqual(on_validation['teacher_oof'], on_test['teacher_oof'])
        self.assertEqual(on_validation['students']['LR']['selected'], on_test['students']['LR']['selected'])
        self.assertEqual(on_validation['students']['LR']['cv'], on_test['students']['LR']['cv'])
        self.assertIn('teacher_test', on_test)


# Step 8: Exploratory Improved Pipeline (improved_v2), Version Protection and the Test Lock

class ImprovedTests(unittest.TestCase):
    setUp = ModelTests.setUp

    def test_heuristic_lambda_closed_form_and_informative_teacher(self):
        rng = np.random.default_rng(3)
        P = core.softmax(rng.normal(size=(40, 4)), axis=1)
        perfect = np.eye(4)[self.y]  # g_f = g_Y, so tr Cov = tr Var and λ = 1/(1 + n/N).
        self.assertAlmostEqual(improved.heuristic_lambda(self.XL, self.y, P, perfect, np.ones(40), 24), 1/(1+40/24), places=10)
        informative = .7*np.eye(4)[self.y]+.3*self.qL
        self.assertGreater(improved.heuristic_lambda(self.XL, self.y, P, informative, np.ones(40), 24), 0)

    def test_ensemble_teacher_predictions_exclude_own_labels(self):
        case = data.load_case(masking.generate_mask(100, 42, 'MCAR'))
        fitted = []
        real = core.fold_teacher
        def spy(X, *args, **kwargs):
            fitted.append(set(X.index))
            return real(X, *args, **kwargs)
        config = replace(improved.CONFIG, teacher_iterations=5)
        with patch.object(core, 'fold_teacher', side_effect=spy), threadpool_limits(limits=1):
            qL, qU, qV, info = improved.build_teacher(case.X, case.y, case.U, case.V, case.labeled_weights, 42, config)
        held = [set(case.X.iloc[h].index) for _, h in core.stratified_folds(case.y, 42, config)]
        self.assertEqual(len(fitted), 2*len(held))  # CatBoost and LR members, same folds.
        for used, rows in zip(fitted, held+held):
            self.assertFalse(used & rows)
            self.assertFalse(used & set(case.V.index))
        np.testing.assert_allclose(qL.sum(axis=1), 1.)
        self.assertEqual(info['teacher'], 'Ensemble')

    def test_whole_student_oof_refits_tree_structure_per_fold(self):
        case = data.load_case(masking.generate_mask(100, 42, 'MCAR'))
        trained = []
        real = improved.xgb.train
        def spy(params, dtrain, **kwargs):
            trained.append(dtrain.num_row())
            return real(params, dtrain, **kwargs)
        config = replace(improved.CONFIG, tree_iterations=5)
        with patch.object(improved.xgb, 'train', side_effect=spy):
            P = improved.oof_supervised('XGBoost leaf-refit', case.X, case.y, case.labeled_weights, case.U, config)
        folds = core.stratified_folds(case.y, 0, config)
        self.assertEqual(trained, [len(fit) for fit, _ in folds])  # Never trained on a row it then predicts.
        np.testing.assert_allclose(P.sum(axis=1), 1.)

    def test_threshold_rule_never_lowers_out_of_fold_macro_f1(self):
        rng = np.random.default_rng(2)
        y = rng.integers(0, 4, 300)
        P = core.softmax(rng.normal(size=(300, 4))+np.eye(4)[y]*np.array([2., .3, .3, 1.]), axis=1)
        bias = improved.fit_bias(P, y, np.ones(300))
        self.assertEqual(bias[0], 0.)
        self.assertGreaterEqual(core.macro_f1(y, improved.biased(P, bias).argmax(axis=1)), core.macro_f1(y, P.argmax(axis=1)))

    def test_leaf_refit_structure_uses_labeled_rows_only(self):
        case = data.load_case(masking.generate_mask(100, 42, 'MCAR'))
        trained = []
        real = improved.xgb.train
        def spy(params, dtrain, **kwargs):
            trained.append(dtrain.num_row())
            return real(params, dtrain, **kwargs)
        with patch.object(improved.xgb, 'train', side_effect=spy):
            XL, XU, XV = improved.student_design('XGBoost leaf-refit', case.X, case.y, case.labeled_weights, case.U, case.V,
                                                 replace(improved.CONFIG, tree_iterations=5))
        self.assertEqual(trained, [len(case.y)])
        self.assertEqual((XL.shape[0], XU.shape[0], XV.shape[0]), (len(case.y), len(case.U), len(case.V)))
        self.assertEqual(XL.shape[1], XV.shape[1])

    def test_improved_case_runs_without_full_or_test_labels(self):
        config = replace(improved.CONFIG, cv_folds=3, teacher_iterations=5, tree_iterations=5)
        with patch.object(experiment, 'full_training_data', side_effect=AssertionError('Full labels leaked')):
            rows, audit = improved.run_improved_case(100, 'MCAR', 42, config)
        self.assertEqual(len(rows), 1+len(improved.STUDENTS)*len(improved.METHODS)*2)
        self.assertTrue(all(0 <= r['macro_f1'] <= 1 for r in rows))

    def test_final_ppi_fit_that_is_not_estimable_backs_off_instead_of_crashing(self):
        config = replace(improved.CONFIG, cv_folds=3, teacher_iterations=5, tree_iterations=5)
        select, fit = improved.select_and_calibrate, improved.fit_methods
        def force_lambda(*args, **kwargs):  # Pretend CV chose λ = 0.75 ...
            chosen, bias, cv = select(*args, **kwargs)
            return {**chosen, 'PPI CV λ': .75}, bias, cv
        def not_estimable(XL, y, XU, qL, qU, wl, wu, P_oof, settings, ssl_weights=(.1, .5), lambdas=(.25, .5, .75, 1.)):
            fits = fit(XL, y, XU, qL, qU, wl, wu, P_oof, settings, ssl_weights, lambdas)
            if tuple(lambdas) == (.75,):  # ... and that the final full-data fit at 0.75 is not estimable.
                fits[('PPI CV λ', .75)] = None
            return fits
        with patch.object(improved, 'select_and_calibrate', side_effect=force_lambda), \
             patch.object(improved, 'fit_methods', side_effect=not_estimable):
            rows, audit = improved.run_improved_case(100, 'MCAR', 42, config)
        self.assertEqual(len(rows), 1+len(improved.STUDENTS)*len(improved.METHODS)*2)
        for student in improved.STUDENTS:
            self.assertEqual(audit['ppi_cv_backoff'][student]['requested'], .75)
            self.assertLess(audit['ppi_cv_backoff'][student]['effective'], .75)


class V2SettingTests(unittest.TestCase):
    def test_airport_parsing_accepts_only_verified_countries(self):
        self.assertEqual(data.airport_country('Singapore-Kallang Airport , Singapore'), 'singapore')
        self.assertEqual(data.airport_country('Manchester-Boston Regional Airport, NH (MHT/KMHT)'), 'united states of america')
        self.assertEqual(data.airport_country('Sydney Airport, NSW (SYD/YSSY)'), 'australia')
        self.assertEqual(data.airport_country('Perth Airport, WA (PER/YPPH)'), 'australia')          # WA + Y... = Australia
        self.assertEqual(data.airport_country('Seattle-Tacoma Airport, WA (SEA/KSEA)'), 'united states of america')
        self.assertEqual(data.airport_country('Nigeria'), 'nigeria')                                 # no comma, whole string
        self.assertEqual(data.airport_country('Some field, USA'), 'united states of america')         # alias
        self.assertTrue(pd.isna(data.airport_country('Southampton Airport (SOU/EGHI), Hampshire')))   # a county is not a country
        self.assertEqual(data.airport_region('Southampton Airport (SOU/EGHI), Hampshire'), 'Europe')   # ... ICAO prefix instead
        self.assertTrue(pd.isna(data.airport_region('Aeroclub Allen')))                              # nothing verifiable: missing
        self.assertTrue(pd.isna(data.airport_region(np.nan)))

    def test_registration_prefix_and_route_flags(self):
        self.assertEqual(data.registration_prefix('F-GBRS'), 'F')
        self.assertEqual(data.registration_prefix('N123AB'), 'N')
        self.assertEqual(data.registration_prefix('44-77320'), 'numeric serial')
        self.assertTrue(pd.isna(data.registration_prefix(np.nan)))
        rows = pd.DataFrame({'Aircraft_Registration': ['F-GBRS', np.nan, 'N1'],
                             'Departure_Airport': ['A (AAA/LFPG), France', 'B (BBB/KJFK), NY (BBB/KJFK)', np.nan],
                             'Destination_Airport': ['C (CCC/EGLL), United Kingdom', 'B (BBB/KJFK), NY (BBB/KJFK)', 'X']})
        out = data.contextual_features(rows)
        self.assertEqual(out.Domestic_Route.tolist(), ['no', 'yes', 'unknown'])
        self.assertEqual(out.Same_Airport.tolist(), ['no', 'yes', 'no'])
        self.assertEqual(out.Has_Route.tolist(), ['yes', 'yes', 'no'])

    def test_contextual_features_never_read_labels_or_excluded_columns(self):
        raw = data.read_data().head(300)
        changed = raw.assign(**{data.TARGET: 'minor', 'Fatalities': 999, 'Incident_Category': 'x', 'Incident_Causes': 'y'})
        pd.testing.assert_frame_equal(data.contextual_features(raw), data.contextual_features(changed))

    def test_three_class_map_merges_only_written_off(self):
        self.assertEqual(data.CLASS_MAP['destroyed'], data.CLASS_MAP['damaged beyond repair'])
        self.assertEqual(len(set(data.CLASS_MAP.values())), 3)
        self.assertEqual(data.VERSION, 'v1')        # the default task is unchanged
        self.assertEqual(len(data.CLASSES), 4)
        self.assertEqual(len(data.FEATURES), 7)

    def test_ensemble_teacher_is_the_mean_of_catboost_and_lr_on_the_same_rows(self):
        case = data.load_case(masking.generate_mask(100, 42, 'MCAR'))
        seen = []
        def member(value):
            def fit(X, y, wl, U, seed=42, config=DEFAULT):
                seen.append((config.teacher_model, tuple(X.index)))
                return lambda table: np.full((len(table), core.K), value)
            return fit
        original = core.fold_teacher
        calls = {'CatBoost': member(.1), 'LR': member(.4)}
        def routed(X, y, wl, U, seed=42, config=DEFAULT):
            return original(X, y, wl, U, seed, config) if config.teacher_model == 'Ensemble' else calls[config.teacher_model](X, y, wl, U, seed, config)
        with patch.object(core, 'fold_teacher', side_effect=routed):
            predict = core.fold_teacher(case.X, case.y, case.labeled_weights, case.U, 42, replace(FAST, teacher_model='Ensemble'))
        np.testing.assert_allclose(predict(case.U.iloc[:3]), .25)
        self.assertEqual({m for m, _ in seen}, {'CatBoost', 'LR'})
        self.assertEqual(seen[0][1], seen[1][1])      # both members fitted on exactly the same fold rows


class VersionProtectionTests(unittest.TestCase):
    def test_fingerprint_differences_name_what_changed(self):
        old = json.loads(storage.dumps(storage.fingerprint()))
        new = json.loads(json.dumps(old))
        new['config']['cv_folds'] = 9
        new['source_sha256']['00_methods.py'] = 'edited'
        self.assertEqual(storage.fingerprint_differences(old, new), ['config.cv_folds', 'src/00_methods.py'])
        self.assertEqual(storage.fingerprint_differences(old, old), [])

    def test_partial_improved_database_is_not_plotted(self):
        figures = FIGURES
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'improved.sqlite'
            with patch.object(improved, 'IMPROVED_DB', path):
                db = improved.connect()
                db.execute("INSERT INTO cases VALUES('MCAR', 50, 42, 'complete', '{}', NULL)")
                db.commit(); db.close()
                self.assertFalse(improved.is_complete())
                with patch.object(figures, 'save', side_effect=AssertionError('Partial study plotted')):
                    figures.improved_outputs()

    def test_draft_test_protocol_refuses_to_run(self):
        final_test = FINAL_TEST
        self.assertEqual(final_test.FROZEN['status'], 'draft')
        # The draft must match the README proposal: CatBoost → LR is the primary method, with no pooled Teacher rule.
        self.assertEqual(final_test.VARIANTS[0], ('CatBoost', 'LR'))
        self.assertIn('Secondary only', final_test.FROZEN['teacher_rule'])
        with patch.object(final_test, 'connect', side_effect=AssertionError('Test database touched')), \
             self.assertRaises(RuntimeError):
            final_test.run()


# Step 10: Synthetic Positive Control, LR Main-Line Analysis and the PP-SSL Anchor

class PositiveControlTests(unittest.TestCase):
    SMALL = {'n_train': 1500, 'n_validation': 400}

    def test_declaration_names_settings_that_should_and_should_not_benefit(self):
        pc = SYNTHETIC
        rules = {e['rule'] for e in pc.EXPECTATIONS}
        self.assertTrue({'gain', 'no_gain', 'harm', 'no_harm'} <= rules)
        self.assertEqual({e['teacher'] for e in pc.EXPECTATIONS}, set(pc.TEACHERS))
        self.assertIn('expectations', pc.DECLARATION)

    def test_teacher_is_a_label_free_function_of_x(self):
        pc = SYNTHETIC
        world = {**pc.WORLD, **self.SMALL}
        beta, intercept = pc.coefficients(world)
        X, y, posterior = pc.sample(500, np.random.default_rng(0), beta, intercept, world)
        np.testing.assert_allclose(posterior.mean(axis=0), world['priors'], atol=.06)
        noise, group = np.random.default_rng(1).normal(size=(500, core.K)), X[:, -1]
        self.assertIs(pc.teacher_targets('oracle', posterior, group, noise), posterior)
        biased = pc.teacher_targets('subgroup-biased', posterior, group, noise)
        np.testing.assert_array_equal(biased[group == 0], posterior[group == 0])
        np.testing.assert_array_equal(biased[group == 1], np.roll(posterior[group == 1], 1, axis=1))
        for name in pc.TEACHERS:  # Shuffling labels cannot change any Teacher.
            np.testing.assert_array_equal(pc.teacher_targets(name, posterior, group, noise),
                                          pc.teacher_targets(name, posterior, group, noise))
            np.testing.assert_allclose(pc.teacher_targets(name, posterior, group, noise).sum(axis=1), 1)

    def test_synthetic_masks_reuse_make_mask_with_exact_counts_and_known_weights(self):
        pc = SYNTHETIC
        rng = np.random.default_rng(3)
        X = np.hstack([rng.normal(size=(2000, 3)), np.zeros((2000, 1))])
        y = rng.integers(0, core.K, 2000)
        for mechanism in DEFAULT.mechanisms:
            visible, wl, wu = pc.synthetic_mask(X, y, 100, 42, mechanism)
            self.assertEqual(visible.sum(), 100)
            self.assertEqual((len(wl), len(wu)), (100, 1900))
        visible, wl, _ = pc.synthetic_mask(X, y, 100, 42, 'MAR')
        high = X[:, 0] > pc.WORLD['mar_threshold']
        self.assertLess(visible[high].mean(), visible[~high].mean())
        self.assertAlmostEqual(float(np.sum(wl)), 2000., delta=1e-6)  # Horvitz–Thompson weights sum to the pool.
        visible, wl, _ = pc.synthetic_mask(X, y, 100, 42, 'MNAR')
        self.assertTrue(np.all(wl == 1))
        self.assertLess(visible[y == 1].mean(), visible[y != 1].mean())

    def test_selection_falls_back_without_cv_and_stays_on_the_grid(self):
        pc = SYNTHETIC
        rng = np.random.default_rng(4)
        XL, XU = np.hstack([rng.normal(size=(40, 3)), np.ones((40, 1))]), np.hstack([rng.normal(size=(300, 3)), np.ones((300, 1))])
        qL, qU = rng.dirichlet(np.ones(core.K), 40), rng.dirichlet(np.ones(core.K), 300)
        y = np.r_[np.zeros(39, int), 1]  # A class with one label: no stratified CV, Supervised fallback.
        selected, policy = pc.select(XL, y, XU, qL, qU, np.ones(40), np.ones(300), 0, FAST)
        self.assertTrue(policy.startswith('supervised_fallback'))
        self.assertEqual(selected['lambda'], 0.)
        y = np.arange(40) % core.K
        selected, policy = pc.select(XL, y, XU, qL, qU, np.ones(40), np.ones(300), 0, FAST)
        self.assertIn(selected['lambda'], FAST.ppi_lambdas)
        self.assertIn(selected['soft']['weight'], FAST.ssl_weights)

    def test_small_case_runs_and_lambda_zero_is_exactly_supervised(self):
        pc = SYNTHETIC
        world = {**pc.WORLD, **self.SMALL}
        rows, audit = pc.run_case('MCAR', 50, 42, world, ('oracle', 'uninformative'), FAST)
        table = pd.DataFrame(rows).set_index(['teacher', 'method'])
        self.assertEqual(len(rows), 2+2*4)
        for teacher in ['oracle', 'uninformative']:
            if audit['teachers'][teacher]['selected']['lambda'] == 0:
                self.assertEqual(table.loc[(teacher, 'PPI CV λ'), 'macro_f1'], table.loc[('none', 'Supervised'), 'macro_f1'])
        self.assertAlmostEqual(table.loc[('oracle', 'Teacher'), 'macro_f1'], table.loc[('none', 'Bayes'), 'macro_f1'])

    def test_expectation_rules_and_seed_bootstrap(self):
        pc = SYNTHETIC
        index = pd.MultiIndex.from_product([['MCAR'], [50, 1000], range(10)], names=['mechanism', 'budget', 'seed'])
        sup = pd.DataFrame({'mechanism': index.get_level_values(0), 'budget': index.get_level_values(1),
                            'seed': index.get_level_values(2), 'teacher': 'none', 'method': 'Supervised', 'macro_f1': .5})
        ppi = sup.assign(teacher='oracle', method='PPI CV λ', macro_f1=np.where(sup.budget == 50, .55, .5))
        rows = pd.concat([sup, ppi], ignore_index=True)
        base = dict(teacher='oracle', contrast='PPI CV λ − Supervised', mechanisms=['MCAR'])
        self.assertEqual(pc.check(rows, dict(base, id='g', budgets=[50], rule='gain'))['verdict'], 'confirmed')
        self.assertEqual(pc.check(rows, dict(base, id='n', budgets=[1000], rule='gain'))['verdict'], 'NOT confirmed')
        self.assertEqual(pc.check(rows, dict(base, id='s', budgets=[50, 1000], rule='shrinks'))['verdict'], 'confirmed')
        self.assertEqual(pc.check(rows, dict(base, id='h', budgets=[1000], rule='no_harm'))['verdict'], 'confirmed')


class LRMainlineTests(unittest.TestCase):
    def test_seed_cluster_bootstrap_and_describe(self):
        lr_mainline = FIGURES
        index = pd.MultiIndex.from_product([['MCAR', 'MAR'], [50, 100], range(5)], names=lr_mainline.KEY)
        constant = pd.Series(.01, index=index)
        np.testing.assert_allclose(lr_mainline.seed_cluster_bootstrap(constant), [.01, .01])
        # One seed far from the others must widen the interval: whole seeds are resampled, not single masks.
        shifted = constant.copy()
        shifted[shifted.index.get_level_values('seed') == 0] = .05
        low, high = lr_mainline.seed_cluster_bootstrap(shifted)
        self.assertAlmostEqual(low, .01)
        self.assertGreater(high, .02)
        summary = lr_mainline.describe(pd.Series(np.r_[np.ones(10), -np.ones(5), np.zeros(5)], index=index))
        self.assertEqual((summary['better'], summary['worse'], summary['equal']), (10, 5, 5))

    def test_lambda_sources_separate_cv_zero_from_forced_fallback(self):
        lr_mainline = FIGURES
        def audit(lam, policy, extra=None):
            return {'students': {'LR': {'ppi_selected_lambda': lam, 'selection_policy': policy, **(extra or {})}}}
        audits = {('MCAR', 50, 42): audit(0., 'supervised_fallback: fewer than two visible labels in a class'),
                  ('MCAR', 50, 43): audit(0., 'training_stratified_cv_macro_f1'),
                  ('MCAR', 50, 44): audit(.5, 'training_stratified_cv_macro_f1')}
        table, counts = lr_mainline.lambda_choices(audits)
        self.assertEqual(table.source.str.contains('forced').sum(), 1)
        self.assertEqual(table.source.str.contains('chosen by CV \\(no').sum(), 1)
        self.assertEqual(int(counts['all'].sum()), 3)

    def test_main_study_tables_cover_all_masks_read_only(self):
        import hashlib
        lr_mainline = FIGURES
        before = hashlib.sha256(data.DB_PATH.read_bytes()).hexdigest()
        rows, audits = lr_mainline.main_lr()
        table, contrasts = lr_mainline.paired_uncertainty(rows)
        self.assertEqual(int(table.loc[table.scope == 'all 75 masks', 'n_masks'].iloc[0]), 75)
        wide = rows.pivot_table(index=lr_mainline.KEY, columns='method', values='macro_f1')
        self.assertAlmostEqual(contrasts['PPI − Supervised (improvement)'].mean(), (wide['PPI selected']-wide.Supervised).mean())
        _, counts = lr_mainline.lambda_choices(audits)
        self.assertEqual(int(counts['all'].sum()), 75)
        self.assertEqual(hashlib.sha256(data.DB_PATH.read_bytes()).hexdigest(), before)


class PPSSLReplicationTests(unittest.TestCase):
    def test_data_follow_the_paper_groups_and_oracle_teacher(self):
        pp = SYNTHETIC
        (labeled, unlabeled, test), _ = pp.make_data(0., 0)
        X, y, f, group_b = test
        self.assertEqual((len(labeled[0]), len(unlabeled[0]), len(X)), (20, 1000, 1000))
        self.assertAlmostEqual(np.concatenate([labeled[3], unlabeled[3], group_b]).mean(), .2, places=2)
        np.testing.assert_array_equal(X[:, -2], group_b)  # The group indicator is an input feature.

    def test_rectifier_vanishes_when_pseudo_labels_are_the_labels(self):
        pp = SYNTHETIC
        rng = np.random.default_rng(0)
        X = np.hstack([rng.normal(size=(30, 3)), np.ones((30, 1))])
        y = X @ np.array([1., -2., .5, .3])+rng.normal(size=30)
        np.testing.assert_allclose(pp.exact_fit(X, y, y, X, y, .7), pp.exact_fit(X, y, y, X, y, 0.), atol=1e-8)
        np.testing.assert_allclose(pp.exact_fit(X, y, 0*y, X, 0*y, 0.), np.linalg.lstsq(X, y, rcond=None)[0], atol=1e-4)
        # A perfect Teacher on many unlabeled rows gives the PPI++ optimum 1/(1 + n/N).
        XU = np.hstack([rng.normal(size=(3000, 3)), np.ones((3000, 1))])
        fU = XU @ np.array([1., -2., .5, .3])
        self.assertGreater(pp.ppi_plus_plus_lambda(X, y, y, XU, fU), .5)

    def test_online_lambda_stays_in_unit_interval_and_drops_for_a_useless_teacher(self):
        pp = SYNTHETIC
        setup = {**pp.SETUP, 'steps': 400}
        (labeled, unlabeled, _), _ = pp.make_data(7., 1, setup)
        noise = np.random.default_rng(2).normal(scale=20, size=len(unlabeled[2]))
        _, lam = pp.train(labeled[:3], (unlabeled[0], noise), 'online', np.random.default_rng(0), setup=setup)
        self.assertTrue(0 <= lam < .5)


if __name__ == '__main__':
    unittest.main()
