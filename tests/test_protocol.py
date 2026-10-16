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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'/'core'))
from config import DEFAULT
import ablations
import experiment
import masking
import model_utils as data
import selection
import ssl_core as core
import storage

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
        from tree_student import signed_derivatives
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
        from tree_student import tree_parameters
        config = replace(FAST, student_model='XGBoost', tree_iterations=20)
        fitted, _ = core.fit_student(self.XL, self.y, self.XU, self.qL, self.qU, config=config)
        matrix = xgb.DMatrix(self.XL, label=self.y, nthread=1)
        reference = xgb.train(tree_parameters(config, 4), matrix, num_boost_round=config.tree_iterations)
        np.testing.assert_allclose(core.probabilities(self.XL, fitted), reference.predict(matrix), atol=2e-6)

    def test_tree_ipw_matches_native_weighted_xgboost(self):
        import xgboost as xgb
        from tree_student import tree_parameters
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
        from tree_student import fit_tree
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

if __name__ == '__main__':
    unittest.main()
