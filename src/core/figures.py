# Step 1: Read Every Study Read-Only; Figures and Analysis Tables Are the Only Written Outputs

import json
import sqlite3
from itertools import product

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ablations import A2_GRID, A3_LABEL, ABLATION_DB
from config import DEFAULT
from experiment import METHODS
from model_utils import CLASSES, DB_PATH, RESULTS_DIR
from teacher_study import TABPFN_GRID, cv_selected_teacher, load as load_teacher_study, load_tabpfn

# Reference categorical slots 1–3 (blue, orange, aqua) validate all-pairs for color-vision deficiency.
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, MUTED, TEACHER_GRAY = '#0b0b0b', '#52514e', '#8a8984'
BUDGETS = list(DEFAULT.budgets)
plt.rcParams.update({'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10, 'axes.titleweight': 'bold'})


def read_only(path):
    return sqlite3.connect(f'file:{path}?mode=ro', uri=True)


def load_main():
    db = read_only(DB_PATH)
    try:
        rows = pd.read_sql_query('SELECT * FROM metrics', db)
        reference = json.loads(db.execute('SELECT value FROM reference').fetchone()[0])
        audits = {(m, b, s): json.loads(a) for m, b, s, a in
                  db.execute("SELECT mechanism,budget,seed,audit FROM cases WHERE status='complete'")}
    finally:
        db.close()
    rows[['student', 'training_method']] = rows.method.str.split(' / ', expand=True)
    return rows, reference, audits


def load_ablations():
    db = read_only(ABLATION_DB)
    try:
        rows = pd.read_sql_query('SELECT * FROM metrics', db)
        a1 = {(m, b, s): json.loads(a) for m, b, s, a in
              db.execute("SELECT mechanism,budget,seed,audit FROM cases WHERE status='complete'")}
        a2 = {(m, b, s): json.loads(a) for m, b, s, a in
              db.execute("SELECT mechanism,budget,seed,audit FROM a2_cases WHERE status='complete'")}
        a3_reference = json.loads(db.execute("SELECT value FROM metadata WHERE key='a3_reference'").fetchone()[0])
    finally:
        db.close()
    return rows, a1, a2, a3_reference


# Step 2: One Shared Six-Panel Template (Columns = Mechanisms, x = Visible Labels, Bands = ±1 SD Over 5 Masks)

MECHANISMS = list(DEFAULT.mechanisms)
VIOLET = '#6250d6'


def six_panel(row_labels):
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.6), sharex=True, sharey='row', layout='constrained')
    for row, label in enumerate(row_labels):
        for column, mechanism in enumerate(MECHANISMS):
            ax = axes[row, column]
            ax.set_xscale('log')
            ax.set_xticks(BUDGETS, [str(b) for b in BUDGETS])
            ax.set_xlim(BUDGETS[0]/1.2, BUDGETS[-1]*1.2)  # Same x range in every figure.
            ax.minorticks_off()
            ax.grid(alpha=.18, linewidth=.8)
            if row == 0:
                ax.set_title(mechanism)
            else:
                ax.set_xlabel('Visible training labels')
            if column == 0:
                ax.set_ylabel(label)
    return fig, axes


def band(ax, values, label, color, marker, linestyle='-'):
    """values: Series indexed by (budget, seed); draws the mean over masks with a ±1 SD band."""
    grouped = values.groupby(level='budget')
    mean, sd = grouped.mean().reindex(BUDGETS), grouped.std().reindex(BUDGETS)
    keep = mean.notna().to_numpy()
    x = np.array(BUDGETS)[keep]
    ax.plot(x, mean[keep], color=color, marker=marker, linestyle=linestyle, linewidth=2, markersize=6, label=label)
    ax.fill_between(x, (mean-sd)[keep], (mean+sd)[keep], color=color, alpha=.10, linewidth=0)


def zero_line(ax):
    ax.axhline(0, color=INK, linewidth=1.2, label='Supervised (difference = 0)')


def finish(fig, axes, title, name):
    handles = {}
    for ax in axes.flat:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(label, handle)
    ncol = len(handles) if len(handles) <= 5 else (len(handles)+1)//2
    fig.legend(handles.values(), handles.keys(), loc='outside lower center', ncol=ncol, frameon=False)
    fig.suptitle(title, fontsize=12)
    save(fig, name)


def save(fig, name):
    fig.savefig(RESULTS_DIR/name, dpi=180, facecolor='white')
    plt.close(fig)
    print(f'Saved: results/{name}', flush=True)


def by_case(rows, mechanism, column='macro_f1'):
    return rows.loc[rows.mechanism == mechanism].set_index(['budget', 'seed'])[column]


# Step 3: Figure 1 — Main Study Learning Curves (Macro F1; LR Top, XGBoost Bottom)

def learning_curves(main, reference, audits):
    teacher = pd.DataFrame([dict(mechanism=m, budget=b, seed=s, macro_f1=a['teacher_validation']['macro_f1'])
                            for (m, b, s), a in audits.items()])
    fig, axes = six_panel([f'{s} Student\nValidation Macro F1' for s in DEFAULT.student_models])
    for row, student in enumerate(DEFAULT.student_models):
        for column, mechanism in enumerate(MECHANISMS):
            ax = axes[row, column]
            for method, color, marker in zip(METHODS, [BLUE, ORANGE, AQUA], ['o', 's', '^']):
                band(ax, by_case(main.loc[(main.student == student) & (main.training_method == method)], mechanism),
                     method, color, marker)
            band(ax, by_case(teacher, mechanism), 'CatBoost Teacher (cross-fit)', TEACHER_GRAY, 'D', ':')
            ax.axhline(reference[student]['macro_f1'], color=INK, linestyle='--', linewidth=1.4,
                       label='Fully labeled Student (same architecture)')
    finish(fig, axes, 'Validation Macro F1 vs visible labels (mean ± SD over 5 masks; MAR uses IPW; '
           'MNAR is an uncorrected stress test)', 'learning_curves.png')


# Step 4: Figure 2 — Are the Teachers Better Than the Student They Teach?

def teacher_quality(main, audits, study_audits, extra):
    catboost = pd.DataFrame([dict(mechanism=m, budget=b, seed=s, macro_f1=a['teacher_validation']['macro_f1'])
                             for (m, b, s), a in audits.items()])
    refit = extra.loc[extra.method == 'Teacher / Refit on visible labels']
    tabpfn = tabpfn_teacher_scores()  # Exploratory pilot (MCAR, 100 and 500 labels); empty until it has run.
    fig, axes = six_panel([f'{s} Student\nValidation Macro F1' for s in DEFAULT.student_models])
    for row, student in enumerate(DEFAULT.student_models):
        same = pd.DataFrame([dict(mechanism=m, budget=b, seed=s, macro_f1=a['teacher_validation']['macro_f1'])
                             for (v, m, b, s), a in study_audits.items() if v == f'{student}→{student} [labeled]'])
        supervised = main.loc[(main.student == student) & (main.training_method == 'Supervised')]
        for column, mechanism in enumerate(MECHANISMS):
            ax = axes[row, column]
            band(ax, by_case(supervised, mechanism), 'Student trained Supervised', BLUE, 'o')
            band(ax, by_case(catboost, mechanism), 'CatBoost Teacher (cross-fit, used for pseudo-labels)', TEACHER_GRAY, 'D', ':')
            band(ax, by_case(refit, mechanism), 'CatBoost Teacher refit on all visible labels', TEACHER_GRAY, 'D', '--')
            band(ax, by_case(same, mechanism), 'Same-architecture Teacher (cross-fit)', VIOLET, 'v', '-.')
            if student == 'LR' and mechanism in TABPFN_GRID['mechanisms'] and not tabpfn.empty:
                band(ax, by_case(tabpfn, mechanism), 'TabPFN-3.5 Teacher (pretrained; cross-fit)', TEACHER_GRAY, 'X', '-')
                band(ax, by_case(tabpfn, mechanism, 'alone'), 'TabPFN-3.5 alone (all visible labels as context)',
                     TEACHER_GRAY, 'P', (0, (1, 1)))
    finish(fig, axes, 'Teacher quality: each Teacher vs the Student trained Supervised '
           '(validation Macro F1; mean ± SD over 5 masks)', 'teacher_quality.png')


def complete_tabpfn():
    """The TabPFN pilot's rows and audits, but only once every declared case is complete (never partial results)."""
    rows, audits = load_tabpfn()
    keys = set(product(TABPFN_GRID['mechanisms'], TABPFN_GRID['budgets'], TABPFN_GRID['seeds']))
    return (rows, audits) if keys <= set(audits) else (pd.DataFrame(), {})


def tabpfn_teacher_scores():
    _, audits = complete_tabpfn()
    return pd.DataFrame([dict(mechanism=m, budget=b, seed=s, macro_f1=a['teacher_validation']['macro_f1'],
                              alone=a['tabpfn_refit_validation']['macro_f1']) for (m, b, s), a in audits.items()])


# Step 5: Figure 3 — Main Result: Gain Over Supervised on the Same Mask

def ppi_gains(rows, split='validation', name='ppi_gains.png', extra=None):
    rows = rows.loc[rows.design == 'labeled'] if 'design' in rows else rows
    wide = rows.pivot_table(index=['teacher', 'student', 'mechanism', 'budget', 'seed'], columns='method', values='macro_f1')
    fixed = None if extra is None else extra.loc[extra.method.str.endswith('/ PPI fixed λ=1') & ~extra.method.str.contains(r'\[')]
    if fixed is not None:
        fixed = fixed.assign(student=fixed.method.str.split(' / ').str[0]).set_index(['student', 'mechanism', 'budget', 'seed']).macro_f1
    pilot, _ = complete_tabpfn()
    tabpfn = pd.DataFrame() if pilot.empty else pilot.pivot_table(index=['mechanism', 'budget', 'seed'], columns='method',
                                                                  values='macro_f1')
    fig, axes = six_panel([f'{s} Student\nMacro F1 − Supervised' for s in DEFAULT.student_models])
    for row, student in enumerate(DEFAULT.student_models):
        catboost, same = wide.loc[('CatBoost', student)], wide.loc[(student, student)]
        for column, mechanism in enumerate(MECHANISMS):
            ax = axes[row, column]
            zero_line(ax)
            c, s = catboost.loc[mechanism], same.loc[mechanism]
            band(ax, c['SSL soft']-c.Supervised, 'SSL soft (CatBoost Teacher)', ORANGE, 's')
            band(ax, c['PPI selected']-c.Supervised, 'PPI (CatBoost Teacher)', AQUA, '^')
            band(ax, s['PPI selected']-s.Supervised, 'PPI (same-architecture Teacher)', AQUA, 'v', '--')
            if fixed is not None:  # A1: CatBoost Teacher with λ fixed at 1 instead of chosen by training CV.
                f = fixed.loc[(student, mechanism)]
                band(ax, f-c.Supervised, 'PPI, λ fixed at 1 (CatBoost Teacher, A1)', AQUA, 'D', '-.')
            if split == 'validation' and student == 'LR' and mechanism in TABPFN_GRID['mechanisms'] and not tabpfn.empty:
                t = tabpfn.loc[mechanism]
                band(ax, t['SSL soft']-t.Supervised, 'SSL soft (TabPFN-3.5 Teacher, pilot)', ORANGE, 'X', ':')
                band(ax, t['PPI selected']-t.Supervised, 'PPI (TabPFN-3.5 Teacher, pilot)', AQUA, 'X', ':')
    finish(fig, axes, f'Gain over the same Student trained Supervised on the same mask '
           f'(above 0 = better; {split} Macro F1; mean ± SD over 5 masks)', name)


# Step 6: Figure 4 — XGBoost Sensitivity on the Pre-Declared Subset (A2 Rounds, A3 Capacity)

def xgboost_sensitivity(main, extra, a2, a3_reference):
    fits = pd.DataFrame([dict(mechanism=m, budget=b, seed=s, rounds=int(r), sup=f['supervised_macro_f1'],
                              ppi=f['ppi_macro_f1'], negative=f['ppi_lambda_1']['empirical_risk_final'] < 0)
                         for (m, b, s), a in a2.items() for r, f in a['fits'].items()])
    large = extra.loc[extra.method == f'{A3_LABEL} / Supervised']
    small = main.loc[(main.method == 'XGBoost / Supervised') & main.budget.isin(A2_GRID['budgets'])]
    fig, axes = six_panel(['XGBoost PPI λ=1\nMacro F1 − Supervised', 'XGBoost Supervised\nValidation Macro F1'])
    for column, mechanism in enumerate(MECHANISMS):
        top, bottom = axes[0, column], axes[1, column]
        if mechanism not in A2_GRID['mechanisms']:
            for ax in (top, bottom):
                ax.text(.5, .5, 'Not run\n(pre-declared subset:\nMCAR and MAR at 100 and 500 labels)',
                        ha='center', va='center', color=MUTED, transform=ax.transAxes)
            continue
        zero_line(top)
        for rounds, linestyle, marker in [(100, '-', '^'), (300, '--', 'v')]:
            f = fits.loc[fits.rounds == rounds]
            band(top, by_case(f.assign(macro_f1=f.ppi-f.sup), mechanism),
                 f'PPI λ=1, {rounds} rounds (risk < 0 in {int(f.negative.sum())}/{len(f)})', AQUA, marker, linestyle)
        band(bottom, by_case(small, mechanism), 'Supervised, 100 rounds, depth 3 (main)', BLUE, 'o')
        band(bottom, by_case(fits.loc[fits.rounds == 300].assign(macro_f1=lambda d: d.sup), mechanism),
             'Supervised, 300 rounds, depth 3', BLUE, 's', '--')
        band(bottom, by_case(large, mechanism), 'Supervised, 300 rounds, depth 5', BLUE, 'D', ':')
        bottom.axhline(a3_reference['main']['macro_f1'], color=INK, linestyle='--', linewidth=1.4,
                       label='Fully labeled, main')
        bottom.axhline(a3_reference['large']['macro_f1'], color=INK, linestyle=':', linewidth=1.4,
                       label='Fully labeled, depth 5')
    finish(fig, axes, 'XGBoost sensitivity on the pre-declared subset: boosting rounds for PPI (top) and Student capacity '
           'for Supervised (bottom); mean ± SD over 5 masks', 'xgboost_sensitivity.png')


# Step 7: Print the Key Numbers (Console Only; Nothing Is Written Besides the Figures)

def pooled(d):
    return f'{d.mean():+.4f} ({int((d > 1e-12).sum())} better / {int((d < -1e-12).sum())} worse / {int((d.abs() <= 1e-12).sum())} equal)'


def print_summary(main, reference, extra, a1, a2, a3_reference, study_rows, study_audits):
    print('\n=== Key results (validation = development set; Test not evaluated) ===')
    print(f"Fully labeled reference Macro F1: LR {reference['LR']['macro_f1']:.4f} | XGBoost {reference['XGBoost']['macro_f1']:.4f} "
          f"(larger XGBoost {a3_reference['large']['macro_f1']:.4f})")
    wide = study_rows.pivot_table(index=['teacher', 'student', 'design', 'mechanism', 'budget', 'seed'],
                                  columns='method', values='macro_f1')
    print('\nPaired differences over 75 masks [Teacher → Student, preprocessing]:')
    for (teacher, student, design), w in wide.groupby(level=['teacher', 'student', 'design']):
        print(f'  {teacher}→{student} [{design}]  PPI−Sup {pooled(w["PPI selected"]-w.Supervised)} | '
              f'SSL−Sup {pooled(w["SSL soft"]-w.Supervised)} | PPI−SSL {pooled(w["PPI selected"]-w["SSL soft"])}')
    for design in ['labeled', 'labeled+unlabeled']:
        for student in DEFAULT.student_models:
            choice = cv_selected_teacher(study_rows, study_audits, student, design)
            print(f'  Teacher chosen by training CV, {student} [{design}]: PPI−Sup {pooled(choice.ppi-choice.supervised)}')
    fixed = extra.loc[extra.method.str.endswith('/ PPI fixed λ=1') & ~extra.method.str.contains(r'\[')]
    for student in DEFAULT.student_models:
        f = fixed.loc[fixed.method == f'{student} / PPI fixed λ=1'].set_index(['mechanism', 'budget', 'seed']).macro_f1
        m = main.loc[main.student == student].pivot_table(index=['mechanism', 'budget', 'seed'], columns='training_method', values='macro_f1')
        print(f'  {student} fixed λ=1: −Sup {pooled(f-m.Supervised)} | −SSL {pooled(f-m["SSL soft"])} | −PPI selected {pooled(f-m["PPI selected"])}')
    replay = sum(a['replay']['teacher'] == 'exact' for a in a1.values())
    print(f'\nReplay checks: Teacher {replay}/{len(a1)} exact; every A1/A2/A3 case reproduced its stored fits before running.')
    neg = {r: sum(a['fits'][str(r)]['ppi_lambda_1']['empirical_risk_final'] < 0 for a in a2.values()) for r in (100, 300)}
    print(f'A2: XGBoost PPI λ=1 training risk negative in {neg[100]}/{len(a2)} fits at 100 rounds vs {neg[300]}/{len(a2)} at 300.')


def main():
    main_rows, reference, audits = load_main()
    extra, a1, a2, a3_reference = load_ablations()
    study_rows, study_audits = load_teacher_study()
    learning_curves(main_rows, reference, audits)
    teacher_quality(main_rows, audits, study_audits, extra)
    ppi_gains(study_rows, extra=extra)
    xgboost_sensitivity(main_rows, extra, a2, a3_reference)
    print_summary(main_rows, reference, extra, a1, a2, a3_reference, study_rows, study_audits)
    improved_outputs()  # Draws only when every declared case is complete; otherwise reports progress.
    lr_mainline_outputs()  # Analysis of the stored LR main line; no retraining.
    tabpfn_summary()  # Exploratory TabPFN-3.5 Teacher pilot (MCAR, 100/500 labels), if it has run.
    positive_control_outputs()  # Synthetic studies draw only when complete.
    ppssl_outputs()


# Step 8: Final Test Evaluation (Written Only After 07_final_test.py Has Run Once)

def final_test_outputs():
    from final_test import FROZEN, load as load_test
    test, reference = load_test()
    ppi_gains(test, 'TEST', 'test_gains.png')
    validation, _ = load_teacher_study()
    print('\n=== Final Test evaluation (frozen protocol, run once) ===')
    print(f"Final method: {FROZEN['final_method']}")
    print(f"Fully labeled Test Macro F1: LR {reference['LR']['macro_f1']:.4f} | XGBoost {reference['XGBoost']['macro_f1']:.4f}")
    for split, rows in [('validation', validation.loc[validation.design == 'labeled']), ('TEST', test)]:
        wide = rows.pivot_table(index=['teacher', 'student', 'mechanism', 'budget', 'seed'], columns='method', values='macro_f1')
        print(f'\n{split}: paired differences over 75 masks')
        for (teacher, student), w in wide.groupby(level=['teacher', 'student']):
            by = (w['PPI selected']-w.Supervised).groupby(level='budget').mean()
            print(f'  {teacher}→{student}: Supervised {w.Supervised.mean():.4f} | PPI−Sup {pooled(w["PPI selected"]-w.Supervised)} | '
                  f'SSL−Sup {pooled(w["SSL soft"]-w.Supervised)} | PPI−SSL {pooled(w["PPI selected"]-w["SSL soft"])} | '
                  'PPI−Sup by budget '+' / '.join(f'{by[b]:+.4f}' for b in BUDGETS))



# Step 9: Exploratory Improved Pipeline — Same Six-Panel Template, Validation Only, Complete Runs Only

def improved_outputs():
    from improved import CONFIG, IMPROVED_DB, PROTOCOL, STUDENTS, is_complete
    if not is_complete():
        if IMPROVED_DB.exists():
            db = read_only(IMPROVED_DB)
            try:
                counts = db.execute("SELECT mechanism, budget, COUNT(*) FROM cases WHERE status='complete' "
                                    "GROUP BY 1, 2").fetchall()
            finally:
                db.close()
            print(f'Improved study incomplete ({sum(c for *_, c in counts)}/{len(CONFIG.mechanisms)*len(CONFIG.budgets)*len(CONFIG.seeds)} cases); no figure drawn. '
                  'Completed seeds per cell: '+', '.join(f'{m}/{b}: {c}' for m, b, c in counts), flush=True)
        return
    db = read_only(IMPROVED_DB)
    try:
        rows = pd.read_sql_query('SELECT mechanism,budget,seed,method,macro_f1 FROM metrics', db)
        audits = [json.loads(a) for (a,) in db.execute("SELECT audit FROM cases WHERE status='complete'")]
    finally:
        db.close()
    wide = rows.pivot_table(index=['mechanism', 'budget', 'seed'], columns='method', values='macro_f1')
    # One figure (argmax). The threshold-rule results are printed below rather than drawn as a second figure.
    for rule, name, label in [('', 'improved_gains.png', 'argmax decisions')]:
        fig, axes = six_panel([f'{s}\nMacro F1 − Supervised' for s in STUDENTS])
        for row, student in enumerate(STUDENTS):
            sup = wide[f'{student} / Supervised{rule}']
            for column, mechanism in enumerate(MECHANISMS):
                ax = axes[row, column]
                zero_line(ax)
                for method, color, marker, style in [('SSL soft', ORANGE, 's', '-'), ('PPI CV λ', AQUA, '^', '-'),
                                                     ('PPI heuristic λ', AQUA, 'v', '--')]:
                    band(ax, (wide[f'{student} / {method}{rule}']-sup).loc[mechanism], method, color, marker, style)
        finish(fig, axes, f'Exploratory {PROTOCOL["name"]}: gain over Supervised on the same mask ({label}; Ensemble '
               f'Teacher; validation; mean ± SD over {len(CONFIG.seeds)} masks)', name)

    print(f'\n=== Exploratory {PROTOCOL["name"]} (validation only; {len(CONFIG.seeds)} seeds) ===')
    for student in STUDENTS:
        lam = pd.DataFrame([(a['n_labeled'], a['students'][student]['heuristic_lambda']) for a in audits],
                           columns=['budget', 'lam'])
        print(f'\n{student}: heuristic λ mean by budget '+' / '.join(f'{b}:{v:.2f}' for b, v in lam.groupby('budget').lam.mean().items()))
        for rule in ['', ' + threshold']:
            s = wide[f'{student} / Supervised{rule}']
            label = 'threshold' if rule else 'argmax'
            print(f'  [{label}] Supervised {s.mean():.4f} | '+' | '.join(
                f'{m}−Sup {pooled(wide[f"{student} / {m}{rule}"]-s)}' for m in ['SSL soft', 'PPI CV λ', 'PPI heuristic λ']))
            for m in ['PPI CV λ', 'PPI heuristic λ']:
                by = (wide[f'{student} / {m}{rule}']-s).groupby(level='budget').mean()
                print(f'    {m}−Sup by budget: '+' / '.join(f'{by[b]:+.4f}' for b in BUDGETS))
        t = wide[f'{student} / Supervised + threshold']-wide[f'{student} / Supervised']
        print(f'  Threshold rule effect on Supervised: {pooled(t)}')


# Step 10: LR Main Line — Paired Uncertainty, λ Sources, Per-Class F1 and Ablations as Questions (Tables Only)

KEY = ['mechanism', 'budget', 'seed']
CAVEAT = ('All 75 masks share one train/validation split and one validation set; seeds vary only the mask (and the '
          'CV folds). Intervals describe mask-to-mask variability on this development set, not generalization.')


def main_lr():
    """LR main study (CatBoost Teacher, labeled-only preprocessing): rows with short method names, and audits."""
    rows, _, audits = load_main()
    rows = rows.loc[rows.student == 'LR'].drop(columns='method').rename(columns={'training_method': 'method'})
    return rows, audits


# Step 10a: LR Main Line — Paired Differences With SE and a Stratified Paired Bootstrap Over Masks

def stratified_bootstrap(d, draws=4000, seed=0):
    """Resample masks with replacement within each (mechanism, budget) cell, keeping the design balanced."""
    rng = np.random.default_rng(seed)
    cells = [g.to_numpy() for _, g in d.groupby(level=['mechanism', 'budget'])]
    total = sum(len(c) for c in cells)
    means = np.zeros(draws)
    for c in cells:
        means += c[rng.integers(0, len(c), (draws, len(c)))].sum(axis=1)
    return np.percentile(means/total, [2.5, 97.5])


def describe(d):
    lo, hi = stratified_bootstrap(d)
    return {'n_masks': len(d), 'mean': d.mean(), 'se': d.std(ddof=1)/np.sqrt(len(d)), 'boot_low': lo, 'boot_high': hi,
            'better': int((d > 1e-12).sum()), 'worse': int((d < -1e-12).sum()), 'equal': int((d.abs() <= 1e-12).sum())}


def scopes(d):
    yield 'all 75 masks', d
    for budget, g in d.groupby(level='budget'):
        yield f'budget {budget}', g
    for mechanism, g in d.groupby(level='mechanism'):
        yield mechanism, g


def paired_uncertainty(rows):
    wide = rows.pivot_table(index=KEY, columns='method', values='macro_f1')
    contrasts = {
        'PPI − Supervised (improvement)': wide['PPI selected']-wide.Supervised,
        'PPI − SSL soft (robustness relative to SSL)': wide['PPI selected']-wide['SSL soft'],
        'SSL soft − Supervised (SSL harm)': wide['SSL soft']-wide.Supervised,
    }
    table = [dict(contrast=name, scope=scope, **describe(g)) for name, d in contrasts.items() for scope, g in scopes(d)]
    return pd.DataFrame(table), contrasts


def harm_rates(rows):
    """Robustness (does PPI avoid SSL's harm?) kept separate from improvement (does PPI beat Supervised?)."""
    wide = rows.pivot_table(index=KEY, columns='method', values='macro_f1')
    ssl, ppi = wide['SSL soft']-wide.Supervised, wide['PPI selected']-wide.Supervised
    harmed = ssl < -1e-12
    return {'masks': len(wide),
            'SSL worse than Supervised': int(harmed.sum()),
            'PPI worse than Supervised': int((ppi < -1e-12).sum()),
            'of SSL-harmed masks, PPI also worse than Supervised': int((harmed & (ppi < -1e-12)).sum()),
            'mean SSL − Sup on SSL-harmed masks': float(ssl[harmed].mean()),
            'mean PPI − Sup on SSL-harmed masks': float(ppi[harmed].mean()),
            'PPI identical to Supervised (λ=0)': int((ppi.abs() <= 1e-12).sum())}


# Step 10b: Where λ = 0 Comes From — Chosen by CV, or Forced by the Small-Class Fallback

def lambda_choices(audits):
    rows = []
    for (m, b, s), audit in audits.items():
        entry = audit['students']['LR']
        lam = entry['ppi_selected_lambda']
        if lam != 0:
            source = 'λ > 0 chosen by CV'
        elif entry['selection_policy'].startswith('supervised_fallback'):
            source = 'λ = 0 forced (a class has < 2 visible labels; no CV)'
        elif 'ppi_final_feasibility_backoff' in entry:
            source = 'λ = 0 after feasibility back-off'
        else:
            source = 'λ = 0 chosen by CV (no candidate beat Supervised)'
        rows.append(dict(mechanism=m, budget=b, seed=s, lambda_value=lam, source=source))
    table = pd.DataFrame(rows)
    counts = table.pivot_table(index='source', columns='budget', values='seed', aggfunc='count', fill_value=0)
    counts['all'] = counts.sum(axis=1)
    return table, counts


# Step 10c: Per-Class F1 Differences (Which Classes Move?)

def per_class(rows):
    f1 = pd.DataFrame(rows.class_f1.map(json.loads).tolist(), columns=CLASSES, index=rows.index)
    long = pd.concat([rows[KEY+['method']], f1], axis=1).melt(id_vars=KEY+['method'], var_name='class', value_name='f1')
    wide = long.pivot_table(index=KEY+['class'], columns='method', values='f1')
    out = []
    for name, d in [('PPI − Supervised', wide['PPI selected']-wide.Supervised),
                    ('PPI − SSL soft', wide['PPI selected']-wide['SSL soft']),
                    ('SSL soft − Supervised', wide['SSL soft']-wide.Supervised)]:
        for cls in CLASSES:
            g = d.xs(cls, level='class')
            for scope, part in [('all budgets', g), ('budgets 50–200', g[g.index.get_level_values('budget') <= 200]),
                                ('budgets 500–1000', g[g.index.get_level_values('budget') >= 500])]:
                out.append(dict(contrast=name, cls=cls, scope=scope, **describe(part)))
    return pd.DataFrame(out).rename(columns={'cls': 'class'})


# Step 10d: Ablations as LR-Mainline Questions (Each Variant Compared With Its Own Supervised Fit)

def ablation_variants():
    """Paired per-mask series for each LR-mainline question, plus the Teacher's validation Macro F1."""
    study, study_audits = load_teacher_study()
    lr = study.loc[study.student == 'LR']
    wide = lr.pivot_table(index=['teacher', 'design']+KEY, columns='method', values='macro_f1').sort_index()
    teacher_f1 = pd.DataFrame([dict(teacher=v.split('→')[0], design=v.split('[')[1][:-1], mechanism=m, budget=b, seed=s,
                                    macro_f1=a['teacher_validation']['macro_f1'])
                               for (v, m, b, s), a in study_audits.items() if '→LR ' in v]
                              ).set_index(['teacher', 'design']+KEY).sort_index().macro_f1
    db = read_only(ABLATION_DB)
    try:
        fixed = pd.read_sql_query("SELECT mechanism,budget,seed,macro_f1 FROM metrics WHERE method='LR / PPI fixed λ=1'", db)
    finally:
        db.close()
    fixed = fixed.set_index(KEY).macro_f1
    main = wide.loc[('CatBoost', 'labeled')]
    choice = cv_selected_teacher(study, study_audits, 'LR', 'labeled').set_index(KEY)
    variants = [
        ('Q0 main: CatBoost Teacher, CV λ, labeled-only preprocessing', 'main', main['PPI selected']-main.Supervised,
         main['PPI selected']-main['SSL soft'], teacher_f1[('CatBoost', 'labeled')]),
        ('Q1 Teacher: LR Teacher instead of CatBoost (also changes Teacher quality)', 'teacher',
         wide.loc[('LR', 'labeled')]['PPI selected']-wide.loc[('LR', 'labeled')].Supervised,
         wide.loc[('LR', 'labeled')]['PPI selected']-wide.loc[('LR', 'labeled')]['SSL soft'], teacher_f1[('LR', 'labeled')]),
        ('Q2 λ: CatBoost Teacher, λ fixed at 1 (A1)', 'lambda', fixed-main.Supervised, fixed-main['SSL soft'],
         teacher_f1[('CatBoost', 'labeled')]),
        ('Q3 preprocessing: CatBoost Teacher, labeled+unlabeled features', 'design',
         wide.loc[('CatBoost', 'labeled+unlabeled')]['PPI selected']-wide.loc[('CatBoost', 'labeled+unlabeled')].Supervised,
         wide.loc[('CatBoost', 'labeled+unlabeled')]['PPI selected']-wide.loc[('CatBoost', 'labeled+unlabeled')]['SSL soft'],
         teacher_f1[('CatBoost', 'labeled+unlabeled')]),
        ('Q3 preprocessing: LR Teacher, labeled+unlabeled features', 'design',
         wide.loc[('LR', 'labeled+unlabeled')]['PPI selected']-wide.loc[('LR', 'labeled+unlabeled')].Supervised,
         wide.loc[('LR', 'labeled+unlabeled')]['PPI selected']-wide.loc[('LR', 'labeled+unlabeled')]['SSL soft'],
         teacher_f1[('LR', 'labeled+unlabeled')]),
        ('Q3 preprocessing effect on Supervised itself (labeled+unlabeled − labeled)', 'design',
         wide.loc[('CatBoost', 'labeled+unlabeled')].Supervised-main.Supervised, None, None),
        ('Q4 Teacher chosen per mask by training CV (CatBoost vs LR; candidate Test rule)', 'teacher',
         choice.ppi-choice.supervised, None, None),
    ]
    return variants, main.Supervised


def ablation_table():
    variants, supervised = ablation_variants()
    rows = []
    for name, group, gain, versus_ssl, teacher in variants:
        rows.append(dict(question=name, group=group, contrast='PPI − own Supervised' if 'Supervised itself' not in name
                         else 'Supervised(l+u) − Supervised(labeled)', **describe(gain.sort_index()),
                         ppi_minus_ssl=None if versus_ssl is None else float(versus_ssl.mean()),
                         teacher_macro_f1=None if teacher is None else float(teacher.mean()),
                         teacher_minus_supervised=None if teacher is None else float((teacher-supervised).mean())))
    return pd.DataFrame(rows)

def write_tables():
    """Two CSVs: paired uncertainty (overall rows, then per class) and the ablation questions. λ sources are printed."""
    rows, audits = main_lr()
    uncertainty, _ = paired_uncertainty(rows)
    classes = per_class(rows)
    uncertainty = pd.concat([uncertainty.assign(**{'class': 'all classes'}), classes], ignore_index=True)
    _, counts = lambda_choices(audits)
    outputs = {'lr_paired_uncertainty.csv': uncertainty, 'lr_ablation_questions.csv': ablation_table()}
    for name, table in outputs.items():
        table.to_csv(RESULTS_DIR/name, index=False, float_format='%.5f')
        print(f'Saved: results/{name}', flush=True)
    return outputs, counts, harm_rates(rows)


def lr_mainline_outputs():
    outputs, counts, harms = write_tables()
    print('\n=== LR main line: paired uncertainty (development set) ===')
    print(CAVEAT)
    table = outputs['lr_paired_uncertainty.csv']
    for _, r in table.loc[(table.scope == 'all 75 masks') & (table['class'] == 'all classes')].iterrows():
        print(f"  {r.contrast}: {r['mean']:+.4f} ± {r.se:.4f} SE, 95% bootstrap [{r.boot_low:+.4f}, {r.boot_high:+.4f}] "
              f"({r.better} better / {r.worse} worse / {r.equal} equal)")
    print('  Robustness vs improvement: '+'; '.join(f'{k}: {v:.4f}' if isinstance(v, float) else f'{k}: {v}' for k, v in harms.items()))
    print('  λ sources by budget:\n'+counts.to_string())
    for _, r in outputs['lr_ablation_questions.csv'].iterrows():
        print(f"  {r.question}: {r['mean']:+.4f} [{r.boot_low:+.4f}, {r.boot_high:+.4f}]")


# Step 11: Synthetic Positive Control — One Figure (Oracle and Weak-but-Unbiased Teachers)

def positive_control_outputs():
    from positive_control import METHODS as PC_METHODS, checks, is_complete, load as load_pc, paired
    if not is_complete():
        print('Positive control incomplete; no figure drawn.', flush=True)
        return
    rows, declaration = load_pc()
    styles = {'SSL soft': (ORANGE, 's', '-'), 'PPI CV λ': (AQUA, '^', '-'), 'PPI λ=1': (AQUA, 'D', ':')}
    # Oracle = the objective works; σ=3 = a Teacher worse than the Student but unbiased. The other conditions
    # (σ=1, uninformative, subgroup-biased) are in positive_control_checks.csv and the console summary.
    pair = ['oracle', 'noise σ=3']
    fig, axes = six_panel([f'{t} Teacher\nMacro F1 − Supervised' for t in pair])
    for row, teacher in enumerate(pair):
        for column, mechanism in enumerate(MECHANISMS):
            ax = axes[row, column]
            zero_line(ax)
            for method in PC_METHODS:
                band(ax, paired(rows, teacher, f'{method} − Supervised').loc[mechanism], method, *styles[method])
    finish(fig, axes, 'Synthetic positive control: oracle Teacher (top) and an unbiased Teacher weaker than the Student (bottom)'
           f'\nLR Student, gain over Supervised on the same mask (mean ± SD over {len(declaration["grid"]["seeds"])} '
           'independent replications)', 'positive_control.png')
    table = checks(rows, declaration)
    table.to_csv(RESULTS_DIR/'positive_control_checks.csv', index=False, float_format='%.5f')
    print('Saved: results/positive_control_checks.csv')
    reference = rows.loc[rows.method.isin(['Teacher', 'Supervised', 'Bayes'])]
    quality = reference.pivot_table(index=['teacher', 'method'], columns='budget', values='macro_f1')
    print('\n=== Synthetic positive control (pre-declared expectations) ===')
    print('Validation Macro F1 by visible labels (Teacher and Bayes do not depend on the budget):')
    print(quality.round(3).to_string())
    for _, r in table.iterrows():
        print(f"  {r['id']} [{r.teacher}] {r.contrast} ({r.metric}, {r.scope}) {r.rule}: {r.estimate:+.4f} "
              f"[{r.ci_low:+.4f}, {r.ci_high:+.4f}] → {r.verdict}")


# Step 12: PP-SSL Anchor — Same 2 × 3 Grid and Colors, Columns Follow the Paper (All / Group A / Group B)

def ppssl_outputs():
    from ppssl_replication import BOUNDARY, SETUP, check_targets, is_complete, load as load_ppssl, with_best_fixed
    if not is_complete():
        print('PP-SSL replication incomplete; no figure drawn.', flush=True)
        return
    runs, best = with_best_fixed(load_ppssl())
    mean = runs.groupby(['method', 'mu'])[['mse', 'mse_a', 'mse_b', 'lam']].mean()
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.6), layout='constrained')
    styles = [('Only Labeled', BLUE, 'o', '-'), ('Classic SSL', ORANGE, 's', '-'), ('Teacher', TEACHER_GRAY, 'D', ':'),
              ('PPI++ (offline λ)', AQUA, 'v', '--'), ('PP-SSL (online λ)', AQUA, '^', '-'),
              ('PPI, λ by training CV (this project)', AQUA, 'P', ':')]
    for column, (metric, title) in enumerate([('mse', 'All test rows'), ('mse_a', 'Group A (Teacher = oracle)'),
                                              ('mse_b', 'Group B (Teacher biased by −μ)')]):
        ax = axes[0, column]
        for method, color, marker, style in styles:
            values = mean.loc[method, metric]
            ax.plot(values.index, values, color=color, marker=marker, linestyle=style, linewidth=2, label=method)
        ax.set_yscale('log')
        ax.set_title(title)
        ax.set_xlabel('Teacher bias μ in group B')
        ax.grid(alpha=.18)
    axes[0, 0].set_ylabel('Test MSE (log scale)\nmean over 100 runs')
    ax = axes[1, 0]
    for method, color, marker, style in styles[3:]:
        values = mean.loc[method, 'lam']
        ax.plot(values.index, values, color=color, marker=marker, linestyle=style, linewidth=2, label=method)
    ax.plot(best.index, [float(b.split('=')[1]) for b in best], color=INK, marker='*', linestyle='', markersize=10,
            label='best fixed λ (infeasible oracle)')
    ax.set(xlabel='Teacher bias μ in group B', ylabel='λ used (mean)', ylim=(-.03, 1.03))
    ax.grid(alpha=.18)
    fixed = runs.loc[runs.method.str.startswith('fixed λ=')].assign(fixed=lambda d: d.method.str.split('=').str[1].astype(float))
    for ax, mu in zip(axes[1, 1:], [1., 5.]):
        curve = fixed.loc[fixed.mu == mu].groupby('fixed').mse.mean()
        ax.plot(curve.index, curve, color=AQUA, marker='o', linestyle=':', linewidth=2, label='PPI gradient, fixed λ')
        ax.axhline(mean.loc[('PP-SSL (online λ)', mu), 'mse'], color=AQUA, linewidth=2, label='PP-SSL (online λ)')
        ax.axhline(mean.loc[('Only Labeled', mu), 'mse'], color=BLUE, linewidth=1.4, label='Only Labeled')
        ax.set(title=f'Fixed-λ sweep at μ = {mu:g}', xlabel='fixed λ', ylabel='Test MSE (all rows)')
        ax.grid(alpha=.18)
    finish(fig, axes, f"PP-SSL anchor (Shoham et al., NeurIPS 2025): two-group synthetic regression, n = {SETUP['n']}, "
           f"N = {SETUP['N']}; linear model, Adam lr {SETUP['adam_lr']:g}, {SETUP['steps']:,} steps", 'ppssl_replication.png')
    table = check_targets(runs.loc[runs.method != 'best fixed λ'])
    table.to_csv(RESULTS_DIR/'ppssl_replication_checks.csv', index=False, float_format='%.5f')
    print('Saved: results/ppssl_replication_checks.csv')
    print('\n=== PP-SSL anchor: pre-declared reproduction targets ===')
    print(mean[['mse', 'mse_a', 'mse_b', 'lam']].unstack('mu').round(3).to_string())
    for _, r in table.iterrows():
        print(f"  {r['id']} μ={r.mu}: {r.contrast} ({r.metric}) {r['diff']:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}] "
              f"{'ok' if r.ok else 'no'} → {r.verdict}")
    for key, text in BOUNDARY.items():
        print(f'  {key}: {text}')


# Step 13: Exploratory TabPFN-3.5 Teacher Pilot — Paired With the CatBoost Teacher on the Same Masks

def tabpfn_summary():
    from teacher_study import TABPFN_DB, TABPFN_GRID
    rows, audits = load_tabpfn()
    keys = {(m, b, s) for m in TABPFN_GRID['mechanisms'] for b in TABPFN_GRID['budgets'] for s in TABPFN_GRID['seeds']}
    if not keys <= set(audits):
        print(f'TabPFN Teacher pilot incomplete ({len(audits)}/{len(keys)} cases); no summary.', flush=True)
        return
    main_rows, _, main_audits = load_main()
    lr = main_rows.loc[main_rows.student == 'LR'].pivot_table(index=['mechanism', 'budget', 'seed'],
                                                              columns='training_method', values='macro_f1')
    tab = rows.pivot_table(index=['mechanism', 'budget', 'seed'], columns='method', values='macro_f1')
    lam = rows.loc[rows.method == 'PPI selected'].set_index(['mechanism', 'budget', 'seed']).lambda_value
    table = []
    for key in sorted(keys):
        a = audits[key]
        table.append(dict(zip(['mechanism', 'budget', 'seed'], key),
                          supervised=tab.loc[key, 'Supervised'],
                          tabpfn_alone=a['tabpfn_refit_validation']['macro_f1'],
                          tabpfn_teacher_crossfit=a['teacher_validation']['macro_f1'],
                          catboost_teacher_crossfit=main_audits[key]['teacher_validation']['macro_f1'],
                          ssl_tabpfn_minus_sup=tab.loc[key, 'SSL soft']-tab.loc[key, 'Supervised'],
                          ppi_tabpfn_minus_sup=tab.loc[key, 'PPI selected']-tab.loc[key, 'Supervised'],
                          ssl_catboost_minus_sup=lr.loc[key, 'SSL soft']-lr.loc[key, 'Supervised'],
                          ppi_catboost_minus_sup=lr.loc[key, 'PPI selected']-lr.loc[key, 'Supervised'],
                          ppi_tabpfn_lambda=lam.loc[key], seconds=a['elapsed_seconds']))
    table = pd.DataFrame(table)
    table.to_csv(RESULTS_DIR/'tabpfn_teacher_summary.csv', index=False, float_format='%.5f')
    print('Saved: results/tabpfn_teacher_summary.csv')
    print('\n=== Exploratory TabPFN-3.5 Teacher pilot (MCAR; validation; 5 masks per budget; descriptive) ===')
    summary = table.drop(columns=['mechanism', 'seed']).groupby('budget').agg(['mean', 'std'])
    print(summary.round(4).T.to_string())
