# Step 1: Read Every Study Read-Only; Figures Are the Only Written Output

import json
import sqlite3

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ablations import A2_GRID, A3_LABEL, ABLATION_DB
from config import DEFAULT
from experiment import METHODS
from model_utils import DB_PATH, RESULTS_DIR
from teacher_study import cv_selected_teacher, load as load_teacher_study

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


# Step 3: Figure 1–2 — Main Study Learning Curves, One Figure per Student

def learning_curves(main, reference, audits):
    teacher = pd.DataFrame([dict(mechanism=m, budget=b, seed=s, accuracy=a['teacher_validation']['accuracy'],
                                 macro_f1=a['teacher_validation']['macro_f1']) for (m, b, s), a in audits.items()])
    for student in DEFAULT.student_models:
        fig, axes = six_panel(['Validation Macro F1', 'Validation Accuracy'])
        for row, metric in enumerate(['macro_f1', 'accuracy']):
            for column, mechanism in enumerate(MECHANISMS):
                ax = axes[row, column]
                for method, color, marker in zip(METHODS, [BLUE, ORANGE, AQUA], ['o', 's', '^']):
                    band(ax, by_case(main.loc[(main.student == student) & (main.training_method == method)], mechanism, metric),
                         method, color, marker)
                band(ax, by_case(teacher, mechanism, metric), 'CatBoost Teacher (cross-fit)', TEACHER_GRAY, 'D', ':')
                ax.axhline(reference[student][metric], color=INK, linestyle='--', linewidth=1.4,
                           label=f'Fully labeled {student} (n={reference[student]["labeled_count"]:,})')
        finish(fig, axes, f'{student} Student: validation performance vs visible labels '
               '(mean ± SD over 5 masks; MAR uses IPW; MNAR is an uncorrected stress test)',
               f'learning_curves_{student.lower()}.png')


# Step 4: Figure 3 — Are the Teachers Better Than the Student They Teach?

def teacher_quality(main, audits, study_audits, extra):
    catboost = pd.DataFrame([dict(mechanism=m, budget=b, seed=s, macro_f1=a['teacher_validation']['macro_f1'])
                             for (m, b, s), a in audits.items()])
    refit = extra.loc[extra.method == 'Teacher / Refit on visible labels']
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
    finish(fig, axes, 'Teacher quality: each Teacher vs the Student trained Supervised '
           '(validation Macro F1; mean ± SD over 5 masks)', 'teacher_quality.png')


# Step 5: Figure 4 — Main Result: Gain Over Supervised on the Same Mask

def ppi_gains(rows, split='validation', name='ppi_gains.png'):
    rows = rows.loc[rows.design == 'labeled'] if 'design' in rows else rows
    wide = rows.pivot_table(index=['teacher', 'student', 'mechanism', 'budget', 'seed'], columns='method', values='macro_f1')
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
    finish(fig, axes, f'Gain over the same Student trained Supervised on the same mask '
           f'(above 0 = better; {split} Macro F1; mean ± SD over 5 masks)', name)


# Step 6: Figure 5 — Does Choosing λ Matter? (A1)

def fixed_lambda(main, extra):
    fixed = extra.loc[extra.method.str.endswith('/ PPI fixed λ=1') & ~extra.method.str.contains(r'\[')]
    fixed = fixed.assign(student=fixed.method.str.split(' / ').str[0], training_method='PPI fixed λ=1')
    wide = pd.concat([main, fixed], ignore_index=True).pivot_table(
        index=['student', 'mechanism', 'budget', 'seed'], columns='training_method', values='macro_f1')
    fig, axes = six_panel([f'{s} Student\nMacro F1 − Supervised' for s in DEFAULT.student_models])
    for row, student in enumerate(DEFAULT.student_models):
        for column, mechanism in enumerate(MECHANISMS):
            ax = axes[row, column]
            w = wide.loc[(student, mechanism)]
            zero_line(ax)
            band(ax, w['PPI selected']-w.Supervised, 'PPI, λ chosen by training CV', AQUA, '^')
            band(ax, w['PPI fixed λ=1']-w.Supervised, 'PPI, λ fixed at 1', AQUA, 'D', ':')
    finish(fig, axes, 'Choosing λ: PPI gain over Supervised with λ chosen by training CV vs fixed at 1 '
           '(CatBoost Teacher; mean ± SD over 5 masks)', 'fixed_lambda.png')


# Step 7: Figure 6 — XGBoost Sensitivity on the Pre-Declared Subset (A2 Rounds, A3 Capacity)

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


# Step 8: Print the Key Numbers (Console Only; Nothing Is Written Besides the Figures)

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
    ppi_gains(study_rows)
    fixed_lambda(main_rows, extra)
    xgboost_sensitivity(main_rows, extra, a2, a3_reference)
    print_summary(main_rows, reference, extra, a1, a2, a3_reference, study_rows, study_audits)
    from improved import IMPROVED_DB
    if IMPROVED_DB.exists():
        improved_outputs()


# Step 9: Final Test Evaluation (Written Only After 06_final_test.py Has Run Once)

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



# Step 10: Improved Pipeline (improved_v1) — Same Six-Panel Template, Validation Only

def improved_outputs():
    from improved import IMPROVED_DB, STUDENTS
    db = read_only(IMPROVED_DB)
    try:
        rows = pd.read_sql_query('SELECT mechanism,budget,seed,method,macro_f1 FROM metrics', db)
        audits = [json.loads(a) for (a,) in db.execute("SELECT audit FROM cases WHERE status='complete'")]
    finally:
        db.close()
    wide = rows.pivot_table(index=['mechanism', 'budget', 'seed'], columns='method', values='macro_f1')
    fig, axes = six_panel([f'{s}\nMacro F1 − Supervised' for s in STUDENTS])
    for row, student in enumerate(STUDENTS):
        sup = wide[f'{student} / Supervised + threshold']
        for column, mechanism in enumerate(MECHANISMS):
            ax = axes[row, column]
            zero_line(ax)
            for method, color, marker, style in [('SSL soft', ORANGE, 's', '-'), ('PPI CV λ', AQUA, '^', '-'),
                                                 ('PPI power λ', AQUA, 'v', '--')]:
                d = (wide[f'{student} / {method} + threshold']-sup).loc[mechanism]
                band(ax, d, f'{method} (Macro-F1 threshold)', color, marker, style)
    finish(fig, axes, 'Improved pipeline: gain over Supervised on the same mask (selected Teacher, 5-fold, threshold rule; '
           'validation; mean ± SD over 15 masks)', 'improved_gains.png')

    print('\n=== Improved pipeline (improved_v1; validation only; 15 seeds) ===')
    teachers = pd.Series([a['teacher']['teacher'] for a in audits]).value_counts()
    print('Selected Teacher (final, per case): '+', '.join(f'{k} {v}' for k, v in teachers.items()))
    for student in STUDENTS:
        lam = pd.DataFrame([(a['n_labeled'], a['students'][student]['power_lambda']) for a in audits], columns=['budget', 'lam'])
        print(f'\n{student}: power λ mean by budget '+' / '.join(f'{b}:{v:.2f}' for b, v in lam.groupby('budget').lam.mean().items()))
        for rule in ['', ' + threshold']:
            s = wide[f'{student} / Supervised{rule}']
            label = 'threshold' if rule else 'argmax'
            print(f'  [{label}] Supervised {s.mean():.4f} | '+' | '.join(
                f'{m}−Sup {pooled(wide[f"{student} / {m}{rule}"]-s)}' for m in ['SSL soft', 'PPI CV λ', 'PPI power λ']))
            by = (wide[f'{student} / PPI power λ{rule}']-s).groupby(level='budget').mean()
            print(f'    PPI power λ−Sup by budget: '+' / '.join(f'{by[b]:+.4f}' for b in BUDGETS))
        t = wide[f'{student} / Supervised + threshold']-wide[f'{student} / Supervised']
        print(f'  Threshold rule effect on Supervised: {pooled(t)}')
