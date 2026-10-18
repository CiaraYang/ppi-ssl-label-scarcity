# Step 6: All Figures and Tables From the Saved Databases (Reads Only)

import json
import sqlite3
import sys
from importlib import import_module
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402  (after choosing the non-interactive backend)
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))  # The numbered files import each other by name.
common = import_module('00_common')
CLASSES, DB_PATH, DEFAULT, RESULTS_DIR = common.CLASSES, common.DB_PATH, common.DEFAULT, common.RESULTS_DIR
methods = import_module('00_methods')
METHODS = methods.METHODS
_ablations = import_module('02_ablations')
A2_GRID, A3_LABEL, ABLATION_DB = _ablations.A2_GRID, _ablations.A3_LABEL, _ablations.ABLATION_DB
_teacher_study = import_module('03_teacher_study')
TABPFN_GRID, cv_selected_teacher = _teacher_study.TABPFN_GRID, _teacher_study.cv_selected_teacher
load_teacher_study, load_tabpfn = _teacher_study.load, _teacher_study.load_tabpfn


# ===== Read Every Study Read-Only; Figures and Analysis Tables Are the Only Written Outputs =====

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


def right_row_label(ax, label):
    # Step 2a: Put every Student / Teacher row label vertically on the outer right edge.
    ax.text(1.065, .5, label, transform=ax.transAxes, rotation=270,
            ha='center', va='center', fontsize=13, fontweight='bold', clip_on=False, gid='row-label')


def six_panel(row_labels, mechanisms=None):
    """2 × 3 grid (MCAR / MAR / MNAR); a study run on fewer mechanisms gets only those columns."""
    mechanisms = MECHANISMS if mechanisms is None else mechanisms
    fig, axes = plt.subplots(2, len(mechanisms), figsize=(14*len(mechanisms)/3+(1 if len(mechanisms) < 3 else 0), 7.6),
                             sharex=True, sharey='row', layout='constrained')
    for row, label in enumerate(row_labels):
        row_name = None
        if '\n' in label:
            row_name, label = label.split('\n', 1)
            row_name = {'LR': 'LR Student', 'XGBoost leaf-refit': 'XGBoost leaf-refit Student',
                        'oracle Teacher': 'Oracle Teacher', 'noise σ=3 Teacher': 'Noisy Teacher (σ = 3)'}.get(row_name, row_name)
        label = label.replace('Validation Macro F1', 'Macro F1')
        for column, mechanism in enumerate(mechanisms):
            ax = axes[row, column]
            ax.set_xscale('log')
            ax.set_xticks(BUDGETS, [str(b) for b in BUDGETS])
            ax.set_xlim(BUDGETS[0]/1.2, BUDGETS[-1]*1.2)  # Same x range in every figure.
            ax.minorticks_off()
            ax.grid(alpha=.18, linewidth=.8)
            if row == 0:
                ax.set_title(mechanism)
            else:
                ax.set_xlabel('Visible training labels (n)')
            if column == 0:
                ax.set_ylabel(label, fontsize=12)
        if row_name:
            right_row_label(axes[row, -1], row_name)
    return fig, axes


def band(ax, values, label, color, marker, linestyle='-'):
    """values: Series indexed by (budget, seed); draws the mean over masks as a line (no shaded band; the spread is
    in the CSV tables). The final color is assigned per label by finish()."""
    mean = values.groupby(level='budget').mean().reindex(BUDGETS)
    keep = mean.notna().to_numpy()
    ax.plot(np.array(BUDGETS)[keep], mean[keep], color=color, marker=marker, linestyle=linestyle, linewidth=2,
            markersize=7, label=label)


def zero_line(ax):
    ax.axhline(0, color=INK, linewidth=1.2, label='Supervised (difference = 0)')


# Validated categorical order (dataviz reference palette; passes adjacent-pair CVD and normal-vision checks on a light
# surface). Every line label in a figure gets its own slot, in order of first appearance; all lines are solid, markers
# stay as a second cue, and black/gray reference lines keep their color.
PALETTE = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']


def recolor(axes):
    slots = {}
    method_colors = {'Supervised': BLUE, 'SSL soft': ORANGE, 'PPI selected': AQUA,
                     'PPI CV λ': AQUA, 'PPI heuristic λ': VIOLET, 'PPI λ=1': VIOLET}
    for ax in axes.flat:
        for line in ax.get_lines():
            label = line.get_label()
            if label.startswith('_'):
                continue
            if label.startswith('Fully labeled') or str(line.get_color()).lower() == INK:
                line.set_linestyle('--')  # Reference / threshold lines: dashed, black (a second reference stays gray).
                continue
            if label in method_colors:
                line.set_color(method_colors[label])
                line.set_linestyle('-')
                continue
            if line.get_linestyle() not in ('None', ''):
                line.set_linestyle('-')  # Data lines: all solid; series differ by color and marker.
            if label not in slots:
                if len(slots) == len(PALETTE):
                    raise ValueError('More than eight series in one figure: split it instead of cycling colors.')
                slots[label] = PALETTE[len(slots)]
            line.set_color(slots[label])


def finish(fig, axes, title, name):
    recolor(axes)
    handles = {}
    for ax in axes.flat:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(label, handle)
    ncol = len(handles) if len(handles) <= 5 else (len(handles)+1)//2
    if axes.shape[1] < 3:  # Narrower figure (fewer mechanisms): a taller legend instead of a clipped one.
        ncol = min(ncol, 3)
    fig.legend(handles.values(), handles.keys(), loc='outside lower center', ncol=ncol, frameon=False)
    fig.suptitle(title, fontsize=12)
    save(fig, name)


def save(fig, name):
    folders = {'teacher_quality.png': '04_teacher_quality', 'ppi_gains.png': '05_ablations',
               'xgboost_sensitivity.png': '05_ablations', 'improved_gains.png': '05_ablations',
               'positive_control.png': '06_synthetic', 'ppssl_replication.png': '06_synthetic'}
    relative = Path('figures')/folders[name]/name if name in folders else Path(name)
    path = RESULTS_DIR/relative
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    for ax in fig.axes:
        for text in ax.texts:
            if text.get_gid() == 'row-label':
                box = text.get_window_extent(fig.canvas.get_renderer())
                if box.x0 < 0 or box.y0 < 0 or box.x1 > fig.bbox.width or box.y1 > fig.bbox.height:
                    raise ValueError(f'Row label clipped in {name}: {text.get_text()}')
    fig.savefig(path, dpi=180, facecolor='white')
    plt.close(fig)
    print(f'Saved: results/{relative}', flush=True)


def by_case(rows, mechanism, column='macro_f1'):
    return rows.loc[rows.mechanism == mechanism].set_index(['budget', 'seed'])[column]


# Step 3: Figure 1 — Main Study Learning Curves (Macro F1; LR Top, XGBoost Bottom)

def learning_curves(main, reference, audits):
    # Step 3a: Fix the Teacher per figure; compare Students in rows and methods within each panel.
    groups = {'CatBoost': main.drop(columns='method').rename(columns={'training_method': 'method'})}
    tabpfn, _, scope = tabpfn_results()
    if scope == 'full':
        groups['TabPFN'] = tabpfn
    for teacher, rows in groups.items():
        fig, axes = six_panel([f'{s} Student\nValidation Macro F1' for s in DEFAULT.student_models])
        for row, student in enumerate(DEFAULT.student_models):
            for column, mechanism in enumerate(MECHANISMS):
                ax = axes[row, column]
                for method, color, marker in zip(METHODS, [BLUE, ORANGE, AQUA], ['o', 's', '^']):
                    selected = rows.loc[(rows.student == student) & (rows.method == method)]
                    band(ax, by_case(selected, mechanism), method, color, marker)
                ax.axhline(reference[student]['macro_f1'], color=INK, linestyle='--', linewidth=1.4,
                           label=f'Fully labeled Student ({reference[student]["labeled_count"]:,} labels)')
        # Share scales across Teachers so visual differences remain comparable.
        for row, student in enumerate(DEFAULT.student_models):
            values = pd.concat([g.loc[g.student == student, 'macro_f1'] for g in groups.values()])
            low, high = min(values.min(), reference[student]['macro_f1']), max(values.max(), reference[student]['macro_f1'])
            axes[row, 0].set_ylim(max(0, low-.02), min(1, high+.02))
        finish(fig, axes, f'{teacher} Teacher: Classification Performance',
               f'figures/01_learning_curves/{teacher.lower()}.png')

    # Step 3b: Two gain curves per panel; black zero line is the matched Supervised baseline.
    differences = {}
    for teacher, rows in groups.items():
        wide = rows.pivot(index=['student', 'mechanism', 'budget', 'seed'], columns='method', values='macro_f1')
        differences[teacher] = wide[['SSL soft', 'PPI selected']].sub(wide.Supervised, axis=0)
    limit = max(.01, max(np.abs(d.groupby(level=['student', 'mechanism', 'budget']).mean().to_numpy()).max()
                         for d in differences.values())+.005)
    for teacher, delta in differences.items():
        fig, axes = six_panel([f'{s} Student\nMacro F1 − Supervised' for s in DEFAULT.student_models])
        for row, student in enumerate(DEFAULT.student_models):
            for column, mechanism in enumerate(MECHANISMS):
                ax = axes[row, column]
                for method, color, marker in [('SSL soft', ORANGE, 's'), ('PPI selected', AQUA, '^')]:
                    band(ax, delta.loc[(student, mechanism), method], method, color, marker)
                zero_line(ax)
                ax.set_ylim(-limit, limit)
        finish(fig, axes, f'{teacher} Teacher: Macro F1 Gain over Supervised',
               f'figures/02_gains/{teacher.lower()}.png')

    # Step 3c: One metric table instead of a separate figure for every metric.
    table = pd.concat([rows.assign(teacher=teacher) for teacher, rows in groups.items()], ignore_index=True)
    metrics = ['macro_f1', 'accuracy', 'balanced_accuracy', 'log_loss']
    summary = table.groupby(['teacher', 'student', 'mechanism', 'budget', 'method'])[metrics].agg(['mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary['n_seeds'] = table.groupby(['teacher', 'student', 'mechanism', 'budget', 'method']).seed.nunique()
    folder = RESULTS_DIR/'tables'
    folder.mkdir(exist_ok=True)
    summary.to_csv(folder/'metrics_summary.csv', float_format='%.6f')
    class_f1_heatmap(groups)
    metric_differences(groups)


def class_f1_heatmap(groups):
    # Step 3d: Keep all budgets and mechanisms visible; average paired class-F1 changes over masks.
    columns = pd.MultiIndex.from_product([MECHANISMS, BUDGETS], names=['mechanism', 'budget'])
    matrices = {}
    for teacher, rows in groups.items():
        for student in DEFAULT.student_models:
            part = rows.loc[rows.student == student].copy()
            for k, label in enumerate(CLASSES):
                part[label] = part.class_f1.map(lambda v: json.loads(v)[k] if isinstance(v, str) else v[k])
            sup = part.loc[part.method == 'Supervised'].set_index(KEY)[CLASSES]
            ppi = part.loc[part.method == 'PPI selected'].set_index(KEY)[CLASSES]
            matrices[teacher, student] = (ppi-sup).groupby(level=['mechanism', 'budget']).mean().reindex(columns).T
    limit = max(.005, max(np.abs(m.to_numpy()).max() for m in matrices.values()))
    fig, axes = plt.subplots(len(groups), 2, figsize=(16, 3.5*len(groups)), squeeze=False, layout='constrained')
    for row, teacher in enumerate(groups):
        for col, student in enumerate(DEFAULT.student_models):
            ax = axes[row, col]
            matrix = matrices[teacher, student]
            im = ax.imshow(matrix, cmap='RdBu', vmin=-limit, vmax=limit, aspect='auto')
            ax.set_title(f'{teacher} Teacher → {student} Student', pad=30)
            ax.set_yticks(range(len(CLASSES)), CLASSES)
            ax.set_xticks(range(len(columns)), [str(b) for _, b in columns], rotation=45)
            for center, mechanism in zip([2, 7, 12], MECHANISMS):
                ax.text(center, 1.02, mechanism, transform=ax.get_xaxis_transform(), ha='center', fontsize=9)
            for boundary in [4.5, 9.5]:
                ax.axvline(boundary, color='white', linewidth=2)
            ax.set_xlabel('Visible training labels')
    fig.colorbar(im, ax=axes.ravel().tolist(), label='Per-class F1 − Supervised (blue = improvement)', shrink=.8)
    fig.suptitle('PPI: Per-class F1 Gain over Supervised', fontsize=12)
    save(fig, 'figures/03_class_f1/ppi_vs_supervised.png')


# Step 4: Figure 2 — Are the Teachers Better Than the Student They Teach?

def teacher_quality(main, audits, study_audits, extra):
    catboost = pd.DataFrame([dict(mechanism=m, budget=b, seed=s, macro_f1=a['teacher_validation']['macro_f1'])
                             for (m, b, s), a in audits.items()])
    refit = extra.loc[extra.method == 'Teacher / Refit on visible labels']
    tabpfn = tabpfn_teacher_scores()  # Full TabPFN study if complete, else the MCAR pilot; empty if neither.
    fig, axes = six_panel([f'{s} Student\nValidation Macro F1' for s in DEFAULT.student_models])
    for row, student in enumerate(DEFAULT.student_models):
        same = pd.DataFrame([dict(mechanism=m, budget=b, seed=s, macro_f1=a['teacher_validation']['macro_f1'])
                             for (v, m, b, s), a in study_audits.items() if v == f'{student}→{student} [labeled]'])
        supervised = main.loc[(main.student == student) & (main.training_method == 'Supervised')]
        for column, mechanism in enumerate(MECHANISMS):
            ax = axes[row, column]
            band(ax, by_case(supervised, mechanism), 'Student trained Supervised', BLUE, 'o')
            band(ax, by_case(catboost, mechanism), 'CatBoost Teacher (cross-fit, used for pseudo-labels)', TEACHER_GRAY, 'D', ':')
            band(ax, by_case(refit, mechanism), 'CatBoost Teacher (trained on n visible labels)', TEACHER_GRAY, 'D', '--')
            band(ax, by_case(same, mechanism), 'Same-architecture Teacher (cross-fit)', VIOLET, 'v', '-.')
            if not tabpfn.empty and mechanism in set(tabpfn.mechanism):
                band(ax, by_case(tabpfn, mechanism), 'TabPFN-3.5 Teacher (pretrained; cross-fit)', MUTED, 'X', '-')
                band(ax, by_case(tabpfn, mechanism, 'alone'), 'TabPFN-3.5 alone (n visible labels as context)',
                     MUTED, 'P', ':')
    finish(fig, axes, 'Teacher and Supervised Student Performance', 'teacher_quality.png')


def tabpfn_results():
    """TabPFN-Teacher rows (student, mechanism, budget, seed, method, macro_f1, lambda_value) and audits.
    Uses the full study (75 masks, LR and XGBoost) once complete, else the complete MCAR/LR pilot, else nothing."""
    full_db = _teacher_study.TABPFN_FULL_DB
    if full_db.exists():
        db = read_only(full_db)
        try:
            done = db.execute("SELECT COUNT(*) FROM cases WHERE status='complete'").fetchone()[0]
            if done == len(DEFAULT.mechanisms)*len(DEFAULT.budgets)*len(DEFAULT.seeds):
                rows = pd.read_sql_query('SELECT * FROM metrics', db)
                audits = {(m, b, s): json.loads(a) for m, b, s, a in
                          db.execute("SELECT mechanism,budget,seed,audit FROM cases WHERE status='complete'")}
                rows[['student', 'method']] = rows.method.str.split(' / ', expand=True)
                return rows, audits, 'full'
        finally:
            db.close()
    rows, audits = load_tabpfn()
    keys = set(product(TABPFN_GRID['mechanisms'], TABPFN_GRID['budgets'], TABPFN_GRID['seeds']))
    if rows.empty or not keys <= set(audits):
        return pd.DataFrame(), {}, None
    return rows.assign(student='LR'), audits, 'pilot'


def tabpfn_teacher_scores():
    _, audits, _ = tabpfn_results()
    return pd.DataFrame([dict(mechanism=m, budget=b, seed=s, macro_f1=a['teacher_validation']['macro_f1'],
                              alone=a['tabpfn_refit_validation']['macro_f1']) for (m, b, s), a in audits.items()])


# Step 5: Figure 3 — Main Result: Gain Over Supervised on the Same Mask

def ppi_gains(rows, split='validation', name='ppi_gains.png', extra=None):
    rows = rows.loc[rows.design == 'labeled'] if 'design' in rows else rows
    wide = rows.pivot_table(index=['teacher', 'student', 'mechanism', 'budget', 'seed'], columns='method', values='macro_f1')
    fixed = None if extra is None else extra.loc[extra.method.str.endswith('/ PPI fixed λ=1') & ~extra.method.str.contains(r'\[')]
    if fixed is not None:
        fixed = fixed.assign(student=fixed.method.str.split(' / ').str[0]).set_index(['student', 'mechanism', 'budget', 'seed']).macro_f1
    found, _, scope = tabpfn_results()
    tabpfn = pd.DataFrame() if found.empty else found.pivot_table(index=['student', 'mechanism', 'budget', 'seed'],
                                                                  columns='method', values='macro_f1')
    tag = '' if scope == 'full' else ', pilot'
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
            if split == 'validation' and not tabpfn.empty and (student, mechanism) in tabpfn.index.droplevel([2, 3]):
                t = tabpfn.loc[(student, mechanism)]
                band(ax, t['SSL soft']-t.Supervised, f'SSL soft (TabPFN-3.5 Teacher{tag})', ORANGE, 'X', ':')
                band(ax, t['PPI selected']-t.Supervised, f'PPI (TabPFN-3.5 Teacher{tag})', AQUA, 'X', ':')
    finish(fig, axes, f'Macro F1 Gain over Supervised Learning ({split.capitalize()})', name)


# Step 6: Figure 4 — XGBoost Sensitivity on the Pre-Declared Subset (A2 Rounds, A3 Capacity)

def xgboost_sensitivity(main, extra, a2, a3_reference):
    fits = pd.DataFrame([dict(mechanism=m, budget=b, seed=s, rounds=int(r), sup=f['supervised_macro_f1'],
                              ppi=f['ppi_macro_f1'], negative=f['ppi_lambda_1']['empirical_risk_final'] < 0)
                         for (m, b, s), a in a2.items() for r, f in a['fits'].items()])
    large = extra.loc[extra.method == f'{A3_LABEL} / Supervised']
    small = main.loc[(main.method == 'XGBoost / Supervised') & main.budget.isin(A2_GRID['budgets'])]
    # A2/A3 test optimization and capacity, which do not depend on the missingness mechanism; they were declared on
    # MCAR and MAR only (MNAR would add uncorrected selection bias), so the figure is 2 × 2.
    fig, axes = six_panel(['XGBoost PPI λ=1\nMacro F1 − Supervised', 'XGBoost Supervised\nValidation Macro F1'],
                          A2_GRID['mechanisms'])
    for column, mechanism in enumerate(A2_GRID['mechanisms']):
        top, bottom = axes[0, column], axes[1, column]
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
        bottom.axhline(a3_reference['large']['macro_f1'], color=MUTED, linewidth=1.4,
                       label='Fully labeled, depth 5')
    finish(fig, axes, 'XGBoost: Boosting Rounds and Model Capacity', 'xgboost_sensitivity.png')


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


METRICS = [('macro_f1', 1, 'Macro F1'), ('balanced_accuracy', 1, 'Balanced Accuracy'),
           ('accuracy', 1, 'Accuracy'), ('log_loss', -1, 'Log Loss Reduction')]


def metric_differences(groups):
    # Step 7a: Paired change vs Supervised for every metric, oriented so that positive = better (log loss is negated).
    rows = []
    for teacher, data in groups.items():
        for student in DEFAULT.student_models:
            part = data.loc[data.student == student]
            for metric, sign, label in METRICS:
                wide = part.pivot_table(index=KEY, columns='method', values=metric)
                for method in ['SSL soft', 'PPI selected']:
                    r = describe(sign*(wide[method]-wide.Supervised))
                    rows.append(dict(teacher=teacher, student=student, metric=label, method=method, **r))
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR/'tables'/'metric_differences.csv', index=False, float_format='%.5f')
    print('Saved: results/tables/metric_differences.csv', flush=True)

    # Main-report figure: LR Student only (XGBoost is in the CSV); rows = Teachers, columns = metrics.
    teachers = list(groups)
    fig, axes = plt.subplots(len(teachers), len(METRICS), figsize=(14, 2.3*len(teachers)+1.2), sharey=True,
                             layout='constrained', squeeze=False)
    lr = table.loc[table.student == 'LR']
    for col, (_, _, label) in enumerate(METRICS):
        values = lr.loc[lr.metric == label, ['boot_low', 'boot_high']].to_numpy()
        span = max(abs(values).max(), .002)*1.25
        for row, teacher in enumerate(teachers):
            ax = axes[row, col]
            ax.axvline(0, color=INK, linestyle='--', linewidth=1.2, label='Supervised (difference = 0)')
            for y, (method, color, marker) in zip([1, 0], [('SSL soft', ORANGE, 's'), ('PPI selected', AQUA, '^')]):
                r = lr.loc[(lr.teacher == teacher) & (lr.metric == label) & (lr.method == method)].iloc[0]
                ax.errorbar(r['mean'], y, xerr=[[r['mean']-r.boot_low], [r.boot_high-r['mean']]], fmt=marker,
                            color=color, markersize=8, capsize=3, linewidth=1.6, label=method)
                ax.annotate(f"{r['mean']:+.4f}", (r['mean'], y), xytext=(0, 9), textcoords='offset points',
                            ha='center', fontsize=9)
            ax.set_xlim(-span, span)
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4, symmetric=True))
            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%+.3f'))
            ax.set_ylim(-.6, 1.6)
            ax.set_yticks([1, 0], ['SSL soft', 'PPI selected'])
            ax.grid(axis='x', alpha=.18)
            if row == 0:
                ax.set_title(label)
            if row == len(teachers)-1:
                ax.set_xlabel('Change vs Supervised (→ better)')
            if col == len(METRICS)-1:
                right_row_label(ax, f'{teacher} Teacher')
    handles = {l: h for ax in axes.flat for h, l in zip(*ax.get_legend_handles_labels())}
    fig.legend(handles.values(), handles.keys(), loc='outside lower center', ncol=len(handles), frameon=False)
    fig.suptitle('LR Student: Metric Changes Relative to Supervised', fontsize=12)
    save(fig, 'figures/07_metrics/paired_differences.png')




# Step 8: Final Test Evaluation (Written Only After 07_final_test.py Has Run Once)

def final_test_outputs():
    _module = import_module('07_final_test')
    FROZEN, load_test = _module.FROZEN, _module.load
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
    _module = import_module('04_improved')
    CONFIG, IMPROVED_DB, PROTOCOL, STUDENTS, is_complete = _module.CONFIG, _module.IMPROVED_DB, _module.PROTOCOL, _module.STUDENTS, _module.is_complete
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
        finish(fig, axes, 'Ensemble Teacher: Student Performance Gains', name)

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
CAVEAT = ('All 75 masks share one train/validation split and one validation set, and masks with the same seed are nested '
          'across budgets. SE and intervals therefore resample whole seeds (5 clusters): coarse, descriptive, and about '
          'mask-to-mask variability on this development set only, not generalization.')


def main_lr():
    """LR main study (CatBoost Teacher, labeled-only preprocessing): rows with short method names, and audits."""
    rows, _, audits = load_main()
    rows = rows.loc[rows.student == 'LR'].drop(columns='method').rename(columns={'training_method': 'method'})
    return rows, audits


# Step 10a: LR Main Line — Paired Differences With SE and a Stratified Paired Bootstrap Over Masks

def seed_cluster_bootstrap(d, draws=4000, seed=0):
    """Resample whole seeds (clusters) with replacement. Masks with the same seed are nested across budgets and share
    CV folds, so they are not independent; resampling masks one by one would understate the uncertainty."""
    rng = np.random.default_rng(seed)
    groups = d.groupby(level='seed')
    sums, counts = groups.sum().to_numpy(), groups.size().to_numpy()
    picks = rng.integers(0, len(sums), (draws, len(sums)))
    return np.percentile(sums[picks].sum(axis=1)/counts[picks].sum(axis=1), [2.5, 97.5])


def describe(d):
    """Mean, seed-cluster SE and seed-cluster bootstrap interval (only 5 seeds: coarse and descriptive)."""
    lo, hi = seed_cluster_bootstrap(d)
    seed_means = d.groupby(level='seed').mean()
    return {'n_masks': len(d), 'n_seeds': len(seed_means), 'mean': d.mean(),
            'cluster_se': seed_means.std(ddof=1)/np.sqrt(len(seed_means)), 'boot_low': lo, 'boot_high': hi,
            'seeds_positive': int((seed_means > 1e-12).sum()),
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
        print(f"  {r.contrast}: {r['mean']:+.4f} ± {r.cluster_se:.4f} seed-cluster SE, 95% seed-cluster bootstrap [{r.boot_low:+.4f}, {r.boot_high:+.4f}] "
              f"({r.better} better / {r.worse} worse / {r.equal} equal)")
    print('  Robustness vs improvement: '+'; '.join(f'{k}: {v:.4f}' if isinstance(v, float) else f'{k}: {v}' for k, v in harms.items()))
    print('  λ sources by budget:\n'+counts.to_string())
    for _, r in outputs['lr_ablation_questions.csv'].iterrows():
        print(f"  {r.question}: {r['mean']:+.4f} [{r.boot_low:+.4f}, {r.boot_high:+.4f}]")


# Step 11: Synthetic Positive Control — One Figure (Oracle and Weak-but-Unbiased Teachers)

def positive_control_outputs():
    _module = import_module('05_synthetic_checks')
    PC_METHODS, checks, is_complete, load_pc, paired = _module.METHODS, _module.checks, _module.is_complete, _module.load, _module.paired
    if not is_complete():
        print('Positive control incomplete; no figure drawn.', flush=True)
        return
    rows, declaration = load_pc()
    styles = {'SSL soft': (ORANGE, 's', '-'), 'PPI CV λ': (AQUA, '^', '-'), 'PPI λ=1': (AQUA, 'D', ':')}
    # Oracle = gains are possible here; σ=3 = a Teacher worse than the Student whose noise is independent of y and x
    # (softmax of noisy logits is NOT an unbiased probability). The other conditions
    # (σ=1, uninformative, subgroup-biased) are in positive_control_checks.csv and the console summary.
    pair = ['oracle', 'noise σ=3']
    fig, axes = six_panel([f'{t} Teacher\nMacro F1 − Supervised' for t in pair])
    for row, teacher in enumerate(pair):
        for column, mechanism in enumerate(MECHANISMS):
            ax = axes[row, column]
            zero_line(ax)
            for method in PC_METHODS:
                band(ax, paired(rows, teacher, f'{method} − Supervised').loc[mechanism], method, *styles[method])
    finish(fig, axes, 'Synthetic Classification: Effect of Teacher Quality', 'positive_control.png')
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
    _module = import_module('05_synthetic_checks')
    BOUNDARY, check_targets, is_complete, load_ppssl, with_best_fixed = _module.BOUNDARY, _module.check_targets, _module.ppssl_complete, _module.load_ppssl, _module.with_best_fixed
    if not is_complete():
        print('PP-SSL replication incomplete; no figure drawn.', flush=True)
        return
    runs, best = with_best_fixed(load_ppssl())
    mean = runs.groupby(['method', 'mu'])[['mse', 'mse_a', 'mse_b', 'lam']].mean()
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.6), layout='constrained')
    styles = [('Only Labeled', BLUE, 'o', '-'), ('Classic SSL', ORANGE, 's', '-'), ('Teacher', TEACHER_GRAY, 'D', ':'),
              ('PPI++ (offline λ)', AQUA, 'v', '--'), ('PP-SSL (online λ)', AQUA, '^', '-'),
              ('PPI, λ by training CV (this project)', AQUA, 'P', ':')]
    for column, (metric, title) in enumerate([('mse', 'Overall'), ('mse_a', 'Group A: Oracle Teacher'),
                                              ('mse_b', 'Group B: Biased Teacher')]):
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
        ax.set(title=f'Effect of λ (μ = {mu:g})', xlabel='fixed λ', ylabel='Test MSE (all rows)')
        ax.grid(alpha=.18)
    finish(fig, axes, 'PP-SSL: Synthetic Regression', 'ppssl_replication.png')
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
    rows, audits, scope = tabpfn_results()
    if scope is None:
        print('TabPFN Teacher results incomplete; no summary.', flush=True)
        return
    main_rows, _, main_audits = load_main()
    cat = main_rows.pivot_table(index=['student', 'mechanism', 'budget', 'seed'], columns='training_method', values='macro_f1')
    tab = rows.pivot_table(index=['student', 'mechanism', 'budget', 'seed'], columns='method', values='macro_f1')
    lam = rows.loc[rows.method == 'PPI selected'].set_index(['student', 'mechanism', 'budget', 'seed']).lambda_value
    table = []
    for key in sorted(tab.index):
        a, m = audits[key[1:]], main_audits[key[1:]]
        table.append(dict(zip(['student', 'mechanism', 'budget', 'seed'], key),
                          supervised=tab.loc[key, 'Supervised'],
                          tabpfn_alone=a['tabpfn_refit_validation']['macro_f1'],
                          tabpfn_teacher_crossfit=a['teacher_validation']['macro_f1'],
                          catboost_teacher_crossfit=m['teacher_validation']['macro_f1'],
                          ssl_tabpfn_minus_sup=tab.loc[key, 'SSL soft']-tab.loc[key, 'Supervised'],
                          ppi_tabpfn_minus_sup=tab.loc[key, 'PPI selected']-tab.loc[key, 'Supervised'],
                          ssl_catboost_minus_sup=cat.loc[key, 'SSL soft']-cat.loc[key, 'Supervised'],
                          ppi_catboost_minus_sup=cat.loc[key, 'PPI selected']-cat.loc[key, 'Supervised'],
                          ppi_tabpfn_lambda=lam.loc[key]))
    table = pd.DataFrame(table)
    table.to_csv(RESULTS_DIR/'tabpfn_teacher_summary.csv', index=False, float_format='%.5f')
    print(f'Saved: results/tabpfn_teacher_summary.csv ({scope} TabPFN Teacher study)')
    print(f'\n=== TabPFN-3.5 Teacher ({scope}; validation; development set; descriptive) ===')
    index = ['mechanism', 'budget', 'seed']
    for student, part in table.groupby('student'):
        part = part.set_index(index)
        print(f'\n{student} Student — mean by budget:')
        print(part.drop(columns='student').groupby(level='budget').mean().round(4).T.to_string())
        for name, d in [('TabPFN Teacher − CatBoost Teacher (cross-fit Macro F1)', part.tabpfn_teacher_crossfit-part.catboost_teacher_crossfit),
                        ('TabPFN alone − Supervised', part.tabpfn_alone-part.supervised),
                        ('PPI − Supervised, TabPFN Teacher', part.ppi_tabpfn_minus_sup),
                        ('PPI − Supervised, CatBoost Teacher', part.ppi_catboost_minus_sup),
                        ('PPI (TabPFN) − PPI (CatBoost)', part.ppi_tabpfn_minus_sup-part.ppi_catboost_minus_sup),
                        ('SSL − Supervised, TabPFN Teacher', part.ssl_tabpfn_minus_sup)]:
            r = describe(d)
            print(f"  {name}: {r['mean']:+.4f} [{r['boot_low']:+.4f}, {r['boot_high']:+.4f}] "
                  f"({r['seeds_positive']}/{r['n_seeds']} seeds positive)")

if __name__ == '__main__':
    main()
