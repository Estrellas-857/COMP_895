"""Presentation-sized descriptive views of frozen results and diagnostics."""
import csv
import os

import numpy as np

from ..evaluation import HistogramAccumulator
from ..report import restore_accumulator
from .shared import BASE, OUT, read, table, verify_execution

METHODS = ('raw_msp', 'raw_entropy', 'lr3', 'lr3_class')
LABELS = ('MSP', 'Entropy', 'LR: three features', 'LR + predicted class')
COLORS = ('#414487', '#777777', '#238a8d', '#da8c28')
NIGHT = ('gopro_0351', 'gopro_0376', 'gopro_0397')


def class_tables():
    with (OUT / 'tables/pixel_class_acceptance.csv').open() as handle:
        classes = list(csv.DictReader(handle))
    with (OUT / 'tables/pixel_group_operating_points.csv').open() as handle:
        operating = list(csv.DictReader(handle))
    table('night_frozen_operating_points.csv', [r for r in operating if r['group'] in NIGHT])
    points = {(r['group'], r['method'], float(r['requested_coverage'])): r for r in operating}
    table('night_low_coverage_class_composition.csv', [r for r in classes
          if r['group'] in NIGHT and float(r['requested_coverage']) == .1])
    selected = {(r['group'], r['method'], r['predicted_class']): r for r in classes
                if float(r['requested_coverage']) == .8}
    contributions = []
    for group in sorted({r['group'] for r in classes}):
        a, b = (points[group, method, .8] for method in ('raw_msp', 'lr3_class'))
        group_rows = []
        for category in range(19):
            x, y = (selected[group, method, str(category)] for method in ('raw_msp', 'lr3_class'))
            group_rows.append(dict(group=group, weather=x['condition'], predicted_class=category,
                predicted_class_name=x['predicted_class_name'],
                msp_actual_coverage=float(a['actual_coverage']), class_actual_coverage=float(b['actual_coverage']),
                msp_risk_contribution=int(x['accepted_class_errors']) / int(a['accepted_pixels']),
                class_risk_contribution=int(y['accepted_class_errors']) / int(b['accepted_pixels'])))
        for row in group_rows:
            row['class_minus_msp_contribution_pp'] = 100 * (row['class_risk_contribution'] - row['msp_risk_contribution'])
        assert np.isclose(sum(r['class_minus_msp_contribution_pp'] for r in group_rows),
                          100 * (float(b['risk']) - float(a['risk'])), atol=1e-12)
        contributions.extend(group_rows)
    table('class_risk80_contributions.csv', contributions)
    macro = []
    for weather in ('fog', 'night'):
        for category in range(19):
            rows = [r for r in contributions if r['weather'] == weather and r['predicted_class'] == category]
            macro.append(dict(weather=weather, predicted_class=category,
                predicted_class_name=rows[0]['predicted_class_name'], groups=len(rows),
                mean_class_minus_msp_contribution_pp=np.mean([r['class_minus_msp_contribution_pp'] for r in rows])))
    table('class_risk80_macro_contributions.csv', macro)


def main():
    verify_execution()
    with (BASE / 'results/report/group_metrics.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    summary = []
    for method in METHODS:
        selected = [r for r in rows if r['method'] == method]
        assert len(selected) == 17
        summary.append(dict(method=method, groups=17, **{
            key: float(np.mean([float(r[key]) for r in selected]))
            for key in ('auroc', 'average_precision', 'aurc', 'retrospective80_risk')},
            training_threshold_coverage_min=min(float(r['training_threshold_actual_coverage']) for r in selected),
            training_threshold_coverage_max=max(float(r['training_threshold_actual_coverage']) for r in selected)))
    table('frozen_macro_results.csv', summary)
    table('night_frozen_group_metrics.csv', [r for r in rows if r['weather'] == 'night'])

    os.environ.setdefault('MPLCONFIGDIR', str(OUT / 'logs/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    pooled = {m: HistogramAccumulator() for m in METHODS}
    for ax, group in zip(axes.flat, NIGHT):
        result = read(BASE / 'results/folds' / group / 'result.json')
        with np.load(BASE / 'results/folds' / group / 'histograms.npz') as saved:
            for method, label, color in zip(METHODS, LABELS, COLORS):
                accumulator = restore_accumulator(result['methods'][method], saved[method + '_correct'], saved[method + '_incorrect'])
                pooled[method].merge(accumulator)
                curve = accumulator.finalize()['risk_coverage']
                ax.stairs(curve['risk'], np.r_[0, curve['coverage']], baseline=None, label=label, color=color, lw=1.6)
                index = int(np.argmin(abs(curve['coverage'] - .8)))
                ax.scatter(curve['coverage'][index], curve['risk'][index], color=color, s=18)
        ax.set_title(f"{group} · {result['num_images']} images")
    ax = axes.flat[-1]
    for method, label, color in zip(METHODS, LABELS, COLORS):
        curve = pooled[method].finalize()['risk_coverage']
        ax.stairs(curve['risk'], np.r_[0, curve['coverage']], baseline=None, label=label, color=color, lw=1.6)
    ax.set_title('Pooled Night · cross-fitted diagnostic only')
    for ax in axes.flat:
        ax.axvspan(0, .2, color='#999999', alpha=.08)
        ax.axvline(.8, color='#999999', linestyle=':', lw=.8)
        ax.set(xlim=(0, 1), ylim=(0, .42), xlabel='Actual accepted coverage', ylabel='Error among accepted pixels')
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=.18)
    axes.flat[0].legend(frameon=False, fontsize=8, loc='upper left')
    fig.suptitle('Night failure exists inside every held-out group\nFull pixels; steps show frozen AURC integration; dots mark actual ≈80% coverage', fontsize=13)
    fig.savefig(OUT / 'figures/night_group_risk_coverage.png', dpi=180)
    plt.close(fig)
    class_tables()
    print('Frozen macro table and Night group figure written; no metric replaced.')


if __name__ == '__main__':
    main()
