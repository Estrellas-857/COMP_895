"""Saved-QA diagnostics separating score binning from true-score tie policy."""
from collections import defaultdict
import time

import numpy as np

from ..evaluation import histogram_indices, predict_scores
from experiments.paired_calibration_v1.metrics import weighted_risk_coverage
from .shared import BASE, OUT, ROWS, dump, models, samples, table, verify_execution


METHODS = ('raw_msp', 'raw_entropy', 'lr3', 'lr3_class')
TARGETS = (.01, .05, .1, .2, .5, .8)
RESOLUTIONS = (10000, 100000, 1000000)
METRICS = ('auroc', 'average_precision', 'aurc')


def grouped_sorted(keys, errors, weights, targets=TARGETS):
    """Ascending keys; accept complete equal-key blocks and integrate right steps."""
    keys, errors, weights = map(np.asarray, (keys, errors, weights))
    if not len(keys) or keys.ndim != 1 or errors.shape != keys.shape or weights.shape != keys.shape:
        raise ValueError('Nonempty, equal-length vectors are required')
    if (not np.isfinite(keys).all() or np.any(keys[1:] < keys[:-1])
            or not np.isfinite(weights).all() or np.any(weights <= 0)
            or not np.isin(errors, [0, 1]).all()):
        raise ValueError('Sorted finite keys, binary errors and positive finite weights required')
    starts = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1]
    mass = np.add.reduceat(weights, starts)
    wrong = np.add.reduceat(weights * errors, starts)
    correct = mass - wrong
    total, positives = mass.sum(), wrong.sum()
    negatives = total - positives
    accepted = np.cumsum(mass)
    cumulative_wrong = np.cumsum(wrong)
    coverage, risk = accepted / total, cumulative_wrong / accepted
    ap_precision = np.cumsum(wrong[::-1]) / np.cumsum(mass[::-1])
    result = dict(auroc=float(np.sum(wrong * (np.cumsum(correct) - correct / 2)) / (positives * negatives))
                      if positives and negatives else None,
                  average_precision=float(np.sum(wrong[::-1] * ap_precision) / positives) if positives else None,
                  aurc=float(np.sum(mass / total * risk)),
                  num_score_blocks=len(starts), largest_accepted_block_mass=float(mass.max() / total),
                  lowest_accepted_block_mass=float(mass[0] / total),
                  zero_key_mass=float(mass[0] / total) if keys[0] == 0 else 0.)
    checkpoints = []
    for target in targets:
        index = int(np.argmin(abs(coverage - target)))
        checkpoints.append(dict(requested_coverage=target, actual_coverage=float(coverage[index]),
                                risk=float(risk[index]), coverage_error=float(coverage[index] - target)))
    return result, checkpoints


def policies(scores, errors, weights, tie_keys):
    """Identical samples and weights; vary only quantization or RC tie handling."""
    scores = np.asarray(scores)
    order = np.argsort(scores, kind='stable')
    s, y, w = scores[order], errors[order], weights[order]
    raw, raw_points = grouped_sorted(s, y, w)
    result = {'raw_grouped': (raw, raw_points)}
    for bins in RESOLUTIONS:
        quantized, points = grouped_sorted(histogram_indices(s, bins), y, w)
        result[f'arcsine_{bins}_grouped'] = quantized, points
    curve = weighted_risk_coverage(scores, errors, weights, tie_keys, TARGETS)
    secondary = dict(raw, aurc=curve['aurc'], num_score_blocks=len(scores),
                     largest_accepted_block_mass=float(weights.max() / weights.sum()),
                     lowest_accepted_block_mass=None, zero_key_mass=raw['zero_key_mass'])
    result['raw_secondary'] = secondary, curve['checkpoints']
    for summary, _ in result.values():
        summary['largest_true_score_tie_mass'] = raw['largest_accepted_block_mass']
        summary['lowest_true_score_tie_mass'] = raw['lowest_accepted_block_mass']
        summary['score_zero_mass'] = raw['zero_key_mass']
    return result


def measure(scope, group, weather, values, error, weights, ties, summary_rows, point_rows):
    for method, scores in values.items():
        for policy, (summary, points) in policies(scores, error, weights, ties).items():
            identity = dict(scope=scope, group=group, weather=weather, method=method, policy=policy)
            summary_rows.append(dict(**identity, sampled_pixels=len(error), population_weight=float(weights.sum()),
                error_prevalence=float(np.average(error, weights=weights)), **summary))
            for point in points:
                point_rows.append(dict(**identity, requested_coverage=point['requested_coverage'],
                    actual_coverage=point['actual_coverage'], accepted_error=point['risk'],
                    coverage_error=point['actual_coverage'] - point['requested_coverage']))


def macro_tables(summary_rows, point_rows):
    metrics, points = [], []
    for weather in ('all', 'fog', 'night', 'rain', 'snow'):
        chosen = [r for r in summary_rows if r['scope'] == 'heldout_family'
                  and (weather == 'all' or r['weather'] == weather)]
        for method in METHODS:
            for policy in sorted({r['policy'] for r in chosen}):
                selected = [r for r in chosen if r['method'] == method and r['policy'] == policy]
                metrics.append(dict(weather=weather, method=method, policy=policy, num_families=len(selected),
                    **{f'mean_{metric}': float(np.mean([r[metric] for r in selected])) for metric in METRICS}))
                for target in TARGETS:
                    selected_points = [r for r in point_rows if r['scope'] == 'heldout_family'
                        and (weather == 'all' or r['weather'] == weather)
                        and r['method'] == method and r['policy'] == policy and r['requested_coverage'] == target]
                    points.append(dict(weather=weather, method=method, policy=policy, num_families=len(selected_points),
                        requested_coverage=target,
                        mean_actual_coverage=float(np.mean([r['actual_coverage'] for r in selected_points])),
                        mean_accepted_error=float(np.mean([r['accepted_error'] for r in selected_points])),
                        max_absolute_coverage_error=float(max(abs(r['coverage_error']) for r in selected_points))))
    return metrics, points


def night_pair_decomposition(summary_rows):
    """Decompose pooled AUROC by positive/negative pairs within/across families."""
    result = []
    for method in METHODS:
        groups = [r for r in summary_rows if r['scope'] == 'heldout_family' and r['weather'] == 'night'
                  and r['method'] == method and r['policy'] == 'raw_grouped']
        pooled = next(r for r in summary_rows if r['scope'] == 'pooled_crossfitted_night'
                      and r['method'] == method and r['policy'] == 'raw_grouped')
        positives = np.array([float(r['population_weight']) * float(r['error_prevalence']) for r in groups])
        negatives = np.array([float(r['population_weight']) for r in groups]) - positives
        within_denominator = float(np.sum(positives * negatives))
        total_denominator = float(positives.sum() * negatives.sum())
        cross_denominator = total_denominator - within_denominator
        within_numerator = float(sum(float(r['auroc']) * p * n for r, p, n in zip(groups, positives, negatives)))
        cross_numerator = float(pooled['auroc']) * total_denominator - within_numerator
        result.append(dict(method=method, num_families=len(groups), policy='raw_grouped',
            pooled_auroc=float(pooled['auroc']), within_family_pair_weight_fraction=within_denominator / total_denominator,
            cross_family_pair_weight_fraction=cross_denominator / total_denominator,
            within_family_pair_weighted_auroc=within_numerator / within_denominator,
            cross_family_pair_weighted_auroc=cross_numerator / cross_denominator,
            positive_population_weight=float(positives.sum()), negative_population_weight=float(negatives.sum()),
            interpretation='Pair-weighted decomposition, not equal-family macro; cross-family comparisons mix composition and model score scales, not causal calibration effects.'))
    return result


def main():
    started = time.perf_counter()
    complete = verify_execution()
    summary_rows, point_rows = [], []
    night = defaultdict(list)
    for group in sorted({r['recording_group'] for r in ROWS}):
        weather = next(r['condition'] for r in ROWS if r['recording_group'] == group)
        sample = samples(group)
        fitted, _ = models(group)
        values = {m: predict_scores(m, sample['X'], fitted.get(m)) for m in METHODS}
        measure('heldout_family', group, weather, values, sample['errors'], sample['weights'],
                sample['tie_keys'], summary_rows, point_rows)
        if weather == 'night':
            for method, scores in values.items():
                night[method].append(scores)
            for key in ('errors', 'weights', 'tie_keys'):
                night[key].append(sample[key])
        print(f'Metric sensitivity completed {group} ({weather}), {time.perf_counter() - started:.1f}s', flush=True)
    pooled = {key: np.concatenate(parts) for key, parts in night.items()}
    measure('pooled_crossfitted_night', 'all_night', 'night', {m: pooled[m] for m in METHODS},
            pooled['errors'], pooled['weights'], pooled['tie_keys'], summary_rows, point_rows)
    macro_metrics, macro_points = macro_tables(summary_rows, point_rows)
    table('metrics_policy_summary.csv', summary_rows)
    table('metrics_policy_coverage.csv', point_rows)
    table('metrics_macro_summary.csv', macro_metrics)
    table('metrics_macro_coverage.csv', macro_points)
    table('metrics_night_auc_pair_decomposition.csv', night_pair_decomposition(summary_rows))
    reference = {(r['scope'], r['group'], r['method']): r for r in summary_rows if r['policy'] == 'raw_grouped'}
    differences = []
    for row in summary_rows:
        if row['policy'] == 'raw_grouped':
            continue
        base = reference[(row['scope'], row['group'], row['method'])]
        differences.append({key: row[key] for key in ('scope', 'group', 'weather', 'method', 'policy')} |
            dict(reference_policy='raw_grouped', comparison='true-score tie policy' if row['policy'] == 'raw_secondary'
                 else 'score quantization only; both accept whole tied blocks',
                 **{f'{metric}_delta': row[metric] - base[metric] for metric in METRICS}))
    table('metrics_policy_differences.csv', differences)
    verify_execution()
    dump('logs/metric_sensitivity.json', dict(status='PASS', families=17,
        methods=list(METHODS), policies=['raw_grouped', *[f'arcsine_{n}_grouped' for n in RESOLUTIONS], 'raw_secondary'],
        coverages=list(TARGETS), execution_source_hashes=complete['source_hashes'],
        elapsed_seconds=time.perf_counter() - started,
        definitions=dict(grouped='Sort scores ascending; aggregate equal scores or transformed bin IDs; accept whole blocks; integrate right steps.',
            secondary='Existing weighted_risk_coverage; independent saved secondary keys order exact score ties; same right-step rule.',
            auroc_ap='True-score ties grouped for raw scores, including raw_secondary; transformed-score ties grouped for binned scores.',
            weights='Identical saved QA samples and inverse inclusion weights within each comparison.',
            group_macro='Arithmetic mean of the per-family metrics; neither pooled pixel metrics nor a confidence interval.'),
        limitations=['Post-hoc sampled diagnostics; do not replace the frozen full-population primary endpoint.',
            'Pooling Night combines group composition and potentially different fold-specific score scales; differences cannot be attributed solely to one factor.',
            'Whole-block curves may miss requested coverages when scores tie; actual coverages are reported.',
            'Three Night filename families are not three proven independent sessions.',
            'No fitting, model selection, neural-network forward pass, or probability tensor storage.']))


if __name__ == '__main__':
    main()
