"""Locate Night LR+class score clusters using only saved weighted QA samples."""
import numpy as np

from ..evaluation import histogram_indices, predict_scores
from .shared import CLASSES, ROWS, dump, models, samples, table, verify_execution


def main():
    verify_execution()
    summaries, compositions = [], []
    for group in sorted({r['recording_group'] for r in ROWS if r['condition'] == 'night'}):
        sample = samples(group)
        fitted, _ = models(group)
        scores = predict_scores('lr3_class', sample['X'], fitted['lr3_class'])
        indices = histogram_indices(scores, 100000)
        mass = np.bincount(indices, weights=sample['weights'], minlength=100000)
        index = int(np.argmax(mass))
        selected = indices == index
        classes = sample['X'][selected, 3].astype(int)
        weights, errors = sample['weights'][selected], sample['errors'][selected]
        by_class = np.bincount(classes, weights=weights, minlength=19)
        class_errors = np.bincount(classes, weights=weights * errors, minlength=19)
        dominant = int(np.argmax(by_class))
        lower = float(np.sin((np.pi / 2) * index / 100000) ** 2)
        upper = float(np.sin((np.pi / 2) * (index + 1) / 100000) ** 2)
        np.testing.assert_allclose(by_class.sum(), mass[index], rtol=1e-12)
        np.testing.assert_allclose(class_errors.sum(), np.sum(weights * errors), rtol=1e-12)
        assert np.all(scores[selected] >= lower - 1e-15) and np.all(scores[selected] < upper + 1e-15)
        summaries.append(dict(group=group, method='lr3_class', bins=100000, largest_bin_index=index,
            original_score_lower=lower, original_score_upper_exclusive=upper,
            observed_score_min=float(scores[selected].min()), observed_score_max=float(scores[selected].max()),
            sampled_pixels=int(selected.sum()), population_weight=float(weights.sum()),
            group_population_mass=float(weights.sum() / sample['weights'].sum()),
            bin_error_rate=float(np.average(errors, weights=weights)),
            dominant_predicted_class_id=dominant, dominant_predicted_class=CLASSES[dominant],
            dominant_class_fraction_within_bin=float(by_class[dominant] / weights.sum()),
            dominant_class_error_rate_within_bin=float(class_errors[dominant] / by_class[dominant])))
        for class_id in np.flatnonzero(by_class):
            compositions.append(dict(group=group, largest_bin_index=index, predicted_class_id=int(class_id),
                predicted_class=CLASSES[class_id], sampled_pixels=int(np.count_nonzero(classes == class_id)),
                population_weight=float(by_class[class_id]), weighted_errors=float(class_errors[class_id]),
                fraction_within_bin=float(by_class[class_id] / weights.sum()),
                error_rate_within_class_and_bin=float(class_errors[class_id] / by_class[class_id])))
    table('metrics_night_largest_bins.csv', summaries)
    table('metrics_night_largest_bins_classes.csv', compositions)
    verify_execution()
    dump('logs/metric_largest_bins.json', dict(status='PASS', groups=3, method='lr3_class',
        scope='Saved weighted QA samples, largest occupied 100k arcsine score bin per Night family.',
        summaries=summaries, definitions='Bin and class masses use the same inverse-inclusion weights; errors are binary segmentation errors; class is predicted class, not GT.',
        limitations='Observed score clusters do not prove a mathematical model score floor or a causal class-offset mechanism. No fit, new normalization, or neural-network forward pass.'))


if __name__ == '__main__':
    main()
