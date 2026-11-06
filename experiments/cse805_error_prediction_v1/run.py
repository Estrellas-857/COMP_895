"""Sequential phase gates, nested grouped fitting and full-pixel evaluation."""
import argparse
import csv
import importlib.metadata
import json
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from .common import ROOT, sha256, save_json, event, preservation_snapshot
from .protocol import load_protocol, read_csv, DATA
from .features import extract, configure_torch, native_cache, load_target, iter_feature_blocks
from .evaluation import (CONFIGS, C_VALUES, TARGETS, fit_pipeline, predict_scores,
                         training_threshold, HistogramAccumulator, exact_metrics,
                         HISTOGRAM_BINS, histogram_indices)


def read_json(path):
    return json.loads(Path(path).read_text())


def source_hashes():
    return {name: sha256(ROOT / name) for name in ('protocol.py', 'common.py', 'features.py', 'evaluation.py', 'run.py', 'data/metric_amendment_001.json')}


def execution_record(action, device):
    versions = {n: importlib.metadata.version(n) for n in
                ('numpy', 'scipy', 'scikit-learn', 'torch', 'transformers', 'joblib', 'threadpoolctl', 'cloudpickle', 'narwhals')}
    record = dict(action=action, device=device, source_hashes=source_hashes(), versions=versions,
                  protocol_sha256=sha256(DATA/'protocol.json'),
                  metric_amendment_sha256=sha256(DATA/'metric_amendment_001.json'),
                  qa_seed=806, train_seed=805, inference_deterministic=True,
                  cuda_available=torch.cuda.is_available(), torch_threads=4,
                  gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    save_json(ROOT / f'logs/environment_{action}.json', record)
    event('phase_start', **record)
    return record


def load_training(rows):
    X, y, weights, groups = [], [], [], []
    for row in rows:
        path = DATA / 'features' / (row['image_id'] + '.npz')
        meta = read_json(path.with_suffix('.json'))
        if meta['sha256'] != sha256(path) or meta['feature_source_sha256'] != sha256(ROOT / 'features.py'):
            raise RuntimeError('Feature sample integrity failed')
        with np.load(path, allow_pickle=False) as saved:
            X.append(saved['train_X'])
            y.append(saved['train_errors'])
            weights.append(saved['train_weights'])
            groups.append(np.repeat(row['recording_group'], len(saved['train_errors'])))
    return tuple(np.concatenate(values) for values in (X, y, weights, groups))


def strip_curve(metrics):
    return {k: v for k, v in metrics.items() if k != 'risk_coverage'}


def quantized_qa(scores, y, weights, bins=HISTOGRAM_BINS):
    """Same weighted QA population; isolate binning from pixel sampling effects."""
    indices = histogram_indices(scores, bins)
    correct = np.bincount(indices, weights=weights * (1-y), minlength=bins)
    incorrect = np.bincount(indices, weights=weights * y, minlength=bins)
    nonempty = correct + incorrect > 0
    accepted = np.cumsum(correct + incorrect)[nonempty]
    wrong = np.cumsum(incorrect)[nonempty]
    coverage, risk = accepted / weights.sum(), wrong / accepted
    checkpoints = []
    for target in TARGETS:
        index = int(np.argmin(np.abs(coverage - target)))
        checkpoints.append(dict(requested_coverage=target, actual_coverage=float(coverage[index]), risk=float(risk[index])))
    return dict(auroc=roc_auc_score(y, indices, sample_weight=weights),
                average_precision=average_precision_score(y, indices, sample_weight=weights),
                aurc=float(np.sum(np.diff(np.r_[0., coverage]) * risk)),
                error_prevalence=float(np.average(y, weights=weights)), checkpoints=checkpoints,
                method='same weighted QA pixels; grouped arcsine-sqrt bins and whole-bin RC; exact-sample RC uses random secondary keys for true ties')


def training_thresholds(X, w, models):
    thresholds = {}
    for method in CONFIGS:
        scores = predict_scores(method, X, models.get(method))
        value = training_threshold(scores, w)
        thresholds[method] = dict(value=value,
            training_sample_coverage=float(w[scores <= value].sum()/w.sum()),
            requested_coverage=.8, source='final outer-training weighted sample scores, no held-out outcomes')
    return thresholds


def evaluate_images(rows, models, thresholds, device, output, group):
    output.mkdir(parents=True, exist_ok=True)
    accumulators = {m: HistogramAccumulator(threshold=thresholds[m]['value']) for m in CONFIGS}
    qa_scores = {m: [] for m in CONFIGS}
    qa_y, qa_w, qa_ties = [], [], []
    images, feature_seconds = [], 0.
    score_seconds = {m: 0. for m in CONFIGS}
    started = time.perf_counter()
    for number, row in enumerate(rows, 1):
        image_started = time.perf_counter()
        native, _, _ = native_cache(row, device)
        gt = load_target(row)
        saved_path = DATA / 'features' / (row['image_id'] + '.npz')
        with np.load(saved_path, allow_pickle=False) as saved:
            sample = {name: saved['qa_' + name] for name in ('X', 'errors', 'weights', 'tie_keys', 'indices')}
        qa_y.append(sample['errors']); qa_w.append(sample['weights']); qa_ties.append(sample['tie_keys'])
        image_acc = {m: HistogramAccumulator(threshold=thresholds[m]['value']) for m in CONFIGS}
        image_features = 0.
        iterator = iter(iter_feature_blocks(native, device))
        while True:
            before = time.perf_counter()
            try:
                first, last, X, _ = next(iterator)
            except StopIteration:
                break
            image_features += time.perf_counter() - before
            lo, hi = np.searchsorted(sample['indices'], [first, last])
            np.testing.assert_array_equal(X[sample['indices'][lo:hi] - first], sample['X'][lo:hi])
            target = gt.ravel()[first:last]
            valid = target < 19
            values = X[valid]
            errors = (values[:, 3] != target[valid]).astype(np.uint8)
            for method in CONFIGS:
                before = time.perf_counter()
                scores = predict_scores(method, values, models.get(method))
                score_seconds[method] += time.perf_counter() - before
                image_acc[method].update(scores, errors)
        feature_seconds += image_features
        image_results = {}
        for method in CONFIGS:
            accumulators[method].merge(image_acc[method])
            image_results[method] = strip_curve(image_acc[method].finalize())
            qa_scores[method].append(predict_scores(method, sample['X'], models.get(method)))
        count_sets = {(r['num_pixels'], r['num_errors']) for r in image_results.values()}
        if len(count_sets) != 1:
            raise RuntimeError('Methods evaluated different pixels or error labels')
        images.append(dict(image_id=row['image_id'], condition=row['condition'], recording_group=row['recording_group'],
                           methods=image_results, feature_seconds=image_features,
                           total_seconds=time.perf_counter()-image_started))
        if number == 1 or number % 25 == 0 or number == len(rows):
            print(f'Evaluate {group}: {number}/{len(rows)} images; {time.perf_counter()-started:.1f}s', flush=True)
            event('evaluation_progress', group=group, completed=number, total=len(rows))
    y, w, ties = np.concatenate(qa_y), np.concatenate(qa_w), np.concatenate(qa_ties)
    results, qa, histograms = {}, {}, {}
    for method in CONFIGS:
        metrics = accumulators[method].finalize()
        curve_path = output / (method + '_curve.npz')
        np.savez_compressed(curve_path, **metrics.pop('risk_coverage'))
        metrics['curve_path'] = str(curve_path)
        results[method] = metrics
        scores = np.concatenate(qa_scores[method])
        qa[method] = dict(exact=strip_curve(exact_metrics(scores, y, w, ties)),
                          quantized=quantized_qa(scores, y, w))
        histograms[method + '_correct'] = accumulators[method].correct
        histograms[method + '_incorrect'] = accumulators[method].incorrect
    np.savez_compressed(output / 'histograms.npz', **histograms)
    save_json(output / 'image_results.json', images)
    result = dict(group=group, weather=rows[0]['condition'], num_images=len(rows), methods=results, qa=qa,
                  protocol_sha256=sha256(DATA/'protocol.json'),
                  metric_amendment_sha256=sha256(DATA/'metric_amendment_001.json'),
                  image_records=str(output / 'image_results.json'), source_hashes=source_hashes(),
                  runtime=dict(total_seconds=time.perf_counter()-started, feature_seconds=feature_seconds,
                               score_seconds=score_seconds,
                               boundary='cached native logits plus GT reading, full feature regeneration, scoring, metrics and serialization; excludes LR fitting/backbone forward'))
    save_json(output / 'result.json', result)
    return result


def select_and_fit(outer, arrays, protocol_digest):
    X, y, w, groups = arrays
    group = outer['eval_group']
    folder = DATA / 'models' / group
    selection_path = folder / 'selection.json'
    model_paths = {m: folder / (m + '.joblib') for m in ('lr3', 'lr3_class')}
    if selection_path.exists():
        selection = read_json(selection_path)
        if selection['source_hashes'] != source_hashes() or selection['protocol_sha256'] != protocol_digest:
            raise RuntimeError('Training source/protocol changed; cannot reuse fitted models')
        for m, p in model_paths.items():
            if sha256(p) != selection['model_hashes'][m]:
                raise RuntimeError('Model bytes changed')
        return {m: joblib.load(p) for m, p in model_paths.items()}, selection
    folder.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    candidates, selected, models = [], {}, {}
    for method in ('lr3', 'lr3_class'):
        for C in C_VALUES:
            scores, records = [], []
            for fold in outer['inner_folds']:
                fit = np.isin(groups, fold['fit_groups'])
                dev = np.isin(groups, fold['dev_groups'])
                assert not np.any(fit & dev) and not np.any((fit | dev) & (groups == group))
                before = time.perf_counter()
                try:
                    model = fit_pipeline(method, C, X[fit], y[fit], w[fit])
                    predicted = predict_scores(method, X[dev], model)
                    if len(np.unique(y[dev])) != 2:
                        raise ValueError('Inner validation has one error class')
                    ap = float(average_precision_score(y[dev], predicted, sample_weight=w[dev]))
                    scores.append(ap)
                    records.append(dict(inner_fold=fold['inner_fold'], ap=ap, diagnostics=model.diagnostics_,
                                        seconds=time.perf_counter()-before, status='PASS'))
                except (RuntimeError, ValueError) as error:
                    records.append(dict(inner_fold=fold['inner_fold'], status='FAILED', exception=str(error)))
            candidate = dict(config=method, C=C, innerfoldAP=scores, inner_records=records,
                             meanAP=float(np.mean(scores)) if len(scores)==3 else None,
                             status='PASS' if len(scores)==3 else 'FAILED')
            candidates.append(candidate)
            event('candidate', outer_group=group, **candidate)
            print(f'Tune {group} {method} C={C}: {candidate["meanAP"]}', flush=True)
        usable = [r for r in candidates if r['config']==method and r['status']=='PASS']
        if not usable:
            save_json(folder / 'failed_selection.json', dict(candidates=candidates))
            raise RuntimeError('Every preregistered candidate failed; no silent model replacement')
        best = max(r['meanAP'] for r in usable)
        chosen = min(r['C'] for r in usable if best-r['meanAP'] <= 1e-12)
        selected[method] = chosen
        fit = np.isin(groups, outer['train_groups'])
        assert not np.any(fit & (groups == group))
        model = fit_pipeline(method, chosen, X[fit], y[fit], w[fit])
        models[method] = model
        joblib.dump(model, model_paths[method])
    fit = np.isin(groups, outer['train_groups'])
    thresholds = training_thresholds(X[fit], w[fit], models)
    selection = dict(group=group, candidates=candidates, selected=selected, thresholds=thresholds,
                     train_groups=outer['train_groups'], source_hashes=source_hashes(),
                     protocol_sha256=protocol_digest, model_hashes={m:sha256(p) for m,p in model_paths.items()},
                     runtime=dict(total_seconds=time.perf_counter()-started),
                     diagnostics={m:model.diagnostics_ for m,model in models.items()})
    save_json(selection_path, selection)
    return models, selection


def smoke(rows, device, digest):
    train, heldout = [], []
    for condition in ('fog', 'night', 'rain', 'snow'):
        candidates = [r for r in rows if r['condition']==condition]
        chosen = next(r for r in candidates if r['native_cache_path'])
        train.append(chosen)
        heldout.append(next(r for r in candidates if not r['native_cache_path'] and r['recording_group'] != chosen['recording_group']))
    assert not {r['recording_group'] for r in train} & {r['recording_group'] for r in heldout}
    extract(train + heldout, device, DATA / 'features', digest)
    X,y,w,_ = load_training(train)
    models = {m:fit_pipeline(m,.1,X,y,w) for m in ('lr3','lr3_class')}
    thresholds = training_thresholds(X,w,models)
    checked=[]
    for row in heldout:
        result = evaluate_images([row],models,thresholds,device,
                                 ROOT / 'smoke' / row['condition'],row['recording_group'])
        counts={(r['num_pixels'],r['num_errors']) for r in result['methods'].values()}
        assert len(counts)==1
        for r in result['methods'].values():
            assert 0 <= r['auroc'] <= 1 and 0 <= r['average_precision'] <= 1
            assert r['fixed_threshold']['complete_population']
        checked.append(dict(image_id=row['image_id'], group=row['recording_group'], counts=list(counts)))
    record=dict(status='PASS', train_image_ids=[r['image_id'] for r in train], heldout=checked,
                protocol_sha256=digest, source_hashes=source_hashes(), device=device,
                limitations='Smoke results only; fixed C=.1 and one image/group, not scientific model selection')
    save_json(ROOT / 'smoke/complete.json',record)
    event('smoke_complete', **record)
    print('End-to-end smoke PASS',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('snapshot','smoke','extract','evaluate','preserve'))
    parser.add_argument('--device', choices=('cpu','cuda'), default='cuda')
    args=parser.parse_args()
    if args.phase in ('snapshot','preserve'):
        current=preservation_snapshot()
        if args.phase=='snapshot':
            path=ROOT/'logs/preservation_before.json'
            if path.exists():
                if read_json(path)!=current: raise RuntimeError('Original files differ from existing snapshot')
            else: save_json(path,current)
        else:
            before=read_json(ROOT/'logs/preservation_before.json')
            if before!=current: raise RuntimeError('Original protected files changed')
            save_json(ROOT/'logs/preservation_after.json',dict(status='PASS',files=len(current),
                      bytes=sum(r['bytes'] for r in current.values())))
        print(args.phase, len(current), 'files',flush=True)
        return
    load_protocol()
    digest=sha256(DATA/'protocol.json')
    amendment=read_json(DATA/'metric_amendment_001.json')
    if amendment['base_protocol_sha256'] != digest or amendment['replacement_metrics']['histogram_bins'] != HISTOGRAM_BINS:
        raise RuntimeError('Metric amendment does not match the frozen protocol/runtime')
    rows=read_csv(DATA/'metadata_train.csv')
    execution_record(args.phase,args.device)
    configure_torch(args.device)
    if args.phase=='smoke':
        smoke(rows,args.device,digest)
        return
    check=read_json(ROOT/'smoke/complete.json')
    if check['status']!='PASS' or check['source_hashes']!=source_hashes() or check['device']!=args.device:
        raise RuntimeError('Smoke must pass with these sources and backend before full execution')
    if args.phase=='extract':
        records=extract(rows,args.device,DATA/'features',digest)
        assert len(records)==1600
        save_json(DATA/'features/complete.json',dict(status='PASS',num_images=len(records),protocol_sha256=digest,
                  source_hashes=source_hashes(),manifest_sha256=sha256(DATA/'features/manifest.json')))
        return
    complete=read_json(DATA/'features/complete.json')
    if complete['source_hashes']!=source_hashes(): raise RuntimeError('Sources changed after extraction')
    arrays=load_training(rows)
    assignments=read_json(DATA/'assignments.json')
    if isinstance(assignments,dict): assignments=assignments['outer_folds']
    for outer in assignments:
        group=outer['eval_group']
        output=ROOT/'results/folds'/group
        if (output/'result.json').exists():
            if read_json(output/'result.json')['source_hashes']!=source_hashes():
                raise RuntimeError('Sources changed after partial evaluation')
            continue
        models,selection=select_and_fit(outer,arrays,digest)
        evaluation_rows=[r for r in rows if r['recording_group']==group]
        result=evaluate_images(evaluation_rows,models,selection['thresholds'],args.device,output,group)
        event('outer_complete', group=group, num_images=len(evaluation_rows), runtime=result['runtime'])
    save_json(ROOT/'results/evaluation_complete.json',dict(status='PASS',num_groups=17,num_images=1600,
              protocol_sha256=digest,metric_amendment_sha256=sha256(DATA/'metric_amendment_001.json'),source_hashes=source_hashes()))


if __name__=='__main__':
    main()
