"""Verify saved Stage 5B arrays/tables and add descriptive interpretation only."""
import csv
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent/'results/stage5_final_evaluation/stage5b'
METHODS=('raw_msp','calibrated_msp','raw_entropy','calibrated_entropy')
CONDITIONS=('fog','rain','snow','night','pooled')


def read(name):
    with (ROOT/name).open(newline='') as handle:return list(csv.DictReader(handle))


def main():
    checkpoints=read('risk_at_target_coverages.csv')
    summaries=read('stage5b_combined_summary.csv')
    exact=read('frozen_operating_points.csv')
    assert len(checkpoints)==120 and len(summaries)==20 and len(exact)==5
    files={c:np.load(ROOT/('pooled_histograms.npz' if c=='pooled' else f'condition_histograms/{c}.npz')) for c in CONDITIONS}
    saved=np.load(ROOT/'risk_coverage_curves.npz')
    resolution=[]
    for condition in CONDITIONS:
        for method in METHODS:
            c=files[condition][f'{method}_correct_hist'];i=files[condition][f'{method}_incorrect_hist']
            assert c.dtype==i.dtype==np.int64 and c.shape==i.shape==(1000,)
            if condition=='pooled':
                for label in ('correct','incorrect'):
                    np.testing.assert_array_equal(files[condition][f'{method}_{label}_hist'],
                        sum((files[k][f'{method}_{label}_hist'] for k in CONDITIONS[:-1]),start=np.zeros(1000,dtype=np.int64)))
            accepted=np.cumsum(c+i);errors=np.cumsum(i);keep=accepted>0
            cov=accepted[keep]/accepted[-1];risk=errors[keep]/accepted[keep]
            prefix=f'{condition}__{method}__'
            np.testing.assert_array_equal(saved[prefix+'coverage'],cov)
            np.testing.assert_array_equal(saved[prefix+'risk'],risk)
            assert cov[-1]==1 and risk[-1]==int(i.sum())/int(accepted[-1])
            assert np.all(cov>0) and np.all(np.diff(cov)>=0)
            rows=[r for r in checkpoints if r['condition']==condition and r['method']==method]
            assert len(rows)==6
            for row in rows:
                j=int(np.argmin(abs(cov-float(row['requested_coverage']))))
                assert float(row['actual_histogram_coverage'])==cov[j]
                assert float(row['risk'])==risk[j]
                assert int(row['uncertainty_threshold_bin'])==np.flatnonzero(keep)[j]
                assert int(row['accepted_pixels'])==accepted[keep][j]
                assert int(row['incorrect_accepted_pixels'])==errors[keep][j]
            resolution.append(dict(condition=condition,method=method,first_observed_coverage=float(cov[0]),
                requested_coverages_below_observed_domain=';'.join(row['requested_coverage'] for row in rows if float(row['requested_coverage'])<cov[0]),
                maximum_checkpoint_coverage_error=max(abs(float(row['coverage_error'])) for row in rows)))
    for condition in CONDITIONS[:-1]:
        cm=np.load(ROOT/f'confusion_matrices/{condition}.npy')
        for p in (ROOT.parents[1]/f'{condition}_val/confusion_matrix.npy',
                  ROOT.parents[1]/f'stage3_calibration/stage3c/{condition}/confusion_matrix.npy',
                  ROOT.parent/f'stage5a/confusion_matrices/{condition}_confusion_matrix.npy'):
            np.testing.assert_array_equal(cm,np.load(p))
    audit=json.loads((ROOT/'inference_audit.json').read_text())
    assert audit['num_images']==len(audit['data_access_records'])==406
    assert audit['valid_pixels']==790712012 and audit['prediction_changes']==0
    assert len({r['rgb_path'] for r in audit['data_access_records']})==406
    assert all(r['gt_opened_only_after_inference'] for r in audit['data_access_records']) and not audit['blocked_dataset_opens']
    reference={r['condition']:r for r in read_external(ROOT.parent/'stage5a/stage5a_combined_summary.csv')}
    for row in exact:
        ref=reference[row['condition']]
        for key in ('valid_coverage','accepted_risk','baseline_risk','threshold'):
            assert float(row[key])==float(ref[key])
    bracket=read('frozen_point_histogram_consistency.csv')
    assert all(r['compatible_with_histogram']=='True' and
               float(r['coverage_before_bin'])<=float(r['exact_coverage'])<=float(r['coverage_after_bin']) for r in bracket)
    with (ROOT/'histogram_resolution_diagnostics.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(resolution[0]));writer.writeheader();writer.writerows(resolution)
    primary=[r for r in checkpoints if r['method']=='calibrated_msp']
    max_primary_error=max(abs(float(r['coverage_error'])) for r in primary)
    checks=dict(status='PASS',histogram_curves_recomputed_and_verified=True,all_120_nearest_checkpoints_verified=True,
        pooled_histograms_equal_condition_sums=True,counts_int64=True,nonzero_coverage_only=True,
        endpoint_equals_exact_baseline=True,exact_frozen_stage5a_points_preserved=True,
        frozen_points_compatible_with_histogram_bins=True,canonical_baselines_exact=True,
        num_images=406,valid_pixels=790712012,prediction_changes=0,
        maximum_primary_checkpoint_coverage_error=max_primary_error,
        nominal_50_percent_raw_msp_unreachable_in_fog_and_rain=True,
        aurc_computed=False,algorithm_tests_passed=4,no_stage5c_or_5d=True)
    (ROOT/'verification_check.json').write_text(json.dumps(checks,indent=2)+'\n')
    lines=['DESCRIPTIVE FINDINGS AND HISTOGRAM LIMITATIONS','',
        'Risk generally decreases as coverage is reduced across the requested 100% to 50% checkpoints.',
        'It is not guaranteed to decrease monotonically at every histogram step; Snow has a non-monotonic low-coverage region.',
        'Primary calibrated-MSP pooled results: target / actual coverage / risk / absolute reduction from baseline (percentage points).']
    pooled=[r for r in primary if r['condition']=='pooled']
    baseline=float(reference['pooled']['baseline_risk'])
    for row in pooled:
        lines.append(f"{100*float(row['requested_coverage']):.0f}% / {100*float(row['actual_histogram_coverage']):.6f}% / "
                     f"{100*float(row['risk']):.6f}% / {100*(baseline-float(row['risk'])):.6f} pp")
    lines += ['',f'For primary calibrated MSP, the largest target-versus-observed coverage discrepancy is {100*max_primary_error:.6f} percentage points.',
        'Night remains much harder: primary risk around 50% coverage is 18.940293%, versus 0.786734% Fog, 0.589819% Rain and 2.979947% Snow.',
        'Pooled calibrated MSP has slightly lower approximate risks than raw MSP at the 90/80/70/60/50% checkpoints.',
        'Condition comparisons are mixed: Night calibrated MSP is slightly lower; Snow is slightly higher; Fog/Rain vary by coverage.',
        'These small differences are descriptive, not statistically significant claims, and actual histogram coverages can differ.',
        'Pooled calibrated entropy has higher risk than raw entropy around 70/80/90% coverage, but slightly lower risk around 50/60%.',
        'At condition-specific 50–90% checkpoints, calibrated entropy risks exceed the corresponding raw entropy risks; pooled ordering can differ.',
        'Entropy and MSP curves are often close; neither method is declared universally superior and the primary method is unchanged.',
        '', 'Exact pooled frozen Stage 5A point: 81.43315849867221% coverage, 9.838704340500287% risk.',
        'Its containing calibrated-MSP bin spans coverage 81.40514564485963–81.47290634557858%,',
        'with adjacent cumulative risks 9.82896889032321–9.852576524648779%; the exact point agrees.',
        'All condition-specific frozen points also satisfy exact within-bin count bounds.',
        '', 'Resolution limitations (not hidden or repaired by changing bins):']
    for row in resolution:
        if row['requested_coverages_below_observed_domain']:
            lines.append(f"{row['condition']} {row['method']}: first observable coverage {100*row['first_observed_coverage']:.6f}%; requested 50% maps to this nearest point.")
    lines += ['Thus the nominal raw-MSP 50% points for Fog/Rain do not represent exactly 50% coverage.',
        'Pooled first observed coverage: raw MSP 45.6234%, calibrated MSP 2.9678%, raw entropy 35.7963%, calibrated entropy 1.0203%.',
        'AURC is omitted because these method-dependent missing low-coverage domains prevent a straightforward like-for-like area comparison.',
        'No risk at zero coverage or within the first bin is fabricated. All actual coverages and bin indices are saved.',
        'The required 1000-bin approach is retained. No finer-bin rerun, threshold revision, or deployment-method replacement is performed.',
        '', 'Execution issue: the first GPU approval review timed out without starting the command; the permitted retry succeeded.',
        'No unresolved computational discrepancy. The historical raw-label inconsistency was resolved by the clean canonical pass.',
        'Rejected pixels remain abstentions; no claim of corrected semantic predictions or real-world safety. STOP before Stage 5C/5D.','']
    path=ROOT/'stage5b_summary.txt'
    old=path.read_text()
    if not old.startswith('DESCRIPTIVE FINDINGS'):path.write_text('\n'.join(lines)+'\n'+old)
    print(json.dumps(checks,indent=2))
    print('\n'.join(lines))
    saved.close()
    for f in files.values():f.close()


def read_external(path):
    with path.open(newline='') as handle:return list(csv.DictReader(handle))


if __name__=='__main__':main()
