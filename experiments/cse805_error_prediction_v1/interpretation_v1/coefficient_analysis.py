"""Post-hoc coefficient and predicted-class diagnostics; never fit a model."""
from collections import Counter
import time

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

from .shared import OUT, BASE, ROWS, CLASSES, read, models, samples, table, dump, verify_execution
from ..evaluation import CONFIGS, predict_scores
from ..common import sha256

FEATURES = ("msp", "normalized_entropy", "margin")


def inspect_pipeline(model, method):
    transformer = model.named_steps["features"]
    classifier = model.named_steps["classifier"]
    if list(classifier.classes_) != [0, 1] or model.diagnostics_["config"] != method:
        raise ValueError("Class/score orientation mismatch")
    continuous = transformer.transformers_[0]
    if continuous[0] != "continuous" or list(continuous[2]) != [0, 1, 2]:
        raise ValueError("Continuous feature order changed")
    scaler = transformer.named_transformers_["continuous"]
    if transformer.output_indices_["continuous"] != slice(0, 3):
        raise ValueError("Transformed continuous column order changed")
    expected_width = 22 if method == "lr3_class" else 3
    if classifier.coef_.shape != (1, expected_width):
        raise ValueError("Unexpected classifier width")
    if method == "lr3_class":
        categorical = transformer.transformers_[1]
        encoder = transformer.named_transformers_["class"]
        if categorical[0] != "class" or list(categorical[2]) != [3]:
            raise ValueError("Predicted-class feature index changed")
        if (transformer.output_indices_["class"] != slice(3, 22) or encoder.drop is not None or
                len(encoder.categories_) != 1 or not np.array_equal(encoder.categories_[0], np.arange(19))):
            raise ValueError("Class one-hot mapping changed")
        probe = np.zeros((19, 4), dtype=np.float32)
        probe[:, 3] = np.arange(19)
        np.testing.assert_array_equal(transformer.transform(probe)[:, 3:], np.eye(19))
    beta = classifier.coef_[0, :3].copy()
    intercept = float(classifier.intercept_[0])
    gamma = classifier.coef_[0, 3:].astype(np.float64) if method == "lr3_class" else np.zeros(19)
    raw_beta = beta / scaler.scale_
    raw_intercept = intercept - np.dot(raw_beta, scaler.mean_)
    return dict(beta=beta, gamma=gamma, scaler_mean=scaler.mean_.copy(), scaler_scale=scaler.scale_.copy(),
                raw_beta=raw_beta, intercept=intercept, raw_intercept=float(raw_intercept),
                centered_raw_intercept=float(raw_intercept + gamma.mean()),
                road_raw_intercept=float(raw_intercept + gamma[0]),
                feature_names=list(transformer.get_feature_names_out()))


def decompose(model, method, algebra, X):
    transformed = model.named_steps["features"].transform(X)
    continuous = (transformed[:, :3] @ algebra["beta"].reshape(-1, 1)).ravel()
    class_component = algebra["gamma"][X[:, 3].astype(np.int64)]
    reconstructed = algebra["intercept"] + continuous + class_component
    reference = model.decision_function(X)
    exact_error = float(np.max(abs(reconstructed - reference))) if len(X) else 0.
    if exact_error > 1e-10:
        raise ValueError("Transformed coefficient decomposition failed")
    raw = X[:, :3].astype(np.float64) @ algebra["raw_beta"] + algebra["raw_intercept"] + class_component
    raw_error = float(np.max(abs(raw - reference))) if len(X) else 0.
    # StandardScaler rounds transformed float32 inputs; raw-coordinate algebra is float64.
    if raw_error > 1e-4:
        raise ValueError("Unexpected raw-coordinate reconstruction discrepancy")
    return dict(logit=reference, continuous=continuous, class_component=class_component,
                transformed_max_abs_error=exact_error, raw_units_max_abs_error=raw_error)


def profile(X, y, weights, scores, mask):
    count = int(np.count_nonzero(mask))
    wrong_count = int(np.count_nonzero(y[mask]))
    w = weights[mask]
    mass = float(w.sum())
    total = float(weights.sum())
    result = dict(sample_pixels=count, sample_errors=wrong_count, weighted_pixels=mass,
                  class_population_share=mass / total,
                  error_rate=float(np.average(y[mask], weights=w)) if count else None,
                  small_sample=count < 100 or wrong_count < 20 or count - wrong_count < 20)
    for i, name in enumerate(FEATURES):
        result[f"mean_{name}"] = float(np.average(X[mask, i], weights=w)) if count else None
    for method in CONFIGS:
        result[f"mean_{method}_score"] = float(np.average(scores[method][mask], weights=w)) if count else None
    return result


def class_profiles(group, weather, train, heldout, train_scores, heldout_scores, decomposition, algebra):
    profiles, high_confidence, ranking = [], [], []
    for class_id, class_name in enumerate(CLASSES):
        train_mask, held_mask = train["X"][:, 3] == class_id, heldout["X"][:, 3] == class_id
        base = dict(group=group, weather=weather, predicted_class_id=class_id, predicted_class=class_name)
        row = {**base, **{"train_" + k: v for k, v in profile(train["X"], train["errors"], train["weights"], train_scores, train_mask).items()},
               **{"heldout_" + k: v for k, v in profile(heldout["X"], heldout["errors"], heldout["weights"], heldout_scores, held_mask).items()},
               "lrclass_conditional_offset_centered": float(algebra["lr3_class"]["gamma"][class_id] - algebra["lr3_class"]["gamma"].mean()),
               "lrclass_conditional_offset_vs_road": float(algebra["lr3_class"]["gamma"][class_id] - algebra["lr3_class"]["gamma"][0])}
        count = np.count_nonzero(held_mask)
        for method in ("lr3", "lr3_class"):
            for component in ("logit", "continuous", "class_component"):
                values = decomposition[method][component]
                row[f"heldout_mean_{method}_{component}"] = float(np.average(values[held_mask], weights=heldout["weights"][held_mask])) if count else None
        profiles.append(row)
        for cutoff in (.9, .99):
            t = train_mask & (train["X"][:, 0] >= cutoff)
            h = held_mask & (heldout["X"][:, 0] >= cutoff)
            high_confidence.append({**base, "msp_cutoff": cutoff,
                **{"train_" + k: v for k, v in profile(train["X"], train["errors"], train["weights"], train_scores, t).items()},
                **{"heldout_" + k: v for k, v in profile(heldout["X"], heldout["errors"], heldout["weights"], heldout_scores, h).items()}})
        y, w = heldout["errors"][held_mask], heldout["weights"][held_mask]
        positives = float(w @ y)
        negatives = float(w.sum() - positives)
        for method in CONFIGS:
            score = heldout_scores[method][held_mask]
            ranking.append({**base, "method": method, "qa_sample_pixels": int(count),
                            "qa_sample_errors": int(np.count_nonzero(y)),
                            "weighted_pixels": float(w.sum()),
                            "error_prevalence": positives / w.sum() if count else None,
                            "small_sample": row["heldout_small_sample"],
                            "auroc": float(roc_auc_score(y, score, sample_weight=w)) if positives and negatives else None,
                            "average_precision": float(average_precision_score(y, score, sample_weight=w)) if positives else None})
    return profiles, high_confidence, ranking


def main():
    started = time.perf_counter()
    complete = verify_execution()
    groups = sorted({r["recording_group"] for r in ROWS})
    train_parts = {group: samples(group, "train") for group in groups}
    train_all = {key: np.concatenate([train_parts[g][key] for g in groups]) for key in ("X", "errors", "weights")}
    group_ids = np.concatenate([np.full(len(train_parts[g]["errors"]), i, dtype=np.int8) for i, g in enumerate(groups)])
    del train_parts
    coefficients, offsets, intercepts, slope_comparisons = [], [], [], []
    profiles, high_confidence, rankings, checks = [], [], [], []
    source_hashes = {}
    with threadpool_limits(limits=4):
        for group_index, group in enumerate(groups):
            weather = next(r["condition"] for r in ROWS if r["recording_group"] == group)
            fitted, selection = models(group)
            if set(selection["train_groups"]) != set(groups) - {group}:
                raise ValueError("Stored outer model training groups disagree")
            mask = group_ids != group_index
            train = {key: value[mask] for key, value in train_all.items()}
            heldout = samples(group, "qa")
            algebra = {m: inspect_pipeline(fitted[m], m) for m in ("lr3", "lr3_class")}
            for name in ("scaler_mean", "scaler_scale"):
                np.testing.assert_array_equal(algebra["lr3"][name], algebra["lr3_class"][name])
            decomposition = {m: decompose(fitted[m], m, algebra[m], heldout["X"]) for m in ("lr3", "lr3_class")}
            for method in ("lr3", "lr3_class"):
                folder = BASE / "data/models" / group
                source_hashes[str((folder / (method + ".joblib")).relative_to(BASE))] = sha256(folder / (method + ".joblib"))
                a = algebra[method]
                if fitted[method].diagnostics_["num_training_pixels"] != len(train["errors"]):
                    raise ValueError("Training sample count differs from fitted model")
                for i, feature in enumerate(FEATURES):
                    coefficients.append(dict(group=group, weather=weather, method=method, feature=feature,
                        standardized_coefficient=float(a["beta"][i]), raw_unit_slope=float(a["raw_beta"][i]),
                        scaler_mean=float(a["scaler_mean"][i]), scaler_scale=float(a["scaler_scale"][i]),
                        C=selection["selected"][method]))
                intercepts.append(dict(group=group, weather=weather, method=method,
                    standardized_intercept=a["intercept"], raw_intercept=a["raw_intercept"],
                    class_offset_mean=float(a["gamma"].mean()),
                    raw_intercept_with_centered_class_offsets=a["centered_raw_intercept"],
                    raw_intercept_with_road_reference=a["road_raw_intercept"]))
                checks.append(dict(group=group, method=method, qa_pixels=len(heldout["errors"]),
                                   feature_names=a["feature_names"],
                                   transformed_max_abs_error=decomposition[method]["transformed_max_abs_error"],
                                   raw_units_max_abs_error=decomposition[method]["raw_units_max_abs_error"]))
            for i, feature in enumerate(FEATURES):
                slope_comparisons.append(dict(group=group, weather=weather, feature=feature,
                    lr3_raw_slope=float(algebra["lr3"]["raw_beta"][i]),
                    lrclass_raw_slope=float(algebra["lr3_class"]["raw_beta"][i]),
                    raw_slope_difference=float(algebra["lr3_class"]["raw_beta"][i] - algebra["lr3"]["raw_beta"][i]),
                    lr3_standardized_coefficient=float(algebra["lr3"]["beta"][i]),
                    lrclass_standardized_coefficient=float(algebra["lr3_class"]["beta"][i])))
            for class_id, class_name in enumerate(CLASSES):
                gamma = algebra["lr3_class"]["gamma"]
                offsets.append(dict(group=group, weather=weather, predicted_class_id=class_id, predicted_class=class_name,
                    original_offset=float(gamma[class_id]), centered_offset=float(gamma[class_id] - gamma.mean()),
                    contrast_vs_road=float(gamma[class_id] - gamma[0])))
            train_scores = {m: predict_scores(m, train["X"], fitted.get(m)) for m in CONFIGS}
            heldout_scores = {m: predict_scores(m, heldout["X"], fitted.get(m)) for m in CONFIGS}
            p, h, r = class_profiles(group, weather, train, heldout, train_scores, heldout_scores, decomposition, algebra)
            profiles.extend(p); high_confidence.extend(h); rankings.extend(r)
            print(f"Coefficient/class diagnostics {group_index+1}/{len(groups)}: {group}; {time.perf_counter()-started:.1f}s", flush=True)
    paths = [table("coefficients_continuous.csv", coefficients), table("coefficients_class_offsets.csv", offsets),
             table("coefficients_intercepts.csv", intercepts), table("coefficients_slopes_comparison.csv", slope_comparisons),
             table("class_profile_train_heldout.csv", profiles), table("class_profile_high_confidence.csv", high_confidence),
             table("within_class_metrics.csv", rankings)]
    verify_execution()
    dump("logs/coefficient_analysis.json", dict(status="PASS", groups=17, models=34,
        elapsed_seconds=time.perf_counter()-started, model_source_hashes=source_hashes,
        script_sha256=sha256(__file__), checks=checks,
        completed_experiment_source_hashes=complete["source_hashes"],
        output_hashes={str(p.relative_to(OUT)): sha256(p) for p in paths},
        sampling="Stored uniform valid-pixel samples, train512/image and QA8192/image, inverse inclusion weights; all metrics here are sample estimates",
        small_class_flag="Descriptive flag: fewer than100 sampled pixels OR fewer than20 sampled errors OR fewer than20 sampled correct pixels; all classes retained",
        high_confidence="Exploratory prespecified diagnostic cutoffs MSP>=0.9 and >=0.99; no model/threshold retuning",
        coefficient_meaning="Conditional log-odds contrasts, not causal importance or class error rates; correlated features and refitted continuous slopes prevent attributing all model differences to class offsets",
        class_share_meaning="Estimated class/subset pixel mass divided by full partition valid-pixel mass; high-confidence shares are not normalized within class",
        independence="No pixel-IID significance tests; filename groups are recording proxies, not proven independent sessions"))
    print("Completed coefficient and class tables", flush=True)


if __name__ == "__main__":
    main()
