"""Freeze a seed-42 whole-sequence split of ACDC train; never read validation."""

import csv
import hashlib
import itertools
import json
import random
import re
import shutil
from collections import Counter
from pathlib import Path

from stage1.evaluate_condition import GT_ROOT, RGB_ROOT

ROOT = Path("results/stage3_calibration/stage3b")
CONDITIONS = ("fog", "rain", "snow", "night")
SEED = 42
SPLIT_METHOD = ("Exhaustive whole-filename-prefix sequence subset allocation minimizing "
                "absolute distance from 50% images per condition; sorted sequence IDs, "
                "Python random.Random(42) per-condition tie-break over optimal masks.")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path, rows):
    with Path(path).open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def inventory():
    rows = []
    for condition in CONDITIONS:
        paths = sorted((RGB_ROOT / condition / "train").rglob("*_rgb_anon.png"))
        if len(paths) != 400:
            raise RuntimeError(f"Expected 400 train images for {condition}, found {len(paths)}")
        for path in paths:
            match = re.fullmatch(r"([A-Z0-9]+)_frame_\d+_rgb_anon\.png", path.name)
            if not match or match[1] != path.parent.name:
                raise RuntimeError(f"Ambiguous sequence ID: {path}")
            # Same suffix and parent-directory mapping as Stage 1; split=train.
            gt = GT_ROOT / condition / "train" / path.parent.name / path.name.replace(
                "_rgb_anon.png", "_gt_labelTrainIds.png")
            if not gt.is_file():
                raise FileNotFoundError(gt)
            rows.append(dict(condition=condition, sequence_id=match[1],
                             rgb_path=str(path), gt_path=str(gt)))
    return rows


def allocate(rows):
    result, summaries = [], []
    for condition in CONDITIONS:
        items = [r for r in rows if r["condition"] == condition]
        counts = Counter(r["sequence_id"] for r in items)
        sequences = sorted(counts)
        allocations = []
        best_distance = float("inf")
        for mask in itertools.product((0, 1), repeat=len(sequences)):
            n = sum(counts[s] for s, selected in zip(sequences, mask) if selected)
            distance = abs(2 * n - len(items))
            if distance < best_distance:
                best_distance, allocations = distance, [mask]
            elif distance == best_distance:
                allocations.append(mask)
        chosen = random.Random(SEED).choice(allocations)
        fit_sequences = {s for s, selected in zip(sequences, chosen) if selected}
        for row in items:
            result.append({**row, "subset": "calibration_fit" if row["sequence_id"]
                           in fit_sequences else "reliability_selection"})
        fit_images = sum(counts[s] for s in fit_sequences)
        summaries.append(dict(condition=condition, num_sequences=len(sequences),
                              num_images=len(items), calibration_fit_sequences=len(fit_sequences),
                              calibration_fit_images=fit_images,
                              reliability_selection_sequences=len(sequences)-len(fit_sequences),
                              reliability_selection_images=len(items)-fit_images))
    return result, summaries


def check_split(rows, source):
    assert len(rows) == len(source) == 1600
    assert len({r["rgb_path"] for r in rows}) == len(rows)
    assert len({r["gt_path"] for r in rows}) == len(rows)
    expected = {r["rgb_path"]: r for r in source}
    for row in rows:
        assert all(row[k] == expected[row["rgb_path"]][k]
                   for k in ("condition", "sequence_id", "rgb_path", "gt_path"))
        assert row["subset"] in ("calibration_fit", "reliability_selection")
        assert "train" in Path(row["rgb_path"]).parts and "val" not in Path(row["rgb_path"]).parts
        assert "train" in Path(row["gt_path"]).parts and "val" not in Path(row["gt_path"]).parts
    fit = [r for r in rows if r["subset"] == "calibration_fit"]
    reserved = [r for r in rows if r["subset"] == "reliability_selection"]
    for field in ("rgb_path", "gt_path", "sequence_id"):
        assert not {r[field] for r in fit} & {r[field] for r in reserved}, field


def main():
    if (ROOT / "splits").exists():
        raise RuntimeError("Split directory already exists; frozen splits must not be overwritten")
    source = inventory()
    rows, summaries = allocate(source)
    assert rows == allocate(source)[0]
    check_split(rows, source)
    protected = sorted(p for p in Path("results").rglob("*")
                       if p.is_file() and ROOT not in p.parents)
    protected += sorted(p for p in Path(".").glob("stage[1-5]/*.py")
                        if p.name not in ("prepare_stage3b_split.py", "fit_temperature.py"))
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "protected_files_before.json").open("x") as handle:
        json.dump({str(p): sha256(p) for p in protected}, handle, indent=2)
    split_dir = ROOT / "splits"
    split_dir.mkdir()
    for subset in ("calibration_fit", "reliability_selection"):
        write_csv(split_dir / f"{subset}.csv", [r for r in rows if r["subset"] == subset])
    write_csv(split_dir / "split_summary.csv", summaries)
    fit_count = sum(r["calibration_fit_images"] for r in summaries)
    projected_bytes = fit_count * (19 * 128 * 256 * 4 + 128)
    free = shutil.disk_usage(ROOT).free
    if projected_bytes > 10 * 2**30 or free < projected_bytes * 2 + 2**30:
        raise RuntimeError(f"Unexpected cache/disk budget: {projected_bytes=}, {free=}")
    lines = ["FROZEN ACDC TRAIN SPLIT - STAGE 3B", f"Seed: {SEED}", SPLIT_METHOD,
             "Sequence ID = exact filename prefix before _frame_, verified against parent directory.",
             "This groups dataset sequence IDs, not inferred longer driving trips.",
             "Only train file inventory used. Labels/predictions/metrics not used for allocation.",
             "Paths in CSVs are relative to the project root.", ""]
    lines += [str(row) for row in summaries]
    lines += ["", "PASS: all 1600 train images assigned exactly once; zero image, GT-path and sequence overlap.",
              "PASS: condition labels preserved; all train GT paths resolve.",
              f"Projected float32 native-logit cache: {projected_bytes:,} bytes ({projected_bytes/2**30:.3f} GiB).",
              f"Free disk before cache: {free:,} bytes ({free/2**30:.1f} GiB).",
              "Split is frozen before caching or fitting. Do not revise based on fitted results."]
    (split_dir / "split_summary.txt").write_text("\n".join(lines) + "\n")
    manifest = {str(p): sha256(p) for p in sorted(split_dir.iterdir()) if p.is_file()}
    (ROOT / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
