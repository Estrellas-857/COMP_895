"""Shared location and portable references for the external ACDC dataset."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_PREFIX = Path("data/ACDC")
DATASET_DIRECTORIES = {"rgb_anon_trainvaltest", "gt_trainval", "gt_trainval_ref"}


def get_acdc_root():
    """Use ACDC_ROOT when set, otherwise ~/Documents/ACDC."""
    configured = os.environ.get("ACDC_ROOT")
    root = Path(configured).expanduser() if configured else Path.home() / "Documents/ACDC"
    if not root.is_absolute():
        raise ValueError("ACDC_ROOT must be an absolute path (or start with ~)")
    return root.resolve()


ACDC_ROOT = get_acdc_root()
RGB_ROOT = ACDC_ROOT / "rgb_anon_trainvaltest/rgb_anon"
GT_ROOT = ACDC_ROOT / "gt_trainval/gt"
REF_GT_ROOT = ACDC_ROOT / "gt_trainval_ref/gt"


def acdc_path(reference):
    """Resolve a current path or a legacy manifest reference into ACDC_ROOT.

    Frozen CSV/JSON records keep their original data/ACDC/... identifiers and
    hashes. That prefix is a logical dataset identifier, not a local directory.
    Dataset-relative paths and former absolute project paths are also supported.
    """
    path = Path(reference).expanduser()
    if path.is_absolute():
        try:
            relative = path.relative_to(ACDC_ROOT)
        except ValueError:
            try:
                relative = path.relative_to(PROJECT_ROOT / LEGACY_PREFIX)
            except ValueError as error:
                raise ValueError(f"Path is outside the configured ACDC dataset: {path}") from error
    elif path.is_relative_to(LEGACY_PREFIX):
        relative = path.relative_to(LEGACY_PREFIX)
    else:
        relative = path
    if not relative.parts or relative.parts[0] not in DATASET_DIRECTORIES:
        raise ValueError(f"Not an ACDC image/label reference: {reference}")
    result = (ACDC_ROOT / relative).resolve()
    if not result.is_relative_to(ACDC_ROOT):
        raise ValueError(f"Dataset reference escapes ACDC_ROOT: {reference}")
    return result


def dataset_reference(path):
    """Return the stable identifier used by frozen split/cache/audit records."""
    relative = acdc_path(path).relative_to(ACDC_ROOT)
    return (LEGACY_PREFIX / relative).as_posix()
