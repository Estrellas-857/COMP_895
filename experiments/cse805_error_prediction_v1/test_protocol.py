"""Structural leakage and parser checks, requiring no dataset or trained model."""
import copy
from pathlib import Path
import unittest

from .protocol import recording_group, parse_image, group_summary, build_assignments, validate_assignments


class ProtocolTests(unittest.TestCase):
    def rows(self):
        rows = []
        for i, (condition, n) in enumerate((("fog", 4), ("fog", 3), ("night", 8),
                ("night", 2), ("rain", 5), ("rain", 1), ("snow", 7), ("snow", 2))):
            for j in range(n):
                rows.append({"image_id": f"{i}/{j}", "dataset_split": "train", "condition": condition,
                             "sequence_id": f"GOPR{i:04d}", "recording_group": f"gopro_{i:04d}"})
        return rows

    def test_possible_chapters_stay_together(self):
        self.assertEqual(recording_group("GOPR0376"), recording_group("GP010376"))
        self.assertEqual(recording_group("GP020376"), "gopro_0376")
        self.assertNotEqual(recording_group("GOPR0376"), recording_group("GOPR0397"))

    def test_unknown_or_mismatched_sequence_fails(self):
        for sequence in ("GOPR376", "OTHER0376", "GP010376extra", "GOPR0376/../../"):
            with self.assertRaises(ValueError):
                recording_group(sequence)
        with self.assertRaises(ValueError):
            parse_image(Path("/data/GOPR0376/GOPR0397_frame_000001_rgb_anon.png"))
        self.assertEqual(parse_image(Path("/data/GP010376/GP010376_frame_000001_rgb_anon.png")),
                         ("GP010376", "gopro_0376"))

    def test_cross_weather_numeric_collision_requires_review(self):
        rows = self.rows()
        rows[-1]["recording_group"] = rows[0]["recording_group"]
        with self.assertRaises(ValueError):
            group_summary(rows)

    def test_nested_assignments_are_complete_label_free_and_order_invariant(self):
        rows = self.rows()
        assignments = build_assignments(rows)
        self.assertEqual(assignments, build_assignments(list(reversed(rows))))
        changed_labels = [{**r, "error_label": 1, "num_valid_pixels": 0} for r in rows]
        self.assertEqual(assignments, build_assignments(changed_labels))
        for outer in assignments:
            held_out_images = {r["image_id"] for r in rows if r["recording_group"] == outer["eval_group"]}
            for fold in outer["inner_folds"]:
                seen = {r["image_id"] for r in rows if r["recording_group"] in fold["fit_groups"] + fold["dev_groups"]}
                self.assertFalse(seen & held_out_images)
                self.assertEqual(len(seen) + len(held_out_images), len(rows))

    def test_tampered_inner_group_leakage_is_rejected(self):
        rows = self.rows()
        assignments = copy.deepcopy(build_assignments(rows))
        assignments[0]["inner_folds"][0]["fit_groups"].append(assignments[0]["eval_group"])
        with self.assertRaises(AssertionError):
            validate_assignments(rows, assignments)

    def test_official_val_cannot_enter_primary_protocol(self):
        rows = self.rows()
        rows[0]["dataset_split"] = "val"
        with self.assertRaises(ValueError):
            build_assignments(rows)


if __name__ == "__main__":
    unittest.main()
