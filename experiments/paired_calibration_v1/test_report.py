"""Synthetic checks for honest reporting and distinct evaluation denominators."""
import csv
import tempfile
import unittest
from pathlib import Path

from .report import generate_report, _paired_image_summary, _write_once_or_identical


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class ReportTests(unittest.TestCase):
    def test_blocked_report_does_not_invent_outcomes(self):
        with tempfile.TemporaryDirectory() as folder:
            result = generate_report(folder, "CUDA driver unavailable", make_plots=False)
            text = (Path(folder) / "findings.txt").read_text()
            self.assertEqual(result["status"], "incomplete")
            self.assertIn("CUDA driver unavailable", text)
            self.assertIn("seven research questions remain unanswered", text)
            self.assertNotIn("balanced_validation_nll", result)

    def test_existing_different_report_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "report.txt"
            _write_once_or_identical(path, "original")
            _write_once_or_identical(path, "original")
            with self.assertRaises(FileExistsError):
                _write_once_or_identical(path, "replacement")
            self.assertEqual(path.read_text(), "original")

    def test_pair_average_is_not_pixel_weighted(self):
        rows = []
        for pair, normal, night, count in (("a", 1, 3, 1000), ("b", 1, 1, 1)):
            for domain, nll in (("normal", normal), ("night", night)):
                rows.append(dict(pair_id=pair, domain=domain, temperature_name="global",
                                 sequence_id="GOPR0351", nll=nll, num_valid=count))
        summary = _paired_image_summary(rows, ["global"])
        self.assertEqual(summary[0]["pairs"], 2)
        self.assertEqual(summary[0]["mean_night_minus_normal_nll"], 1)
        self.assertIsNone(summary[0]["mean_night_minus_normal_accepted_risk"])

    def test_conflicting_calibration_and_rejection_are_not_conflated(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            calibration = []
            for name, nll in (("raw", .6), ("global", .5), ("NightPair", .4), ("NormalPair", .7), ("Mix", .45)):
                for domain in ("normal", "night"):
                    calibration.append(dict(domain=domain, temperature_name=name, sequence_id="pooled", temperature=2,
                                            nll=nll, ece=.1, signed_gap=.05, coverage_valid=.7, coverage_all=.65,
                                            accepted_risk=.2, num_valid=100, num_correct=70, num_high_conf=20,
                                            num_high_conf_errors=5, high_conf_error_rate=.25,
                                            error_high_conf_fraction=1/6))
            sequences = [dict(row, sequence_id=sequence) for row in calibration for sequence in ("GOPR0351", "GOPR0356")]
            matched = [dict(domain=row["domain"], temperature_name=row["temperature_name"], sequence_id="pooled",
                            requested_coverage=.8, actual_coverage=.81 if row["temperature_name"] == "NightPair" else .8,
                            risk=.3 if row["temperature_name"] == "NightPair" else .2) for row in calibration]
            aurc = [dict(domain=row["domain"], temperature_name=row["temperature_name"], sequence_id="pooled", aurc=.1)
                    for row in calibration]
            for name, rows in (("calibration_comparison.csv", calibration), ("frozen_policy_comparison.csv", calibration),
                               ("sequence_metrics.csv", sequences), ("matched_coverage.csv", matched), ("aurc.csv", aurc)):
                write_csv(directory / name, rows)
            result = generate_report(directory, make_plots=False)
            text = (directory / "findings.txt").read_text()
            self.assertEqual(result["status"], "complete")
            self.assertIn("5 wrong pixels among 20", text)
            self.assertIn("25.0000%", text)
            self.assertIn("16.6667% of all errors", text)
            self.assertIn("81.0000% versus 80.0000%", text)
            self.assertIn("delta +10.0000 percentage points", text)
            self.assertIn("NLL is lower on both", text)
            self.assertIn("Fixed-threshold risk changes alone cannot establish", text)
            self.assertEqual(result["comparison_deltas"]["night_NightPair_minus_NormalPair"]["nll"], .4-.7)
            generate_report(directory, make_plots=False)


if __name__ == "__main__":
    unittest.main()
