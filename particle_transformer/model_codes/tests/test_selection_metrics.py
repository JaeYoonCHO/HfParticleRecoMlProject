"""Selection metrics and their zero-denominator behavior."""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import compute_selection_metrics


class SelectionMetricTests(unittest.TestCase):
    def test_known_counts_and_inclusive_threshold(self):
        result = compute_selection_metrics(
            [1, 1, 1, 0, 0, 0, 0], [0.95, 0.85, 0.1, 0.9, 0.3, 0.2, 0.1], 0.85)
        self.assertEqual(result["selected_signal"], 2)
        self.assertEqual(result["selected_background"], 1)
        self.assertEqual(result["num_signal"], 3)
        self.assertEqual(result["num_background"], 4)
        self.assertAlmostEqual(result["purity"], 2 / 3)
        self.assertAlmostEqual(result["signal_efficiency"], 2 / 3)
        self.assertEqual(result["background_efficiency"], 0.25)
        self.assertEqual(result["background_rejection"], 0.75)

    def test_no_selection(self):
        result = compute_selection_metrics([1, 0], [0.3, 0.2])
        self.assertIsNone(result["purity"])
        self.assertEqual(result["signal_efficiency"], 0)
        self.assertEqual(result["background_efficiency"], 0)
        self.assertEqual(result["background_rejection"], 1)

    def test_missing_class_and_empty_sample(self):
        result = compute_selection_metrics([1, 1], [0.9, 0.1])
        self.assertEqual(result["purity"], 1)
        self.assertEqual(result["signal_efficiency"], 0.5)
        self.assertIsNone(result["background_efficiency"])
        self.assertIsNone(result["background_rejection"])
        result = compute_selection_metrics([0], [0.9])
        self.assertEqual(result["purity"], 0)
        self.assertIsNone(result["signal_efficiency"])
        result = compute_selection_metrics([], [])
        self.assertIsNone(result["purity"])
        self.assertIsNone(result["signal_efficiency"])

    def test_cut_changes_metrics_without_scaling_counts(self):
        labels, scores = [1, 1, 0, 0], [0.9, 0.6, 0.7, 0.1]
        self.assertEqual(compute_selection_metrics(labels, scores, 0.5)["signal_efficiency"], 1)
        self.assertEqual(compute_selection_metrics(labels, scores, 0.85)["signal_efficiency"], 0.5)
        self.assertEqual(compute_selection_metrics(labels, scores, 0)["purity"], 0.5)

    def test_invalid_inputs(self):
        for threshold in (-0.1, 1.1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                compute_selection_metrics([1, 0], [0.9, 0.1], threshold)
        for labels, scores in [([2], [0.9]), ([1], [np.nan]), ([1], [1.1]), ([1, 0], [0.9])]:
            with self.assertRaises(ValueError):
                compute_selection_metrics(labels, scores)


if __name__ == "__main__":
    unittest.main()
