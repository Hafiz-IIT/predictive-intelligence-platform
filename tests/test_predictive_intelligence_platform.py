import unittest

from predictive_intelligence_platform import exponential_smoothing, mae, moving_average_forecast, rolling_anomalies


class PredictivePlatformTests(unittest.TestCase):
    def test_moving_average(self):
        self.assertEqual(moving_average_forecast([1, 2, 3, 4], 2), 3.5)

    def test_exponential_smoothing_tracks_latest(self):
        result = exponential_smoothing([0, 10], 0.8)
        self.assertEqual(result, 8.0)

    def test_anomaly_detection(self):
        anomalies = rolling_anomalies([10, 10, 10, 10, 10, 50], window=5, threshold=3)
        self.assertEqual([a.index for a in anomalies], [5])

    def test_invalid_windows_and_nonfinite_values_are_rejected(self):
        with self.assertRaises(ValueError):
            moving_average_forecast([1, 2], 1.5)
        with self.assertRaises(ValueError):
            rolling_anomalies([1, 2, 3], window=2, threshold=-1)
        with self.assertRaises(ValueError):
            exponential_smoothing([1, float("nan")], 0.5)

    def test_zero_variance_outlier_has_no_infinite_z_score(self):
        anomalies = rolling_anomalies([5, 5, 5, 5, 5, 10], window=5)
        self.assertEqual(len(anomalies), 1)
        self.assertIsNone(anomalies[0].z_score)

    def test_mae_rejects_nonfinite_values(self):
        with self.assertRaises(ValueError):
            mae([1, float("inf")], [1, 2])

    def test_extreme_finite_series_uses_stable_scaling(self):
        self.assertEqual(moving_average_forecast([1e308, 1e308], 2), 1e308)
        anomalies = rolling_anomalies([1e308, 1e308, 1e308, 1e308, 1e308, -1e308], window=5)
        self.assertEqual(len(anomalies), 1)

    def test_zero_baseline_outlier_is_detected(self):
        anomalies = rolling_anomalies([0, 0, 0, 0, 0, 1], window=5)
        self.assertEqual(len(anomalies), 1)
        self.assertEqual(anomalies[0].index, 5)
        self.assertIsNone(anomalies[0].z_score)

    def test_extreme_candidate_to_history_ratio_never_emits_infinite_z_score(self):
        anomalies = rolling_anomalies([1e-308, 1e-308, 1e-308, 1e-308, 1e-308, 1e308], window=5)
        self.assertEqual(len(anomalies), 1)
        self.assertIsNone(anomalies[0].z_score)

    def test_extremely_large_integer_inputs_are_rejected_cleanly(self):
        huge = 10 ** 10000
        with self.assertRaises(ValueError):
            moving_average_forecast([1, huge], 1)
        with self.assertRaises(ValueError):
            exponential_smoothing([1, 2], huge)
        with self.assertRaises(ValueError):
            rolling_anomalies([1, 2, 3], window=2, threshold=huge)

    def test_mae_rejects_unrepresentable_difference(self):
        with self.assertRaises(OverflowError):
            mae([1e308], [-1e308])

    def test_mae(self):
        self.assertEqual(mae([1, 2], [2, 2]), 0.5)


if __name__ == "__main__":
    unittest.main()
