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

    def test_mae(self):
        self.assertEqual(mae([1, 2], [2, 2]), 0.5)


if __name__ == "__main__":
    unittest.main()
