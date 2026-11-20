import unittest
from predictive_intelligence_platform import exponential_smoothing, mae, moving_average_forecast, rolling_anomalies
class PredictivePlatformTests(unittest.TestCase):
    def test_moving_average(self): self.assertEqual(moving_average_forecast([1,2,3,4],2),3.5)
    def test_exponential_smoothing_tracks_latest(self): self.assertEqual(exponential_smoothing([0,10],0.8),8.0)
    def test_anomaly_detection(self): self.assertEqual([a.index for a in rolling_anomalies([10,10,10,10,10,50],window=5,threshold=3)],[5])
    def test_mae(self): self.assertEqual(mae([1,2],[2,2]),0.5)
    def test_nonfinite_series_rejected(self):
        for values in ([1,float("nan")],[float("inf")]):
            with self.assertRaises(ValueError): moving_average_forecast(values,1)
    def test_invalid_window_and_threshold_rejected(self):
        with self.assertRaises(ValueError): rolling_anomalies([1,2,3],window=1)
        with self.assertRaises(ValueError): rolling_anomalies([1,2,3],threshold=float("nan"))
if __name__ == "__main__": unittest.main()
