import unittest

from backtest import compare_windows, walk_forward_moving_average


class BacktestTests(unittest.TestCase):
    def test_walk_forward_produces_aligned_predictions(self):
        result = walk_forward_moving_average([1, 2, 3, 4, 5], window=2)
        self.assertEqual(result["indices"], [2, 3, 4])
        self.assertEqual(len(result["actual"]), len(result["predicted"]))

    def test_window_comparison_sorted_by_mae(self):
        rows = compare_windows([1, 2, 3, 4, 5, 6], [1, 2, 3])
        self.assertLessEqual(rows[0]["mae"], rows[-1]["mae"])


if __name__ == "__main__":
    unittest.main()
