import unittest
import os
import pandas as pd
import numpy as np
import src.config as config

class TestFinalPJME49ColumnsDataset(unittest.TestCase):

    def setUp(self):
        self.final_path = config.FINAL_DATASET_PATH
        self.report_path = config.FINAL_REPORT_PATH
        self.assertTrue(os.path.exists(self.final_path), "Final 49-column dataset file missing!")
        self.df = pd.read_csv(self.final_path)

    def test_exact_dimensions_and_zero_nans(self):
        self.assertEqual(len(self.df), 145198, "Row count must be exactly 145,198 after dropping first 168 initialization rows.")
        self.assertEqual(len(self.df.columns), 49, "Column count must be 49.")
        self.assertEqual(list(self.df.columns), config.FINAL_COLUMN_LIST, "Column names/order do not match final schema.")
        self.assertEqual(self.df.isna().sum().sum(), 0, "Total missing values across all columns must be 0.")

    def test_required_key_columns_exist(self):
        required_cols = ['timestamp', 'demand', 'temperature', 'season', 'is_holiday', 'split']
        for col in required_cols:
            self.assertIn(col, self.df.columns, f"Required column '{col}' missing from final dataset.")

    def test_split_column_values_and_counts(self):
        split_vals = set(self.df['split'].unique())
        self.assertEqual(split_vals, {'train', 'val', 'test'}, "Split values must be exactly train, val, test.")

        train_cnt = (self.df['split'] == 'train').sum()
        val_cnt = (self.df['split'] == 'val').sum()
        test_cnt = (self.df['split'] == 'test').sum()

        self.assertEqual(train_cnt, 101638, "Train set must have 101,638 rows (70% of 145,198).")
        self.assertEqual(val_cnt, 21780, "Val set must have 21,780 rows (15% of 145,198).")
        self.assertEqual(test_cnt, 21780, "Test set must have 21,780 rows (15% of 145,198).")
        self.assertEqual(train_cnt + val_cnt + test_cnt, 145198)

    def test_season_column(self):
        season_vals = set(self.df['season'].unique())
        self.assertEqual(season_vals, {'winter', 'spring', 'summer', 'fall'}, "Season values must be winter, spring, summer, fall.")

    def test_is_holiday_column(self):
        holiday_vals = set(self.df['is_holiday'].unique())
        self.assertTrue(holiday_vals.issubset({0, 1}), "is_holiday must contain binary 0/1 flags.")
        self.assertGreater(self.df['is_holiday'].sum(), 0, "is_holiday must flag US holiday hours.")

    def test_chronological_ordering(self):
        dt_series = pd.to_datetime(self.df['timestamp'])
        self.assertTrue(dt_series.is_monotonic_increasing, "Timestamp is not chronologically sorted.")

    def test_all_46_original_features_preserved(self):
        orig_features = [
            'timestamp', 'demand', 'hour', 'day', 'day_of_week', 'day_of_month', 'week_of_year',
            'month', 'quarter', 'year', 'is_weekend', 'is_weekday', 'is_month_start',
            'is_month_end', 'is_year_start', 'is_year_end', 'sin_hour', 'cos_hour',
            'sin_day_of_week', 'cos_day_of_week', 'sin_month', 'cos_month',
            'lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12', 'lag_24', 'lag_48', 'lag_72', 'lag_168',
            'rolling_mean_6', 'rolling_mean_12', 'rolling_mean_24', 'rolling_mean_48', 'rolling_mean_168',
            'rolling_std_24', 'rolling_min_24', 'rolling_max_24', 'peak_threshold', 'is_peak',
            'temperature', 'relative_humidity_2m', 'precipitation', 'cloud_cover', 'wind_speed_10m'
        ]
        for f in orig_features:
            self.assertIn(f, self.df.columns, f"Original feature {f} missing from final dataset.")

    def test_train_only_peak_threshold(self):
        train_q85 = self.df.loc[self.df['split'] == 'train', 'demand'].quantile(0.85)
        stored_pthresh = self.df['peak_threshold'].iloc[0]
        self.assertTrue(np.isclose(stored_pthresh, train_q85), f"peak_threshold ({stored_pthresh}) must match train-only quantile ({train_q85}).")

if __name__ == '__main__':
    unittest.main()
