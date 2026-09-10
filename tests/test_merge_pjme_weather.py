import unittest
import os
import pandas as pd

class TestPJMEWeatherMerge(unittest.TestCase):

    def setUp(self):
        self.orig_features_path = "data/features/PJME_hourly_features.csv"
        self.weather_features_path = "data/features/PJME_hourly_features_weather.csv"
        self.report_path = "reports/PJME_weather_merge_report.csv"
        self.raw_weather_path = "weather/PJME_weather_2002_2018.csv"

    def test_file_existence(self):
        self.assertTrue(os.path.exists(self.orig_features_path), "Original PJME features CSV must exist.")
        self.assertTrue(os.path.exists(self.weather_features_path), "Merged PJME features with weather CSV must exist.")
        self.assertTrue(os.path.exists(self.report_path), "Merge report CSV must exist.")
        self.assertTrue(os.path.exists(self.raw_weather_path), "Raw weather file must exist.")

    def test_row_count_preservation(self):
        df_orig = pd.read_csv(self.orig_features_path)
        df_merged = pd.read_csv(self.weather_features_path)
        self.assertEqual(len(df_orig), len(df_merged), f"Row count changed: {len(df_orig)} vs {len(df_merged)}")

    def test_column_addition(self):
        df_orig = pd.read_csv(self.orig_features_path)
        df_merged = pd.read_csv(self.weather_features_path)

        # 41 original + 5 weather = 46 total
        self.assertEqual(len(df_orig.columns), 41)
        self.assertEqual(len(df_merged.columns), 46)

        weather_cols = ['temperature_2m', 'relative_humidity_2m', 'precipitation', 'cloud_cover', 'wind_speed_10m']
        for col in weather_cols:
            self.assertIn(col, df_merged.columns, f"Weather column {col} missing from merged dataset.")

        for col in df_orig.columns:
            self.assertIn(col, df_merged.columns, f"Original column {col} lost in merge.")

    def test_zero_missing_weather_values(self):
        df_merged = pd.read_csv(self.weather_features_path)
        weather_cols = ['temperature_2m', 'relative_humidity_2m', 'precipitation', 'cloud_cover', 'wind_speed_10m']
        missing_count = df_merged[weather_cols].isna().sum().sum()
        self.assertEqual(missing_count, 0, f"Expected 0 missing weather values, found {missing_count}")

    def test_report_accuracy(self):
        df_report = pd.read_csv(self.report_path)
        self.assertEqual(df_report.loc[0, 'merge_status'], "SUCCESS")
        self.assertEqual(df_report.loc[0, 'energy_rows'], 145366)
        self.assertEqual(df_report.loc[0, 'final_rows'], 145366)
        self.assertEqual(df_report.loc[0, 'missing_weather_values'], 0)

if __name__ == '__main__':
    unittest.main()
