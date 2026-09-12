import unittest
import os
import pandas as pd


class TestPJMEWeatherMerge(unittest.TestCase):

    def setUp(self):
        self.final_features_path = "data/features/features.csv"

    def test_file_existence(self):
        self.assertTrue(
            os.path.exists(self.final_features_path),
            "Final features.csv must exist."
        )

    def test_final_row_count(self):
        df = pd.read_csv(self.final_features_path)

        self.assertEqual(
            len(df),
            145198,
            "Final dataset must contain 145,198 rows."
        )

    def test_weather_columns_present(self):
        df = pd.read_csv(self.final_features_path)

        expected_weather_cols = [
            "temperature",
            "relative_humidity_2m",
            "precipitation",
            "cloud_cover",
            "wind_speed_10m"
        ]

        for col in expected_weather_cols:
            self.assertIn(
                col,
                df.columns,
                f"Weather column '{col}' missing from final dataset."
            )

    def test_final_column_count(self):
        df = pd.read_csv(self.final_features_path)

        self.assertEqual(
            len(df.columns),
            49,
            "Final dataset must contain exactly 49 columns."
        )

    def test_zero_missing_weather_values(self):
        df = pd.read_csv(self.final_features_path)

        weather_cols = [
            "temperature",
            "relative_humidity_2m",
            "precipitation",
            "cloud_cover",
            "wind_speed_10m"
        ]

        missing_count = df[weather_cols].isna().sum().sum()

        self.assertEqual(
            missing_count,
            0,
            f"Expected 0 missing weather values, found {missing_count}"
        )


if __name__ == "__main__":
    unittest.main()