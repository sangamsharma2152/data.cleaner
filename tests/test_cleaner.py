import unittest

import pandas as pd

from modules.cleaner import DataCleaner


class DataCleanerTests(unittest.TestCase):
    def test_clean_all_fills_missing_and_removes_duplicates(self):
        df = pd.DataFrame(
            {
                "name": ["Aarav", "Aarav", None],
                "age": [20, 20, 30],
            }
        )

        cleaner = DataCleaner(df)
        cleaned = cleaner.clean_all()
        report = cleaner.get_report()

        self.assertEqual(cleaned.isna().sum().sum(), 0)
        self.assertEqual(len(cleaned), 2)
        self.assertEqual(report["duplicates_removed"], 1)
        self.assertEqual(report["missing_values_filled"], 1)

    def test_cleaner_does_not_mutate_original_dataframe(self):
        df = pd.DataFrame({"name": ["Aarav", None]})

        cleaner = DataCleaner(df)
        cleaner.clean_all()

        self.assertEqual(df.isna().sum().sum(), 1)


if __name__ == "__main__":
    unittest.main()
