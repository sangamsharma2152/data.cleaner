import unittest

import pandas as pd

from modules.preprocessing import ml_ready


class PreprocessingTests(unittest.TestCase):
    def test_ml_ready_returns_train_test_and_report(self):
        df = pd.DataFrame(
            {
                "city": ["Delhi", "Mumbai", "Delhi", "Pune", "Chennai"],
                "age": [28, 34, 42, 29, 31],
                "constant": ["same", "same", "same", "same", "same"],
            }
        )

        train, test, report = ml_ready(df, test_size=0.4, random_state=1)

        self.assertEqual(len(train), 3)
        self.assertEqual(len(test), 2)
        self.assertIn("city", report["encoded_columns"])
        self.assertIn("constant", report["dropped_constant_columns"])

    def test_ml_ready_rejects_single_row_dataset(self):
        df = pd.DataFrame({"value": [1]})

        with self.assertRaisesRegex(ValueError, "At least two rows"):
            ml_ready(df)


if __name__ == "__main__":
    unittest.main()
