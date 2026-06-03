import unittest

import pandas as pd

from modules.merger import generate_ai_id, merge_on_columns


class MergerTests(unittest.TestCase):
    def test_merge_on_columns_uses_selected_columns(self):
        left = pd.DataFrame({"customer_id": ["C001", "C002"], "name": ["Aarav", "Meera"]})
        right = pd.DataFrame({"customer_id": ["C001"], "order_value": [24000]})

        merged = merge_on_columns(left, right, "customer_id", "customer_id", how="left")

        self.assertEqual(len(merged), 2)
        self.assertIn("order_value", merged.columns)
        self.assertEqual(merged.loc[0, "order_value"], 24000)

    def test_generate_ai_id_adds_stable_id_without_mutating_original(self):
        df = pd.DataFrame({"a": ["one"], "b": ["two"], "c": ["three"]})

        result = generate_ai_id(df)

        self.assertIn("AI_GENERATED_ID", result.columns)
        self.assertNotIn("AI_GENERATED_ID", df.columns)
        self.assertEqual(len(result.loc[0, "AI_GENERATED_ID"]), 10)


if __name__ == "__main__":
    unittest.main()
