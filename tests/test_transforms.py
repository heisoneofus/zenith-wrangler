from __future__ import annotations

import unittest

import pandas as pd

from src.tools.transforms import flatten_nested


class FlattenNestedTests(unittest.TestCase):
    def test_flatten_nested_explodes_stringified_scalar_lists(self) -> None:
        df = pd.DataFrame(
            {
                "movie": ["M1", "M2"],
                "genres": ["['Drama']", "['Crime', 'Drama']"],
            }
        )

        result = flatten_nested(df)

        self.assertEqual(len(result), 3)
        self.assertIn("genres", result.columns)
        self.assertEqual(result["genres"].tolist(), ["Drama", "Crime", "Drama"])

    def test_flatten_nested_supports_dict_and_list_of_dict(self) -> None:
        dict_df = pd.DataFrame(
            {
                "id": [1, 2],
                "payload": [{"a": 1}, {"a": 2}],
            }
        )
        list_dict_df = pd.DataFrame(
            {
                "id": [1],
                "items": [[{"k": "x"}, {"k": "y"}]],
            }
        )

        dict_result = flatten_nested(dict_df)
        list_dict_result = flatten_nested(list_dict_df)

        self.assertIn("payload.a", dict_result.columns)
        self.assertEqual(dict_result["payload.a"].tolist(), [1, 2])
        self.assertIn("items.k", list_dict_result.columns)
        self.assertEqual(list_dict_result["items.k"].tolist(), ["x", "y"])

    def test_flatten_nested_ignores_malformed_stringified_values(self) -> None:
        df = pd.DataFrame(
            {
                "movie": ["M1", "M2"],
                "genres": ["['Drama'", "not a nested value"],
            }
        )

        result = flatten_nested(df)

        self.assertEqual(result.equals(df), True)


if __name__ == "__main__":
    unittest.main()
