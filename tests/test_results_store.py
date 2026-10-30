import os
import tempfile
import unittest
from unittest import mock

import pandas as pd

import results_store
from results_store import read_results, results_exist, write_results


def _frame():
    return pd.DataFrame(
        {
            "StageID": [7530, 7530],
            "Player1": ["A", "B"],
            "Player1ID": [12.0, None],
            "SourceTournamentID": ["0102/overum-open", "0102/overum-open"],
            "SourceMatchID": ["443270.0", None],
        }
    )


class ResultsStoreTests(unittest.TestCase):
    def test_parquet_round_trip_keeps_fixed_types(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "results.parquet")
            write_results(path, _frame())

            back = read_results(path)
            self.assertEqual(str(back["StageID"].dtype), "Int64")
            self.assertEqual(str(back["Player1ID"].dtype), "Int64")
            self.assertEqual(back["Player1ID"].tolist()[0], 12)
            self.assertTrue(pd.isna(back["Player1ID"].tolist()[1]))
            self.assertEqual(back["SourceMatchID"].tolist()[0], "443270")

            text = read_results(path, as_text=True)
            self.assertEqual(text["Player1ID"].tolist(), ["12", ""])

    def test_text_round_trip_matches_typed_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "results.parquet")
            write_results(path, _frame())
            first = read_results(path)
            write_results(path, read_results(path, as_text=True))

            pd.testing.assert_frame_equal(read_results(path), first)

    def test_default_path_falls_back_to_legacy_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            parquet = os.path.join(tmp, "data", "results.parquet")
            legacy = os.path.join(tmp, "legacy.csv")
            _frame().to_csv(legacy, index=False, encoding="utf-8-sig")

            with mock.patch.object(results_store, "RESULTS_PATH", parquet), mock.patch.object(
                results_store, "LEGACY_CSV", legacy
            ):
                self.assertTrue(results_exist(parquet))
                self.assertEqual(len(read_results(parquet)), 2)
                write_results(parquet, read_results(parquet))
                self.assertTrue(os.path.exists(parquet))

    def test_missing_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(results_exist(os.path.join(tmp, "x.parquet")))


if __name__ == "__main__":
    unittest.main()
