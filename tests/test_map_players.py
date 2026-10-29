import unittest

import pandas as pd

from fetch_api_player_ids import participants_from_export
from map_players import apply_entry_map, attach_api_ids, map_entries, map_names

GITHUB = pd.DataFrame(
    {
        "PlayerID": [1, 2, 3, 4, 5, 6, 7],
        "Name": ["Johan Johansson", "Johan Johansson", "Kimmo Eriksson", "Erik Lund", "Twin A", "Twin B", "Martin Vojáček"],
        "RankingID": [None, 665520.0, 654720.0, None, 700000.0, 700000.0, None],
    }
)


def entries(*rows):
    return pd.DataFrame(rows, columns=["SourceTournamentID", "PlayerName", "Rows", "ITHF", "SBHF"])


def mapped(entry_df, manual=None, offset=None):
    manual = manual or {}
    name_map = map_names(sorted(entry_df["PlayerName"].unique()), GITHUB, manual)
    return map_entries(entry_df, GITHUB, name_map, manual, ithf_only_offset=offset)


class ExportTests(unittest.TestCase):
    def test_reads_names_and_ids_and_skips_unnamed_participants(self):
        export = {
            "participants": [
                {"id": "p1", "name": " Johan Johansson ", "ids": {"ithf": "665520", "sbhf": "2724"}},
                {"id": "p2", "name": "No Ids"},
                {"id": "p3"},
            ]
        }
        rows = participants_from_export("2627/sommarpucken", export)
        self.assertEqual([r["PlayerName"] for r in rows], ["Johan Johansson", "No Ids"])
        self.assertEqual((rows[0]["ITHF"], rows[0]["SBHF"]), ("665520", "2724"))
        self.assertIsNone(rows[1]["ITHF"])


class AttachTests(unittest.TestCase):
    def test_matches_name_within_tournament_and_drops_ambiguous_names(self):
        api = pd.DataFrame(
            {
                "SourceTournamentID": ["t1", "t1", "t1", "t2"],
                "PlayerName": ["Johan  johansson", "Twin", "Twin", "Other"],
                "ITHF": ["665520", "1", "2", "3"],
                "SBHF": ["2724", None, None, None],
            }
        )
        base = pd.DataFrame(
            {"SourceTournamentID": ["t1", "t1", "t2"], "PlayerName": ["Johan Johansson", "Twin", "Johan Johansson"], "Rows": [1, 1, 1]}
        )
        out = attach_api_ids(base, api)
        self.assertEqual(out["ITHF"].tolist()[0], "665520")
        self.assertTrue(pd.isna(out["ITHF"].tolist()[1]))
        self.assertTrue(pd.isna(out["ITHF"].tolist()[2]))


class MapEntriesTests(unittest.TestCase):
    def test_ithf_id_beats_same_name_duplicate_profile(self):
        out = mapped(entries(("t1", "Johan Johansson", 5, "665520", None)))
        self.assertEqual((out["PlayerID"][0], out["Method"][0]), (2, "ITHF"))

    def test_name_match_with_other_ranking_id_is_rejected(self):
        out = mapped(entries(("t1", "Kimmo Eriksson", 5, "280954", None)))
        self.assertTrue(pd.isna(out["PlayerID"][0]))
        self.assertEqual(out["Method"][0], "Conflict")

    def test_name_match_without_ranking_id_is_kept(self):
        out = mapped(entries(("t1", "Erik Lund", 5, "123", None), ("t2", "Erik Lund", 1, None, None)))
        self.assertEqual(out["PlayerID"].tolist(), [4, 4])

    def test_accent_folded_name_beats_ithf_only_id(self):
        out = mapped(entries(("t1", "Martin Vojacek", 5, "670657", None)), offset=1_000_000)
        self.assertEqual((out["PlayerID"][0], out["Method"][0]), (7, "Folded"))

    def test_one_ithf_player_per_name_matched_profile(self):
        out = mapped(entries(("t1", "Erik Lund", 5, "123", None), ("t2", "Erik Lund", 2, "456", None)))
        self.assertEqual(out["PlayerID"][0], 4)
        self.assertTrue(pd.isna(out["PlayerID"][1]))

    def test_duplicate_ranking_id_falls_back_to_name(self):
        out = mapped(entries(("t1", "Twin A", 5, "700000", None)))
        self.assertEqual((out["PlayerID"][0], out["Method"][0]), (5, "Exact"))

    def test_manual_ithf_override_and_ithf_only_offset(self):
        out = mapped(
            entries(("t1", "Kimmo Eriksson", 5, "280954", None), ("t1", "New Player", 5, "999", None)),
            manual={"ithf:280954": 3},
            offset=1_000_000,
        )
        self.assertEqual(out["PlayerID"].tolist(), [3, 1_000_999])
        self.assertEqual(out["Method"].tolist(), ["Manual", "ITHF-only"])

    def test_apply_entry_map_sets_ids_per_tournament(self):
        results = pd.DataFrame(
            {
                "SourceTournamentID": ["t1", "t2"],
                "Player1": ["Fredrik Johansson", "Fredrik Johansson"],
                "Player1ID": ["", "7"],
                "Player2": ["Erik Lund", "Erik Lund"],
                "Player2ID": ["4", "4"],
            }
        )
        entry_map = pd.DataFrame(
            {
                "SourceTournamentID": ["t1", "t2", "t1", "t2"],
                "PlayerName": ["Fredrik Johansson", "Fredrik Johansson", "Erik Lund", "Erik Lund"],
                "PlayerID": pd.array([10, 11, 4, pd.NA], dtype="Int64"),
            }
        )
        out = apply_entry_map(results, entry_map)
        self.assertEqual(out["Player1ID"].tolist(), ["10", "11"])
        self.assertEqual(out["Player2ID"].tolist(), ["4", ""])


if __name__ == "__main__":
    unittest.main()
