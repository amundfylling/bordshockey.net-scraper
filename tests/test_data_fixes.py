import os
import tempfile
import unittest

import pandas as pd

import check_results
import clean_results
import export_index
import map_players
from map_players import LocalIds, apply_entry_map, fill_ithf, id_changes, map_entries, map_names, row_keys

SCORPION = pd.DataFrame(
    {
        "PlayerID": [1, 2, 3, 4, 5],
        "Name": ["Gunnarsson Fredrik", "Igor Novikov", "Igor Novikov", "Trond Ove Gangsøy", "Anna-Maria Israelsson"],
        "RankingID": [None, 654709.0, 658005.0, 220094.0, 672575.0],
        "Country": [None, "Russia", "Russia", "Norway", "Sweden"],
    }
)


def entries(*rows):
    return pd.DataFrame(rows, columns=["SourceTournamentID", "PlayerName", "Rows", "ITHF", "SBHF", "Nation"])


def run(entry_df, local_ids=None, manual=None):
    manual = manual or {}
    entry_df = fill_ithf(entry_df, manual)
    name_map = map_names(sorted(entry_df["PlayerName"].unique()), SCORPION, manual)
    return map_entries(entry_df, SCORPION, name_map, manual, local_ids=local_ids)


class FillIthfTests(unittest.TestCase):
    def test_sbhf_lends_its_ithf_id(self):
        out = fill_ithf(entries(("1819/a", "Annika Vikdahl", 1, None, "2909", "SWE"), ("1819/b", "Annika Vikdahl", 1, "670321", "2909", "SWE")))
        self.assertEqual(out["ITHF"].tolist(), ["670321", "670321"])
        self.assertEqual(out["ITHFSource"].tolist(), ["SBHF", "API"])

    def test_name_lends_ithf_id_with_middle_name_and_folded_letters(self):
        out = fill_ithf(entries(("1314/a", "Trond Gangsoy", 1, None, "4819", "NOR"), ("1718/b", "Trond Ove Gangsøy", 1, "220094", None, "NOR")))
        self.assertEqual(out["ITHF"][0], "220094")
        self.assertEqual(out["ITHFSource"][0], "Name")

    def test_name_needs_close_seasons_same_nation_and_one_holder(self):
        far = fill_ithf(entries(("9900/a", "Lars Berg", 1, None, None, "SWE"), ("2526/b", "Lars Berg", 1, "1", None, "SWE")))
        self.assertTrue(pd.isna(far["ITHF"][0]))
        nation = fill_ithf(entries(("1819/a", "Lars Berg", 1, None, None, "NOR"), ("1819/b", "Lars Berg", 1, "1", None, "SWE")))
        self.assertTrue(pd.isna(nation["ITHF"][0]))
        two = fill_ithf(entries(("1819/a", "Lars Berg", 1, None, None, None), ("1819/b", "Lars Berg", 1, "1", None, None), ("1819/c", "Lars Berg", 1, "2", None, None)))
        self.assertTrue(pd.isna(two["ITHF"][0]))
        manual_null = fill_ithf(entries(("1819/a", "Lars Berg", 1, None, None, None), ("1819/b", "Lars Berg", 1, "1", None, None)), {"Lars Berg": None})
        self.assertTrue(pd.isna(manual_null["ITHF"][0]))

    def test_correction_replaces_api_ithf(self):
        out = fill_ithf(entries(("2324/x", "Marcus Lönnqvist", 1, "675313", "3152", "SWE")), corrections={("2324/x", "Marcus Lönnqvist"): "675314"})
        self.assertEqual((out["ITHF"][0], out["ITHFSource"][0]), ("675314", "Correction"))


class MapEntriesTests(unittest.TestCase):
    def test_word_order_name_match_beats_ithf_only(self):
        out = run(entries(("1819/a", "Fredrik Gunnarsson", 5, "660147", None, "SWE")))
        self.assertEqual((out["PlayerID"][0], out["Method"][0]), (1, "ScorpionName"))

    def test_ambiguous_scorpion_name_is_not_a_name_match(self):
        out = run(entries(("1819/a", "Novikov Igor", 5, "999", None, "RUS")))
        self.assertEqual(out["PlayerID"][0], 9_000_999)

    def test_one_sbhf_with_two_ithf_ids_uses_the_one_scorpion_knows(self):
        out = run(entries(("2122/a", "Anna Israelsson", 5, "673130", "3017", "SWE"), ("2021/b", "Anna-Maria Israelsson", 5, "672575", "3017", "SWE")))
        self.assertEqual(out["PlayerID"].tolist(), [5, 5])

    def test_two_entries_of_a_tournament_never_share_an_id(self):
        # Both name-match "Gunnarsson Fredrik"; the exact match keeps it, the other falls back to a local id.
        out = run(entries(("1819/a", "Fredrik Gunnarsson", 9, None, None, None), ("1819/a", "Gunnarsson Fredrik", 1, "5", None, None)), local_ids=LocalIds(None))
        self.assertEqual(out["PlayerID"].tolist(), [10_000_001, 1])

    def test_local_ids_are_stable_and_never_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "local.csv")
            ids = LocalIds(path)
            out = run(entries(("2223/a", "Eva Hallin", 5, None, "4817", "SWE"), ("2223/a", "No Ids", 5, None, None, None)), local_ids=ids)
            self.assertEqual(out["PlayerID"].tolist(), [10_000_001, 10_000_002])
            self.assertEqual(out["Method"].tolist(), ["Local", "Local"])
            ids.save()
            again = LocalIds(path)
            out2 = run(entries(("2324/b", "Eva Hallin", 5, None, "4817", "SWE"), ("2324/b", "Newcomer", 5, None, None, None)), local_ids=again)
            self.assertEqual(out2["PlayerID"].tolist(), [10_000_001, 10_000_003])

    def test_placeholder_names_get_no_id(self):
        out = run(entries(("1415/a", "Struken", 5, "663858", None, None), ("1415/a", "?", 1, None, None, None)), local_ids=LocalIds(None))
        self.assertTrue(out["PlayerID"].isna().all())


class ApplyTests(unittest.TestCase):
    def results(self):
        return pd.DataFrame(
            {
                "SourceTournamentID": ["1819/kursk", "1819/kursk"],
                "SourceMatchID": ["11", ""],
                "Stage": ["G", "G"],
                "RoundNumber": ["1", "2"],
                "PlayoffGameNumber": ["", ""],
                "Player1": ["Igor Novikov", "Igor Novikov"],
                "Player1ID": ["1334", "1334"],
                "Player2": ["Igor Novikov", "Someone"],
                "Player2ID": ["1334", "7"],
                "GoalsPlayer1": ["3", "1"],
                "GoalsPlayer2": ["1", "0"],
            }
        )

    def test_row_overrides_split_namesakes(self):
        df = self.results()
        keys = row_keys(df)
        self.assertEqual(keys.tolist()[0], "m:11")
        self.assertTrue(keys.tolist()[1].startswith("t:G|2|"))
        entry_map = pd.DataFrame(
            {"SourceTournamentID": ["1819/kursk", "1819/kursk"], "PlayerName": ["Igor Novikov", "Someone"], "PlayerID": pd.array([3, 7], dtype="Int64")}
        )
        out = apply_entry_map(df, entry_map, {("1819/kursk", "m:11", 2): 2})
        self.assertEqual(out["Player1ID"].tolist(), ["3", "3"])
        self.assertEqual(out["Player2ID"].tolist(), ["2", "7"])

    def test_id_changes_lists_only_whole_moves(self):
        old = pd.DataFrame({"SourceTournamentID": ["a", "a", "b"], "Player1": ["X", "Y", "X"], "Player1ID": ["9", "8", "9"], "Player2": ["Z", "Z", "Y"], "Player2ID": ["", "5", "8"]})
        new = old.assign(Player1ID=["1", "8", "1"], Player2ID=["4", "5", "6"])
        whole, detail = id_changes(old, new)
        self.assertEqual(whole.values.tolist(), [[9, 1]])  # 8 moved only partly, "" was not an id
        self.assertEqual(set(detail["Old"]), {8, 9})


class CleanTests(unittest.TestCase):
    def test_season_and_display_name(self):
        self.assertEqual(clean_results.season_label("1314/sm"), "2013/14")
        self.assertEqual(clean_results.season_label("9900/x"), "1999/00")
        self.assertEqual(clean_results.display_name("Svenska Mästerskapen", "1314/svenska-masterskapen"), "Svenska Mästerskapen 2013/14")
        self.assertEqual(clean_results.display_name("Kursk Cup 2019", "1819/kursk"), "Kursk Cup 2019")
        self.assertEqual(clean_results.display_name("SM", "2526/sm_veteran", "Veteran"), "SM Veteran 2025/26")
        self.assertEqual(clean_results.display_name("SM", "2526/sm", "Öppna"), "SM 2025/26")
        once = clean_results.display_name("SM", "2526/sm_dam", "Dam")
        self.assertEqual(clean_results.display_name(once, "2526/sm_dam", "Dam"), once)

    def test_tournament_meta_defaults_to_sweden(self):
        df = pd.DataFrame({"SourceTournamentID": ["1819/kursk", "0001/x"], "TournamentName": ["Kursk Cup", "X"]})
        meta = pd.DataFrame([{"SourceTournamentID": "1819/kursk", "ClassName": "Öppna", "City": "Kursk", "Nation": "RUS"}])
        out = clean_results.add_tournament_meta(df, meta)
        self.assertEqual(out["Country"].tolist(), ["Russia", "Sweden"])
        self.assertEqual(out["City"].tolist()[0], "Kursk")
        self.assertTrue(pd.isna(out["City"].tolist()[1]))
        self.assertEqual(out["TournamentName"].tolist(), ["Kursk Cup 2018/19", "X 2000/01"])

    def test_placement_game_listed_twice(self):
        base = {"SourceTournamentID": "t", "SourceStageID": "1", "StageType": "playoff", "Player1": "A", "Player2": "B",
                "GoalsPlayer1": "3", "GoalsPlayer2": "2", "PlayoffGameNumber": "1"}
        df = pd.DataFrame([dict(base, Stage="17-22 Plats 23-24", SourceMatchID="55988"), dict(base, Stage="17-22 Plats 21-22", SourceMatchID="55987")])
        self.assertEqual(clean_results.placement_duplicates(df), [0])

    def test_carried_copies(self):
        rows = []
        players = ["A", "B", "C", "D"]
        pairs = [(a, b) for i, a in enumerate(players) for b in players[i + 1:]]
        for n, (a, b) in enumerate(pairs):
            rows.append(dict(Stage="Kval", StageSequence="1", RoundNumber=str(n + 1), Player1=a, Player2=b, GoalsPlayer1=str(n % 4), GoalsPlayer2="2"))
        for n, (a, b) in enumerate(pairs):  # final group: a copy of each kval game, then a new game
            rows.append(dict(Stage="Final", StageSequence="2", RoundNumber=str(n + 1), Player1=b, Player2=a, GoalsPlayer1="2", GoalsPlayer2=str(n % 4)))
            rows.append(dict(Stage="Final", StageSequence="2", RoundNumber=str(n + 7), Player1=a, Player2=b, GoalsPlayer1="9", GoalsPlayer2="9"))
        df = pd.DataFrame(rows).assign(SourceTournamentID="t", StageType="round-robin")
        drop, report = clean_results.carried_copies(df)
        self.assertEqual(len(drop), 6)
        self.assertTrue(all(df.loc[drop, "Stage"] == "Final") and all(df.loc[drop, "GoalsPlayer1"] == "2"))
        # A stage where only a few results repeat by chance keeps its games.
        df2 = df[~((df["Stage"] == "Final") & (df["GoalsPlayer1"] == "2"))]
        self.assertEqual(clean_results.carried_copies(df2)[0], [])

    def test_impossible_overtime(self):
        df = pd.DataFrame({"Overtime": ["Yes", "Yes", "Yes", "No"], "GoalsPlayer1": ["4", "2", "3", "5"], "GoalsPlayer2": ["2", "2", "2", "5"]})
        self.assertEqual(clean_results.impossible_overtime(df).tolist(), [True, True, False, False])

    def test_source_correction_drop(self):
        df = pd.DataFrame({"SourceTournamentID": ["t", "t"], "SourceMatchID": ["1", "2"], "Player1": ["A", "A"], "Player2": ["A", "B"]})
        out, n = clean_results.apply_source_corrections(df, [{"SourceTournamentID": "t", "SourceMatchID": "1", "where": {"Player2": "A"}, "action": "drop"}])
        self.assertEqual((n, out["SourceMatchID"].tolist()), (1, ["2"]))


class ExportTests(unittest.TestCase):
    EXPORT = {
        "participants": [
            {"id": "p1", "name": "Igor Novikov"},
            {"id": "p2", "name": "Igor Novikov", "ids": {"ithf": "654709"}},
            {"id": "p3", "name": "Anna"},
        ],
        "parts": [{"stages": [
            {"name": "Grupp", "groups": [{"name": "G1", "matches": [
                {"round": 1, "home": "p1", "away": "p2", "homeScore": 3, "awayScore": 1},
                {"round": 1, "home": "p3", "away": "p1", "homeScore": 5, "awayScore": 0, "walkover": True},
                {"round": 2, "home": "p3", "away": "p2", "homeScore": 1, "awayScore": 1, "carriedOver": True},
            ]}]},
            {"name": "Slutspel", "playoff": {"rounds": [{"name": "Final", "series": [
                {"home": "p3", "away": "p2", "games": [{"game": 1, "homeScore": 4, "awayScore": 3, "suddenDeath": True}]}]}]}},
        ]}],
    }

    def test_records_and_join(self):
        recs = export_index.records_frame(export_index.export_records("t", self.EXPORT))
        self.assertEqual(len(recs), 4)
        results = pd.DataFrame(
            {
                "SourceTournamentID": ["t", "t", "t"],
                "StageType": ["round-robin", "round-robin", "playoff"],
                "RoundNumber": ["1", "1", "1"],
                "PlayoffGameNumber": ["", "", "1"],
                "Player1": ["Igor Novikov", "Anna", "Anna"],
                "Player2": ["Igor Novikov", "Igor Novikov", "Igor Novikov"],
                "GoalsPlayer1": ["3", "5", "4"],
                "GoalsPlayer2": ["1", "0", "3"],
            }
        )
        joined = export_index.join_results(results, recs)
        self.assertEqual(joined["Walkover"].tolist(), [False, True, False])
        self.assertEqual(joined["P1Key"].tolist(), ["p1", "p3", "p3"])
        self.assertEqual(joined["P2ITHF"][0], "654709")
        self.assertTrue(pd.isna(joined["P2ITHF"][1]))


class CheckTests(unittest.TestCase):
    def test_self_matches_and_name_conflicts(self):
        df = pd.DataFrame(
            {
                "SourceTournamentID": ["t", "t", "u"],
                "Player1": ["Nils Kullberg Michels", "Mikael Eklund", "Michael Gellerstam"],
                "Player1ID": [9675313, 1, 1],
                "Player2": ["Marcus Lönnqvist", "Mikael Eklund,", "X"],
                "Player2ID": [9675313, 1, 2],
            }
        )
        self.assertEqual(len(check_results.self_matches(df)), 2)
        conflicts = check_results.name_conflicts(df)
        self.assertEqual(conflicts["PlayerID"].tolist(), [9675313])
        self.assertEqual(len(check_results.name_conflicts(df, per_tournament=False)), 2)


if __name__ == "__main__":
    unittest.main()
