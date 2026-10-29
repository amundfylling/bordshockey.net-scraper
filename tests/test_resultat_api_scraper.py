import unittest

from bordshockey_scraper import _generate_id
from resultat_api_scraper import ApiTournament, rows_from_group_stage, rows_from_playoff_stage


TOURNAMENT = ApiTournament(
    id=9097,
    name="Swedish Masters",
    season="2526",
    permalink="swedish-masters",
    start_date="2026-02-07",
)
RESULT_URL = "https://bordshockey.net/tavlingar/2526/swedish-masters/resultat/"
NAMES = {1: "Astrid Lindqvist", 2: "Björn Ekholm", 3: "Cecilia Wahl"}


def _match(id, round, home, away, hs, as_, **extra):
    return {
        "id": id, "round": round, "tableNo": 1, "homeId": home, "awayId": away,
        "homeScore": hs, "awayScore": as_, "walkOver": False, "suddenDeath": False, "hidden": False, **extra,
    }


class TournamentTests(unittest.TestCase):
    def test_urls_and_ids_match_html_scraper(self):
        self.assertEqual(TOURNAMENT.tournament_url, "https://bordshockey.net/tavlingar/2526/swedish-masters/")
        self.assertEqual(TOURNAMENT.result_url, RESULT_URL)
        self.assertEqual(TOURNAMENT.source_tournament_id, "2526/swedish-masters")


class GroupStageTests(unittest.TestCase):
    stage = {"id": 11218, "name": "Kvalgrupper", "permalink": "kvalgrupper", "stageGroup": 1, "kind": "group"}

    def test_multi_group_rows(self):
        detail = {
            "groups": [
                {"name": "Grupp 1", "permalink": "grupp-1", "rounds": [
                    {"round": 1, "matches": [
                        _match(443270, 1, 1, 2, 0, 2),
                        _match(443271, 1, 3, 1, None, None),
                    ]},
                    {"round": 2, "matches": [_match(443272, 2, 2, 3, 4, 3, suddenDeath=True)]},
                ]},
                {"name": "Grupp 2", "permalink": "grupp-2", "rounds": [
                    {"round": 1, "matches": [_match(443280, 1, 3, 2, 1, 1, hidden=True)]},
                ]},
            ]
        }

        rows = rows_from_group_stage(TOURNAMENT, self.stage, detail, NAMES)

        self.assertEqual(len(rows), 2)
        first, second = rows
        url = f"{RESULT_URL}kvalgrupper/grupp-1/?matcher=1"
        self.assertEqual(first["Stage"], "Kvalgrupper - Grupp 1")
        self.assertEqual(first["StageURL"], url)
        self.assertEqual(first["StageID"], _generate_id(url))
        self.assertEqual(first["SourceStageID"], 11218)
        self.assertEqual(first["SourceMatchID"], 443270)
        self.assertEqual((first["Player1"], first["Player2"]), ("Astrid Lindqvist", "Björn Ekholm"))
        self.assertEqual((first["GoalsPlayer1"], first["GoalsPlayer2"]), (0, 2))
        self.assertEqual(first["StageType"], "round-robin")
        self.assertEqual(first["StageSequence"], 1)
        self.assertEqual(first["TournamentID"], _generate_id(TOURNAMENT.tournament_url))
        self.assertEqual(second["RoundNumber"], 2)
        self.assertEqual(second["Overtime"], "Yes")

    def test_single_group_stage_uses_stage_url(self):
        stage = {"id": 11464, "name": "Finalgrupp", "permalink": "finalgrupp", "stageGroup": 2, "kind": "group"}
        detail = {"groups": [{"name": "Grupp 1", "permalink": "grupp-1", "rounds": [
            {"round": 1, "matches": [_match(1, 1, 1, 2, 3, 0)]},
        ]}]}

        rows = rows_from_group_stage(TOURNAMENT, stage, detail, NAMES)

        self.assertEqual(rows[0]["Stage"], "Finalgrupp")
        self.assertEqual(rows[0]["StageURL"], f"{RESULT_URL}finalgrupp/?matcher=1")


class PlayoffStageTests(unittest.TestCase):
    def test_one_row_per_played_game_from_series_home_side(self):
        stage = {"id": 11229, "name": "A-slutspel", "permalink": "a-slutspel", "stageGroup": 3, "kind": "playoff"}
        detail = {"playoff": {"rounds": [{"name": "Final", "series": [{
            "homeId": 1, "awayId": 2,
            "games": [
                _match(446460, 1, 1, 2, 5, 3),
                _match(446461, 2, 2, 1, 4, 1, suddenDeath=True),
                _match(446462, 3, 1, 2, None, None),
            ],
        }]}]}}

        rows = rows_from_playoff_stage(TOURNAMENT, stage, detail, NAMES)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["Stage"], "A-slutspel Final")
        self.assertEqual(rows[0]["StageID"], 11229)
        self.assertEqual(rows[0]["StageURL"], f"{RESULT_URL}a-slutspel/")
        self.assertEqual(rows[0]["PlayoffGameNumber"], 1)
        self.assertEqual(rows[0]["RoundNumber"], 1)
        self.assertEqual(rows[1]["PlayoffGameNumber"], 2)
        self.assertEqual((rows[1]["Player1"], rows[1]["GoalsPlayer1"], rows[1]["GoalsPlayer2"]), ("Astrid Lindqvist", 1, 4))
        self.assertEqual(rows[1]["Overtime"], "Yes")


if __name__ == "__main__":
    unittest.main()
