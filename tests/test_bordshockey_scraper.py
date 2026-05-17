import unittest

import pandas as pd
from bs4 import BeautifulSoup

from bordshockey_scraper import (
    StageMeta,
    TournamentMeta,
    _canonical_url,
    _deduplicate_results,
    _parse_result_overview,
    _rows_from_group_soup,
    _rows_from_playoff_soup,
)


RESULT_URL = "https://bordshockey.net/tavlingar/2526/swedish-masters/resultat/"
TOURNAMENT = TournamentMeta(
    name="Swedish Masters",
    date="2026-02-07",
    tournament_id=123,
    tournament_url="https://bordshockey.net/tavlingar/2526/swedish-masters/",
    result_url=RESULT_URL,
    source_tournament_id="2526/swedish-masters",
)


class OverviewParserTests(unittest.TestCase):
    def test_extracts_sequences_ids_group_urls_and_stage_types(self):
        soup = BeautifulSoup(
            """
            <table>
              <tr class="stage_group"><td>1</td></tr>
              <tr class="stage stage-11218">
                <td class="stage_name"><a href="qualification-groups/">Qualification groups</a></td>
                <td class="status">Avslutad</td>
                <td class="placements"></td>
                <td class="links">
                  <a href="qualification-groups/">Tabeller</a>
                  <a href="qualification-groups/?korstabell=1">Korstabeller</a>
                </td>
              </tr>
              <tr class="group">
                <td class="group_name"><a href="qualification-groups/grupp-1/?matcher=1">Grupp 1</a></td>
                <td class="links"><a href="qualification-groups/grupp-1/?matcher=1">Tabell och matcher</a></td>
              </tr>
              <tr class="stage_group"><td>2</td></tr>
              <tr class="stage stage-11223">
                <td class="stage_name"><a href="final-groups-a/">Final groups A</a></td>
                <td class="links"><a href="final-groups-a/">Tabeller</a></td>
              </tr>
              <tr class="group">
                <td class="group_name"><a href="final-groups-a/grupp-1/?matcher=1">Grupp 1</a></td>
                <td class="links"><a href="final-groups-a/grupp-1/?matcher=1">Tabell och matcher</a></td>
              </tr>
              <tr class="stage_group"><td>3</td></tr>
              <tr class="stage stage-11229">
                <td class="stage_name"><a href="playoff-a/">Playoff A</a></td>
                <td class="links"><a href="playoff-a/">Slutspel</a></td>
              </tr>
            </table>
            """,
            "lxml",
        )

        stages = _parse_result_overview(soup, RESULT_URL)

        self.assertEqual(len(stages), 3)

        qualification = stages[0]
        self.assertEqual(qualification.name, "Qualification groups - Grupp 1")
        self.assertEqual(qualification.stage_type, "round-robin")
        self.assertEqual(qualification.stage_sequence, 1)
        self.assertEqual(qualification.source_stage_id, "11218")
        self.assertNotEqual(qualification.stage_id, 11218)
        self.assertEqual(
            qualification.stage_url,
            _canonical_url(f"{RESULT_URL}qualification-groups/grupp-1/?matcher=1"),
        )

        final_group = stages[1]
        self.assertEqual(final_group.name, "Final groups A - Grupp 1")
        self.assertEqual(final_group.stage_type, "round-robin")
        self.assertEqual(final_group.stage_sequence, 2)

        playoff = stages[2]
        self.assertEqual(playoff.name, "Playoff A")
        self.assertEqual(playoff.stage_type, "playoff")
        self.assertEqual(playoff.stage_id, 11229)
        self.assertEqual(playoff.source_stage_id, "11229")
        self.assertEqual(playoff.stage_sequence, 3)


class RoundRobinParserTests(unittest.TestCase):
    def test_extracts_match_rows_round_numbers_scores_and_source_ids(self):
        stage = StageMeta(
            name="Qualification groups - Grupp 1",
            stage_type="round-robin",
            stage_url=_canonical_url(f"{RESULT_URL}qualification-groups/grupp-1/?matcher=1"),
            stage_id=456,
            stage_sequence=1,
            source_stage_id="11218",
        )
        soup = BeautifulSoup(
            """
            <div class="round_matches round_2">
              <table>
                <tbody>
                  <tr class="round_header"><td colspan="9">Omgång 2 av 17</td></tr>
                  <tr class="match-443278">
                    <td class="table_no">1</td>
                    <td class="home_name participant-69921">Axel Lönnqvist</td>
                    <td class="name_sep">-</td>
                    <td class="away_name participant-69989">Krisjanis Zemnickis</td>
                    <td class="home_score">2</td>
                    <td class="score_sep">-</td>
                    <td class="away_score">2</td>
                    <td class="matchinfo">(SD)</td>
                  </tr>
                </tbody>
              </table>
              <p class="noplay">Bye: Someone</p>
            </div>
            """,
            "lxml",
        )

        rows = _rows_from_group_soup(soup, stage, TOURNAMENT)

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["StageType"], "round-robin")
        self.assertEqual(row["StageURL"], stage.stage_url)
        self.assertEqual(row["SourceURL"], stage.stage_url)
        self.assertEqual(row["SourceStageID"], "11218")
        self.assertEqual(row["SourceMatchID"], "443278")
        self.assertEqual(row["RoundNumber"], 2)
        self.assertIsNone(row["PlayoffGameNumber"])
        self.assertEqual(row["Player1"], "Axel Lönnqvist")
        self.assertEqual(row["Player2"], "Krisjanis Zemnickis")
        self.assertEqual(row["GoalsPlayer1"], 2)
        self.assertEqual(row["GoalsPlayer2"], 2)
        self.assertEqual(row["Overtime"], "Yes")


class PlayoffParserTests(unittest.TestCase):
    def test_extracts_one_row_per_played_game_and_ignores_mobile_rows(self):
        stage = StageMeta(
            name="Playoff A",
            stage_type="playoff",
            stage_url=_canonical_url(f"{RESULT_URL}playoff-a/"),
            stage_id=11229,
            stage_sequence=3,
            source_stage_id="11229",
        )
        soup = BeautifulSoup(
            """
            <div class="stage stage-11229 playoff_stage">
              <p>Matcherna spelas i bäst av 7</p>
              <div class="playoff_round">
                <h3>16-delsfinal</h3>
                <table class="playoff_round">
                  <tbody>
                    <tr class="match">
                      <td class="table_no">1</td>
                      <td class="home_name winner participant-69909">Rainers Kalnins</td>
                      <td class="name_sep">-</td>
                      <td class="away_name participant-70005">Emils Liepins</td>
                      <td class="match match_played"><span class="result">4 - 3</span><div class="matchinfo">(SD)</div></td>
                      <td class="match match_played"><span class="result">8 - 2</span></td>
                      <td class="match"></td>
                    </tr>
                    <tr class="mobile_matches">
                      <td class="match"><span class="result">4 - 3</span></td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
            """,
            "lxml",
        )

        rows = _rows_from_playoff_soup(soup, stage, TOURNAMENT)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["StageType"], "playoff")
        self.assertEqual(rows[0]["Stage"], "Playoff A 16-delsfinal")
        self.assertEqual(rows[0]["PlayoffGameNumber"], 1)
        self.assertEqual(rows[0]["GoalsPlayer1"], 4)
        self.assertEqual(rows[0]["GoalsPlayer2"], 3)
        self.assertEqual(rows[0]["Overtime"], "Yes")
        self.assertEqual(rows[1]["PlayoffGameNumber"], 2)
        self.assertEqual(rows[1]["GoalsPlayer1"], 8)
        self.assertEqual(rows[1]["GoalsPlayer2"], 2)
        self.assertIsNone(rows[0]["SourceMatchID"])
        self.assertEqual(rows[0]["SourceStageID"], "11229")


class DedupeTests(unittest.TestCase):
    def test_source_match_ids_survive_csv_numeric_round_trip(self):
        base = {
            "StageID": 1,
            "Player1": "A",
            "Player1ID": None,
            "Player2": "B",
            "Player2ID": None,
            "GoalsPlayer1": 1,
            "GoalsPlayer2": 2,
            "Overtime": "No",
            "Stage": "Qualification groups - Grupp 1",
            "RoundNumber": 1,
            "PlayoffGameNumber": None,
            "Date": "2026-02-07",
            "TournamentName": "Swedish Masters",
            "TournamentID": 123,
            "StageSequence": 1,
            "StageType": "round-robin",
            "TournamentURL": TOURNAMENT.tournament_url,
            "ResultURL": TOURNAMENT.result_url,
            "StageURL": f"{RESULT_URL}qualification-groups/grupp-1/?matcher=1",
            "SourceURL": f"{RESULT_URL}qualification-groups/grupp-1/?matcher=1",
            "Source": "bordshockey.net",
            "SourceTournamentID": "2526/swedish-masters",
            "SourceStageID": "11218",
            "SourceMatchID": 443270.0,
        }
        newer = dict(base)
        newer["SourceMatchID"] = "443270"
        newer["GoalsPlayer1"] = 3

        deduped = _deduplicate_results(pd.DataFrame([base, newer]))

        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped.iloc[0]["GoalsPlayer1"], 3)


if __name__ == "__main__":
    unittest.main()
