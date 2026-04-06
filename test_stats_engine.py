"""Unit tests for stats_engine with a small fixture of 4 teams, 3 categories, 2 weeks."""

import pytest
import pandas as pd
from stats_engine import compute_all_tables


@pytest.fixture
def sample_league():
    """4 teams, 3 categories (G normal, A normal, GAA inverse+ratio), 2 weeks."""
    return {
        "league_id": "test",
        "season": "2024",
        "teams": ["Alpha", "Beta", "Gamma", "Delta"],
        "categories": [
            {"name": "G", "is_inverse": False, "is_ratio": False},
            {"name": "A", "is_inverse": False, "is_ratio": False},
            {"name": "GAA", "is_inverse": True, "is_ratio": True},
        ],
        "matchups": [
            # Week 1: Alpha vs Beta, Gamma vs Delta
            {
                "week": 1,
                "team1": "Alpha", "team2": "Beta",
                "team1_stats": {"G": 10, "A": 20, "GAA": 2.5},
                "team2_stats": {"G": 8, "A": 25, "GAA": 3.0},
            },
            {
                "week": 1,
                "team1": "Gamma", "team2": "Delta",
                "team1_stats": {"G": 12, "A": 15, "GAA": 2.0},
                "team2_stats": {"G": 6, "A": 30, "GAA": 3.5},
            },
            # Week 2: Alpha vs Gamma, Beta vs Delta
            {
                "week": 2,
                "team1": "Alpha", "team2": "Gamma",
                "team1_stats": {"G": 14, "A": 18, "GAA": 2.0},
                "team2_stats": {"G": 10, "A": 22, "GAA": 2.5},
            },
            {
                "week": 2,
                "team1": "Beta", "team2": "Delta",
                "team1_stats": {"G": 9, "A": 28, "GAA": 2.8},
                "team2_stats": {"G": 7, "A": 26, "GAA": 3.2},
            },
        ],
    }


class TestComputeAllTables:
    def test_averages_pooler(self, sample_league):
        tables = compute_all_tables(sample_league, "pooler")
        avg = tables["averages"]

        assert len(avg) == 4
        assert list(avg.columns) == ["G", "A", "GAA"]

        # Alpha: G=(10+14)/2=12, A=(20+18)/2=19, GAA=(2.5+2.0)/2=2.25
        assert avg.loc["Alpha", "G"] == pytest.approx(12.0)
        assert avg.loc["Alpha", "A"] == pytest.approx(19.0)
        assert avg.loc["Alpha", "GAA"] == pytest.approx(2.25)

        # Beta: G=(8+9)/2=8.5, A=(25+28)/2=26.5, GAA=(3.0+2.8)/2=2.9
        assert avg.loc["Beta", "G"] == pytest.approx(8.5)
        assert avg.loc["Beta", "A"] == pytest.approx(26.5)
        assert avg.loc["Beta", "GAA"] == pytest.approx(2.9)

    def test_totals_pooler(self, sample_league):
        tables = compute_all_tables(sample_league, "pooler")
        totals = tables["totals"]

        # Alpha: G=10+14=24, A=20+18=38, GAA=avg=2.25 (ratio)
        assert totals.loc["Alpha", "G"] == pytest.approx(24.0)
        assert totals.loc["Alpha", "A"] == pytest.approx(38.0)
        assert totals.loc["Alpha", "GAA"] == pytest.approx(2.25)

        # Delta: G=6+7=13, A=30+26=56, GAA=avg=3.35 (ratio)
        assert totals.loc["Delta", "G"] == pytest.approx(13.0)
        assert totals.loc["Delta", "A"] == pytest.approx(56.0)
        assert totals.loc["Delta", "GAA"] == pytest.approx(3.35)

    def test_rankings_respect_inverse(self, sample_league):
        tables = compute_all_tables(sample_league, "pooler")
        rankings = tables["rankings"]

        # G averages: Alpha=12, Gamma=11, Beta=8.5, Delta=6.5
        # Rank (higher=better): Alpha=1, Gamma=2, Beta=3, Delta=4
        assert rankings.loc["Alpha", "G"] == 1.0
        assert rankings.loc["Delta", "G"] == 4.0

        # GAA averages: Gamma=2.25, Alpha=2.25, Beta=2.9, Delta=3.35
        # Rank (lower=better, ascending): Gamma/Alpha=1, Beta=3, Delta=4
        assert rankings.loc["Gamma", "GAA"] == 1.0 or rankings.loc["Alpha", "GAA"] == 1.0
        assert rankings.loc["Delta", "GAA"] == 4.0

    def test_losses_pooler(self, sample_league):
        tables = compute_all_tables(sample_league, "pooler")
        losses = tables["losses"]

        # Week 1: Alpha(G=10) vs Beta(G=8) -> Beta loses G
        # Week 1: Alpha(A=20) vs Beta(A=25) -> Alpha loses A
        # Week 1: Alpha(GAA=2.5) vs Beta(GAA=3.0) -> Beta loses GAA (higher=worse for inverse)
        # Week 1: Gamma(G=12) vs Delta(G=6) -> Delta loses G
        # Week 1: Gamma(A=15) vs Delta(A=30) -> Gamma loses A
        # Week 1: Gamma(GAA=2.0) vs Delta(GAA=3.5) -> Delta loses GAA
        # Week 2: Alpha(G=14) vs Gamma(G=10) -> Gamma loses G
        # Week 2: Alpha(A=18) vs Gamma(A=22) -> Alpha loses A
        # Week 2: Alpha(GAA=2.0) vs Gamma(GAA=2.5) -> Gamma loses GAA
        # Week 2: Beta(G=9) vs Delta(G=7) -> Delta loses G
        # Week 2: Beta(A=28) vs Delta(A=26) -> Delta loses A
        # Week 2: Beta(GAA=2.8) vs Delta(GAA=3.2) -> Delta loses GAA

        assert losses.loc["Alpha", "G"] == 0  # Alpha never lost G
        assert losses.loc["Alpha", "A"] == 2  # Alpha lost A both weeks
        assert losses.loc["Delta", "G"] == 2  # Delta lost G both weeks
        assert losses.loc["Delta", "GAA"] == 2  # Delta lost GAA both weeks

    def test_win_pct_pooler(self, sample_league):
        tables = compute_all_tables(sample_league, "pooler")
        win_pct = tables["win_pct"]

        # All values should be between 0 and 1
        assert (win_pct >= 0).all().all()
        assert (win_pct <= 1).all().all()

        # Alpha won G in both weeks -> 100%
        assert win_pct.loc["Alpha", "G"] == pytest.approx(1.0)
        # Alpha lost A in both weeks -> 0%
        assert win_pct.loc["Alpha", "A"] == pytest.approx(0.0)

    def test_classement_is_sum_of_ranks(self, sample_league):
        tables = compute_all_tables(sample_league, "pooler")
        rankings = tables["rankings"]
        classement = tables["classement"]

        for team in sample_league["teams"]:
            expected = rankings.loc[team].sum()
            assert classement.loc[team, "Somme des rangs"] == pytest.approx(expected)

    def test_ecart(self, sample_league):
        tables = compute_all_tables(sample_league, "pooler")
        classement = tables["classement"]

        min_val = classement["Somme des rangs"].min()
        for team in sample_league["teams"]:
            assert classement.loc[team, "Écart"] == pytest.approx(
                classement.loc[team, "Somme des rangs"] - min_val
            )

    def test_classement_has_rank(self, sample_league):
        tables = compute_all_tables(sample_league, "pooler")
        classement = tables["classement"]
        n = len(sample_league["teams"])

        assert "Classement" in classement.columns
        assert classement["Classement"].min() == 1
        assert classement["Classement"].max() <= n

    def test_max_week_filter(self, sample_league):
        tables_w1 = compute_all_tables(sample_league, "pooler", max_week=1)
        tables_all = compute_all_tables(sample_league, "pooler")

        # With max_week=1, Alpha only has week 1 stats: G=10
        assert tables_w1["averages"].loc["Alpha", "G"] == pytest.approx(10.0)
        # With all weeks, Alpha: G=(10+14)/2=12
        assert tables_all["averages"].loc["Alpha", "G"] == pytest.approx(12.0)

    def test_adversaire_mode(self, sample_league):
        tables = compute_all_tables(sample_league, "adversaire")
        avg = tables["averages"]

        # Alpha faced Beta(w1) and Gamma(w2)
        # Adversaire stats for Alpha: Beta's w1 stats and Gamma's w2 stats
        # G: (8 + 10) / 2 = 9, A: (25 + 22) / 2 = 23.5, GAA: (3.0 + 2.5) / 2 = 2.75
        assert avg.loc["Alpha", "G"] == pytest.approx(9.0)
        assert avg.loc["Alpha", "A"] == pytest.approx(23.5)
        assert avg.loc["Alpha", "GAA"] == pytest.approx(2.75)

    def test_ranks_go_1_to_n(self, sample_league):
        tables = compute_all_tables(sample_league, "pooler")
        rankings = tables["rankings"]
        n = len(sample_league["teams"])

        for cat in rankings.columns:
            vals = sorted(rankings[cat].values)
            assert vals[0] >= 1.0
            assert vals[-1] <= n
