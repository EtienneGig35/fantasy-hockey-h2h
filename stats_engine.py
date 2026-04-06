"""Stats engine for Yahoo Fantasy Hockey H2H analysis.

Computes 7 tables from league JSON data:
1. Averages per category per team
2. Totals (sum, except ratio categories stay as average)
3. Performance ranking (rank 1-N by category on averages)
4. Loss count (per category, weeks where the pooler lost the H2H)
5. Win percentage (wins / weeks played per category)
6. Ranking (sum of ranks from table 3)
7. Gap (ranking minus minimum ranking)
"""

import pandas as pd
import numpy as np


def _build_team_week_stats(league_data: dict, mode: str, max_week: int = None) -> dict:
    """Build a dict of {team: [{cat: val, ...}, ...]} from matchups.

    mode="pooler": stats of the team itself
    mode="adversaire": stats of the team's opponents
    """
    teams = league_data["teams"]
    matchups = league_data["matchups"]
    categories = [c["name"] for c in league_data["categories"]]

    # {team_name: [list of stat dicts, one per week played]}
    team_stats = {t: [] for t in teams}

    for m in matchups:
        week = m["week"]
        if max_week is not None and week > max_week:
            continue

        t1, t2 = m["team1"], m["team2"]
        s1, s2 = m["team1_stats"], m["team2_stats"]

        if mode == "pooler":
            if t1 in team_stats:
                team_stats[t1].append(s1)
            if t2 in team_stats:
                team_stats[t2].append(s2)
        else:  # adversaire
            if t1 in team_stats:
                team_stats[t1].append(s2)
            if t2 in team_stats:
                team_stats[t2].append(s1)

    return team_stats


def _compute_averages(team_stats: dict, categories: list[str]) -> pd.DataFrame:
    """Table 1: Average per category per team."""
    rows = {}
    for team, weeks in team_stats.items():
        if not weeks:
            rows[team] = {c: 0.0 for c in categories}
            continue
        avgs = {}
        for cat in categories:
            vals = [w.get(cat, 0.0) for w in weeks]
            avgs[cat] = np.mean(vals) if vals else 0.0
        rows[team] = avgs
    return pd.DataFrame.from_dict(rows, orient="index")[categories]


def _compute_totals(team_stats: dict, categories_info: list[dict]) -> pd.DataFrame:
    """Table 2: Totals (sum for counting stats, average for ratio stats)."""
    categories = [c["name"] for c in categories_info]
    ratio_cats = {c["name"] for c in categories_info if c["is_ratio"]}

    rows = {}
    for team, weeks in team_stats.items():
        if not weeks:
            rows[team] = {c: 0.0 for c in categories}
            continue
        totals = {}
        for cat in categories:
            vals = [w.get(cat, 0.0) for w in weeks]
            if cat in ratio_cats:
                totals[cat] = np.mean(vals) if vals else 0.0
            else:
                totals[cat] = sum(vals)
        rows[team] = totals
    return pd.DataFrame.from_dict(rows, orient="index")[categories]


def _compute_rankings(averages: pd.DataFrame, categories_info: list[dict]) -> pd.DataFrame:
    """Table 3: Rank 1 to N per category based on averages. Respects is_inverse."""
    rankings = pd.DataFrame(index=averages.index)
    for cat_info in categories_info:
        cat = cat_info["name"]
        if cat not in averages.columns:
            continue
        ascending = cat_info["is_inverse"]  # inverse = lower is better = ascending rank
        rankings[cat] = averages[cat].rank(ascending=ascending, method="min")
    return rankings


def _compute_losses(league_data: dict, mode: str, max_week: int = None) -> pd.DataFrame:
    """Table 4: Number of H2H losses per category per team."""
    teams = league_data["teams"]
    matchups = league_data["matchups"]
    categories_info = league_data["categories"]
    categories = [c["name"] for c in categories_info]
    inverse_cats = {c["name"] for c in categories_info if c["is_inverse"]}

    losses = {t: {c: 0 for c in categories} for t in teams}

    for m in matchups:
        week = m["week"]
        if max_week is not None and week > max_week:
            continue

        t1, t2 = m["team1"], m["team2"]
        s1, s2 = m["team1_stats"], m["team2_stats"]

        if mode == "pooler":
            my_stats_1, opp_stats_1 = s1, s2
            my_stats_2, opp_stats_2 = s2, s1
        else:  # adversaire
            my_stats_1, opp_stats_1 = s2, s1
            my_stats_2, opp_stats_2 = s1, s2

        for cat in categories:
            v1 = my_stats_1.get(cat, 0.0)
            o1 = opp_stats_1.get(cat, 0.0)
            v2 = my_stats_2.get(cat, 0.0)
            o2 = opp_stats_2.get(cat, 0.0)

            if cat in inverse_cats:
                # Lower is better: loss when your value > opponent's
                if v1 > o1 and t1 in losses:
                    losses[t1][cat] += 1
                if v2 > o2 and t2 in losses:
                    losses[t2][cat] += 1
            else:
                # Higher is better: loss when your value < opponent's
                if v1 < o1 and t1 in losses:
                    losses[t1][cat] += 1
                if v2 < o2 and t2 in losses:
                    losses[t2][cat] += 1

    return pd.DataFrame.from_dict(losses, orient="index")[categories]


def _compute_win_pct(league_data: dict, mode: str, max_week: int = None) -> pd.DataFrame:
    """Table 5: Win percentage per category per team."""
    teams = league_data["teams"]
    matchups = league_data["matchups"]
    categories_info = league_data["categories"]
    categories = [c["name"] for c in categories_info]
    inverse_cats = {c["name"] for c in categories_info if c["is_inverse"]}

    wins = {t: {c: 0 for c in categories} for t in teams}
    games = {t: 0 for t in teams}

    for m in matchups:
        week = m["week"]
        if max_week is not None and week > max_week:
            continue

        t1, t2 = m["team1"], m["team2"]
        s1, s2 = m["team1_stats"], m["team2_stats"]

        if mode == "pooler":
            my_stats_1, opp_stats_1 = s1, s2
            my_stats_2, opp_stats_2 = s2, s1
        else:
            my_stats_1, opp_stats_1 = s2, s1
            my_stats_2, opp_stats_2 = s1, s2

        if t1 in wins:
            games[t1] += 1
        if t2 in wins:
            games[t2] += 1

        for cat in categories:
            v1 = my_stats_1.get(cat, 0.0)
            o1 = opp_stats_1.get(cat, 0.0)
            v2 = my_stats_2.get(cat, 0.0)
            o2 = opp_stats_2.get(cat, 0.0)

            if cat in inverse_cats:
                if v1 < o1 and t1 in wins:
                    wins[t1][cat] += 1
                if v2 < o2 and t2 in wins:
                    wins[t2][cat] += 1
            else:
                if v1 > o1 and t1 in wins:
                    wins[t1][cat] += 1
                if v2 > o2 and t2 in wins:
                    wins[t2][cat] += 1

    pct = {}
    for team in teams:
        g = games.get(team, 0)
        pct[team] = {c: (wins[team][c] / g if g > 0 else 0.0) for c in categories}

    return pd.DataFrame.from_dict(pct, orient="index")[categories]


def _compute_classement(rankings: pd.DataFrame) -> pd.DataFrame:
    """Table 6+7 merged: Rank, sum of ranks, and gap."""
    somme = rankings.sum(axis=1)
    min_val = somme.min()
    df = pd.DataFrame({
        "Somme des rangs": somme,
        "Écart": somme - min_val,
    })
    df["Classement"] = df["Somme des rangs"].rank(method="min").astype(int)
    df = df[["Classement", "Somme des rangs", "Écart"]].sort_values("Classement")
    return df


def _compute_h2h_points(league_data: dict, max_week: int = None,
                        exclude_categories: list[str] = None) -> pd.DataFrame:
    """Get H2H points from standings if available, otherwise compute from matchups."""
    teams = league_data["teams"]
    standings = league_data.get("standings", [])
    max_possible_week = max(m["week"] for m in league_data["matchups"]) if league_data["matchups"] else 0

    # Use official standings if available and we're looking at all weeks
    if standings and (max_week is None or max_week >= max_possible_week):
        standings_lookup = {s["name"]: s for s in standings}
        pts_dict = {}
        for team in teams:
            s = standings_lookup.get(team)
            if s:
                pts_dict[team] = s["points"]
            else:
                pts_dict[team] = 0
        pts_series = pd.Series(pts_dict, name="Points H2H")
        rank_series = pts_series.rank(ascending=False, method="min").astype(int)
        df = pd.DataFrame({
            "Points H2H": pts_series,
            "Rang H2H": rank_series,
        }).sort_values("Rang H2H")
        return df

    # Fallback: compute from matchups
    matchups = league_data["matchups"]
    categories_info = league_data["categories"]
    if exclude_categories is None:
        exclude_categories = []
    # Also exclude categories marked as display_only in the JSON
    display_only = {c["name"] for c in categories_info if c.get("is_display_only", False)}
    all_excluded = set(exclude_categories) | display_only
    categories = [c["name"] for c in categories_info if c["name"] not in all_excluded]
    inverse_cats = {c["name"] for c in categories_info if c["is_inverse"]}

    points = {t: 0 for t in teams}

    for m in matchups:
        week = m["week"]
        if max_week is not None and week > max_week:
            continue

        t1, t2 = m["team1"], m["team2"]
        s1, s2 = m["team1_stats"], m["team2_stats"]

        for cat in categories:
            v1 = s1.get(cat, 0.0)
            v2 = s2.get(cat, 0.0)

            if cat in inverse_cats:
                if v1 < v2:
                    w1, w2 = 2, 0
                elif v1 > v2:
                    w1, w2 = 0, 2
                else:
                    w1, w2 = 1, 1
            else:
                if v1 > v2:
                    w1, w2 = 2, 0
                elif v1 < v2:
                    w1, w2 = 0, 2
                else:
                    w1, w2 = 1, 1

            if t1 in points:
                points[t1] += w1
            if t2 in points:
                points[t2] += w2

    pts_series = pd.Series(points, name="Points H2H")
    rank_series = pts_series.rank(ascending=False, method="min").astype(int)
    df = pd.DataFrame({
        "Points H2H": pts_series,
        "Rang H2H": rank_series,
    }).sort_values("Rang H2H")
    return df


def compute_all_tables(league_data: dict, mode: str, max_week: int = None,
                       exclude_from_ranking: list[str] = None) -> dict:
    """Compute all 7 tables.

    Args:
        league_data: dict loaded from league JSON
        mode: "pooler" or "adversaire"
        max_week: optional, filter weeks 1 to max_week
        exclude_from_ranking: list of category names to exclude from rankings/classement/ecart

    Returns:
        dict with keys: averages, totals, rankings, losses, win_pct, classement, ecart
    """
    if exclude_from_ranking is None:
        exclude_from_ranking = []

    team_stats = _build_team_week_stats(league_data, mode, max_week)
    categories_info = league_data["categories"]
    categories = [c["name"] for c in categories_info]

    averages = _compute_averages(team_stats, categories)
    totals = _compute_totals(team_stats, categories_info)
    losses = _compute_losses(league_data, mode, max_week)
    win_pct = _compute_win_pct(league_data, mode, max_week)

    # Rankings exclude display-only categories (from JSON metadata + explicit list)
    display_only = {c["name"] for c in categories_info if c.get("is_display_only", False)}
    all_excluded = set(exclude_from_ranking) | display_only
    ranking_cats_info = [c for c in categories_info if c["name"] not in all_excluded]
    rankings = _compute_rankings(averages, ranking_cats_info)
    classement = _compute_classement(rankings)
    h2h_points = _compute_h2h_points(league_data, max_week, list(all_excluded))

    return {
        "averages": averages,
        "totals": totals,
        "rankings": rankings,
        "losses": losses,
        "win_pct": win_pct,
        "classement": classement,
        "h2h_points": h2h_points,
    }
