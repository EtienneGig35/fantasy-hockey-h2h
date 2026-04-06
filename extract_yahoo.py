#!/usr/bin/env python3
"""Extract H2H matchup data from a Yahoo Fantasy Hockey league using yfpy."""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from yfpy.query import YahooFantasySportsQuery

# Fallback ratio categories (when API doesn't specify)
RATIO_CATEGORIES = {"GAA", "SV%", "Save Percentage", "AVG"}


def get_query(league_id: str, season: str) -> YahooFantasySportsQuery:
    """Create an authenticated YahooFantasySportsQuery."""
    load_dotenv()

    consumer_key = os.getenv("YAHOO_CONSUMER_KEY")
    consumer_secret = os.getenv("YAHOO_CONSUMER_SECRET")

    if not consumer_key or not consumer_secret:
        print("Error: YAHOO_CONSUMER_KEY and YAHOO_CONSUMER_SECRET must be set in .env")
        sys.exit(1)

    auth_dir = Path(".yfpy")
    auth_dir.mkdir(exist_ok=True)

    # First create a query without game_id to look up the correct one for the season
    query = YahooFantasySportsQuery(
        auth_dir=auth_dir,
        league_id=league_id,
        game_code="nhl",
        game_id=None,
        consumer_key=consumer_key,
        consumer_secret=consumer_secret,
        browser_callback=True,
    )

    # Resolve the game_id for the requested season
    game_key = query.get_game_key_by_season(int(season))
    print(f"  Resolved game key for season {season}: {game_key}")

    # Recreate query with the correct game_id
    query = YahooFantasySportsQuery(
        auth_dir=auth_dir,
        league_id=league_id,
        game_code="nhl",
        game_id=int(game_key),
        consumer_key=consumer_key,
        consumer_secret=consumer_secret,
        browser_callback=True,
    )
    return query


def detect_categories(query) -> list[dict]:
    """Detect scoring categories from league settings."""
    settings = query.get_league_settings()
    categories = []

    for stat_cat in settings.stat_categories.stats:
        stat = stat_cat
        name = stat.display_name if hasattr(stat, "display_name") else stat.name
        stat_id = stat.stat_id

        # Detect inverse from API sort_order (0=lower is better, 1=higher is better)
        is_inverse = hasattr(stat, "sort_order") and str(stat.sort_order) == "0"

        # Detect display-only from API
        is_display_only = hasattr(stat, "is_only_display_stat") and bool(stat.is_only_display_stat)

        # Detect ratio (averaged, not summed)
        is_ratio = name in RATIO_CATEGORIES

        categories.append({
            "name": name,
            "stat_id": stat_id,
            "is_inverse": is_inverse,
            "is_ratio": is_ratio,
            "is_display_only": is_display_only,
        })

    return categories


def detect_teams(query) -> list[dict]:
    """Get all teams in the league."""
    teams = query.get_league_teams()
    result = []
    for team in teams:
        result.append({
            "team_key": team.team_key,
            "name": team.name.decode() if isinstance(team.name, bytes) else str(team.name),
        })
    return result


def extract_standings(query, teams_lookup: dict) -> list[dict]:
    """Extract official H2H standings from Yahoo API."""
    standings = []
    try:
        league_standings = query.get_league_standings()
        for team in league_standings.teams:
            name = team.name.decode() if isinstance(team.name, bytes) else str(team.name)
            ts = team.team_standings
            record = ts.outcome_totals if hasattr(ts, "outcome_totals") else None

            wins = int(record.wins) if record and hasattr(record, "wins") else 0
            losses = int(record.losses) if record and hasattr(record, "losses") else 0
            ties = int(record.ties) if record and hasattr(record, "ties") else 0
            pts = wins * 2 + ties
            pct = float(record.percentage) if record and hasattr(record, "percentage") else 0.0

            standings.append({
                "name": name,
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "points": pts,
                "pct": pct,
                "rank": int(ts.rank) if hasattr(ts, "rank") else 0,
            })
    except Exception as e:
        print(f"  Warning: could not extract standings: {e}")

    return standings


def _fetch_team_stats_raw(query, team_key: str, week: int, retries: int = 3) -> dict:
    """Fetch per-category stats for a team for a given week via raw Yahoo API."""
    import time
    url = (
        f"https://fantasysports.yahooapis.com/fantasy/v2/"
        f"team/{team_key}/stats;type=week;week={week}?format=json"
    )
    for attempt in range(retries):
        resp = query.oauth.session.get(url)
        if resp.status_code == 200:
            try:
                data = resp.json()
                break
            except Exception:
                pass
        wait = 2 ** attempt
        print(f"    Retry {attempt + 1} for {team_key} week {week} (status {resp.status_code}), waiting {wait}s...")
        time.sleep(wait)
    else:
        print(f"    FAILED to fetch {team_key} week {week} after {retries} retries")
        return {}
    team_data = data["fantasy_content"]["team"]

    # team_data[1] contains team_stats
    stats_block = team_data[1].get("team_stats", {}).get("stats", [])
    result = {}
    for entry in stats_block:
        s = entry["stat"]
        result[str(s["stat_id"])] = s["value"]
    return result


def extract_matchups(query, num_weeks: int, teams_lookup: dict, categories: list[dict]) -> list[dict]:
    """Extract all matchups for all weeks using raw API for stats."""
    matchups = []
    league_key = list(teams_lookup.keys())[0].rsplit(".t.", 1)[0]  # e.g. 453.l.20281

    for week in range(1, num_weeks + 1):
        try:
            week_matchups = query.get_league_matchups_by_week(week)
        except Exception as e:
            print(f"  Week {week}: error fetching matchups - {e}")
            continue

        week_count = 0
        for matchup in week_matchups:
            if not hasattr(matchup, "teams") or not matchup.teams:
                continue

            teams_in_matchup = matchup.teams
            if len(teams_in_matchup) < 2:
                continue

            team1_key = teams_in_matchup[0].team_key
            team2_key = teams_in_matchup[1].team_key

            team1_name = teams_lookup.get(team1_key, team1_key)
            team2_name = teams_lookup.get(team2_key, team2_key)

            # Fetch actual stats via raw API (with throttle to avoid rate limiting)
            import time
            time.sleep(0.5)
            raw1 = _fetch_team_stats_raw(query, team1_key, week)
            time.sleep(0.5)
            raw2 = _fetch_team_stats_raw(query, team2_key, week)

            # Map stat_id -> category name
            t1 = {}
            t2 = {}
            for cat in categories:
                sid = str(cat["stat_id"])
                name = cat["name"]
                for raw, target in [(raw1, t1), (raw2, t2)]:
                    v = raw.get(sid, "0")
                    try:
                        target[name] = float(v) if v not in (None, "", "-") else 0.0
                    except (ValueError, TypeError):
                        target[name] = 0.0

            matchups.append({
                "week": week,
                "team1": team1_name,
                "team2": team2_name,
                "team1_stats": t1,
                "team2_stats": t2,
            })
            week_count += 1

        print(f"  Week {week}: {week_count} matchups")

    return matchups


def is_json_complete(path: Path) -> bool:
    """Check if a league JSON exists and is complete."""
    if not path.exists():
        return False
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)

        # Must have standings
        if not d.get("standings"):
            print(f"  {path.name}: pas de standings -> ré-extraction nécessaire")
            return False

        # Must have is_display_only on categories
        if not any("is_display_only" in c for c in d.get("categories", [])):
            print(f"  {path.name}: pas de is_display_only -> ré-extraction nécessaire")
            return False

        # Check no zero-stat matchups
        for m in d.get("matchups", []):
            if all(v == 0 for v in m.get("team1_stats", {}).values()):
                print(f"  {path.name}: stats à 0 semaine {m['week']} -> ré-extraction nécessaire")
                return False

        # Check expected number of weeks (matchups per week = n_teams / 2)
        n_teams = len(d.get("teams", []))
        matchups = d.get("matchups", [])
        weeks = set(m["week"] for m in matchups)
        matchups_per_week = n_teams // 2
        for w in weeks:
            week_matchups = [m for m in matchups if m["week"] == w]
            if len(week_matchups) < matchups_per_week:
                print(f"  {path.name}: semaine {w} incomplète ({len(week_matchups)}/{matchups_per_week}) -> ré-extraction nécessaire")
                return False

        print(f"  {path.name}: complet ({len(weeks)} semaines, {n_teams} équipes, standings OK)")
        return True
    except (json.JSONDecodeError, KeyError):
        return False


def main():
    parser = argparse.ArgumentParser(description="Extract Yahoo Fantasy Hockey H2H data")
    parser.add_argument("league_id", help="Yahoo league ID")
    parser.add_argument("season", help="Season year (e.g. 2024)")
    parser.add_argument("--force", action="store_true", help="Force re-extraction even if JSON is complete")
    args = parser.parse_args()

    output_path = Path("leagues") / f"{args.league_id}_{args.season}.json"
    if not args.force and is_json_complete(output_path):
        print(f"Skipping: {output_path.name} is already complete. Use --force to re-extract.")
        sys.exit(0)

    print(f"Connecting to Yahoo Fantasy API for league {args.league_id}, season {args.season}...")
    query = get_query(args.league_id, args.season)

    print("Detecting teams...")
    teams = detect_teams(query)
    teams_lookup = {t["team_key"]: t["name"] for t in teams}
    team_names = [t["name"] for t in teams]
    print(f"  Found {len(teams)} teams")

    print("Detecting categories...")
    categories = detect_categories(query)
    print(f"  Found {len(categories)} categories: {[c['name'] for c in categories]}")

    # Detect league name and number of weeks played
    metadata = query.get_league_metadata()
    league_name = metadata.name
    if isinstance(league_name, bytes):
        league_name = league_name.decode()
    league_name = str(league_name)
    print(f"  League name: {league_name}")

    settings = query.get_league_settings()
    current_week = int(metadata.current_week)
    print(f"  Current week: {current_week}")

    # Try to get the number of weeks from settings, fallback to current_week
    num_weeks = current_week
    if hasattr(settings, "playoff_start_week") and settings.playoff_start_week:
        playoff_week = int(settings.playoff_start_week)
        # Regular season weeks
        num_weeks = min(current_week, playoff_week - 1)
    print(f"  Extracting {num_weeks} weeks of matchups...")

    print("Extracting matchups...")
    matchups = extract_matchups(query, num_weeks, teams_lookup, categories)

    print("Extracting standings...")
    standings = extract_standings(query, teams_lookup)
    print(f"  Found standings for {len(standings)} teams")

    # Clean categories for output (remove stat_id)
    categories_out = [
        {
            "name": c["name"],
            "is_inverse": c["is_inverse"],
            "is_ratio": c["is_ratio"],
            "is_display_only": c.get("is_display_only", False),
        }
        for c in categories
    ]

    output = {
        "league_id": args.league_id,
        "league_name": league_name,
        "season": args.season,
        "teams": team_names,
        "categories": categories_out,
        "matchups": matchups,
        "standings": standings,
    }

    output_dir = Path("leagues")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{args.league_id}_{args.season}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {output_path}")
    print(f"  {len(team_names)} teams, {len(categories_out)} categories, {len(matchups)} matchups")


if __name__ == "__main__":
    main()
