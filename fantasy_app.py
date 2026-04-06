"""Application Streamlit pour l'analyse H2H de hockey fantasy Yahoo."""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from stats_engine import compute_all_tables

LEAGUES_DIR = Path("leagues")


def get_display_only_cats(league_data: dict) -> list[str]:
    """Get display-only categories from league metadata."""
    return [c["name"] for c in league_data.get("categories", []) if c.get("is_display_only", False)]


def get_scoring_categories_info(league_data: dict) -> list[dict]:
    """Get only scoring categories (exclude display-only)."""
    return [c for c in league_data.get("categories", []) if not c.get("is_display_only", False)]


def filter_display_only(df: pd.DataFrame, display_only: list[str]) -> pd.DataFrame:
    """Remove display-only columns from a DataFrame."""
    cols_to_drop = [c for c in display_only if c in df.columns]
    return df.drop(columns=cols_to_drop) if cols_to_drop else df


def is_league_json_complete(path: Path) -> bool:
    """Check if a league JSON is complete (inline version for sidebar)."""
    if not path.exists():
        return False
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        if not d.get("standings"):
            return False
        if not any("is_display_only" in c for c in d.get("categories", [])):
            return False
        for m in d.get("matchups", []):
            if all(v == 0 for v in m.get("team1_stats", {}).values()):
                return False
        n_teams = len(d.get("teams", []))
        matchups = d.get("matchups", [])
        weeks = set(m["week"] for m in matchups)
        matchups_per_week = n_teams // 2
        for w in weeks:
            week_matchups = [m for m in matchups if m["week"] == w]
            if len(week_matchups) < matchups_per_week:
                return False
        return True
    except (json.JSONDecodeError, KeyError):
        return False


def load_league(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def list_leagues() -> list[Path]:
    if not LEAGUES_DIR.exists():
        return []
    return sorted(LEAGUES_DIR.glob("*.json"))


def make_format_dict(columns: list[str], fmt: str, categories_info: list = None) -> dict:
    """Build a {col: format_string} dict for Styler.format()."""
    fmt_dict = {}
    if fmt == "pct":
        for col in columns:
            fmt_dict[col] = "{:.1%}"
    elif fmt == "avg":
        if categories_info:
            ratio_cats = {c["name"] for c in categories_info if c["is_ratio"]}
            for col in columns:
                fmt_dict[col] = "{:.3f}" if col in ratio_cats else "{:.1f}"
        else:
            for col in columns:
                fmt_dict[col] = "{:.2f}"
    elif fmt in ("int", "rank"):
        for col in columns:
            fmt_dict[col] = "{:.0f}"
    elif fmt == "classement":
        for col in columns:
            fmt_dict[col] = "{:.0f}"
    return fmt_dict


def generate_highlights(df: pd.DataFrame, categories_info: list[dict], table_type: str,
                        mode: str = "pooler") -> list[str]:
    """Generate 5 bullet-point highlights for a table."""
    highlights = []
    inverse_cats = {c["name"] for c in categories_info if c["is_inverse"]}
    is_adv = mode == "adversaire"
    cats = [c for c in df.columns if c not in ("Total", "Classement", "Somme des rangs", "Écart",
                                                 "Somme des rangs des adversaires")]

    if not cats and table_type != "classement":
        return highlights

    if table_type in ("avg", "totals", "win_pct"):
        # Best and worst performers
        for cat in cats[:3]:
            if cat in inverse_cats:
                best_team = df[cat].idxmin()
                best_val = df.loc[best_team, cat]
                worst_team = df[cat].idxmax()
            else:
                best_team = df[cat].idxmax()
                best_val = df.loc[best_team, cat]
                worst_team = df[cat].idxmin()
            if table_type == "win_pct":
                highlights.append(f"**{best_team}** domine en **{cat}** avec un taux de victoire de {best_val:.0%}")
            else:
                highlights.append(f"**{best_team}** mène en **{cat}** ({best_val:.2f})")
            if len(highlights) >= 2:
                break

        # Most dominant team (most categories where ranked #1)
        top_counts = {}
        for cat in cats:
            if cat in inverse_cats:
                best = df[cat].idxmin()
            else:
                best = df[cat].idxmax()
            top_counts[best] = top_counts.get(best, 0) + 1
        if top_counts:
            most_dominant = max(top_counts, key=top_counts.get)
            highlights.append(f"**{most_dominant}** est en tête dans {top_counts[most_dominant]}/{len(cats)} catégories")

        # Highest variance category
        variances = {cat: df[cat].std() / (df[cat].mean() + 1e-9) for cat in cats}
        most_var = max(variances, key=variances.get)
        highlights.append(f"**{most_var}** est la catégorie avec le plus grand écart entre les équipes")

        least_var = min(variances, key=variances.get)
        highlights.append(f"**{least_var}** est la catégorie la plus serrée")

    elif table_type == "rank":
        if "Total" in df.columns:
            best = df["Total"].idxmin()
            worst = df["Total"].idxmax()
            if is_adv:
                highlights.append(f"**{best}** a affronté les adversaires les plus forts (total {int(df.loc[best, 'Total'])})")
                highlights.append(f"**{worst}** a eu le calendrier le plus facile (total {int(df.loc[worst, 'Total'])})")
            else:
                highlights.append(f"**{best}** a la meilleure somme de rangs ({int(df.loc[best, 'Total'])})")
                highlights.append(f"**{worst}** a la pire somme de rangs ({int(df.loc[worst, 'Total'])})")

        # Teams with most #1 ranks
        rank_1_counts = {}
        for cat in cats:
            for team in df.index:
                if df.loc[team, cat] == 1.0:
                    rank_1_counts[team] = rank_1_counts.get(team, 0) + 1
        if rank_1_counts:
            top = max(rank_1_counts, key=rank_1_counts.get)
            if is_adv:
                highlights.append(f"**{top}** a affronté les meilleurs adversaires dans {rank_1_counts[top]} catégories")
            else:
                highlights.append(f"**{top}** est classé #1 dans {rank_1_counts[top]} catégories")

        # Teams with most last-place ranks
        n = len(df)
        last_counts = {}
        for cat in cats:
            for team in df.index:
                if df.loc[team, cat] == float(n):
                    last_counts[team] = last_counts.get(team, 0) + 1
        if last_counts:
            bottom = max(last_counts, key=last_counts.get)
            if is_adv:
                highlights.append(f"**{bottom}** a affronté les pires adversaires dans {last_counts[bottom]} catégories")
            else:
                highlights.append(f"**{bottom}** est dernier dans {last_counts[bottom]} catégories")

        # Most balanced team (lowest std of ranks)
        if cats:
            stds = {team: df.loc[team, cats].std() for team in df.index}
            most_balanced = min(stds, key=stds.get)
            if is_adv:
                highlights.append(f"**{most_balanced}** a affronté des adversaires de force similaire dans toutes les catégories")
            else:
                highlights.append(f"**{most_balanced}** est l'équipe la plus équilibrée (rangs les plus constants)")

    elif table_type == "classement":
        col_somme = [c for c in df.columns if "Somme" in c]
        if col_somme:
            sc = col_somme[0]
            best = df[sc].idxmin()
            worst = df[sc].idxmax()
            if is_adv:
                highlights.append(f"**{best}** a eu le calendrier le plus difficile (somme {int(df.loc[best, sc])})")
                highlights.append(f"**{worst}** a eu le calendrier le plus facile (somme {int(df.loc[worst, sc])})")
            else:
                highlights.append(f"**{best}** domine le classement avec une somme de {int(df.loc[best, sc])}")
                highlights.append(f"**{worst}** ferme la marche avec une somme de {int(df.loc[worst, sc])}")
            gap = int(df.loc[worst, sc] - df.loc[best, sc])
            highlights.append(f"L'écart entre le premier et le dernier est de **{gap}** points de rang")
        if "Écart" in df.columns:
            mid = df.sort_values("Écart").iloc[len(df) // 2]
            if is_adv:
                highlights.append(f"L'équipe médiane a un écart de **{int(mid['Écart'])}** par rapport au calendrier le plus dur")
            else:
                highlights.append(f"L'équipe médiane a un écart de **{int(mid['Écart'])}** par rapport au leader")
        highlights.append(f"**{len(df)}** équipes au classement")

    return highlights[:5]


def _green_red_scale(ratio):
    """Retourne un style CSS avec un dégradé vert (0) -> blanc (0.5) -> rouge (1)."""
    ratio = max(0.0, min(1.0, ratio))
    if ratio <= 0.5:
        # Vert: intensité décroissante de 0 à 0.5
        intensity = 1 - ratio * 2  # 1.0 -> 0.0
        r = int(255 - intensity * 100)
        g = int(255 - intensity * 20)
        b = int(255 - intensity * 100)
        tr = int(max(0, 50 * intensity))
        tg = int(80 + 20 * intensity)
        tb = int(max(0, 50 * intensity))
        return f"background-color: rgba({155 + int(100 * (1 - intensity))}, {235 + int(20 * (1 - intensity))}, {155 + int(100 * (1 - intensity))}, {0.1 + 0.7 * intensity}); color: rgb(0, {60 + int(40 * intensity)}, 0)"
    else:
        # Rouge: intensité croissante de 0.5 à 1
        intensity = (ratio - 0.5) * 2  # 0.0 -> 1.0
        return f"background-color: rgba(255, {199 - int(70 * intensity)}, {199 - int(70 * intensity)}, {0.1 + 0.7 * intensity}); color: rgb({120 + int(36 * intensity)}, 0, 0)"


def style_gradient(df_raw: pd.DataFrame, n_teams: int,
                   categories_info: list, higher_is_better: bool = True, cols: list = None,
                   fmt: str = None, fmt_categories_info: list = None):
    """Colorer les cellules avec un dégradé continu vert-blanc-rouge basé sur le rang.

    Returns a Styler with formatting and color applied on the raw numeric DataFrame.
    """
    styles = pd.DataFrame("", index=df_raw.index, columns=df_raw.columns)
    target_cols = cols if cols else list(df_raw.columns)

    for col in target_cols:
        if col not in df_raw.columns:
            continue
        is_inverse = False
        for c in categories_info:
            if c["name"] == col and c["is_inverse"]:
                is_inverse = True
                break
        if is_inverse:
            ranks = df_raw[col].rank(ascending=True, method="average")
        else:
            ranks = df_raw[col].rank(ascending=not higher_is_better, method="average")

        for idx in ranks.index:
            ratio = (ranks[idx] - 1) / max(n_teams - 1, 1)
            styles.loc[idx, col] = _green_red_scale(ratio)

    styler = df_raw.style.apply(lambda _: styles, axis=None)
    if fmt:
        fmt_dict = make_format_dict(df_raw.columns.tolist(), fmt, fmt_categories_info)
        if fmt_dict:
            styler = styler.format(fmt_dict)
    return styler


def style_diff_gradient(df: pd.DataFrame, col: str, max_abs: int):
    """Colorer une colonne de différentiel: positif=vert, négatif=rouge, intensité proportionnelle."""
    styles = pd.DataFrame("", index=df.index, columns=df.columns)
    for idx in df.index:
        try:
            v = int(df.loc[idx, col])
        except (ValueError, TypeError):
            continue
        if v == 0 or max_abs == 0:
            continue
        if v > 0:
            intensity = min(abs(v) / max_abs, 1.0)
            styles.loc[idx, col] = f"background-color: rgba(155, 235, 155, {0.2 + 0.6 * intensity}); color: rgb(0, {60 + int(40 * intensity)}, 0)"
        else:
            intensity = min(abs(v) / max_abs, 1.0)
            styles.loc[idx, col] = f"background-color: rgba(255, {199 - int(70 * intensity)}, {199 - int(70 * intensity)}, {0.2 + 0.6 * intensity}); color: rgb({120 + int(36 * intensity)}, 0, 0)"

    def fmt_diff(v):
        try:
            vi = int(v)
            return f"+{vi}" if vi > 0 else str(vi)
        except (ValueError, TypeError):
            return str(v)

    return df.style.apply(lambda _: styles, axis=None).format({col: fmt_diff})


def _show_highlights(highlights: list[str]):
    """Render highlights as a single compact markdown block."""
    if highlights:
        st.markdown("<br>".join(f"&bull; {h}" for h in highlights), unsafe_allow_html=True)


def display_tables(tables: dict, categories_info: list, n_teams: int, mode: str, league_data: dict):
    """Afficher les tableaux avec titres en français."""
    mode_label = "du pooler" if mode == "pooler" else "des adversaires"
    display_only = get_display_only_cats(league_data)
    scoring_cats_info = get_scoring_categories_info(league_data)

    # --- 1. Moyennes ---
    st.subheader("1. Moyennes par catégorie")
    if mode == "adversaire":
        st.caption(
            "Moyenne des stats des adversaires affrontés par chaque pooler. "
            "Vert = adversaires plus forts affrontés, rouge = adversaires plus faibles."
        )
    else:
        st.caption(
            "Moyenne du pooler pour chaque catégorie sur l'ensemble des semaines sélectionnées. "
            "Vert = meilleure performance relative, rouge = moins bonne."
        )
    avg = filter_display_only(tables["averages"], display_only)
    st.dataframe(
        style_gradient(avg, n_teams, scoring_cats_info, higher_is_better=True,
                       fmt="avg", fmt_categories_info=scoring_cats_info),
        use_container_width=True,
    )

    # --- 2. Totaux ---
    st.subheader("2. Totaux")
    if mode == "adversaire":
        st.caption(
            "Somme des stats des adversaires affrontés (les catégories ratio restent en moyenne). "
            "Permet de voir le volume cumulé des adversaires."
        )
    else:
        st.caption(
            "Somme des stats du pooler sur toutes les semaines (les catégories ratio restent en moyenne). "
            "Permet de voir le volume cumulé plutôt que la moyenne hebdomadaire."
        )
    totals = filter_display_only(tables["totals"], display_only)
    st.dataframe(
        style_gradient(totals, n_teams, scoring_cats_info, higher_is_better=True,
                       fmt="avg", fmt_categories_info=scoring_cats_info),
        use_container_width=True,
    )

    # --- 3. Classement performances (rang) ---
    st.subheader("3. Classement performances (rang)")
    if mode == "adversaire":
        st.caption(
            f"Rang de 1 à {n_teams} par catégorie, basé sur les moyennes des adversaires affrontés. "
            "La colonne Total est la somme des rangs. Plus le total est bas, plus le pooler a affronté des adversaires forts."
        )
    else:
        st.caption(
            f"Rang de 1 à {n_teams} par catégorie, basé sur les moyennes. "
            "La colonne Total est la somme des rangs. Plus le total est bas, meilleure est la performance globale."
        )
    rankings = filter_display_only(tables["rankings"], display_only)
    rankings_with_total = rankings.copy()
    rankings_with_total.insert(0, "Total", rankings.sum(axis=1).astype(int))
    raw_with_total = rankings_with_total.astype(float)
    st.dataframe(
        style_gradient(raw_with_total, n_teams, categories_info=[], higher_is_better=False, fmt="rank"),
        use_container_width=True,
    )
    _show_highlights(generate_highlights(rankings_with_total, scoring_cats_info, "rank", mode))

    # --- 4. % de victoire par catégorie ---
    st.subheader("4. % de victoire par catégorie")
    st.caption(
        f"Pourcentage de semaines où le {mode_label.replace('du ', '').replace('des ', '')} a gagné chaque catégorie en H2H. "
        "100% = invaincu dans cette catégorie, 0% = jamais gagné."
    )
    win_pct = filter_display_only(tables["win_pct"], display_only)
    st.dataframe(
        style_gradient(win_pct, n_teams, categories_info=[], higher_is_better=True, fmt="pct"),
        use_container_width=True,
    )
    _show_highlights(generate_highlights(win_pct, scoring_cats_info, "win_pct", mode))

    # --- 5. Classement (somme des rangs) ---
    label_somme = "Somme des rangs des adversaires" if mode == "adversaire" else "Somme des rangs"
    st.subheader(f"5. Classement ({label_somme.lower()})")
    if mode == "adversaire":
        st.caption(
            "Classement global basé sur la somme des rangs des adversaires (tableau 3). "
            "Un total bas signifie que le pooler a affronté des adversaires forts (calendrier difficile). "
            "L'écart indique la distance par rapport au calendrier le plus difficile."
        )
    else:
        st.caption(
            "Classement global basé sur la somme des rangs du tableau 3. "
            "L'écart indique la distance par rapport à la meilleure équipe."
        )
    cls = tables["classement"].copy()
    cls = cls.rename(columns={"Somme des rangs": label_somme})
    if mode == "adversaire":
        st.dataframe(
            cls.style.format({c: "{:.0f}" for c in cls.columns}),
            use_container_width=True,
        )
    else:
        st.dataframe(
            style_gradient(cls, n_teams, categories_info=[], higher_is_better=False,
                           cols=["Classement"], fmt="classement"),
            use_container_width=True,
        )
    _show_highlights(generate_highlights(cls, scoring_cats_info, "classement", mode))


def _analyze_combined_score(h2h_rank: pd.Series, perf_rank: pd.Series,
                            diff: pd.Series, n_teams: int) -> list[str]:
    """Generate insights for the combined score table."""
    highlights = []

    # Dominant teams: top 3 in BOTH rankings
    top_half = n_teams // 2
    for team in h2h_rank.sort_values().index:
        hr = int(h2h_rank[team])
        pr = int(perf_rank[team])
        if hr <= 3 and pr <= 3:
            highlights.append(
                f"**{team}** domine sur tous les plans (#{hr} H2H, #{pr} performance)")
        elif hr <= top_half and pr <= top_half and hr <= 3:
            highlights.append(
                f"**{team}** est solidement en tête (#{hr} H2H, #{pr} performance)")

    # Luckiest: biggest negative diff (H2H rank much better than perf rank)
    lucky = diff.sort_values()
    for team in lucky.head(3).index:
        d = int(diff[team])
        if d < -1:
            hr = int(h2h_rank[team])
            pr = int(perf_rank[team])
            highlights.append(
                f"**{team}** bénéficie de chance : #{hr} au H2H malgré une performance de #{pr} "
                f"({abs(d)} rangs au-dessus de sa valeur réelle)")

    # Unluckiest: biggest positive diff (perf rank much better than H2H rank)
    unlucky = diff.sort_values(ascending=False)
    for team in unlucky.head(3).index:
        d = int(diff[team])
        if d > 1:
            hr = int(h2h_rank[team])
            pr = int(perf_rank[team])
            highlights.append(
                f"**{team}** est malchanceux : #{pr} en performance mais seulement #{hr} au H2H "
                f"({d} rangs en dessous de sa valeur réelle)")

    # Teams where both ranks match perfectly
    exact_matches = [t for t in diff.index if diff[t] == 0]
    if exact_matches:
        names = ", ".join(f"**{t}**" for t in exact_matches[:3])
        highlights.append(
            f"{names} — rang H2H = rang performance (classement fidèle à la réalité)")

    return highlights


def _analyze_power_profile(norm_data: pd.DataFrame, win_pct: pd.DataFrame,
                           cats: list[str], h2h_rank: pd.Series,
                           inverse_cats: set) -> list[str]:
    """Generate detailed per-category insights from power profile."""
    highlights = []

    # For each team, compute avg gap between normalized perf and win%
    # A high perf but low win% = unlucky in that category
    luck_by_team_cat = {}
    for team in norm_data.index:
        for cat in cats:
            perf = norm_data.loc[team, cat]
            wp = win_pct.loc[team, cat] * 100
            gap = perf - wp  # positive = perf > wins (unlucky), negative = wins > perf (lucky)
            luck_by_team_cat[(team, cat)] = gap

    # Biggest unlucky spots: high perf, low win%
    sorted_unlucky = sorted(luck_by_team_cat.items(), key=lambda x: x[1], reverse=True)
    seen_teams_unlucky = set()
    for (team, cat), gap in sorted_unlucky:
        if gap >= 25 and team not in seen_teams_unlucky:
            perf = int(norm_data.loc[team, cat])
            wp = int(win_pct.loc[team, cat] * 100)
            highlights.append(
                f"**{team}** est malchanceux en **{cat}** : performance de {perf}/100 mais seulement {wp}% de victoires")
            seen_teams_unlucky.add(team)
        if len(seen_teams_unlucky) >= 3:
            break

    # Biggest lucky spots: low perf, high win%
    sorted_lucky = sorted(luck_by_team_cat.items(), key=lambda x: x[1])
    seen_teams_lucky = set()
    for (team, cat), gap in sorted_lucky:
        if gap <= -25 and team not in seen_teams_lucky:
            perf = int(norm_data.loc[team, cat])
            wp = int(win_pct.loc[team, cat] * 100)
            highlights.append(
                f"**{team}** est chanceux en **{cat}** : performance de {perf}/100 mais {wp}% de victoires")
            seen_teams_lucky.add(team)
        if len(seen_teams_lucky) >= 3:
            break

    # Dominant categories: teams with both perf >= 80 and win% >= 80
    dominators = {}
    for team in norm_data.index:
        dom_cats = []
        for cat in cats:
            if norm_data.loc[team, cat] >= 80 and win_pct.loc[team, cat] >= 0.80:
                dom_cats.append(cat)
        if len(dom_cats) >= 2:
            dominators[team] = dom_cats
    for team, dom_cats in sorted(dominators.items(), key=lambda x: -len(x[1])):
        highlights.append(
            f"**{team}** domine dans {', '.join(f'**{c}**' for c in dom_cats)} "
            f"(perf. et victoires > 80%)")
    if len(highlights) > 10:
        highlights = highlights[:10]

    # Weak categories: teams with both perf <= 20 and win% <= 20
    weak_teams = {}
    for team in norm_data.index:
        weak_cats = []
        for cat in cats:
            if norm_data.loc[team, cat] <= 20 and win_pct.loc[team, cat] <= 0.20:
                weak_cats.append(cat)
        if len(weak_cats) >= 2:
            weak_teams[team] = weak_cats
    for team, weak_cats in sorted(weak_teams.items(), key=lambda x: -len(x[1])):
        highlights.append(
            f"**{team}** est en difficulté dans {', '.join(f'**{c}**' for c in weak_cats)} "
            f"(perf. et victoires < 20%)")
        if len(highlights) >= 12:
            break

    # Most polarized team (biggest spread between best and worst categories)
    spreads = {}
    for team in norm_data.index:
        vals = [norm_data.loc[team, c] for c in cats]
        spreads[team] = max(vals) - min(vals)
    most_polarized = max(spreads, key=spreads.get)
    best_cat = max(cats, key=lambda c: norm_data.loc[most_polarized, c])
    worst_cat = min(cats, key=lambda c: norm_data.loc[most_polarized, c])
    highlights.append(
        f"**{most_polarized}** est le plus polarisé : {int(norm_data.loc[most_polarized, best_cat])}/100 en "
        f"**{best_cat}** vs {int(norm_data.loc[most_polarized, worst_cat])}/100 en **{worst_cat}**")

    return highlights


def page_resume(league_data: dict, max_week: int):
    """Page Résumé : score combiné en premier, puis sous-classements en dépliants, puis visualisations."""
    n_teams = len(league_data["teams"])

    pooler = compute_all_tables(league_data, "pooler", max_week, exclude_from_ranking=get_display_only_cats(league_data))
    adversaire = compute_all_tables(league_data, "adversaire", max_week, exclude_from_ranking=get_display_only_cats(league_data))

    # Rang H2H basé sur les points (victoire=2, nulle=1, défaite=0 par catégorie)
    h2h = pooler["h2h_points"]
    h2h_rank = h2h["Rang H2H"]
    h2h_pts = h2h["Points H2H"]

    # Rang Performance (basé sur somme des rangs pooler, moins = mieux)
    perf_rank = pooler["classement"]["Somme des rangs"].rank(method="min").astype(int)

    # Différentiel = Rang Performance - Rang H2H
    diff = perf_rank - h2h_rank

    # --- Score combiné ---
    st.subheader("Score combiné")
    st.caption(
        "Vue d'ensemble combinant le classement H2H réel et la performance par catégorie. "
        "Le différentiel montre combien de rangs une équipe gagne (+) ou perd (-) "
        "au classement H2H par rapport à sa performance réelle."
    )
    combined = pd.DataFrame({
        "Rang H2H": h2h_rank,
        "Rang Performance": perf_rank,
        "Différentiel": diff,
    })
    combined = combined.sort_values("Rang H2H")
    max_abs_diff = max(abs(diff.max()), abs(diff.min()), 1)
    st.dataframe(
        style_diff_gradient(combined, "Différentiel", max_abs_diff),
        use_container_width=True,
    )

    # --- Analyse du score combiné ---
    combined_highlights = _analyze_combined_score(h2h_rank, perf_rank, diff, n_teams)
    _show_highlights(combined_highlights)

    # --- Sous-classements en dépliants ---
    with st.expander("Classement H2H"):
        st.caption(
            "Basé sur les points H2H cumulés : chaque catégorie vaut 2 pts (victoire), 1 pt (nulle) "
            "ou 0 pt (défaite) par semaine. Plus de points = meilleur rang."
        )
        h2h_df = pd.DataFrame({
            "Points H2H": h2h_pts.astype(int),
            "Rang": h2h_rank,
        }).sort_values("Rang")
        st.dataframe(h2h_df, use_container_width=True)

    with st.expander("Classement Performance"):
        st.caption(
            "Basé sur la somme des rangs de chaque équipe dans chaque catégorie (moyennes). "
            "Une somme basse signifie que l'équipe performe bien globalement. SV et SA sont exclus."
        )
        perf_df = pd.DataFrame({
            "Somme des rangs": pooler["classement"]["Somme des rangs"].astype(int),
            "Rang": perf_rank,
        }).sort_values("Rang")
        st.dataframe(perf_df, use_container_width=True)

    with st.expander("Indice de force des adversaires"):
        st.caption(
            "Mesure la force des adversaires affrontés. Un rang bas signifie que l'équipe "
            "a affronté des adversaires plus faibles (calendrier favorable). SV et SA sont exclus."
        )
        adv_rank = adversaire["classement"]["Somme des rangs"].rank(ascending=False, method="min").astype(int)
        adv_df = pd.DataFrame({
            "Somme des rangs adversaires": adversaire["classement"]["Somme des rangs"].astype(int),
            "Rang": adv_rank,
        }).sort_values("Rang")
        st.dataframe(adv_df, use_container_width=True)

    # --- Visualisations ---
    st.markdown("---")
    st.subheader("Visualisations")

    # 1. Graphique à barres : Différentiel Performance vs H2H
    st.markdown("**Différentiel Performance vs H2H**")
    st.caption(
        "Les équipes en vert performent mieux que leur classement H2H ne le suggère (malchanceuses). "
        "Les équipes en rouge ont un classement H2H meilleur que leur performance réelle (chanceuses)."
    )
    diff_df = pd.DataFrame({
        "Équipe": diff.index,
        "Différentiel": diff.values,
    }).sort_values("Différentiel", ascending=True)
    colors = ["#2ecc71" if v > 0 else "#e74c3c" if v < 0 else "#95a5a6" for v in diff_df["Différentiel"]]
    fig_diff = go.Figure(go.Bar(
        x=diff_df["Différentiel"],
        y=diff_df["Équipe"],
        orientation="h",
        marker_color=colors,
        text=diff_df["Différentiel"].map(lambda x: f"+{x}" if x > 0 else str(x)),
        textposition="outside",
    ))
    fig_diff.update_layout(
        height=max(400, n_teams * 35),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Différentiel (Rang Perf. - Rang H2H)",
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig_diff, use_container_width=True)

    # 2. Radar chart : Top 3 et Bottom 3 par catégorie
    st.markdown("**Profil par catégorie — Top 3 vs Bottom 3**")
    st.caption(
        "Comparaison du profil des 3 meilleures et 3 pires équipes au classement H2H. "
        "Le pourcentage de victoire par catégorie montre les forces et faiblesses."
    )
    win_pct = filter_display_only(pooler["win_pct"], get_display_only_cats(league_data))
    ranking_cats = list(win_pct.columns)
    top3 = h2h_rank.nsmallest(3).index.tolist()
    bot3 = h2h_rank.nlargest(3).index.tolist()

    fig_radar = go.Figure()
    for team in top3:
        vals = [win_pct.loc[team, c] * 100 for c in ranking_cats]
        vals.append(vals[0])
        fig_radar.add_trace(go.Scatterpolar(
            r=vals, theta=ranking_cats + [ranking_cats[0]],
            name=f"{team} (#{h2h_rank[team]})",
            line=dict(width=2),
        ))
    for team in bot3:
        vals = [win_pct.loc[team, c] * 100 for c in ranking_cats]
        vals.append(vals[0])
        fig_radar.add_trace(go.Scatterpolar(
            r=vals, theta=ranking_cats + [ranking_cats[0]],
            name=f"{team} (#{h2h_rank[team]})",
            line=dict(width=2, dash="dash"),
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        height=500,
        margin=dict(l=40, r=40, t=40, b=40),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # 3. Profil de puissance : performance réelle normalisée + % victoire
    st.markdown("**Profil de puissance par catégorie**")
    st.caption(
        "Chaque cellule montre la performance réelle normalisée (0-100) avec un indicateur du % de victoire H2H. "
        "La couleur reflète la performance réelle, le chiffre entre parenthèses est le % de victoire. "
        "Un écart entre les deux révèle de la chance ou de la malchance."
    )
    averages = filter_display_only(pooler["averages"], get_display_only_cats(league_data))
    scoring_cats_info = get_scoring_categories_info(league_data)
    inverse_cats = {c["name"] for c in scoring_cats_info if c["is_inverse"]}

    # Normalize averages to 0-100 per category (best=100, worst=0)
    norm_data = pd.DataFrame(index=averages.index, columns=ranking_cats, dtype=float)
    for cat in ranking_cats:
        col = averages[cat]
        cmin, cmax = col.min(), col.max()
        if cmax == cmin:
            norm_data[cat] = 50.0
        elif cat in inverse_cats:
            norm_data[cat] = ((cmax - col) / (cmax - cmin) * 100).round(0)
        else:
            norm_data[cat] = ((col - cmin) / (cmax - cmin) * 100).round(0)

    # Sort by H2H rank
    norm_data["_rank"] = h2h_rank
    norm_data = norm_data.sort_values("_rank").drop(columns=["_rank"])
    win_pct_sorted = win_pct.loc[norm_data.index]

    # Build custom text: "perf (win%)"
    text_matrix = []
    for team in norm_data.index:
        row = []
        for cat in ranking_cats:
            perf = int(norm_data.loc[team, cat])
            wp = int(win_pct_sorted.loc[team, cat] * 100)
            row.append(f"{perf} ({wp}%)")
        text_matrix.append(row)

    fig_power = go.Figure(data=go.Heatmap(
        z=norm_data.values,
        x=ranking_cats,
        y=list(norm_data.index),
        text=text_matrix,
        texttemplate="%{text}",
        colorscale="RdYlGn",
        zmin=0, zmax=100,
        colorbar=dict(title="Perf. normalisée"),
    ))
    fig_power.update_layout(
        height=max(450, n_teams * 40),
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig_power, use_container_width=True)

    # Analyse du profil de puissance
    power_highlights = _analyze_power_profile(norm_data, win_pct_sorted, ranking_cats, h2h_rank, inverse_cats)
    _show_highlights(power_highlights)

    # 4. Heatmap : % victoire par catégorie pour toutes les équipes
    st.markdown("**Heatmap — % de victoire par catégorie**")
    st.caption(
        "Vue complète du taux de victoire de chaque équipe dans chaque catégorie. "
        "Les zones foncées indiquent les points forts, les zones claires les faiblesses."
    )
    heatmap_data = win_pct[ranking_cats].copy()
    heatmap_data["_rank"] = h2h_rank
    heatmap_data = heatmap_data.sort_values("_rank").drop(columns=["_rank"])
    fig_heat = px.imshow(
        (heatmap_data * 100).round(0),
        labels=dict(x="Catégorie", y="Équipe", color="% Victoire"),
        color_continuous_scale="RdYlGn",
        aspect="auto",
        text_auto=True,
    )
    fig_heat.update_layout(height=max(400, n_teams * 35), margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_heat, use_container_width=True)


# --- Mise en page ---
st.set_page_config(page_title="Fantasy Hockey H2H Analyzer", layout="wide")

# Sélection de la ligue (avant le titre pour pouvoir afficher le nom)
league_files = list_leagues()
league_data = None
league_title = "Fantasy Hockey H2H Analyzer"

if league_files:
    leagues_info = []
    for f in league_files:
        d = load_league(f)
        league_name = d.get("league_name", d.get("league_id", "?"))
        season = d.get("season", "?")
        complete = is_league_json_complete(f)
        marker = "\u2705" if complete else "\u26a0\ufe0f"
        leagues_info.append({
            "path": f,
            "league_name": league_name,
            "season": season,
            "label": f"{marker} {season} - {league_name}",
            "data": d,
            "complete": complete,
        })

    leagues_info.sort(key=lambda x: (x["league_name"], x["season"]), reverse=True)
    labels = [li["label"] for li in leagues_info]

    selected_idx = st.sidebar.selectbox("Ligue", range(len(labels)), format_func=lambda i: labels[i],
                                        key="selected_league")
    league_data = leagues_info[selected_idx]["data"]
    league_title = f"{leagues_info[selected_idx]['season']} — {leagues_info[selected_idx]['league_name']}"
else:
    st.warning("Aucune ligue trouvée. Importez une ligue via la barre latérale.")

st.title(league_title)

# Navigation
page = st.sidebar.radio("Navigation", ["Stats par pooler", "Stats des adversaires", "Résumé"],
                        key="navigation_page")

# Section d'importation
st.sidebar.markdown("---")
st.sidebar.subheader("Importer une ligue")
st.sidebar.caption(
    "L'identifiant de ligue dans l'URL Yahoo n'est pas le même que celui de l'API. "
    "Cliquez sur **Lister mes ligues** pour voir vos vrais identifiants."
)

if st.sidebar.button("Lister mes ligues Yahoo"):
    with st.spinner("Connexion à l'API Yahoo..."):
        try:
            result = subprocess.run(
                [sys.executable, "-c", """
import json
from pathlib import Path
from yfpy.query import YahooFantasySportsQuery
game_keys = [396, 403, 411, 419, 427, 453, 465]
seasons =   [2019, 2020, 2021, 2022, 2023, 2024, 2025]
q = YahooFantasySportsQuery(auth_dir=Path('.yfpy'), league_id='1', game_code='nhl', game_id=453,
    consumer_key=open('.env').read().split('YAHOO_CONSUMER_KEY=')[1].split('\\n')[0],
    consumer_secret=open('.env').read().split('YAHOO_CONSUMER_SECRET=')[1].split('\\n')[0],
    browser_callback=True)
results = []
for gk, season in zip(game_keys, seasons):
    try:
        for l in q.get_user_leagues_by_game_key(gk):
            name = l.name.decode() if isinstance(l.name, bytes) else str(l.name)
            results.append({"league_id": str(l.league_id), "name": name, "season": str(season), "game_key": str(gk)})
    except: pass
print(json.dumps(results))
"""],
                capture_output=True, text=True, cwd=str(Path(__file__).parent),
                timeout=60,
            )
            if result.returncode == 0 and result.stdout.strip():
                st.session_state["yahoo_leagues"] = json.loads(result.stdout.strip())
            else:
                st.sidebar.error("Erreur lors de la connexion. Vérifiez vos identifiants dans .env")
        except Exception as e:
            st.sidebar.error(f"Erreur : {e}")

if "yahoo_leagues" in st.session_state and st.session_state["yahoo_leagues"]:
    yahoo_leagues = st.session_state["yahoo_leagues"]

    # Add completeness markers to importable leagues
    for yl in yahoo_leagues:
        json_path = LEAGUES_DIR / f"{yl['league_id']}_{yl['season']}.json"
        if is_league_json_complete(json_path):
            yl["_marker"] = "\u2705"
        elif json_path.exists():
            yl["_marker"] = "\u26a0\ufe0f"
        else:
            yl["_marker"] = "\u274c"

    league_names = list(dict.fromkeys(l["name"] for l in yahoo_leagues))
    selected_name = st.sidebar.selectbox("Ligue à importer", league_names, key="import_league_name")

    available_seasons = [l["season"] for l in yahoo_leagues if l["name"] == selected_name]
    season_labels = []
    for s in available_seasons:
        m = next(l for l in yahoo_leagues if l["name"] == selected_name and l["season"] == s)
        season_labels.append(f"{m['_marker']} {s}")
    selected_season_idx = st.sidebar.selectbox("Saison", range(len(available_seasons)),
                                                format_func=lambda i: season_labels[i], key="import_season")
    selected_import_season = available_seasons[selected_season_idx]

    match = next(l for l in yahoo_leagues if l["name"] == selected_name and l["season"] == selected_import_season)
    st.sidebar.text(f"Identifiant API : {match['league_id']}")

    if st.sidebar.button("Importer cette ligue"):
        with st.spinner(f"Extraction de {selected_name} ({selected_import_season})..."):
            result = subprocess.run(
                [sys.executable, "extract_yahoo.py", match["league_id"], match["season"]],
                capture_output=True, text=True, cwd=str(Path(__file__).parent),
                timeout=600,
            )
            if result.returncode == 0:
                st.sidebar.success("Extraction réussie !")
                st.rerun()
            else:
                st.sidebar.error(f"Erreur :\n{result.stderr[-500:]}")
else:
    st.sidebar.info("Cliquez sur **Lister mes ligues** pour commencer.")

# Contenu principal
if league_data:
    max_possible_week = max(m["week"] for m in league_data["matchups"])
    max_week = st.slider("Semaines (1 à max)", 1, max_possible_week, max_possible_week)
    n_teams = len(league_data["teams"])
    categories_info = league_data["categories"]

    if page == "Stats par pooler":
        st.header("Stats par pooler")
        tables = compute_all_tables(league_data, "pooler", max_week, exclude_from_ranking=get_display_only_cats(league_data))
        display_tables(tables, categories_info, n_teams, mode="pooler", league_data=league_data)

    elif page == "Stats des adversaires":
        st.header("Stats des adversaires")
        tables = compute_all_tables(league_data, "adversaire", max_week, exclude_from_ranking=get_display_only_cats(league_data))
        display_tables(tables, categories_info, n_teams, mode="adversaire", league_data=league_data)

    elif page == "Résumé":
        st.header("Résumé")
        page_resume(league_data, max_week)
