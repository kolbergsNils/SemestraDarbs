import math
import random
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


LATVIA_FORWARDS_FILE = "latvia_forwards_ratings.csv"
LATVIA_DEFENDERS_FILE = "latvia_defenders_ratings.csv"
LATVIA_GOALIES_FILE = "latvia_goalies_ratings.csv"

SWISS_FORWARDS_FILE = "swiss_forwards_ratings.csv"
SWISS_DEFENDERS_FILE = "swiss_defenders_ratings.csv"
SWISS_GOALIES_FILE = "swiss_goalies_ratings.csv"

TEAM_A = "Latvia"
TEAM_B = "Switzerland"

FORWARDS_NEEDED = 12
DEFENDERS_NEEDED = 8
GOALIES_NEEDED = 2

N_SIMULATIONS = 500


#palīgfunkcijas

def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def load_players():
    latvia_forwards = pd.read_csv(LATVIA_FORWARDS_FILE)
    latvia_defenders = pd.read_csv(LATVIA_DEFENDERS_FILE)
    latvia_goalies = pd.read_csv(LATVIA_GOALIES_FILE)

    swiss_forwards = pd.read_csv(SWISS_FORWARDS_FILE)
    swiss_defenders = pd.read_csv(SWISS_DEFENDERS_FILE)
    swiss_goalies = pd.read_csv(SWISS_GOALIES_FILE)

    forwards = pd.concat([latvia_forwards, swiss_forwards], ignore_index=True)
    defenders = pd.concat([latvia_defenders, swiss_defenders], ignore_index=True)
    goalies = pd.concat([latvia_goalies, swiss_goalies], ignore_index=True)

    return forwards, defenders, goalies


def average_rating(players, default=60):
    if len(players) == 0:
        return default

    return sum(player["rating"] for player in players) / len(players)


def choose_weighted(players, weight_column="rating"):
    if len(players) == 0:
        return {
            "name": "Emergency Player",
            "rating": 60,
            "offense_rating": 60,
            "defense_rating": 60
        }

    weights = [max(player[weight_column], 1) for player in players]

    return random.choices(players, weights=weights, k=1)[0]


def choose_goalie(roster):
    if len(roster["goalies"]) == 0:
        return {
            "name": "Emergency Goalie",
            "rating": 60
        }

    return max(roster["goalies"], key=lambda player: player["rating"])


#sastāva izvēle

def select_position_players(players, team_name, needed):
    team_players = players[players["team"] == team_name]

    selected = []

    for _, player in team_players.iterrows():
        if random.random() < player["roster_probability"]:
            selected.append(player)

    selected = sorted(selected, key=lambda player: player["rating"], reverse=True)

    if len(selected) < needed:
        already_selected_names = {player["name"] for player in selected}

        remaining = []

        for _, player in team_players.iterrows():
            if player["name"] not in already_selected_names:
                remaining.append(player)

        remaining = sorted(remaining, key=lambda player: player["rating"], reverse=True)
        selected.extend(remaining[:needed - len(selected)])

    return selected[:needed]


def select_roster(forwards, defenders, goalies, team_name):
    return {
        "forwards": select_position_players(forwards, team_name, FORWARDS_NEEDED),
        "defenders": select_position_players(defenders, team_name, DEFENDERS_NEEDED),
        "goalies": select_position_players(goalies, team_name, GOALIES_NEEDED)
    }


#uzbrucēju maiņas, aizsargu pāri

def create_forward_lines(roster):
    forwards = sorted(roster["forwards"], key=lambda player: player["rating"], reverse=True)

    return [
        forwards[0:3],
        forwards[3:6],
        forwards[6:9],
        forwards[9:12]
    ]


def create_defensive_pairs(roster):
    defenders = sorted(
        roster["defenders"],
        key=lambda player: player["rating"],
        reverse=True
    )

    return [
        defenders[0:2],
        defenders[2:4],
        defenders[4:6],
        defenders[6:8]
    ]


def choose_forward_line(roster):
    lines = create_forward_lines(roster)
    line_weights = [20/60, 17/60, 14/60, 9/60]

    return random.choices(lines, weights=line_weights, k=1)[0]


def choose_defensive_pair(roster):
    pairs = create_defensive_pairs(roster)

    pair_weights = [20/60, 17/60, 14/60, 9/60]

    return random.choices(pairs, weights=pair_weights, k=1)[0]


#spēles darbības princips

def shot_rate(attacking_roster, defending_roster, overtime=False):
    attack_strength = average_rating(attacking_roster["forwards"], default=60)
    defence_strength = average_rating(defending_roster["defenders"], default=60)

    base_rate = 0.478
    rate = base_rate + 0.015 * (attack_strength - defence_strength)

    if overtime:
        rate *= 1.30

    return max(0.20, min(0.95, rate))


def goal_probability(shooter, defender, goalie, overtime=False):
    shooter_rating = shooter["rating"]
    defender_rating = defender["rating"]
    goalie_rating = goalie["rating"]

    x = (
        -2.25
        + 0.075 * (shooter_rating - defender_rating)
        - 0.085 * (goalie_rating - 65)
    )

    if overtime:
        x += 0.20

    probability = sigmoid(x)

    return max(0.02, min(0.35, probability))


MISS_PROBABILITY = 0.2552
BLOCK_PROBABILITY = 0.2672


def shot_result(shooter, defender, goalie, overtime=False):
    p_goal = goal_probability(
        shooter,
        defender,
        goalie,
        overtime=overtime
    )

    r = random.random()

    if r < p_goal:
        return "goal", p_goal

    elif r < p_goal + BLOCK_PROBABILITY:
        return "blocked", p_goal

    elif r < p_goal + BLOCK_PROBABILITY + MISS_PROBABILITY:
        return "missed", p_goal

    else:
        return "saved", p_goal

def format_time(minute, second):
    return f"{minute:02d}:{second:02d}"


def simulate_shot(
    attacking_team,
    defending_team,
    attacking_roster,
    defending_roster,
    period,
    time_string,
    overtime=False
):
    forward_line = choose_forward_line(attacking_roster)
    defensive_pair = choose_defensive_pair(defending_roster)

    shooter = choose_weighted(forward_line, "rating")
    defender = choose_weighted(defensive_pair, "rating")
    goalie = choose_goalie(defending_roster)

    result, p_goal = shot_result(shooter, defender, goalie, overtime=overtime)

    return {
        "period": period,
        "time": time_string,
        "team": attacking_team,
        "shooter": shooter["name"],
        "defender": defender["name"],
        "goalie": goalie["name"],
        "result": result,
        "goal_probability": p_goal
    }


def simulate_period(
    team_a,
    team_b,
    roster_a,
    roster_b,
    period_name,
    minutes=20,
    sudden_death=False,
    overtime=False,
    stats=None,
    events=None
):
    if stats is None:
        stats = {}

    if events is None:
        events = []

    for minute in range(minutes):
        attacking_order = [
            (team_a, roster_a, team_b, roster_b),
            (team_b, roster_b, team_a, roster_a)
        ]

        random.shuffle(attacking_order)

        for team, attacking_roster, defending_team, defending_roster in attacking_order:
            rate = shot_rate(attacking_roster, defending_roster, overtime=overtime)
            number_of_shots = np.random.poisson(rate)

            for _ in range(number_of_shots):
                second = random.randint(0, 59)
                time_string = format_time(minute, second)

                event = simulate_shot(
                    attacking_team=team,
                    defending_team=defending_team,
                    attacking_roster=attacking_roster,
                    defending_roster=defending_roster,
                    period=period_name,
                    time_string=time_string,
                    overtime=overtime
                )

                events.append(event)
                stats[team]["shots"] += 1

                if event["result"] == "goal":
                    stats[team]["goals"] += 1
                    stats[team]["goal_scorers"].append(event["shooter"])

                    if sudden_death:
                        return stats, events, team

                elif event["result"] == "saved":
                    stats[defending_team]["saves"] += 1

                elif event["result"] == "blocked":
                    stats[defending_team]["blocks"] += 1

                elif event["result"] == "missed":
                    stats[team]["misses"] += 1

    return stats, events, None


#bullīši

def shootout_attempt(shooter, goalie):
    shooter_rating = shooter["rating"]
    goalie_rating = goalie["rating"]

    x = (
        -1.30
        + 0.085 * (shooter_rating - 65)
        - 0.095 * (goalie_rating - 65)
    )

    p_goal = max(0.08, min(0.55, sigmoid(x)))
    is_goal = random.random() < p_goal

    return is_goal, p_goal


def simulate_shootout(team_a, team_b, roster_a, roster_b):
    shooters_a = sorted(roster_a["forwards"], key=lambda player: player["rating"], reverse=True)
    shooters_b = sorted(roster_b["forwards"], key=lambda player: player["rating"], reverse=True)

    goalie_a = choose_goalie(roster_a)
    goalie_b = choose_goalie(roster_b)

    shootout_log = []

    score_a = 0
    score_b = 0

    team_a_first = random.choice([True, False])

    for round_number in range(1, 6):
        shooter_a = shooters_a[(round_number - 1) % len(shooters_a)]
        shooter_b = shooters_b[(round_number - 1) % len(shooters_b)]

        order = [
            (team_a, shooter_a, goalie_b),
            (team_b, shooter_b, goalie_a)
        ]

        if not team_a_first:
            order.reverse()

        for team, shooter, goalie in order:
            is_goal, p_goal = shootout_attempt(shooter, goalie)

            if team == team_a and is_goal:
                score_a += 1
            elif team == team_b and is_goal:
                score_b += 1

            shootout_log.append({
                "round": round_number,
                "team": team,
                "shooter": shooter["name"],
                "goalie": goalie["name"],
                "result": "goal" if is_goal else "no_goal",
                "goal_probability": p_goal
            })

        shots_left_a = 5 - round_number
        shots_left_b = 5 - round_number

        if score_a > score_b + shots_left_b:
            return team_a, shootout_log

        if score_b > score_a + shots_left_a:
            return team_b, shootout_log

    round_number = 6

    while True:
        shooter_a = shooters_a[(round_number - 1) % len(shooters_a)]
        shooter_b = shooters_b[(round_number - 1) % len(shooters_b)]

        order = [
            (team_a, shooter_a, goalie_b),
            (team_b, shooter_b, goalie_a)
        ]

        if not team_a_first:
            order.reverse()

        results = {}

        for team, shooter, goalie in order:
            is_goal, p_goal = shootout_attempt(shooter, goalie)
            results[team] = is_goal

            shootout_log.append({
                "round": round_number,
                "team": team,
                "shooter": shooter["name"],
                "goalie": goalie["name"],
                "result": "goal" if is_goal else "no_goal",
                "goal_probability": p_goal
            })

        if results[team_a] and not results[team_b]:
            return team_a, shootout_log

        if results[team_b] and not results[team_a]:
            return team_b, shootout_log

        round_number += 1


#pilna spele

def simulate_iihf_game(team_a, team_b, roster_a, roster_b):
    stats = {
        team_a: {
            "goals": 0,
            "shots": 0,
            "saves": 0,
            "blocks": 0,
            "misses": 0,
            "goal_scorers": []
        },
        team_b: {
            "goals": 0,
            "shots": 0,
            "saves": 0,
            "blocks": 0,
            "misses": 0,
            "goal_scorers": []
        }
    }

    events = []

    for period in [1, 2, 3]:
        stats, events, _ = simulate_period(
            team_a,
            team_b,
            roster_a,
            roster_b,
            period_name=f"P{period}",
            minutes=20,
            sudden_death=False,
            overtime=False,
            stats=stats,
            events=events
        )

    regulation_goals_a = stats[team_a]["goals"]
    regulation_goals_b = stats[team_b]["goals"]

    shootout_log = []
    winner = None
    win_type = None

    if stats[team_a]["goals"] > stats[team_b]["goals"]:
        winner = team_a
        win_type = "regulation"

    elif stats[team_b]["goals"] > stats[team_a]["goals"]:
        winner = team_b
        win_type = "regulation"

    else:
        stats, events, winner = simulate_period(
            team_a,
            team_b,
            roster_a,
            roster_b,
            period_name="OT",
            minutes=10,
            sudden_death=True,
            overtime=True,
            stats=stats,
            events=events
        )

        if winner is not None:
            win_type = "overtime"

        else:
            winner, shootout_log = simulate_shootout(
                team_a,
                team_b,
                roster_a,
                roster_b
            )

            win_type = "shootout"

            stats[winner]["goals"] += 1
            stats[winner]["goal_scorers"].append("Shootout decisive goal")

    return {
        "team_a": team_a,
        "team_b": team_b,
        "winner": winner,
        "win_type": win_type,
        "score": {
            team_a: stats[team_a]["goals"],
            team_b: stats[team_b]["goals"]
        },
        "regulation_score": {
            team_a: regulation_goals_a,
            team_b: regulation_goals_b
        },
        "stats": stats,
        "events": events,
        "shootout": shootout_log
    }


#monte karlo

def simulate_one_game(forwards, defenders, goalies):
    roster_a = select_roster(forwards, defenders, goalies, TEAM_A)
    roster_b = select_roster(forwards, defenders, goalies, TEAM_B)

    return simulate_iihf_game(
        TEAM_A,
        TEAM_B,
        roster_a,
        roster_b,
    )


def empty_team_results():
    return {
        "wins": 0,
        "regulation_wins": 0,
        "overtime_wins": 0,
        "shootout_wins": 0,
        "goals": 0,
        "shots": 0
    }


def monte_carlo_simulation(n_simulations=N_SIMULATIONS):
    forwards, defenders, goalies = load_players()

    results = {
        TEAM_A: empty_team_results(),
        TEAM_B: empty_team_results()
    }

    scorelines = Counter()
    goal_scorers = Counter()

    history = {
        "simulation": [],
        f"{TEAM_A}_win_probability": [],
        f"{TEAM_B}_win_probability": []
    }

    team_a_wins = 0
    team_b_wins = 0

    for simulation in range(1, n_simulations + 1):
        game_result = simulate_one_game(forwards, defenders, goalies)

        winner = game_result["winner"]
        win_type = game_result["win_type"]

        results[winner]["wins"] += 1

        if winner == TEAM_A:
            team_a_wins += 1
        elif winner == TEAM_B:
            team_b_wins += 1

        if win_type == "regulation":
            results[winner]["regulation_wins"] += 1
        elif win_type == "overtime":
            results[winner]["overtime_wins"] += 1
        elif win_type == "shootout":
            results[winner]["shootout_wins"] += 1

        for team in [TEAM_A, TEAM_B]:
            results[team]["goals"] += game_result["score"][team]
            results[team]["shots"] += game_result["stats"][team]["shots"]

            for scorer in game_result["stats"][team]["goal_scorers"]:
                if scorer != "Shootout decisive goal":
                    goal_scorers[scorer] += 1

        score_a = game_result["score"][TEAM_A]
        score_b = game_result["score"][TEAM_B]
        scorelines[f"{TEAM_A} {score_a} - {score_b} {TEAM_B}"] += 1

        history["simulation"].append(simulation)
        history[f"{TEAM_A}_win_probability"].append(team_a_wins / simulation * 100)
        history[f"{TEAM_B}_win_probability"].append(team_b_wins / simulation * 100)

    return results, scorelines, goal_scorers, history


#izvade par rezultatiem

def print_monte_carlo_results(results, scorelines, goal_scorers, n_simulations):
    print("MONTE CARLO RESULTS")
    print()
    print("Simulāciju skaits:", n_simulations)
    print()

    for team in [TEAM_A, TEAM_B]:
        wins = results[team]["wins"]
        regulation_wins = results[team]["regulation_wins"]
        overtime_wins = results[team]["overtime_wins"]
        shootout_wins = results[team]["shootout_wins"]

        print(team)
        print()
        print("Uzvaras varbūtība:", round(wins / n_simulations * 100, 2), "%")
        print("Pamatlaika uzvaras varbūtība:", round(regulation_wins / n_simulations * 100, 2), "%")
        print("Papildlaika uzvaras varbūtība:", round(overtime_wins / n_simulations * 100, 2), "%")
        print("Bullīšu uzvaras varbūtība:", round(shootout_wins / n_simulations * 100, 2), "%")
        print("Vidējais vārtu skaits :", round(results[team]["goals"] / n_simulations, 3))
        print("Vidējais metienu skaits:", round(results[team]["shots"] / n_simulations, 3))
        print()
        print()

    print("Biežākie spēles rezultāti")
    print()
    for scoreline, count in scorelines.most_common(10):
        print(scoreline, "-", round(count / n_simulations * 100, 2), "%")

    print()

    print("Biežākie vārtu guvēji")
    print()
    for scorer, goals in goal_scorers.most_common(15):
        print(scorer, "-", goals, "vārti,", round(goals / n_simulations, 3), "vārti uz spēli")


#plt grafiki

plt.rcParams.update({
    "font.family": "Times New Roman",
    "mathtext.fontset": "stix",
    "font.size": 16
})

def plot_convergence(history):
    plt.figure(figsize=(9, 5))

    plt.plot(
        history["simulation"],
        history[f"{TEAM_A}_win_probability"],
        label=f"{TEAM_A} uzvaras varbūtība"
    )

    plt.plot(
        history["simulation"],
        history[f"{TEAM_B}_win_probability"],
        label=f"{TEAM_B} uzvaras varbūtība"
    )

    plt.xlabel("Monte Karlo simulācijas kārtas numurs")
    plt.ylabel("Uzvaras varbūtība, %")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("grafiks1.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_top_goal_scorers(goal_scorers, top_n=15):
    top_scorers = goal_scorers.most_common(top_n)


    players = [item[0] for item in top_scorers]
    goals = [item[1] for item in top_scorers]

    plt.figure(figsize=(10, 6))
    plt.bar(players, goals)
    plt.xlabel("Spēlētājs")
    plt.ylabel("Savāktie vārti visās simulācijās (kopā)")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig("grafiks2.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_common_scorelines(scorelines, top_n=15):
    top_scorelines = scorelines.most_common(top_n)

    labels = [item[0] for item in top_scorelines]
    values = [item[1] for item in top_scorelines]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, values)
    plt.xlabel("Rezultāts")
    plt.ylabel("Biežums")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig("grafiks3.png", dpi=300, bbox_inches="tight")
    plt.show()


def make_graphs(scorelines, goal_scorers, history):
    plot_convergence(history)
    plot_top_goal_scorers(goal_scorers)
    plot_common_scorelines(scorelines)


#main

def main():
    results, scorelines, goal_scorers, history = monte_carlo_simulation(
        n_simulations=N_SIMULATIONS
    )

    print_monte_carlo_results(
        results=results,
        scorelines=scorelines,
        goal_scorers=goal_scorers,
        n_simulations=N_SIMULATIONS
    )

    make_graphs(scorelines, goal_scorers, history)


if __name__ == "__main__":
    main()
