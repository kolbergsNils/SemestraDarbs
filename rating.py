import math
import random
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


league_factor = {
    "NHL": 1.00,
    "KHL": 0.92,
    "SHL": 0.89,
    "AHL": 0.86,
    "SM-LIIGA": 0.85,
    "NLA": 0.84,
    "CZECH1": 0.84,
    "DEL": 0.83,
    "EBEL": 0.80,
    "ECHL": 0.80,
    "ALLSVENSKAN": 0.77,
    "NLB": 0.77,
    "VHL": 0.77,
    "SLOVAKIA1": 0.73,
    "LEAGUE MAGNUS": 0.72,
    "EIHL": 0.71,
    "DEL2": 0.70,
    "CZECH2": 0.70,
    "MESTIS": 0.69,
    "NORWAY1": 0.67,
    "LATVIA1": 0.67,
    "DENMARK1": 0.67,
    "ALPSHL": 0.65,
    "SPHL": 0.65,
    "ASIA LEAGUE": 0.63,
    "OBERLIGA": 0.62,
    "HUNGARY1": 0.61,
    "POLAND1": 0.61,
    "SLOVAKIA2": 0.61,
    "SWE3": 0.60,
    "FRANCE2": 0.59,
    "SUI3": 0.57,
    "RUS3": 0.57,
    "NEIHL": 0.55,
    "BELARUS1": 0.55,
    "CZECH3": 0.53,
    "FIN3": 0.53,
    "ITALY2": 0.52,
    "DENMARK2": 0.52,
    "NORWAY2": 0.51,
    "FRANCE3": 0.51,
    "SWITZERLAND4": 0.51,
    "CROATIA1": 0.51,
    "BENELIGAUE": 0.50,
    "AUSTRALIA": 0.50,
    "GER4": 0.48,
    "Unknown": 0.50
}


def get_league_factor(player):
    return league_factor.get(player["league"], league_factor["Unknown"])


def get_national_team_games(player):
    if "latvia_games" in player:
        return player["latvia_games"]
    elif "swiss_games" in player:
        return player["swiss_games"]
    else:
        return 0


def age_factor(age):

    if age < 20:
        return 0.80
    elif age < 22:
        return 0.95
    elif age < 25:
        return 1.00
    elif age < 28:
        return 0.97
    elif age < 30:
        return 0.92
    elif age < 32:
        return 0.82
    elif age < 34:
        return 0.75
    elif age < 36:
        return 0.60
    else:
        return 0.50


#laukuma spēletāji

def skater_offense_rating(player):
    gp = max(player["games"], 1)

    goals = player["goals"]
    assists = player["assists"]
    points = player["points"]

    gpg = goals / gp
    apg = assists / gp
    ppg = points / gp

    league = get_league_factor(player)

    adjusted_gpg = gpg * league
    adjusted_apg = apg * league
    adjusted_ppg = ppg * league

    offense = (
        55
        + 18 * adjusted_ppg
        + 8 * adjusted_gpg
        + 4 * adjusted_apg
    )

    return max(50, min(90, offense))


def skater_defense_rating(player):
    gp = max(player["games"], 1)

    plus_minus = player["plus_minus"]
    pim = player["pim"]

    plus_minus_pg = plus_minus / gp
    pim_pg = pim / gp

    league = get_league_factor(player)
    games_factor = min(gp / 40, 1)

    defense = (
        55
        + 8 * league
        + 6 * plus_minus_pg
        + 2 * games_factor
        - 0.35 * pim_pg
    )

    return max(50, min(90, defense))


def skater_rating(player):
    offense = player["offense_rating"]
    defense = player["defense_rating"]

    if player["position"] == "F":
        overall = 0.65 * offense + 0.35 * defense
    elif player["position"] == "D":
        overall = 0.35 * offense + 0.65 * defense
    else:
        overall = 0.50 * offense + 0.50 * defense

    return max(50, min(90, overall))


#vārtsargi

def goalie_rating(player):
    sv = player["save_percentage"]
    gaa = player["gaa"]
    gp = max(player["games"], 1)
    shutouts = player["shutouts"]

    league = get_league_factor(player)
    games_factor = min(gp / 40, 1)

    rating = (
        60
        + 180 * (sv - 0.900)
        - 2.5 * (gaa - 2.9)
        + 1.5 * shutouts
        + 6 * league
        + 2 * games_factor
    )

    return max(50, min(90, rating))


#iespēja tikt sastāvā

def roster_probability(player):
    rating = player["rating"]

    nt = min(get_national_team_games(player) / 10, 1)

    og = min(player["olympic_qualifier_games"] / 4, 1)

    league = get_league_factor(player)

    club_gp = min(player["games"] / 40, 1)

    age = player.get("age", None)
    age_score = age_factor(age)

    score = (
        -4.0
        + 0.06 * rating
        + 0.8 * nt
        + 0.3 * og
        + 0.6 * club_gp
        + 0.7 * league
        + 0.4 * age_score
    )

    probability = 1 / (1 + math.exp(-score))

    return probability


def calculate_skater_profiles(filename):
    players = pd.read_csv(filename)

    players["offense_rating"] = players.apply(skater_offense_rating, axis=1)
    players["defense_rating"] = players.apply(skater_defense_rating, axis=1)
    players["rating"] = players.apply(skater_rating, axis=1)
    players["roster_probability"] = players.apply(roster_probability, axis=1)

    return players


def calculate_goalie_profiles(filename):
    goalies = pd.read_csv(filename)

    goalies["rating"] = goalies.apply(goalie_rating, axis=1)
    goalies["roster_probability"] = goalies.apply(roster_probability, axis=1)

    return goalies



def show_skaters(players, title):
    players_sorted = players.sort_values(by="rating", ascending=False)

    print(title)
    print()

    for index, player in players_sorted.iterrows():
        print(player["name"])
        print("Club:", player["club"])
        print("League:", player["league"])
        print("Offense rating:", round(player["offense_rating"], 2))
        print("Defense rating:", round(player["defense_rating"], 2))
        print("Overall rating:", round(player["rating"], 2))
        print("Roster probability:", round(player["roster_probability"] * 100, 2), "%")
        print()


def show_goalies(goalies):
    goalies_sorted = goalies.sort_values(by="rating", ascending=False)

    print("GOALIES")
    print()

    for index, player in goalies_sorted.iterrows():
        print(player["name"])
        print("Club:", player["club"])
        print("League:", player["league"])
        print("Rating:", round(player["rating"], 2))
        print("Roster probability:", round(player["roster_probability"] * 100, 2), "%")
        print()


#main

def main():
    countries = ["latvia", "swiss"]

    for country in countries:
        defenders_file = country + "_defenders.csv"
        forwards_file = country + "_forwards.csv"
        goalies_file = country + "_goalies.csv"

        if (
            os.path.exists(defenders_file)
            and os.path.exists(forwards_file)
            and os.path.exists(goalies_file)
        ):
            defenders = calculate_skater_profiles(defenders_file)
            forwards = calculate_skater_profiles(forwards_file)
            goalies = calculate_goalie_profiles(goalies_file)

            print()
            print(country.upper())
            print("=" * len(country))

            show_skaters(defenders, "DEFENDERS")
            show_skaters(forwards, "FORWARDS")
            show_goalies(goalies)

            defenders.to_csv(country + "_defenders_ratings.csv", index=False)
            forwards.to_csv(country + "_forwards_ratings.csv", index=False)
            goalies.to_csv(country + "_goalies_ratings.csv", index=False)


main()  