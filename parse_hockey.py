import pandas
import os
import json


output_directory = "data"
data = pandas.read_csv("./data/HockeyFights_RegularSeason_20082012_times.csv", delimiter=",")

team_wise = {}
all_fights = {}

player_ids = {}
team_ids = {}
for _, row in data.iterrows():
    season = row["Season"]
    # fighter1 = row["Home Fighter"]
    # fighter2 = row["Away Fighter"]
    team1 = row["Home Team"]
    team2 = row["Away Team"]
    winning_team = row["Winning Team"]

    if season not in team_wise.keys():
        team_wise[season] = { "draws": [], "not draws": [] }

    if team1 not in team_ids.keys():
        team_ids[team1] = len(team_ids)
    if team2 not in team_ids.keys():
        team_ids[team2] = len(team_ids)

    team1_id = team_ids[team1]
    team2_id = team_ids[team2]
    if winning_team == "Draw":
        team_wise[season]["draws"].append([team1_id, team2_id])
    elif winning_team == team1:
        team_wise[season]["not draws"].append([team1_id, team2_id])
    elif winning_team == team2:
        team_wise[season]["not draws"].append([team2_id, team1_id])
    else:
        raise ValueError("Unable to determine result of fight.")

with open(os.path.join(output_directory, "season_wise_teams.json"), "w") as file_stream:
    json.dump(team_wise, file_stream)

with open(os.path.join(output_directory, "team_mappings.json"), "w") as file_stream:
    json.dump(team_ids, file_stream)
