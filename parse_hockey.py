import pandas
import os
import json


output_directory = "data"
data = pandas.read_csv("./data/HockeyFights_RegularSeason_20082012_times.csv", delimiter=",")

season_wise = {
        "draws": {},
        "not draws": {}
    }
all_fights = {
        "draws": [],
        "not draws": []
    }
for season, fighter1, fighter2, winner in\
            zip(data["Season"], data["Home Fighter"], data["Away Fighter"], data["Winning Fighter"]):

    if season not in season_wise["draws"].keys():
        season_wise["draws"][season] = []
        season_wise["not draws"][season] = []

    if winner == "Draw":
        season_wise["draws"][season].extend([fighter1, fighter2])
        all_fights["draws"].extend([fighter1, fighter2])
    elif fighter1 == winner:
        season_wise["not draws"][season].extend([fighter1, fighter2])
        all_fights["not draws"].extend([fighter1, fighter2])
    elif fighter2 == winner:
        season_wise["not draws"][season].extend([fighter2, fighter1])
        all_fights["not draws"].extend([fighter2, fighter1])
    else:
        raise ValueError("Unable to determine result of fight.")


with open(os.path.join(output_directory, "season_wise_fights.json"), "w") as file_stream:
    json.dump(season_wise, file_stream)

with open(os.path.join(output_directory, "all_fights.json"), "w") as file_stream:
    json.dump(all_fights, file_stream)
