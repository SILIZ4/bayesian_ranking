import random
import itertools
import json

from scipy import stats
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt

from modeling import fit_graph, generate_data_with_ties, get_scores_matrix, ranking_distance, rank, generate_graph


with open("./data/season_wise_players.json", "r") as file_stream:
    hockey_data = json.load(file_stream)

merged_data = {
        "draws": [],
    "not draws": [],
}
for season, data in hockey_data.items():
    merged_data["draws"].extend(data["draws"])
    merged_data["not draws"].extend(data["not draws"])

n_players = max([np.max(merged_data["draws"]), np.max(merged_data["not draws"])]) + 1
fit = fit_graph(n_players, draws=merged_data["draws"], not_draws=merged_data["not draws"], verbose=True)
draws = fit.draws_pd()

median_scores = np.median(get_scores_matrix(draws, n_players), axis=0)
np.savetxt(f"results/playerwise_merged.txt", median_scores)

exit()

seed = 10
np.random.seed(seed)
random.seed(seed)


n_players = 10
sigma = 0.01
threshold = 0.1
scores = np.random.normal(loc=0, scale=1, size=n_players)

edges = generate_graph(n_players, 0.3, 4)
not_draws, draws = generate_data_with_ties(scores, edges, sigma, threshold)
print("wins", len(not_draws))
print("tie", len(draws))

fit = fit_graph(n_players, draws=draws, not_draws=not_draws, verbose=True)
draws = fit.draws_pd()

average_scores = np.median(get_scores_matrix(draws, n_players), axis=0)

print(ranking_distance(rank(average_scores), rank(scores)))
print(stats.spearmanr(average_scores, scores)[0])


fig, ax = plt.subplots()

inferred_mu = []
std = []
for i, score  in enumerate(scores):
    sample = draws[f"mu[{i+1}]"]
    inferred_mu.append(np.average(sample))
    std.append(np.std(sample))

ax.errorbar(scores, inferred_mu, yerr=std, marker="o", ls="none")


plt.show()
