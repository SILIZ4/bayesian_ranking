import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
import pickle
from tqdm import tqdm

from modeling import (ranking_distance, fit_hypergraph,
                      get_scores_matrix, predict_rankings,
                      rank, generate_data)
from hypergraph import generate_binomial_assortative_hypergraph, generate_uniform_hyperedges


with open("./data/hypergraph_list.pkl", "rb") as file_stream:
    hypergraph_list = pickle.load(file_stream)

scores = np.loadtxt("./data/scores.csv")

number_players = 50
k = 4
repetitions = 20

rank_correlations = []
lows, highs = [], []
for y in tqdm(hypergraph_list):
    errors = []
    for hyperedges in [y]:
        for repetition in tqdm(np.arange(repetitions), leave=False):
            h = list(map(lambda x: list(x)*5, hyperedges))
            games = generate_data(scores, h, 0.5)

            draws = fit_hypergraph(games, number_players).draws_pd()
            median_scores = np.median(get_scores_matrix(draws, number_players), axis=0)

            errors.append(stats.spearmanr(median_scores, scores)[0])
    median = np.median(errors)
    low, high = np.percentile(errors, [25, 75])
    lows.append(median-low)
    highs.append(high-median)
    rank_correlations.append(median)

assortativity_parameter = [0, 1, 2, 3]
fig, ax = plt.subplots()

ax.set_xlabel("Rank assortativity")
ax.set_ylabel("Spearman's rank correlation")
ax.errorbar(assortativity_parameter, rank_correlations, yerr=np.array([lows, highs]), marker="o")
fig.savefig("figures/assortativity_hypergraph_correlation.svg")
plt.show()
