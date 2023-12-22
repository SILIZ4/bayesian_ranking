import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from scipy import stats

from modeling import fit_hypergraph, get_scores_matrix, generate_data
from hypergraph import generate_regular_hyperedges


seed = 42
np.random.seed(seed)
number_players = 20
repetitions = 100
sigma = 1

scores = np.random.normal(loc=0, scale=1, size=number_players)

fig, ax = plt.subplots()

hyperedge_size = 4
average_degrees = np.arange(10) + 1
hyperedge_sizes = [2, 5, 10, 20]
colors = ["#1D2F6F", "#8390FA", "#FAC748", "#F88DAD", "#A40606"]
for hyperedge_size, color in tqdm(zip(hyperedge_sizes, colors), total=len(hyperedge_sizes)):
    median_errors = []
    for average_degree in tqdm(average_degrees, total=len(average_degrees), leave=False):
        number_games = round(average_degree*number_players/hyperedge_size)
        errors = []
        for repetition in range(repetitions):
            hyperedges = generate_regular_hyperedges(number_players, number_games, hyperedge_size)
            games = generate_data(scores, hyperedges, sigma)

            draws = fit_hypergraph(games, number_players).draws_pd()

            average_scores = np.average(get_scores_matrix(draws, number_players), axis=0)
            errors.append(stats.spearmanr(average_scores, scores)[0])
        lower, upper = np.percentile(errors, [25, 75])
        median = np.median(errors)
        ax.errorbar([average_degree], [median], yerr=[[median-lower], [upper-median]],
                    color=color, marker="o")
        median_errors.append(median)
    ax.plot(average_degrees, median_errors, color=color, label=f"Hyperedges of size {hyperedge_size}")

ax.set_ylabel("Spearman's rank correlation")
ax.set_xlabel("Expected degree $\\mathbb{E}[k]$")
ax.legend()
fig.savefig(f"figures/games_size_number_n={number_players}.pdf")
plt.show()
