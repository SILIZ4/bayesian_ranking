import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from scipy import stats

from modeling import fit, get_scores_matrix, generate_data
from hypergraph import generate_regular_hyperedges


seed = 42
np.random.seed(seed)
number_players = 20
repetitions = 20
sigma = 1

scores = np.random.normal(loc=0, scale=1, size=number_players)

fig, ax = plt.subplots()

hyperedge_size = 4
average_degrees = np.linspace(1, 10, 5)
hyperedge_sizes = [2, 5, 10, 20]
colors = ["#A4031F", "#240B36", "#DB9065", "#F2A359", "#F2DC5D"]
for hyperedge_size, color in tqdm(zip(hyperedge_sizes, colors), total=len(hyperedge_sizes)):
    median_errors = []
    for average_degree in tqdm(average_degrees, total=repetitions, leave=False):
        number_games = round(average_degree*number_players/hyperedge_size)
        hyperedges = generate_regular_hyperedges(number_players, number_games, hyperedge_size)

        errors = []
        for repetition in range(repetitions):
            games = generate_data(scores, hyperedges, sigma)

            draws = fit(games, number_players).draws_pd()

            average_scores = np.average(get_scores_matrix(draws, number_players), axis=0)
            errors.append(stats.spearmanr(average_scores, scores)[0])
        lower, upper = np.percentile(errors, [25, 75])
        median = np.median(errors)
        ax.errorbar([average_degree], [median], yerr=[[median-lower], [upper-median]],
                    color=color, marker="o")
        median_errors.append(median)
    ax.plot(average_degrees, median_errors, color=color, label=f"size={hyperedge_size}")

ax.set_ylabel("average ranking L1 distance")
ax.set_xlabel("$\\mathbb{E}[k]$")
ax.legend()
fig.savefig(f"figures/games_size_number_n={number_players}.pdf")
plt.show()
