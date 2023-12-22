import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from scipy import stats

from modeling import fit_hypergraph, get_scores_matrix, generate_data
from hypergraph import generate_uniform_hyperedges


seed = 42
np.random.seed(seed)
number_players = 10
repetitions = 20
number_games = 100
max_size = number_players//2

scores = np.random.normal(loc=0, scale=1, size=number_players)
hyperedges = generate_uniform_hyperedges(number_players, number_games, max_size)

fig, ax = plt.subplots()

sigmas = np.linspace(0.1, 10, 10)
for sigma in tqdm(sigmas):
    errors = []
    for repetition in tqdm(range(repetitions), total=repetitions, leave=False):
        games = generate_data(scores, hyperedges, sigma)

        draws = fit_hypergraph(games, number_players).draws_pd()

        average_scores = np.average(get_scores_matrix(draws, number_players), axis=0)
        errors.append(stats.spearmanr(average_scores, scores)[0])
    lower, upper = np.percentile(errors, [25, 75])
    median = np.median(errors)
    ax.errorbar([sigma], [median], yerr=[[median-lower], [upper-median]], color="blue", marker="o")

ax.set_ylabel("average ranking L1 distance")
ax.set_xlabel("$\\sigma$")
plt.show()
