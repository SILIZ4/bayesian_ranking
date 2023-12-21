import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from scipy import stats

from modeling import (dataset_ranking_distances, fit,
                      get_scores_matrix, predict_rankings,
                      rank, generate_hyperedges, generate_data)


seed = 42
np.random.seed(seed)
number_players = 50
repetitions = 20
max_size = number_players//2
sigma = 1

scores = np.random.normal(loc=0, scale=1, size=number_players)

fig, ax = plt.subplots()

for n_games in tqdm([10, 50, 100, 500, 1000]):
    hyperedges = generate_hyperedges(number_players, n_games, max_size)

    errors = []
    for repetition in tqdm(range(repetitions), total=repetitions, leave=False):
        games = generate_data(scores, hyperedges, sigma)

        draws = fit(games, number_players).draws_pd()

        average_scores = np.average(get_scores_matrix(draws, number_players), axis=0)
        errors.append(stats.spearmanr(average_scores, scores)[0])
    lower, upper = np.percentile(errors, [25, 75])
    median = np.median(errors)
    ax.errorbar([n_games], [median], yerr=[[median-lower], [upper-median]], color="blue", marker="o")

ax.set_ylabel("average ranking L1 distance")
ax.set_xlabel("Number of games")
plt.show()
