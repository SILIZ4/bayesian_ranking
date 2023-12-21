import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from scipy import stats

from modeling import fit, get_scores_matrix, generate_data
from hypergraph import generate_uniform_hyperedges


seed = 42
np.random.seed(seed)
number_games = 100
repetitions = 20
sigma = 1


fig, ax = plt.subplots()

for n_players in tqdm([10, 20, 30, 40]):
    scores = np.random.normal(loc=0, scale=1, size=n_players)
    max_size = n_players//2
    hyperedges = generate_uniform_hyperedges(n_players, number_games, max_size)

    errors = []
    for repetition in tqdm(range(repetitions), total=repetitions, leave=False):
        games = generate_data(scores, hyperedges, sigma)

        draws = fit(games, n_players).draws_pd()


        average_scores = np.average(get_scores_matrix(draws, n_players), axis=0)
        errors.append(stats.spearmanr(average_scores, scores)[0])
    lower, upper = np.percentile(errors, [25, 75])
    median = np.median(errors)
    ax.errorbar([n_players], [median], yerr=[[median-lower], [upper-median]], color="blue", marker="o")

ax.set_ylabel("average ranking L1 distance")
ax.set_xlabel("Number of players")
plt.show()
