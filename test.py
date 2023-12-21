import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

from modeling import (ranking_distance, fit,
                      get_scores_matrix, predict_rankings,
                      rank, generate_data)
from hypergraph import generate_binomial_assortative_hypergraph, generate_uniform_hyperedges


seed = 200
# np.random.seed(seed)
number_players = 5
number_games = 20

sigma = 0.1
scores = np.random.normal(loc=0, scale=1, size=number_players)

# hyperedges = generate_binomial_assortative_hypergraph(number_players, scores, max_game_size=5, max_game_number=4)
# print(len(hyperedges))
# weights = np.random.normal(loc=-3, scale=2, size=number_players)
# hyperedges = generate_binomial_beta_hypergraph(number_players, weights, max_game_size=5, max_game_number=4)
hyperedges = generate_uniform_hyperedges(number_players, number_games, 5)
games = generate_data(scores, hyperedges, sigma)

draws = fit(games, number_players).draws_pd()
average_scores = np.median(get_scores_matrix(draws, number_players), axis=0)

print(ranking_distance(rank(average_scores), rank(scores)))
print(stats.spearmanr(average_scores, scores))
print(stats.spearmanr(average_scores, scores)[0])


fig, ax = plt.subplots()

inferred_mu = []
std = []
for i, score  in enumerate(scores):
    sample = draws[f"mu[{i+1}]"]
    inferred_mu.append(np.average(sample))
    std.append(np.std(sample))

ax.errorbar(scores, inferred_mu, yerr=std, marker="o", ls="none")
print("sigma", np.average(draws["sigma"]))
plt.show()

exit()

# Rank whole sample
fig, ax = plt.subplots()

sample_distances = []
for sample_scores in get_scores_matrix(draws, number_players):
    predicted_rankings = predict_rankings(sample_scores, games_rankings)
    sample_distances.append(dataset_ranking_distances(predicted_rankings, games_rankings))

ax.hist(sample_distances)

plt.show()
