from itertools import chain
import logging

import cmdstanpy as stan
import numpy as np


logger = logging.getLogger('cmdstanpy')
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.CRITICAL)


def generate_data(scores, hyperedges, sigma):
    results = []
    for hyperedge in hyperedges:
        effective_scores = [np.random.normal(loc=scores[j], scale=sigma) for j in hyperedge]
        ranking = np.array(hyperedge)[np.argsort(effective_scores)]
        results.append(ranking.tolist()[::-1])
    return results


def fit(games_rankings, number_players):
    model = stan.CmdStanModel(stan_file="./stan_models/hypergraph_ranking.stan")

    flattened_games = list(map(lambda x: x+1, chain.from_iterable(games_rankings)))
    data = {
        "n_games": len(games_rankings),
        "players": flattened_games,
        "games_sizes": list(map(len, games_rankings)),
        "n_players": len(flattened_games),
        "N": number_players
    }
    fit = model.sample(data=data, iter_sampling=200, show_progress=False)
    return fit


def rank_in_game(scores, game):
    game_scores = np.asarray(scores)[np.asarray(game)]
    return np.argsort(game_scores)[::-1].tolist()


def rank(scores):
    return np.argsort(scores)[::-1].tolist()


def get_scores_matrix(draws, number_players):
    scores = []
    for i in range(number_players):
        scores.append(draws[f"mu[{i+1}]"])
    return np.array(scores).T


def predict_rankings(scores, games):
    predicted_rankings = []
    for game in games:
        predicted_rankings.append(rank(scores, game))
    return predicted_rankings


def ranking_distance(ranking1, ranking2):
    return np.average(np.abs(np.asarray(ranking1)-np.asarray(ranking2)))
