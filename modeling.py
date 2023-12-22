from itertools import chain, combinations
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


def generate_graph(n, p, k):
    edges = []
    for i, j in combinations(np.arange(n), 2):
        multiplicity = stats.binom.rvs(k, p)
        for _ in range(multiplicity):
            edges.append([i, j])
    return edges


def generate_data_with_ties(scores, edges, sigma, threshold):
    not_draws = []
    draws = []
    for i, j in edges:
        z = np.random.normal(loc=scores[i]-scores[j], scale=np.sqrt(2)*sigma)
        if abs(z) < threshold:
            draws.append([i, j])
        elif z > 0:
            not_draws.append([i, j])
        else:
            not_draws.append([j, i])
    return not_draws, draws


def fit_hypergraph(games_rankings, number_players, verbose=False):
    model = stan.CmdStanModel(stan_file="./stan_models/hypergraph_ranking.stan")

    flattened_games = list(map(lambda x: x+1, chain.from_iterable(games_rankings)))
    data = {
        "n_games": len(games_rankings),
        "players": flattened_games,
        "games_sizes": list(map(len, games_rankings)),
        "n_players": len(flattened_games),
        "N": number_players
    }
    return model.sample(data=data, iter_sampling=200, show_progress=verbose)


def fit_graph(n_players, not_draws, draws, verbose=False):
    data = {
        "n_players": n_players,
        "n_not_draws": len(not_draws),
        "not_draws": not_draws,
        "n_draws": len(draws),
        "draws": draws,
    }
    model = stan.CmdStanModel(stan_file="./stan_models/graph_ranking_ties.stan")
    return model.sample(data=data, iter_sampling=200, show_progress=verbose)


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
