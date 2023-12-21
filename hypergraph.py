import numpy as np
import itertools


def generate_uniform_hyperedges(number_players, number_games, max_size):
    hyperedges = []
    for _ in range(number_games):
        n_players_in_game = np.random.randint(low=2, high=max_size)
        players_in_game = np.random.choice(number_players, size=n_players_in_game)
        hyperedges.append(players_in_game.tolist())
    return hyperedges


def generate_regular_hyperedges(number_players, number_games, size):
    hyperedges = []
    for _ in range(number_games):
        players_in_game = np.random.choice(number_players, size=size)
        hyperedges.append(players_in_game.tolist())
    return hyperedges


def generate_binomial_beta_hypergraph(number_players, weights, max_game_size, max_game_number):
    hyperedges = []
    print("Generating beta hypergraph...", end="")
    for size in range(2, max_game_size):
        for hyperedge in itertools.combinations(np.arange(number_players), size):
            total_weight = sum([weights[v] for v in hyperedge])
            p = min([1, np.exp(total_weight) / (1+np.exp(total_weight)) ])
            n_duplicates = np.random.binomial(max_game_number, p)
            for _ in range(n_duplicates):
                hyperedges.append(hyperedge)
    print("done")
    return hyperedges


def generate_binomial_assortative_hypergraph(number_players, scores, max_game_size, max_game_number):
    hyperedges = []
    print("Generating beta hypergraph...", end="")
    for size in range(2, max_game_size):
        for hyperedge in itertools.combinations(np.arange(number_players), size):
            total_weight = np.average([-np.abs(scores[comb[0]]-scores[comb[1]])
                                for comb in itertools.combinations(hyperedge, 2)])
            p = min([1, np.exp(total_weight) / (1+np.exp(total_weight)) ])
            n_duplicates = np.random.binomial(max_game_number, p)
            for _ in range(n_duplicates):
                hyperedges.append(hyperedge)
    print("done")
    return hyperedges
