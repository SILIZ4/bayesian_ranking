from tqdm import tqdm
import numpy as np
import networkx as nx
from scipy import stats

from matplotlib import pyplot as plt

from modeling import fit_graph, generate_data_with_ties, get_scores_matrix, ranking_distance, rank, generate_graph


def generate_weights(num_elements, heterogeneity_parameter):

    """Normalized weights with power law distribution"""

    concentration_parameters = 1/np.arange(1,num_elements+1)**heterogeneity_parameter
    return concentration_parameters/np.sum(concentration_parameters)

def generate_synthetic_data(t = 0.2,degree_het = 1,idx=None):

    """Generate a graph based on a tunable degree heterogeneity parameter"""

    t = 0.2
    num_players=100
    num_games = 500
    sigma = 1.0

    scores = np.random.normal(0,1,size=100)

    t = 0.1
    G = nx.DiGraph()
    G.add_nodes_from(range(num_players))
    num_ties = 0
    p = generate_weights(num_players,degree_het)

    for _ in range(num_games):
        i,j = np.random.choice(range(num_players),size=2,p=p,replace=False)
        outcome = np.random.normal(scores[i]-scores[j],np.sqrt(2)*sigma)

        if -t < outcome < t:
            G.add_edge(i,j,weight=0.5)
            G.add_edge(j,i,weight=0.5)
            num_ties += 1
        elif outcome > 0:
            G.add_edge(i,j,weight=1)
        else:
            G.add_edge(j,i,weight=1)

    return G, scores


if __name__ == "__main__":
    rank_correlations = []
    percentiles = []
    degree_hets = np.arange(0,8)
    lows, highs = [], []
    for degree_het in tqdm(degree_hets):
        errors = []
        for idx in range(20):
            G, scores = generate_synthetic_data(degree_het=degree_het,idx=idx)

            draws = [[u, v] for u,v in G.edges if G.get_edge_data(u, v) == 0.5]
            not_draws = [[u, v] for u,v in G.edges if G.get_edge_data(u, v) != 0.5]

            sigma = 1
            n_players = len(G)
            fit = fit_graph(n_players, draws=draws, not_draws=not_draws)
            draws = fit.draws_pd()
            median_scores = np.median(get_scores_matrix(draws, n_players), axis=0)

            errors.append(stats.spearmanr(median_scores, scores)[0])
        median = np.median(errors)
        low, high = np.percentile(errors, [25, 75])
        lows.append(median-low)
        highs.append(high-median)
        rank_correlations.append(median)

    fig, ax = plt.subplots()

    ax.set_xlabel("Degree heterogeneity")
    ax.set_ylabel("Spearman's rank correlation")
    ax.errorbar(degree_hets, rank_correlations, yerr=np.array([lows, highs]), marker="o")
    fig.savefig("figures/heterogeneity_graph_correlation.svg")
    plt.show()
