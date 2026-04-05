import numpy as np

def simulate(beta, gamma, rho, N=200, p_edge=0.05, n_infected0=5, T=200, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    # Build Erdos-Renyi graph as dense boolean adjacency matrix
    upper = rng.random((N, N)) < p_edge
    upper = np.triu(upper, k=1)
    adj = (upper | upper.T)  # shape (N, N), symmetric bool

    state = np.zeros(N, dtype=np.int8)
    state[rng.choice(N, size=n_infected0, replace=False)] = 1

    infected_fraction = np.zeros(T + 1)
    rewire_counts = np.zeros(T + 1, dtype=np.int64)
    infected_fraction[0] = state.sum() / N

    for t in range(1, T + 1):
        S_mask = (state == 0)
        I_mask = (state == 1)

        # --- Phase 1: Infection ---
        # For each S node, find which of its neighbors are I,
        # then do independent Bernoulli(beta) per S-I edge
        # adj[S_mask] shape: (n_S, N); mask to I columns gives S-I submatrix
        si_submatrix = adj[np.ix_(np.where(S_mask)[0], np.where(I_mask)[0])]
        # For each S-I edge present, flip coin with prob beta
        transmission = (rng.random(si_submatrix.shape) < beta) & si_submatrix
        # S node gets infected if ANY of its I-neighbors transmitted
        newly_infected_local = transmission.any(axis=1)
        s_indices = np.where(S_mask)[0]
        state[s_indices[newly_infected_local]] = 1

        # --- Phase 2: Recovery ---
        I_mask2 = (state == 1)
        i_indices = np.where(I_mask2)[0]
        recover = rng.random(len(i_indices)) < gamma
        state[i_indices[recover]] = 2

        # --- Phase 3: Rewiring ---
        S_mask2 = (state == 0)
        I_mask2 = (state == 1)
        s_nodes_all = np.where(S_mask2)[0]
        i_nodes_all = np.where(I_mask2)[0]

        # Find all S-I edges
        si_sub = adj[np.ix_(s_nodes_all, i_nodes_all)]
        s_local, i_local = np.where(si_sub)
        s_nodes = s_nodes_all[s_local]
        i_nodes = i_nodes_all[i_local]

        rewire_count = 0
        # Decide which S-I edges rewire (vectorized coin flip)
        rewire_mask = rng.random(len(s_nodes)) < rho
        s_rewire = s_nodes[rewire_mask]
        i_rewire = i_nodes[rewire_mask]

        for s, i in zip(s_rewire, i_rewire):
            if not adj[s, i]:
                continue  # already removed earlier this step
            adj[s, i] = False
            adj[i, s] = False
            # All non-neighbors of s (excluding s itself)
            candidates = np.where(~adj[s])[0]
            candidates = candidates[candidates != s]
            if len(candidates):
                k = rng.choice(candidates)
                adj[s, k] = True
                adj[k, s] = True
                rewire_count += 1

        infected_fraction[t] = (state == 1).sum() / N
        rewire_counts[t] = rewire_count

    # Degree histogram
    degrees = adj.sum(axis=1).astype(int)
    degree_histogram = np.zeros(31, dtype=np.int64)
    np.add.at(degree_histogram, np.minimum(degrees, 30), 1)

    return infected_fraction, rewire_counts, degree_histogram