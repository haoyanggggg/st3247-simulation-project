import numpy as np
from numba import njit

@njit
def _numba_simulate(beta, gamma, rho, N, p_edge, n_infected0, T, seed):
    np.random.seed(seed)
    
    # 1. Build initial graph (Adjacency Matrix)
    adj = np.zeros((N, N), dtype=np.bool_)
    for i in range(N):
        for j in range(i + 1, N):
            if np.random.random() < p_edge:
                adj[i, j] = True
                adj[j, i] = True
                
    # 2. Initialize health state
    # 0=S, 1=I, 2=R
    state = np.zeros(N, dtype=np.int8)
    initial_indices = np.arange(N)
    np.random.shuffle(initial_indices)
    for i in range(n_infected0):
        state[initial_indices[i]] = 1
        
    infected_fraction = np.zeros(T + 1)
    rewire_counts = np.zeros(T + 1, dtype=np.int64)
    
    # Record t=0
    inf_count = 0
    for i in range(N):
        if state[i] == 1:
            inf_count += 1
    infected_fraction[0] = inf_count / N
    
    # 3. Main Loop
    for t in range(1, T + 1):
        # Phase 1: Infection (Synchronous)
        # We need to know who was infected at the start of the step
        currently_infected = []
        for i in range(N):
            if state[i] == 1:
                currently_infected.append(i)
        
        if len(currently_infected) == 0:
            break
            
        new_infections = np.zeros(N, dtype=np.bool_)
        for i in currently_infected:
            for j in range(N):
                if adj[i, j] and state[j] == 0:
                    if np.random.random() < beta:
                        new_infections[j] = True
                        
        # Apply infections
        for j in range(N):
            if new_infections[j]:
                state[j] = 1
                
        # Phase 2: Recovery
        # Re-scan for infected (including new ones)
        infected_nodes = []
        for i in range(N):
            if state[i] == 1:
                infected_nodes.append(i)
                
        for i in infected_nodes:
            if np.random.random() < gamma:
                state[i] = 2 # Recovered
                
        # Phase 3: Rewiring
        # S-I edges
        rewire_count = 0
        if rho > 0.0:
            # Re-scan for current infected after recovery
            current_infected = []
            for i in range(N):
                if state[i] == 1:
                    current_infected.append(i)
            
            # Find all S-I edges
            # We store them to avoid modifying adj while iterating if needed, 
            # though here we can just process them.
            # To match the logic: collect all S-I edges, then decide on rewiring.
            s_nodes = []
            i_nodes = []
            for i_node in current_infected:
                for s_node in range(N):
                    if adj[i_node, s_node] and state[s_node] == 0:
                        s_nodes.append(s_node)
                        i_nodes.append(i_node)
            
            for idx in range(len(s_nodes)):
                s_node = s_nodes[idx]
                i_node = i_nodes[idx]
                
                if np.random.random() < rho:
                    # Double check edge still exists (logic from original)
                    if adj[s_node, i_node]:
                        # Remove
                        adj[s_node, i_node] = False
                        adj[i_node, s_node] = False
                        
                        # New partner
                        # Rejection sampling
                        attempts = 0
                        while attempts < 1000:
                            new_partner = np.random.randint(0, N)
                            if new_partner != s_node and not adj[s_node, new_partner]:
                                adj[s_node, new_partner] = True
                                adj[new_partner, s_node] = True
                                rewire_count += 1
                                break
                            attempts += 1
                            
        # Final counts for step t
        inf_count = 0
        for i in range(N):
            if state[i] == 1:
                inf_count += 1
        infected_fraction[t] = inf_count / N
        rewire_counts[t] = rewire_count
        
    # Final Degree Histogram
    degree_histogram = np.zeros(31, dtype=np.int64)
    for i in range(N):
        deg = 0
        for j in range(N):
            if adj[i, j]:
                deg += 1
        if deg >= 30:
            degree_histogram[30] += 1
        else:
            degree_histogram[deg] += 1
            
    return infected_fraction, rewire_counts, degree_histogram

def simulate(beta, gamma, rho, N=200, p_edge=0.05, n_infected0=5, T=200, rng=None, simulation_context=None):
    # Handle seed for Numba
    if rng is not None:
        # Get a seed from the generator
        s = rng.integers(0, 2**31)
    else:
        s = np.random.randint(0, 2**31)
        
    return _numba_simulate(beta, gamma, rho, N, p_edge, n_infected0, T, s)
