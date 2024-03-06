import homcloud.interface as hc 
import numpy as np

# exports
__all__ = [
    "homology_cycles",
    "persistence_scaffold",
    "persistence_centrality",
    "degree_centrality"
]

def homology_cycles(
        fc, 
        fname,
        weighted,
        noise_level,
        n_tries
    ):
    """
    Compute 1-homology cycles using persistent homology.

    Args:
        - fc (numpy.ndarray): The input functional connectivity matrix.
        - fname (str): The filename to save the graph.
        - weighted (bool): Whether to compute weighted optimal 1-cycles.
        - noise_level (float): The maximum noise level to add to the matrix.
        - n_tries (int): The maximum number of tries to compute homology cycles.

    Returns:
        - dict: A dictionary containing the computed homology cycles, survival values, and the updated pseudo-distance matrix.

    Raises:
        - RuntimeError: If failed to compute homology cycles after the maximum number of tries. 
        The cause is very likely due to the input fc matrix not being positive semidefinite.
    """

    still_trying = True
    tries = 0
    while still_trying and tries < n_tries:
        try:
            pdlist = hc.PDList.from_rips_filtration(
                np.clip(1-fc, None, 1), # pseudo-distance matrix
                maxdim=1, 
                save_graph=True,
                save_to=fname
            )
            perd = pdlist.dth_diagram(1)
            survival = perd.deaths - perd.births
            ranks = np.argsort(survival)[::-1]
            survival = survival[ranks]
            all_cycles = []
            for i in range(len(survival)):
                pair = perd.nearest_pair_to(
                    perd.births[ranks[i]], 
                    perd.deaths[ranks[i]]
                )
                optimal = pair.optimal_1_cycle(weighted=weighted)
                all_cycles.append(optimal.path_vertices)
            still_trying = False
        except(RuntimeError): # matrix is not positive semidefinite
            U = np.random.uniform(
                low=0, 
                high=noise_level, 
                size=fc.shape
            )
            noise = np.tril(U) + np.tril(U, -1).T
            fc = fc + noise
            tries += 1
    if still_trying:
        raise RuntimeError("Failed to compute homology cycles. Try increase noise_level.")
    return {
        "all_cycles": all_cycles,
        "survival": survival,
        'fc' : fc
    }


def persistence_scaffold(
        survival, 
        all_cycles,
        fc,
        fname
    ):
    """
    Calculate the persistence homological scaffold and save it to a file.

    Args:
        - survival (list): List of persistence values for each cycle.
        - all_cycles (list): List of cycles.
        - fc (numpy.ndarray): Matrix representing the functional connectivity.
        - fname (str): File name to save the persistence homological scaffold.

    Returns:
        - numpy.ndarray: The persistence homological scaffold matrix.
    """
    H = np.zeros_like(fc)
    for c, cycle in enumerate(all_cycles):
        persistence = survival[c]
        edges = zip(cycle, cycle[1:] + [cycle[0]])
        for a, b in edges:
            H[a][b] += persistence
            H[b][a] += persistence
    # save persistence homological scaffold to fname
    np.save(fname, H)
    return H


def persistence_centrality(hs_average):
    """
    Calculate the persistence centrality of a group average persistence homological scaffold.

    Args:
        - hs_average (numpy.ndarray): Group average persistence homological scaffold.

    Returns:
        - numpy.ndarray: The persistence centrality of the input scaffold.
    """
    return np.sum(hs_average, axis=-1) / np.sum(hs_average)


def degree_centrality(fc_average):
    """
    Calculate the degree centrality of a group average functional connectivity matrix.

    Args:
        - fc_average (numpy.ndarray): The group average functional connectivity matrix.

    Returns:
        - numpy.ndarray: The degree centrality values.
    """
    # remove diagonal
    np.fill_diagonal(fc_average, 0)
    return np.sum(fc_average, axis=-1) / np.sum(fc_average)