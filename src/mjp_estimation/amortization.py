from typing import Sequence
from collections import defaultdict
import numpy as np
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

def construct_amortization_groups_vectorized(states: Sequence[int], times: Sequence[float], period: float):
    """
    Group edges by (left_time, right_time) keys.
    Returns a dictionary mapping keys to (row_indices, col_indices) arrays.

    Parameters
    ----------
    states:
        Sequence of observed states
    times:
        Sequence of corresponding observation times
    period:
        Period of the MJP rate matrix

    Returns
    -------
    
    """
    if period > 0:
        multiples, left_times = np.divmod(times, period)
        right_times = times[1:] - multiples[:-1] * period
    
    elif period == 0:
        left_times = np.zeros(len(times))
        right_times = np.diff(times)
    
    left_rounded = np.round(left_times[:-1], decimals=10)
    right_rounded = np.round(right_times, decimals=10)
    
    # Get edges
    row_indices = states[:-1]
    col_indices = states[1:]
    
    key_array = np.column_stack([left_rounded, right_rounded])
    sort_idx = np.lexsort((right_rounded, left_rounded))
    sorted_keys = key_array[sort_idx]
    sorted_rows = row_indices[sort_idx]
    sorted_cols = col_indices[sort_idx]
    
    # Find group boundaries
    key_changes = np.concatenate([[True], 
                                  (sorted_keys[1:] != sorted_keys[:-1]).any(axis=1),
                                  [True]])
    group_boundaries = np.flatnonzero(key_changes)
    
    amortization_groups = {}
    for i in range(len(group_boundaries) - 1):
        start, end = group_boundaries[i], group_boundaries[i + 1]
        key = tuple(sorted_keys[start])
        amortization_groups[key] = (sorted_rows[start:end], sorted_cols[start:end])
    
    return amortization_groups

def construct_matrix_from_group(row_indices, col_indices, shape=None):
    """
    Construct a sparse matrix from pre-computed indices.
    
    Parameters
    ----------
    row_indices : ndarray
        Row indices of edges
    col_indices : ndarray  
        Column indices of edges
    shape : tuple, optional
        Shape of the matrix. If None, inferred from max indices.
    
    Returns
    -------
    A : csr_matrix
        Sparse adjacency matrix in CSR format for efficient operations
    """
    n_edges = len(row_indices)
    data = np.ones(n_edges, dtype=np.float64)
    
    if shape is None:
        shape = (row_indices.max() + 1, col_indices.max() + 1)
    
    # Create in COO format then convert to CSR for efficiency
    A = scipy.sparse.coo_array((data, (row_indices, col_indices)), shape=shape)
    return A.tocsr()

def process_all_groups(states, times, period):
    """
    Compute minimum vertex covers for each amortization group. 
    """
    # Build groups once
    groups = construct_amortization_groups_vectorized(
        states, times, period
    )
    max_state = states.max()
    shape = (max_state + 1, max_state + 1)
    
    # Process each group
    results = {}
    for key, (row_idx, col_idx) in groups.items():
        A = construct_matrix_from_group(row_idx, col_idx, shape=shape)
        forward_indices, backward_indices = minimum_vertex_cover_bipartite(A)
        results[key] = (A, forward_indices, backward_indices)
    
    return results

def minimum_vertex_cover_bipartite(A):
    """
    Compute minimum vertex cover of a bipartite graph, applying König's theorem.
    
    Parameters
    ----------
    A : csr_matrix
        Bipartite adjacency matrix of shape (n_top, n_bottom)
        where A[i, j] = 1 means an edge between U[i] and V[j].
    
    Returns
    -------
    cover_U : set
        Indices (in 0..top_nodes-1) of U in the minimum vertex cover.
    cover_V : set
        Indices (in 0..A.shape[1]-1) of V in the minimum vertex cover.
    """
    U_size, V_size = A.shape
    A_csr = A.tocsr() if not isinstance(A, csr_matrix) else A
    matching_U = maximum_bipartite_matching(A_csr, perm_type='column')
    
    # build reverse matching 
    matching_V = np.full(V_size, -1, dtype=np.int32)
    matched_idx = np.flatnonzero(matching_U != -1)
    matching_V[matching_U[matched_idx]] = matched_idx
    
    visited_U = np.zeros(U_size, dtype=np.uint8)
    visited_V = np.zeros(V_size, dtype=np.uint8)
    
    # find unmatched vertices and mark as visited
    unmatched_U = np.flatnonzero(matching_U == -1)
    visited_U[unmatched_U] = 1
    
    queue = np.empty(U_size, dtype=np.int32)
    queue[:len(unmatched_U)] = unmatched_U
    head, tail = 0, len(unmatched_U)
    
    indices = A_csr.indices
    indptr = A_csr.indptr
    
    while head < tail:
        u = queue[head]
        head += 1
        
        start, end = indptr[u], indptr[u + 1]
        neighbors_v = indices[start:end]

        for v in neighbors_v:
            if not visited_V[v]:
                visited_V[v] = 1
                matched_u = matching_V[v]
                
                if matched_u != -1 and not visited_U[matched_u]:
                    visited_U[matched_u] = 1
                    queue[tail] = matched_u
                    tail += 1
    
    cover_U = set(np.flatnonzero(visited_U == 0))
    cover_V = set(np.flatnonzero(visited_V))
    
    return cover_U, cover_V

