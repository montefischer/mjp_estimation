from abc import abstractmethod
from typing import Sequence, Tuple
import itertools
from functools import partial
from collections import defaultdict

import numpy as np
from scipy import sparse
from scipy.optimize import LinearConstraint
from scipy.integrate import quad
from tqdm import tqdm

from .base_functionality import MJPModel, FLOAT, INT

# arbitrary hardcoded upper / lower bounds on rates
QMIN = .01
QMAX = 100

STATE_FUNCTIONAL_REGISTRY = {}
def register_state_functional(method):
    """Decorator that registers the state functional "h" or "r" in FUNCTIONAL_REGISTRY."""
    STATE_FUNCTIONAL_REGISTRY[method.__name__] =method
    return method

def encode(x: Sequence[int], d_vec: Sequence[int]) -> int:
    D = len(d_vec)
    return np.sum(np.array(x, dtype=INT) * np.concatenate([np.ones(1, dtype=INT), np.cumprod(np.array(d_vec))])[:-1])

def decode(x_num: int, d_vec: Sequence[int]) -> Sequence[int]:
    x = np.zeros(len(d_vec), dtype=INT)
    r = x_num
    for i, d in enumerate(d_vec):
        r, xi = np.divmod(r, d)
        x[i] = xi
    return x

def vec_encode(X: np.ndarray, d_vec: Sequence[int]) -> np.ndarray:
    """
    Vectorized encode.
    X: (n, k) array of n states as decoded k-tuples
    d_vec: (k,) integer array of base sizes
    returns: (n,) encoded integers
    """
    d_vec = np.asarray(d_vec, dtype=INT)
    bases = np.concatenate(([1], np.cumprod(d_vec[:-1], dtype=INT)))  # (k,)
    return (X * bases).sum(axis=1, dtype=INT)


def vec_decode(encoded: np.ndarray, d_vec: Sequence[int]) -> np.ndarray:
    """
    Vectorized decode.
    encoded: (n,) integer array
    d_vec: (k,) integer array of base sizes
    returns: (n, k) decoded states
    """
    d_vec = np.asarray(d_vec, dtype=INT)
    n = encoded.shape[0]
    k = len(d_vec)

    X = np.zeros((n, k), dtype=INT)
    r = encoded.copy()
    for i, d in enumerate(d_vec):
        r, X[:, i] = np.divmod(r, d)
    return X


def possible_transitions(x, d_vec, R, L):
    D = len(d_vec)
    pos = []
    pos_arrival = []
    pos_arrival_idx = []

    pos_service_routing = []
    pos_service_routing_from_idx = []
    pos_service_routing_to_idx = []

    pos_pure_service = []
    pos_pure_service_idx = []

    is_room = INT(x < d_vec - 1)
    for i in range(D):
        ei = np.zeros_like(x, INT)
        ei[i] = 1
        if x[i] < d_vec[i] - 1 and L[i] > 0:
            pos.append(x + ei)
            pos_arrival.append(x + ei)
            pos_arrival_idx.append(i)

        if x[i] > 0:
            if R[i].sum() < 1:
                pos.append(x - ei)
                pos_pure_service.append(x - ei)
                pos_pure_service_idx.append(i)
            for j in range(D):
                ej = np.zeros_like(x, INT)
                ej[j] = 1
                if i == j:
                    continue
                if is_room[j] and R[i,j] > 0:
                    pos.append(x - ei + ej)
                    pos_service_routing.append(x - ei + ej)
                    pos_service_routing_from_idx.append(i)
                    pos_service_routing_to_idx.append(j)
    pos_dict = {
        'initial_state': x,
        'is_room': is_room,

        'pos': pos,
        'pos_arrival': pos_arrival,
        'pos_arrival_idx': pos_arrival_idx,

        'pos_service_routing': pos_service_routing,
        'pos_service_routing_from_idx': pos_service_routing_from_idx,
        'pos_service_routing_to_idx': pos_service_routing_to_idx,

        'pos_pure_service': pos_pure_service,
        'pos_pure_service_idx': pos_pure_service_idx,
    }

    return pos_dict


class JacksonNetwork(MJPModel):
    def __init__(self, capacities: Sequence[int]):
        self.capacities = capacities
        self.d_vec = np.array(capacities, INT) + 1
        self.D = len(self.d_vec)
        self._d = np.prod(self.d_vec)

    def setup_jackson_network(self, routing_matrix, stations_with_arrivals):
        self.R = routing_matrix
        self.L = stations_with_arrivals
        self.generate_transition_structure(self.d_vec, routing_matrix, stations_with_arrivals)

    def generate_transition_structure(self, d_vec, R, L) -> None:
        D = len(d_vec)

        # Generate all (encoded) states as (n_states, D) array 
        all_states = np.array(list(itertools.product(*[range(d) for d in d_vec])), dtype=INT)
        n_states = len(all_states)
        all_encoded = vec_encode(all_states, d_vec)
        has_room = all_states < (d_vec - 1) 

        arr_from_list = []
        arr_to_list = []
        arr_idx_list = []

        route_from_list = []
        route_to_list = []
        route_idx_from_list = []
        route_idx_to_list = []
        route_num_at_from_list = []

        exit_from_list = []
        exit_to_list = []
        exit_idx_list = []
        exit_is_room_list = []
        exit_num_at_from_list = []

        # Process lists for arrivals, routing, and exits
        # arrivals
        for i in range(D):
            if L[i] == 0:
                continue  # No arrivals at this station

            can_arrive = has_room[:, i]  # (n_states,)
            state_indices = np.where(can_arrive)[0]

            if len(state_indices) == 0:
                continue

            to_states = all_states[state_indices].copy()
            to_states[:, i] += 1
            to_encoded = vec_encode(to_states, d_vec)

            arr_from_list.append(all_encoded[state_indices])
            arr_to_list.append(to_encoded)
            arr_idx_list.append(np.full(len(state_indices), i, dtype=INT))

        # routing
        for i in range(D):
            for j in range(D):
                if i == j or R[i, j] == 0:
                    continue

                can_route = (all_states[:, i] > 0) & has_room[:, j]
                state_indices = np.where(can_route)[0]

                if len(state_indices) == 0:
                    continue

                to_states = all_states[state_indices].copy()
                to_states[:, i] -= 1
                to_states[:, j] += 1
                to_encoded = vec_encode(to_states, d_vec)

                route_from_list.append(all_encoded[state_indices])
                route_to_list.append(to_encoded)
                route_idx_from_list.append(np.full(len(state_indices), i, dtype=INT))
                route_idx_to_list.append(np.full(len(state_indices), j, dtype=INT))
                route_num_at_from_list.append(all_states[state_indices, i])

        # exits
        for i in range(D):
            if R[i].sum() >= 1:
                continue  # All customers route internally

            can_exit = all_states[:, i] > 0
            state_indices = np.where(can_exit)[0]

            if len(state_indices) == 0:
                continue

            to_states = all_states[state_indices].copy()
            to_states[:, i] -= 1
            to_encoded = vec_encode(to_states, d_vec)

            exit_from_list.append(all_encoded[state_indices])
            exit_to_list.append(to_encoded)
            exit_idx_list.append(np.full(len(state_indices), i, dtype=INT))
            exit_num_at_from_list.append(all_states[state_indices, i])

            # exit_is_room needs the has_room array for each from-state
            exit_is_room_list.append(has_room[state_indices])

        # --- Concatenate and sort all transition arrays ---
        # Arrivals
        if arr_from_list:
            arr_from = np.concatenate(arr_from_list)
            arr_to = np.concatenate(arr_to_list)
            arr_idx = np.concatenate(arr_idx_list)
            # Sort by from state for efficient indexing
            sort_order = np.argsort(arr_from)
            self.arr_from = arr_from[sort_order]
            self.arr_to = arr_to[sort_order]
            self.arr_idx = arr_idx[sort_order]
        else:
            self.arr_from = np.array([], dtype=INT)
            self.arr_to = np.array([], dtype=INT)
            self.arr_idx = np.array([], dtype=INT)

        # Routing
        if route_from_list:
            route_from = np.concatenate(route_from_list)
            route_to = np.concatenate(route_to_list)
            route_idx_from = np.concatenate(route_idx_from_list)
            route_idx_to = np.concatenate(route_idx_to_list)
            route_num = np.concatenate(route_num_at_from_list)
            # Sort by from state
            sort_order = np.argsort(route_from)
            self.route_from = route_from[sort_order]
            self.route_to = route_to[sort_order]
            self.route_idx_from = route_idx_from[sort_order]
            self.route_idx_to = route_idx_to[sort_order]
            self.route_idx_number_in_system_at_from_idx = route_num[sort_order]
        else:
            self.route_from = np.array([], dtype=INT)
            self.route_to = np.array([], dtype=INT)
            self.route_idx_from = np.array([], dtype=INT)
            self.route_idx_to = np.array([], dtype=INT)
            self.route_idx_number_in_system_at_from_idx = np.array([], dtype=INT)

        # Exits
        if exit_from_list:
            exit_from = np.concatenate(exit_from_list)
            exit_to = np.concatenate(exit_to_list)
            exit_idx = np.concatenate(exit_idx_list)
            exit_num = np.concatenate(exit_num_at_from_list)
            exit_is_room = np.vstack(exit_is_room_list)
            # Sort by from state
            sort_order = np.argsort(exit_from)
            self.exit_from = exit_from[sort_order]
            self.exit_to = exit_to[sort_order]
            self.exit_idx = exit_idx[sort_order]
            self.exit_is_room = exit_is_room[sort_order]
            self.exit_idx_number_in_system = exit_num[sort_order]
        else:
            self.exit_from = np.array([], dtype=INT)
            self.exit_to = np.array([], dtype=INT)
            self.exit_idx = np.array([], dtype=INT)
            self.exit_is_room = np.array([], dtype=INT).reshape(0, D)
            self.exit_idx_number_in_system = np.array([], dtype=INT)

        # Build row_map 
        self.row_map = {}

        if len(self.arr_from) > 0:
            arr_starts = np.searchsorted(self.arr_from, np.arange(n_states), side='left')
            arr_ends = np.searchsorted(self.arr_from, np.arange(n_states), side='right')
        else:
            arr_starts = np.zeros(n_states, dtype=INT)
            arr_ends = np.zeros(n_states, dtype=INT)

        if len(self.route_from) > 0:
            route_starts = np.searchsorted(self.route_from, np.arange(n_states), side='left')
            route_ends = np.searchsorted(self.route_from, np.arange(n_states), side='right')
        else:
            route_starts = np.zeros(n_states, dtype=INT)
            route_ends = np.zeros(n_states, dtype=INT)

        if len(self.exit_from) > 0:
            exit_starts = np.searchsorted(self.exit_from, np.arange(n_states), side='left')
            exit_ends = np.searchsorted(self.exit_from, np.arange(n_states), side='right')
        else:
            exit_starts = np.zeros(n_states, dtype=INT)
            exit_ends = np.zeros(n_states, dtype=INT)

        self.xmap = {}
        for x in range(n_states):
            arr_slice = slice(arr_starts[x], arr_ends[x])
            route_slice = slice(route_starts[x], route_ends[x])
            exit_slice = slice(exit_starts[x], exit_ends[x])

            self.row_map[x] = {
                "arr": np.arange(arr_starts[x], arr_ends[x]),
                "route": np.arange(route_starts[x], route_ends[x]),
                "exit": np.arange(exit_starts[x], exit_ends[x]),
            }

            self.xmap[x] = defaultdict(set)
            for station in self.arr_idx[arr_slice]:
                self.xmap[x]['arr'].add(int(station))
            for station in self.route_idx_from[route_slice]:
                self.xmap[x]['serv'].add(int(station))
            for station in self.exit_idx[exit_slice]:
                self.xmap[x]['serv'].add(int(station))

    @property
    def state_space_size(self) -> int:
        return self._d
    
    @abstractmethod
    def get_lambda(self, theta, t, left=True): pass

    @abstractmethod
    def get_lambda_grad(self, theta, t, left=True): pass

    @abstractmethod 
    def get_service(self, theta, t, left=True): pass 

    @abstractmethod 
    def get_service_grad(self, theta, t, left=True): pass 

    @abstractmethod 
    def get_num_servers(self, t, left=True): pass 

    @abstractmethod 
    def get_R_t(self, theta, t, left=True): pass

    @abstractmethod 
    def get_R_t_grad(self, theta, t, left=True): pass


    def get_Q(self, t: float, theta: np.ndarray, left: bool = True) -> sparse.coo_array:
        """
        Construct entire (sparse) Q matrix for given time and parameter
        """
        lambda_t = self.get_lambda(theta, t, left)
        mu_t = self.get_service(theta, t, left)
        ci_t = self.get_num_servers(t, left)
        R_t = self.get_R_t(theta, t, left)

        # arrival rate
        arrival_rate = lambda_t[self.arr_idx]

        # routing rate
        server_cap = ci_t[self.route_idx_from]
        active_servers = np.minimum(server_cap, self.route_idx_number_in_system_at_from_idx)
        total_service_rate = active_servers * mu_t[self.route_idx_from]
        routing_prob = R_t[self.route_idx_from, self.route_idx_to]
        effective_routing_rate = total_service_rate * routing_prob

        # pure exit rate
        r_t = 1. - np.sum(R_t[self.exit_idx] * self.exit_is_room, axis=1)
        exit_server_cap = ci_t[self.exit_idx]
        exit_active_servers = np.minimum(exit_server_cap, self.exit_idx_number_in_system)
        exit_total_service_rate = exit_active_servers * mu_t[self.exit_idx]

        effective_exit_rate = r_t * exit_total_service_rate

        # construct diagonal
        arr_rates = np.bincount(self.arr_from, weights=arrival_rate, minlength=self._d)
        route_rates = np.bincount(self.route_from, weights=effective_routing_rate, minlength=self._d)
        pure_rates = np.bincount(self.exit_from, effective_exit_rate, minlength=self._d)

        diag_rates = -(arr_rates + route_rates + pure_rates)
        diag_coords = np.arange(self._d, dtype=INT)

        # assemble matrix
        coords = np.concatenate([
            [self.arr_from, self.arr_to], 
            [self.route_from, self.route_to], 
            [self.exit_from, self.exit_to], 
            [diag_coords, diag_coords]
            ], axis=1)
        data = np.concatenate([arrival_rate, effective_routing_rate, effective_exit_rate, diag_rates])
        Q = sparse.coo_array((data, coords))
        
        return Q

    def get_Q_grad(self, t: float, theta: np.ndarray, left: bool = True) -> Sequence[sparse.coo_array]:
        """
        Construct gradients of Q matrix with respect to parameter theta.
        """
        mu_t = self.get_service(theta, t, left) # D 
        R_t = self.get_R_t(theta, t, left) # D x D 

        lambda_t_grad = self.get_lambda_grad(theta, t, left) # D x p 
        mu_t_grad = self.get_service_grad(theta, t, left) # D x p 
        R_t_grad = self.get_R_t_grad(theta, t, left) # D x D x p

        ci_t = self.get_num_servers(t, left) # D

        server_cap = ci_t[self.route_idx_from]
        active_servers = np.minimum(server_cap, self.route_idx_number_in_system_at_from_idx)
        routing_prob = R_t[self.route_idx_from, self.route_idx_to]

        r_t = 1. - np.sum(R_t[self.exit_idx] * self.exit_is_room, axis=1)
        exit_server_cap = ci_t[self.exit_idx]
        exit_active_servers = np.minimum(exit_server_cap, self.exit_idx_number_in_system)
        exit_total_service_rate = exit_active_servers * mu_t[self.exit_idx]

        gradients = []
        for i in range(len(theta)):
            # arrival rate
            arrival_rate_grad = lambda_t_grad[self.arr_idx, i] 

            # routing rate
            total_service_rate = mu_t[self.route_idx_from]
            total_service_rate_grad = mu_t_grad[self.route_idx_from, i]
            routing_prob_grad = R_t_grad[self.route_idx_from, self.route_idx_to, i]
            effective_routing_rate_grad = active_servers * (
                total_service_rate_grad * routing_prob + total_service_rate * routing_prob_grad
            )

            # pure exit rate 
            r_t_grad = -1. * np.sum(R_t_grad[self.exit_idx, :, i] * self.exit_is_room, axis=1)
            exit_total_service_rate_grad = exit_active_servers * mu_t_grad[self.exit_idx, i]

            effective_exit_rate_grad = (
                r_t_grad * exit_total_service_rate + r_t * exit_total_service_rate_grad
            )

            # construct diagonal
            arr_rates = np.bincount(self.arr_from, weights=arrival_rate_grad, minlength=self._d)
            route_rates = np.bincount(self.route_from, weights=effective_routing_rate_grad, minlength=self._d)
            pure_rates = np.bincount(self.exit_from, effective_exit_rate_grad, minlength=self._d)

            diag_rates = -(arr_rates + route_rates + pure_rates)
            diag_coords = np.arange(self._d, dtype=INT)

            # assemble matrix
            coords = np.concatenate([
                [self.arr_from, self.arr_to], 
                [self.route_from, self.route_to], 
                [self.exit_from, self.exit_to], 
                [diag_coords, diag_coords]
                ], axis=1)
            data = np.concatenate([arrival_rate_grad, effective_routing_rate_grad, effective_exit_rate_grad, diag_rates])
            Q_grad = sparse.coo_array((data, coords)) 

            gradients.append(Q_grad)

        return gradients

    def get_Q_row(self, x: int, t, theta) -> sparse.coo_array:
        """
        Compute a single row of the generator matrix Q for state x.
        """
        lambda_t = self.get_lambda(theta, t)
        mu_t = self.get_service(theta, t)
        ci_t = self.get_num_servers(t)
        R_t = self.get_R_t(theta, t)

        row_info = self.row_map[x]
        arr_idx = row_info["arr"]
        route_idx = row_info["route"]
        exit_idx = row_info["exit"]

        arr_to = self.arr_to[arr_idx]
        arr_rate = lambda_t[self.arr_idx[arr_idx]]

        server_cap = ci_t[self.route_idx_from[route_idx]]
        active_servers = np.minimum(server_cap, self.route_idx_number_in_system_at_from_idx[route_idx])
        total_service_rate = active_servers * mu_t[self.route_idx_from[route_idx]]
        routing_prob = R_t[self.route_idx_from[route_idx], self.route_idx_to[route_idx]]
        route_to = self.route_to[route_idx]
        route_rate = total_service_rate * routing_prob

        r_t = 1. - np.sum(R_t[self.exit_idx[exit_idx]] * self.exit_is_room[exit_idx], axis=1)
        exit_server_cap = ci_t[self.exit_idx[exit_idx]]
        exit_active_servers = np.minimum(exit_server_cap, self.exit_idx_number_in_system[exit_idx])
        exit_total_service_rate = exit_active_servers * mu_t[self.exit_idx[exit_idx]]
        exit_to = self.exit_to[exit_idx]
        exit_rate = r_t * exit_total_service_rate

        # --- diagonal element ---
        diag_rate = -(np.sum(arr_rate) + np.sum(route_rate) + np.sum(exit_rate))

        # --- assemble single-row sparse matrix ---
        row_idx = np.full(len(arr_to) + len(route_to) + len(exit_to) + 1, x, dtype=INT)
        col_idx = np.concatenate([arr_to, route_to, exit_to, [x]])
        data = np.concatenate([arr_rate, route_rate, exit_rate, [diag_rate]])

        return sparse.coo_array((data, (row_idx, col_idx)), shape=(self._d, self._d))
    
    def get_Q_grad_row(self, x: int, t, theta) -> Sequence[sparse.coo_array]:
        """
        Efficiently compute a single-row gradient of Q for state x.
        Returns a list of sparse 1xd rows (one per parameter).
        """
        lambda_t = self.get_lambda(theta, t)
        mu_t = self.get_service(theta, t)
        R_t = self.get_R_t(theta, t)

        lambda_t_grad = self.get_lambda_grad(theta, t)
        mu_t_grad = self.get_service_grad(theta, t)
        R_t_grad = self.get_R_t_grad(theta, t)

        ci_t = self.get_num_servers(t)

        row_info = self.row_map[x]
        arr_idx = row_info["arr"]
        route_idx = row_info["route"]
        exit_idx = row_info["exit"]

        # --- shared components ---
        # routing
        server_cap = ci_t[self.route_idx_from[route_idx]]
        active_servers = np.minimum(server_cap, self.route_idx_number_in_system_at_from_idx[route_idx])
        routing_prob = R_t[self.route_idx_from[route_idx], self.route_idx_to[route_idx]]

        # exit
        r_t = 1.0 - np.sum(R_t[self.exit_idx[exit_idx]] * self.exit_is_room[exit_idx], axis=1)
        exit_server_cap = ci_t[self.exit_idx[exit_idx]]
        exit_active_servers = np.minimum(exit_server_cap, self.exit_idx_number_in_system[exit_idx])
        exit_total_service_rate = exit_active_servers * mu_t[self.exit_idx[exit_idx]]

        gradients = []
        n_params = len(theta)

        for i in range(n_params):
            # arrivals 
            arr_rate_grad = lambda_t_grad[self.arr_idx[arr_idx], i]

            # routing
            total_service_rate = mu_t[self.route_idx_from[route_idx]]
            total_service_rate_grad = mu_t_grad[self.route_idx_from[route_idx], i]
            routing_prob_grad = R_t_grad[self.route_idx_from[route_idx], self.route_idx_to[route_idx], i]

            route_rate_grad = active_servers * (
                total_service_rate_grad * routing_prob
                + total_service_rate * routing_prob_grad
            )

            # exits
            r_t_grad = -np.sum(
                R_t_grad[self.exit_idx[exit_idx], :, i] * self.exit_is_room[exit_idx], axis=1
            )
            exit_total_service_rate_grad = exit_active_servers * mu_t_grad[self.exit_idx[exit_idx], i]
            exit_rate_grad = r_t_grad * exit_total_service_rate + r_t * exit_total_service_rate_grad

            # diagonal
            diag_rate_grad = -(np.sum(arr_rate_grad) + np.sum(route_rate_grad) + np.sum(exit_rate_grad))

            # assemble sparse row
            row_idx = np.full(
                len(arr_idx) + len(route_idx) + len(exit_idx) + 1,
                x,
                dtype=INT,
            )
            col_idx = np.concatenate([
                self.arr_to[arr_idx],
                self.route_to[route_idx],
                self.exit_to[exit_idx],
                [x],
            ])
            data = np.concatenate([
                arr_rate_grad,
                route_rate_grad,
                exit_rate_grad,
                [diag_rate_grad],
            ])

            Q_grad_row = sparse.coo_array((data, (row_idx, col_idx)), shape=(self._d, self._d))
            gradients.append(Q_grad_row)

        return gradients

    def get_Q_row_compact(self, x: int, theta, t: float, left: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute a Q row for state x in compact sparse format.
        Avoids allocating dense arrays of size self._d.
        Returns:
            col_indices : np.ndarray of column indices where transitions exist (last entry is diagonal)
            rates       : np.ndarray of corresponding transition rates (last entry is diagonal)
        """
        # Get all instantaneous rates
        lambda_t = self.get_lambda(theta, t, left)
        mu_t = self.get_service(theta, t, left)
        R_t = self.get_R_t(theta, t, left)
        ci_t = self.get_num_servers(t, left)

        # Lookup pre-indexed transitions
        idx = self.row_map[x]
        arr_idx = idx["arr"]
        route_idx = idx["route"]
        exit_idx = idx["exit"]

        # arrivals
        arr_to = self.arr_to[arr_idx]
        arr_rate = lambda_t[self.arr_idx[arr_idx]]

        # routing
        cap = ci_t[self.route_idx_from[route_idx]]
        active = np.minimum(cap, self.route_idx_number_in_system_at_from_idx[route_idx])
        total_service_rate = active * mu_t[self.route_idx_from[route_idx]]
        routing_prob = R_t[self.route_idx_from[route_idx], self.route_idx_to[route_idx]]
        route_rate = total_service_rate * routing_prob
        route_to = self.route_to[route_idx]

        # exit
        r_t = 1.0 - np.sum(R_t[self.exit_idx[exit_idx]] * self.exit_is_room[exit_idx], axis=1)
        exit_cap = ci_t[self.exit_idx[exit_idx]]
        active_exit = np.minimum(exit_cap, self.exit_idx_number_in_system[exit_idx])
        exit_rate = r_t * active_exit * mu_t[self.exit_idx[exit_idx]]
        exit_to = self.exit_to[exit_idx]

        diag_rate = -(np.sum(arr_rate) + np.sum(route_rate) + np.sum(exit_rate))

        col_indices = np.concatenate([arr_to, route_to, exit_to, [x]])
        rates = np.concatenate([arr_rate, route_rate, exit_rate, [diag_rate]])

        return col_indices, rates

    def get_Q_grad_row_compact(self, x: int, theta: np.ndarray, t: float, left: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute gradient rows for state x in compact sparse format.
        Avoids allocating dense arrays of size self._d.
        Returns:
            col_indices    : np.ndarray of shape (n_transitions,) - column indices where transitions exist (final entry is the diagonal)
            gradient_rates : np.ndarray of shape (n_transitions, n_params) - gradient values
        """
        # Get instantaneous quantities
        mu_t = self.get_service(theta, t, left)
        R_t = self.get_R_t(theta, t, left)
        ci_t = self.get_num_servers(t, left)

        lambda_t_grad = self.get_lambda_grad(theta, t, left)
        mu_t_grad = self.get_service_grad(theta, t, left)
        R_t_grad = self.get_R_t_grad(theta, t, left)

        # Transition subsets
        idx = self.row_map[x]
        arr_idx = idx["arr"]
        route_idx = idx["route"]
        exit_idx = idx["exit"]

        # Shared components (computed once for all parameters)
        cap = ci_t[self.route_idx_from[route_idx]]
        active = np.minimum(cap, self.route_idx_number_in_system_at_from_idx[route_idx])
        routing_prob = R_t[self.route_idx_from[route_idx], self.route_idx_to[route_idx]]

        r_t = 1.0 - np.sum(R_t[self.exit_idx[exit_idx]] * self.exit_is_room[exit_idx], axis=1)
        exit_cap = ci_t[self.exit_idx[exit_idx]]
        active_exit = np.minimum(exit_cap, self.exit_idx_number_in_system[exit_idx])
        exit_total_service_rate = active_exit * mu_t[self.exit_idx[exit_idx]]

        # --- Column indices (same for all parameters) ---
        arr_to = self.arr_to[arr_idx]
        route_to = self.route_to[route_idx]
        exit_to = self.exit_to[exit_idx]
        col_indices = np.concatenate([arr_to, route_to, exit_to, [x]])

        n_params = mu_t_grad.shape[1]
        n_transitions = len(col_indices)
        gradient_rates = np.zeros((n_transitions, n_params), dtype=float)

        # Precompute slice sizes (constant across parameters)
        n_arr = len(arr_to)
        n_route = len(route_to)
        n_exit = len(exit_to)

        for i in range(n_params):
            # arrivals
            arr_rate_grad = lambda_t_grad[self.arr_idx[arr_idx], i]

            # routing
            total_service_rate = mu_t[self.route_idx_from[route_idx]]
            total_service_rate_grad = mu_t_grad[self.route_idx_from[route_idx], i]
            routing_prob_grad = R_t_grad[self.route_idx_from[route_idx], self.route_idx_to[route_idx], i]
            route_rate_grad = active * (
                total_service_rate_grad * routing_prob + total_service_rate * routing_prob_grad
            )

            # exits
            r_t_grad = -np.sum(
                R_t_grad[self.exit_idx[exit_idx], :, i] * self.exit_is_room[exit_idx], axis=1
            )
            exit_total_service_rate_grad = active_exit * mu_t_grad[self.exit_idx[exit_idx], i]
            exit_rate_grad = r_t_grad * exit_total_service_rate + r_t * exit_total_service_rate_grad


            # fill off-diagonals
            gradient_rates[:n_arr, i] = arr_rate_grad
            gradient_rates[n_arr:n_arr+n_route, i] = route_rate_grad
            gradient_rates[n_arr+n_route:n_arr+n_route+n_exit, i] = exit_rate_grad

            # diagonals
            gradient_rates[-1, i] = -np.sum(gradient_rates[:-1, i])

        return col_indices, gradient_rates

    def get_exit_rates(self, t, left: bool = True, theta = None):
        """
        Sum over exit rates
        """
        lambda_t = self.get_lambda(theta, t, left)
        mu_t = self.get_service(theta, t, left) # D dim
        ci_t = self.get_num_servers(t, left) # D dim
        R_t = self.get_R_t(theta, t, left) # D x D dim

        # arrival rate
        arrival_rate = lambda_t[self.arr_idx]

        # routing rate
        server_cap = ci_t[self.route_idx_from]
        active_servers = np.minimum(server_cap, self.route_idx_number_in_system_at_from_idx)
        total_service_rate = active_servers * mu_t[self.route_idx_from]
        routing_prob = R_t[self.route_idx_from, self.route_idx_to]
        effective_routing_rate = total_service_rate * routing_prob

        # pure exit rate
        r_t = 1. - np.sum(R_t[self.exit_idx] * self.exit_is_room, axis=1)

        exit_server_cap = ci_t[self.exit_idx]
        exit_active_servers = np.minimum(exit_server_cap, self.exit_idx_number_in_system)
        exit_total_service_rate = exit_active_servers * mu_t[self.exit_idx]

        effective_exit_rate = r_t * exit_total_service_rate

        row_sums = np.bincount(
            np.concatenate([self.arr_from, self.route_from, self.exit_from]), 
            weights=np.concatenate([arrival_rate, effective_routing_rate, effective_exit_rate]),
            minlength=self._d)

        return row_sums

    @abstractmethod
    def get_lambda_time_anti(self, theta, t, left=True): pass  

    @abstractmethod
    def get_service_time_anti(self, theta, t, left=True): pass

    @abstractmethod
    def get_lambda_grad_time_anti(self, theta, t, left=True): pass  

    @abstractmethod
    def get_service_grad_time_anti(self, theta, t, left=True): pass

    # TODO: Account for discontinuities -- the number of servers may change at a fixed time t between t0 and t1
    def integrate_exit_rate_at_state(self, theta: np.ndarray, t0: float, t1: float, x: int):
        ci_t = self.get_num_servers(t0) # array of dimension equal to the number of stations
        if not np.all(ci_t == self.get_num_servers(t1)):
            raise NotImplementedError("integrate_exit_rate_at_state does not account for time-varying number of servers")
        x_dec = decode(x, self.d_vec)
        active_servers = np.minimum(ci_t, x_dec)

        arridx = list(self.xmap[x]['arr'])
        servidx = list(self.xmap[x]['serv'])

        arrivals = self.get_lambda_time_anti(theta, t1) - self.get_lambda_time_anti(theta, t0)
        per_server = self.get_service_time_anti(theta, t1) - self.get_service_time_anti(theta, t0)

        service = active_servers * per_server

        return np.sum(arrivals[arridx], axis=0) + np.sum(service[servidx], axis=0) # never forget to sum axis=0

    # TODO: Account for discontinuities -- the number of servers may change at a fixed time t between t0 and t1
    def integrate_grad_exit_rate_at_state(self, theta: np.ndarray, t0: float, t1: float, x: int):
        ci_t = self.get_num_servers(t0) # array of dimension equal to the number of stations
        if not np.all(ci_t == self.get_num_servers(t1)):
            raise NotImplementedError("integrate_exit_rate_at_state does not account for time-varying number of servers")
        x_dec = decode(x, self.d_vec)
        active_servers = np.minimum(ci_t, x_dec)


        arridx = list(self.xmap[x]['arr'])
        servidx = list(self.xmap[x]['serv'])

        arrivals_grad = self.get_lambda_grad_time_anti(theta, t1) - self.get_lambda_grad_time_anti(theta, t0)
        service_grad = self.get_service_grad_time_anti(theta, t1) - self.get_service_grad_time_anti(theta, t0)

        # broadcast multiply gradients by active servers (which do not depend on theta)
        service_grad = active_servers.reshape(-1, 1) * service_grad

        return np.sum(arrivals_grad[arridx], axis=0) + np.sum(service_grad[servidx], axis=0) # never forget to sum axis=0

    def integrate_exit_rate_and_grad_at_state(self, theta: np.ndarray, t0: float, t1: float, x: int):
        """
        Compute both the integral of exit rate and its gradient simultaneously.
        This is more efficient than calling integrate_exit_rate_at_state and
        integrate_grad_exit_rate_at_state separately.

        Returns:
            value: scalar - integrated exit rate
            grad: np.ndarray - gradient of integrated exit rate
        """
        ci_t = self.get_num_servers(t0)
        if not np.all(ci_t == self.get_num_servers(t1)):
            raise NotImplementedError("integrate_exit_rate_at_state does not account for time-varying number of servers")
        x_dec = decode(x, self.d_vec)
        active_servers = np.minimum(ci_t, x_dec)

        arridx = list(self.xmap[x]['arr'])
        servidx = list(self.xmap[x]['serv'])

        # Compute value
        arrivals = self.get_lambda_time_anti(theta, t1) - self.get_lambda_time_anti(theta, t0)
        per_server = self.get_service_time_anti(theta, t1) - self.get_service_time_anti(theta, t0)
        service = active_servers * per_server
        value = np.sum(arrivals[arridx], axis=0) + np.sum(service[servidx], axis=0)

        # Compute gradient
        arrivals_grad = self.get_lambda_grad_time_anti(theta, t1) - self.get_lambda_grad_time_anti(theta, t0)
        service_grad = self.get_service_grad_time_anti(theta, t1) - self.get_service_grad_time_anti(theta, t0)
        service_grad = active_servers.reshape(-1, 1) * service_grad
        grad = np.sum(arrivals_grad[arridx], axis=0) + np.sum(service_grad[servidx], axis=0)

        return value, grad

    def get_external_event_intensities(self, t: float, left: bool = True) -> np.ndarray:
        beta = np.zeros(self._d)
        return beta

    def get_initial_dist(self):
        return self.mu

    def get_initial_state(self) -> int: 
        return np.random.choice(self._d, p=self.get_initial_dist())
    

class TandemQueue(JacksonNetwork): 
    def __init__(self, capacities: Sequence[int], feedback_p: float = 0.):
        super().__init__(capacities)
        routing_matrix = np.diag(np.ones(self.D-1), k=1)
        if feedback_p > 0:
            routing_matrix[self.D-1, 0] = feedback_p
        self.feedback_p = feedback_p
        stations_with_arrivals = np.zeros(shape=self.D, dtype=INT)
        stations_with_arrivals[0] = 1

        self.setup_jackson_network(routing_matrix, stations_with_arrivals)


class TandemQueuePeriodicArrivalsConstantService(TandemQueue):
    def __init__(
            self, 
            capacities: Sequence[int], 
            num_servers: Sequence[int], 
            period: float, 
            arrival_base_rate: float, 
            arrival_amplitude: float, 
            service_base_rates: Sequence[float],
            initial_dist=None,
            feedback_p: float = 0.
    ):
        super().__init__(capacities, feedback_p=feedback_p)

        if not len(num_servers) == self.D:
            raise ValueError(f"Dimension of num_servers must match D={self.D}, but has dimension {len(num_servers)}")
        if not len(service_base_rates) == self.D:
            raise ValueError(f"Dimension of service_base_rates must match D={self.D}, but has dimension {len(service_base_rates)}")

        self.arrival_base_rate = arrival_base_rate
        self.arrival_amplitude = arrival_amplitude
        self.service_base_rates = service_base_rates

        self.theta_true = np.concatenate([[arrival_base_rate, arrival_amplitude], service_base_rates], dtype=FLOAT)

        self.num_servers = np.array(num_servers)
        self.p = len(self.theta_true)
        
        self.period = float(period)

        self.num_lambda_params = 2

        self.slice_arrival = slice(0,self.num_lambda_params)
        self.slice_service = slice(self.num_lambda_params, None)

        A_arr = np.array([[1, 1], [1, -1]])
        A_ser = np.eye(self.D)
        Z1 = np.zeros((2, self.D))
        Z2 = np.zeros((self.D, 2))

        self.constraints = [
            LinearConstraint(
                A=np.block([[A_arr, Z1], [Z2, A_ser]]), 
                lb=np.full(shape=self.p, fill_value=QMIN), 
                ub=np.full(shape=self.p, fill_value=QMAX)
            )]

        # initialize to empty distribution
        if initial_dist is None:
            dist = np.zeros(self._d, dtype=FLOAT)
            dist[0] = 1.
            self.mu = dist
        else:
            self.mu = initial_dist

        self.sanity_check(self.theta_true)

    # system input rate to tandem queue (input only at first station)
    def get_input_rate(self, theta, t):
        return theta[self.slice_arrival][0] + theta[self.slice_arrival][1] * np.cos(2*np.pi/self.period * t)
    
    def get_input_rate_time_anti(self, theta, t):
        return t * theta[self.slice_arrival][0] + self.period / (2 * np.pi) * np.sin(2*np.pi/self.period * t) * theta[self.slice_arrival][1]

    def get_input_rate_grad(self, theta, t):
        return np.concatenate([np.array([1., np.cos(2*np.pi/self.period * t)]), np.zeros(shape=self.D)])
    
    def get_input_rate_grad_time_anti(self, theta, t):
        return np.concatenate([np.array([t, self.period / (2 * np.pi) * np.sin(2*np.pi/self.period * t)]), np.zeros(shape=self.D)])

    # format input rate for lambda representation (requires rates at every station)
    def get_lambda(self, theta, t, left=True):
        in_rate = self.get_input_rate(theta, t)
        return np.concatenate([[in_rate], np.zeros(shape=self.D-1)])
    
    def get_lambda_time_anti(self, theta, t, left=True):
        in_rate_time_anti = self.get_input_rate_time_anti(theta, t)
        return np.concatenate([[in_rate_time_anti], np.zeros(shape=self.D-1)])

    def get_lambda_grad(self, theta, t, left=True):
        in_rate_grad = self.get_input_rate_grad(theta, t) # p dim
        return np.concatenate([[in_rate_grad], np.zeros(shape=(self.D-1, self.p))])
    
    def get_lambda_grad_time_anti(self, theta, t, left=True):
        in_rate_grad_time_anti = self.get_input_rate_grad_time_anti(theta, t)
        return np.concatenate([[in_rate_grad_time_anti], np.zeros(shape=(self.D-1, self.p))])

    def get_service(self, theta, t, left=True) -> np.ndarray:
        """
        Returns:
            Per-server service times. These must be multiplied by the number of active servers to yield the correct rate.
        """
        return theta[self.slice_service]
    
    def get_service_time_anti(self, theta, t, left=True) -> np.ndarray:
        """
        Returns:
            Per-server service times. These must be multiplied by the number of active servers to yield the correct rate.
        """
        return t * self.get_service(theta, t, left=True) # note: we can do this because in this model, service rates are constant

    def get_service_grad(self, theta, t, left=True) -> np.ndarray:
        """
        Returns:
            Per-server service times. These must be multiplied by the number of active servers to yield the correct rate.
        """
        return np.concatenate([np.zeros(shape=(self.D, self.num_lambda_params)), np.eye(self.D)], axis=1)
    
    def get_service_grad_time_anti(self, theta, t, left=True) -> np.ndarray:
        """
        Returns:
            Per-server service times. These must be multiplied by the number of active servers to yield the correct rate.
        """
        return np.concatenate([np.zeros(shape=(self.D, self.num_lambda_params)), t * np.eye(self.D)], axis=1)

    def get_num_servers(self, t, left=True):
        return self.num_servers
    
    def get_R_t(self, theta, t, left=True):
        return self.R

    def get_R_t_grad(self, theta, t, left=True):
        return np.zeros(shape=(self.D, self.D, self.p), dtype=np.float64) # hardcoded: transition probs are not parameterized

    def get_max_total_rate(self, theta):
        if len(theta) != len(self.theta_true):
            raise ValueError("Dimension mismatch between theta and theta_true")
        exogenous_arrival_max = np.sum(np.abs(theta[self.slice_arrival]))
        service_max = np.sum(np.abs(theta[self.slice_service] * self.num_servers))
        return exogenous_arrival_max + service_max

    def compute_state_dependent_rate_bound(self, theta) -> np.ndarray:
        if len(theta) != len(self.theta_true):
            raise ValueError("Dimension mismatch between theta and theta_true")
        # exogenous arrivals occur at state-independent rate
        exogenous_arrival_max = np.sum(np.abs(theta[self.slice_arrival]))
        # services are state dependent
        states_decoded = vec_decode(np.arange(self._d), self.d_vec) # _d x d
        service_upper_bound = np.minimum(self.num_servers, states_decoded) @ theta[self.slice_service]
        return exogenous_arrival_max + service_upper_bound

    def get_discontinuity_times(self, t0, t1):
        return super().get_discontinuity_times(t0, t1)

    def save(self, filepath):
        np.savez(
            filepath, 
            class_name=self.__class__.__name__,
            capacities=self.capacities,
            num_servers=self.num_servers,
            period=self.period,
            arrival_base_rate=self.arrival_base_rate,
            arrival_amplitude=self.arrival_amplitude,
            service_base_rates=self.service_base_rates,
            initial_dist=self.mu,
            feedback_p=self.feedback_p
        )
        return filepath
    
    @classmethod
    def load(cls, filepath):
        data = np.load(filepath, allow_pickle=True)

        return cls(
            capacities=data["capacities"],
            num_servers=data["num_servers"],
            period=float(data["period"]),
            arrival_base_rate=float(data["arrival_base_rate"]),
            arrival_amplitude=data["arrival_amplitude"],
            service_base_rates=data["service_base_rates"],
            initial_dist=data["initial_dist"],
            feedback_p=float(data["feedback_p"])
        )
    

class TandemQueuePeriodicArrivalsConstantFixedService(TandemQueue):
    def __init__(
            self, 
            capacities: Sequence[int], 
            num_servers: Sequence[int], 
            period: float, 
            arrival_base_rate: float, 
            arrival_amplitude: float, 
            service_base_rates: Sequence[float],
            initial_dist=None,
            feedback_p: float = 0.
        ):
        super().__init__(capacities, feedback_p=feedback_p)

        if not len(num_servers) == self.D:
            raise ValueError(f"Dimension of num_servers must match D={self.D}, but has dimension {len(num_servers)}")
        if not len(service_base_rates) == self.D:
            raise ValueError(f"Dimension of service_base_rates must match D={self.D}, but has dimension {len(service_base_rates)}")

        self.arrival_base_rate = arrival_base_rate
        self.arrival_amplitude = arrival_amplitude
        self.service_base_rates = np.array(service_base_rates)

        self.theta_true = np.array([arrival_base_rate, arrival_amplitude], dtype=FLOAT)

        self.num_servers = np.array(num_servers)
        self.p = len(self.theta_true)
        
        self.period = float(period)

        self.num_lambda_params = 2

        self.slice_arrival = slice(0,self.num_lambda_params)
        self.slice_service = slice(self.num_lambda_params, None)

        A_arr = np.array([[1, 1], [1, -1]])
 
        self.constraints = [
            LinearConstraint(
                A=A_arr,
                lb=np.full(shape=self.p, fill_value=QMIN), 
                ub=np.full(shape=self.p, fill_value=QMAX) #np.max(self.theta_true) + 20)
            )]

        # initialize to empty distribution
        if initial_dist is None:
            dist = np.zeros(self._d, dtype=FLOAT)
            dist[0] = 1.
            self.mu = dist
        else:
            self.mu = initial_dist

        self.discontinuity_times = []

        self.sanity_check(self.theta_true)

    # system input rate to tandem queue (input only at first station)
    def get_input_rate(self, theta, t):
        return theta[self.slice_arrival][0] + theta[self.slice_arrival][1] * np.cos(2*np.pi/self.period * t)
    
    def get_input_rate_time_anti(self, theta, t):
        return t * theta[self.slice_arrival][0] + self.period / (2 * np.pi) * np.sin(2*np.pi/self.period * t) * theta[self.slice_arrival][1]

    def get_input_rate_grad(self, theta, t):
        return np.array([1., np.cos(2*np.pi/self.period * t)])
    
    def get_input_rate_grad_time_anti(self, theta, t):
        return np.array([t, self.period / (2 * np.pi) * np.sin(2*np.pi/self.period * t)])

    # format input rate for lambda representation (requires rates at every station)
    def get_lambda(self, theta, t, left=True):
        in_rate = self.get_input_rate(theta, t)
        return np.concatenate([[in_rate], np.zeros(shape=self.D-1)])
    
    def get_lambda_time_anti(self, theta, t, left=True):
        in_rate_time_anti = self.get_input_rate_time_anti(theta, t)
        return np.concatenate([[in_rate_time_anti], np.zeros(shape=self.D-1)])

    def get_lambda_grad(self, theta, t, left=True):
        in_rate_grad = self.get_input_rate_grad(theta, t) # p dim
        return np.concatenate([[in_rate_grad], np.zeros(shape=(self.D-1, self.p))])
    
    def get_lambda_grad_time_anti(self, theta, t, left=True):
        in_rate_grad_time_anti = self.get_input_rate_grad_time_anti(theta, t)
        return np.concatenate([[in_rate_grad_time_anti], np.zeros(shape=(self.D-1, self.p))])

    def get_service(self, theta, t, left=True):
        return self.service_base_rates
    
    def get_service_time_anti(self, theta, t, left=True):
        return t * self.service_base_rates # note: we can do this because in this model, service rates are constant

    def get_service_grad(self, theta, t, left=True):
        return np.zeros(shape=(self.D, self.p))
    
    def get_service_grad_time_anti(self, theta, t, left=True):
        return np.zeros(shape=(self.D, self.p))
    
    def get_num_servers(self, t, left=True):
        return self.num_servers
    
    def get_R_t(self, theta, t, left=True):
        return self.R

    def get_R_t_grad(self, theta, t, left=True):
        return np.zeros(shape=(self.D, self.D, self.p), dtype=np.float64) # hardcoded: transition probs are not parameterized

    def get_max_total_rate(self, theta):
        return np.sum(np.abs(theta)) + np.sum(self.service_base_rates * self.num_servers) + self.feedback_p * np.max(self.service_base_rates)

    def compute_state_dependent_rate_bound(self, theta) -> np.ndarray:
        if len(theta) != len(self.theta_true):
            raise ValueError("Dimension mismatch between theta and theta_true")
        # exogenous arrivals occur at state-independent rate
        exogenous_arrival_max = np.sum(np.abs(theta[self.slice_arrival]))
        # services are state dependent
        states_decoded = vec_decode(np.arange(self._d), self.d_vec) # _d x d
        service_upper_bound = np.minimum(self.num_servers, states_decoded) @ self.service_base_rates
        return exogenous_arrival_max + service_upper_bound

    def get_discontinuity_times(self, t0, t1):
        return super().get_discontinuity_times(t0, t1)

    def save(self, filepath):
        np.savez(
            filepath, 
            class_name=self.__class__.__name__,
            capacities=self.capacities,
            num_servers=self.num_servers,
            period=self.period,
            arrival_base_rate=self.arrival_base_rate,
            arrival_amplitude=self.arrival_amplitude,
            service_base_rates=self.service_base_rates,
            initial_dist=self.mu,
            feedback_p=self.feedback_p
        )
        return filepath
    
    @classmethod
    def load(cls, filepath):
        data = np.load(filepath, allow_pickle=True)

        return cls(
            capacities=data["capacities"],
            num_servers=data["num_servers"],
            period=float(data["period"]),
            arrival_base_rate=float(data["arrival_base_rate"]),
            arrival_amplitude=data["arrival_amplitude"],
            service_base_rates=data["service_base_rates"],
            initial_dist=data["initial_dist"],
            feedback_p=float(data["feedback_p"])
        )


class TandemQueueSecularTrendArrivalsConstantFixedService(TandemQueue):
    """
    Tandem queue model with trend of the form theta_0 + theta_1 * t
    """
    def __init__(
            self, 
            capacities: Sequence[int], 
            num_servers: Sequence[int], 
            arrival_base_rate: float, 
            arrival_trend: float, 
            service_base_rates: Sequence[float],
            initial_dist=None,
            feedback_p: float = 0.,
            TMAX: float = 1000.
        ):
        super().__init__(capacities, feedback_p=feedback_p)

        self.TMAX=TMAX

        if not len(num_servers) == self.D:
            raise ValueError(f"Dimension of num_servers must match D={self.D}, but has dimension {len(num_servers)}")
        if not len(service_base_rates) == self.D:
            raise ValueError(f"Dimension of service_base_rates must match D={self.D}, but has dimension {len(service_base_rates)}")

        self.arrival_base_rate = arrival_base_rate
        self.arrival_trend = arrival_trend
        self.service_base_rates = np.array(service_base_rates)

        self.theta_true = np.array([arrival_base_rate, arrival_trend], dtype=FLOAT)

        self.num_servers = np.array(num_servers)
        self.p = 2

        self.num_lambda_params = 2

        self.slice_arrival = slice(0,self.num_lambda_params)
        self.slice_service = slice(self.num_lambda_params, None)

        A_arr = np.array([[1., TMAX], [1., 0.]])
 
        self.constraints = [
            LinearConstraint(
                A=A_arr,
                lb=np.full(shape=1, fill_value=QMIN), 
                ub=np.full(shape=1, fill_value=QMAX)
            )]

        # initialize to empty distribution
        if initial_dist is None:
            dist = np.zeros(self._d, dtype=FLOAT)
            dist[0] = 1.
            self.mu = dist
        else:
            self.mu = initial_dist

        self.discontinuity_times = []

        self.sanity_check(self.theta_true)

    # system input rate to tandem queue (input only at first station)
    def get_input_rate(self, theta, t):
        return theta[self.slice_arrival][0] + theta[self.slice_arrival][1] * t
    
    def get_input_rate_time_anti(self, theta, t):
        return t * theta[self.slice_arrival][0] + (t**2 / 2) * theta[self.slice_arrival][1]

    def get_input_rate_grad(self, theta, t):
        return np.array([1., t], dtype=FLOAT)
    
    def get_input_rate_grad_time_anti(self, theta, t):
        return np.array([t, t**2 / 2], dtype=FLOAT)

    # format input rate for lambda representation (requires rates at every station)
    def get_lambda(self, theta, t, left=True):
        in_rate = self.get_input_rate(theta, t)
        return np.concatenate([[in_rate], np.zeros(shape=self.D-1)])
    
    def get_lambda_time_anti(self, theta, t, left=True):
        in_rate_time_anti = self.get_input_rate_time_anti(theta, t)
        return np.concatenate([[in_rate_time_anti], np.zeros(shape=self.D-1)])

    def get_lambda_grad(self, theta, t, left=True):
        in_rate_grad = self.get_input_rate_grad(theta, t) # p dim
        return np.concatenate([[in_rate_grad], np.zeros(shape=(self.D-1, self.p))])
    
    def get_lambda_grad_time_anti(self, theta, t, left=True):
        in_rate_grad_time_anti = self.get_input_rate_grad_time_anti(theta, t)
        return np.concatenate([[in_rate_grad_time_anti], np.zeros(shape=(self.D-1, self.p))])

    def get_service(self, theta, t, left=True):
        return self.service_base_rates
    
    def get_service_time_anti(self, theta, t, left=True):
        return t * self.service_base_rates # note: in this model, service rates are constant

    def get_service_grad(self, theta, t, left=True):
        return np.zeros(shape=(self.D, self.p))
    
    def get_service_grad_time_anti(self, theta, t, left=True):
        return np.zeros(shape=(self.D, self.p))
    
    def get_num_servers(self, t, left=True):
        return self.num_servers
    
    def get_R_t(self, theta, t, left=True):
        return self.R

    def get_R_t_grad(self, theta, t, left=True):
        return np.zeros(shape=(self.D, self.D, self.p), dtype=np.float64) #hardcoded: transition probs are not parameterized

    def get_max_total_rate(self, theta):
        """
        Hardcoded: maximum time t = self.TMAX
        """
        return theta[0] + np.max([0, theta[1] * self.TMAX]) + np.sum(self.service_base_rates * self.num_servers) + self.feedback_p * np.max(self.service_base_rates)

    def compute_state_dependent_rate_bound(self, theta) -> np.ndarray:
        if len(theta) != len(self.theta_true):
            raise ValueError("Dimension mismatch between theta and theta_true")
        # exogenous arrivals occur at state-independent rate
        exogenous_arrival_max = theta[0] + np.max([0, theta[1] * self.TMAX])
        # services are state dependent
        states_decoded = vec_decode(np.arange(self._d), self.d_vec) # _d x d
        service_upper_bound = np.minimum(self.num_servers, states_decoded) @ self.service_base_rates
        return exogenous_arrival_max + service_upper_bound

    def get_discontinuity_times(self, t0, t1):
        return super().get_discontinuity_times(t0, t1)
    
    def save(self, filepath):
        np.savez(
            filepath, 
            class_name=self.__class__.__name__,
            capacities=self.capacities,
            num_servers=self.num_servers,
            arrival_base_rate=self.arrival_base_rate,
            arrival_trend=self.arrival_trend,
            service_base_rates=self.service_base_rates,
            initial_dist=self.mu,
            feedback_p=self.feedback_p
        )
        return filepath

    @classmethod
    def load(cls, filepath):
        data = np.load(filepath, allow_pickle=True)

        return cls(
            capacities=data["capacities"],
            num_servers=data["num_servers"],
            arrival_base_rate=float(data["arrival_base_rate"]),
            arrival_trend=data["arrival_trend"],
            service_base_rates=data["service_base_rates"],
            initial_dist=data["initial_dist"],
            feedback_p=float(data["feedback_p"])
        )



class TandemQueueSecularTrendFixedBaselineArrivalsConstantFixedService(TandemQueue):
    """
    Tandem queue model with trend of the form theta_0_fixed + theta[0] * t
    where theta_0_fixed is a user-specified constant baseline arrival rate
    and theta[0] is the only parameter (the trend slope).
    Service rates are constant and fixed.
    """
    def __init__(
            self,
            capacities: Sequence[int],
            num_servers: Sequence[int],
            arrival_base_rate: float,
            arrival_trend: float,
            service_base_rates: Sequence[float],
            initial_dist=None,
            feedback_p: float = 0.,
            TMAX: float = 1000.
        ):
        super().__init__(capacities, feedback_p=feedback_p)

        self.TMAX=TMAX

        if not len(num_servers) == self.D:
            raise ValueError(f"Dimension of num_servers must match D={self.D}, but has dimension {len(num_servers)}")
        if not len(service_base_rates) == self.D:
            raise ValueError(f"Dimension of service_base_rates must match D={self.D}, but has dimension {len(service_base_rates)}")

        self.arrival_base_rate = arrival_base_rate  # fixed constant, not a parameter
        self.arrival_trend = arrival_trend
        self.service_base_rates = np.array(service_base_rates)

        self.theta_true = np.array([arrival_trend], dtype=FLOAT)  # only trend is a parameter

        self.num_servers = np.array(num_servers)
        self.p = 1  # only one parameter (the trend)

        self.num_lambda_params = 1

        self.slice_arrival = slice(0, self.num_lambda_params)
        self.slice_service = slice(self.num_lambda_params, None)

        # Constraint: arrival_base_rate + theta[0] * t >= QMIN for all t in [0, TMAX]
        # This means: arrival_base_rate + theta[0] * 0 >= QMIN and arrival_base_rate + theta[0] * TMAX >= QMIN
        # Simplifies to: arrival_base_rate >= QMIN and arrival_base_rate + theta[0] * TMAX >= QMIN
        # The second constraint is: theta[0] >= (QMIN - arrival_base_rate) / TMAX
        # Also need: arrival_base_rate + theta[0] * TMAX <= QMAX
        # This gives: theta[0] <= (QMAX - arrival_base_rate) / TMAX

        A_arr = np.array([[TMAX], [0.]])  # Evaluate at t=TMAX and t=0

        self.constraints = [
            LinearConstraint(
                A=A_arr,
                lb=np.array([QMIN - arrival_base_rate, QMIN - arrival_base_rate]),
                ub=np.array([QMAX - arrival_base_rate, QMAX - arrival_base_rate])
            )]

        # initialize to empty distribution
        if initial_dist is None:
            dist = np.zeros(self._d, dtype=FLOAT)
            dist[0] = 1.
            self.mu = dist
        else:
            self.mu = initial_dist

        self.discontinuity_times = []

        self.sanity_check(self.theta_true)

    # system input rate to tandem queue (input only at first station)
    def get_input_rate(self, theta, t):
        return self.arrival_base_rate + theta[0] * t

    def get_input_rate_time_anti(self, theta, t):
        return t * self.arrival_base_rate + (t**2 / 2) * theta[0]

    def get_input_rate_grad(self, theta, t):
        return np.array([t], dtype=FLOAT)

    def get_input_rate_grad_time_anti(self, theta, t):
        return np.array([t**2 / 2], dtype=FLOAT)

    # format input rate for lambda representation (requires rates at every station)
    def get_lambda(self, theta, t, left=True):
        in_rate = self.get_input_rate(theta, t)
        return np.concatenate([[in_rate], np.zeros(shape=self.D-1)])

    def get_lambda_time_anti(self, theta, t, left=True):
        in_rate_time_anti = self.get_input_rate_time_anti(theta, t)
        return np.concatenate([[in_rate_time_anti], np.zeros(shape=self.D-1)])

    def get_lambda_grad(self, theta, t, left=True):
        in_rate_grad = self.get_input_rate_grad(theta, t) # p dim
        return np.concatenate([[in_rate_grad], np.zeros(shape=(self.D-1, self.p))])

    def get_lambda_grad_time_anti(self, theta, t, left=True):
        in_rate_grad_time_anti = self.get_input_rate_grad_time_anti(theta, t)
        return np.concatenate([[in_rate_grad_time_anti], np.zeros(shape=(self.D-1, self.p))])

    def get_service(self, theta, t, left=True):
        return self.service_base_rates

    def get_service_time_anti(self, theta, t, left=True):
        return t * self.service_base_rates # note: in this model, service rates are constant

    def get_service_grad(self, theta, t, left=True):
        return np.zeros(shape=(self.D, self.p))

    def get_service_grad_time_anti(self, theta, t, left=True):
        return np.zeros(shape=(self.D, self.p))

    def get_num_servers(self, t, left=True):
        return self.num_servers

    def get_R_t(self, theta, t):
        return self.R

    def get_R_t_grad(self, theta, t):
        return np.zeros(shape=(self.D, self.D, self.p), dtype=np.float64) #hardcoded: transition probs are not parameterized

    def get_max_total_rate(self, theta):
        """
        Hardcoded: maximum time t = self.TMAX
        """
        return self.arrival_base_rate + np.max([0, theta[0] * self.TMAX]) + np.sum(self.service_base_rates * self.num_servers) + self.feedback_p * np.max(self.service_base_rates)

    def compute_state_dependent_rate_bound(self, theta) -> np.ndarray:
        if len(theta) != len(self.theta_true):
            raise ValueError("Dimension mismatch between theta and theta_true")
        # exogenous arrivals occur at state-independent rate
        exogenous_arrival_max = self.arrival_base_rate + np.max([0, theta[0] * self.TMAX])
        # services are state dependent
        states_decoded = vec_decode(np.arange(self._d), self.d_vec) # _d x d
        service_upper_bound = np.minimum(self.num_servers, states_decoded) @ self.service_base_rates
        return exogenous_arrival_max + service_upper_bound

    def get_discontinuity_times(self, t0, t1):
        return super().get_discontinuity_times(t0, t1)

    def save(self, filepath):
        np.savez(
            filepath,
            class_name=self.__class__.__name__,
            capacities=self.capacities,
            num_servers=self.num_servers,
            arrival_base_rate=self.arrival_base_rate,
            arrival_trend=self.arrival_trend,
            service_base_rates=self.service_base_rates,
            initial_dist=self.mu,
            feedback_p=self.feedback_p,
            TMAX=self.TMAX
        )
        return filepath

    @classmethod
    def load(cls, filepath):
        data = np.load(filepath, allow_pickle=True)

        return cls(
            capacities=data["capacities"],
            num_servers=data["num_servers"],
            arrival_base_rate=float(data["arrival_base_rate"]),
            arrival_trend=data["arrival_trend"],
            service_base_rates=data["service_base_rates"],
            initial_dist=data["initial_dist"],
            feedback_p=float(data["feedback_p"]),
            TMAX=float(data["TMAX"])
        )


class TandemQueueSecularTrendArrivalsFixedService(TandemQueue):
    """
    Tandem queue model with trend of the form theta_0 + theta_1 * t, and parameterized service rate
    """
    def __init__(
            self, 
            capacities: Sequence[int], 
            num_servers: Sequence[int], 
            arrival_base_rate: float, 
            arrival_trend: float, 
            service_base_rates: Sequence[float],
            initial_dist=None,
            feedback_p: float = 0.,
            TMAX: float = 1000.
        ):
        super().__init__(capacities, feedback_p=feedback_p)

        self.TMAX=TMAX

        if not len(num_servers) == self.D:
            raise ValueError(f"Dimension of num_servers must match D={self.D}, but has dimension {len(num_servers)}")
        if not len(service_base_rates) == self.D:
            raise ValueError(f"Dimension of service_base_rates must match D={self.D}, but has dimension {len(service_base_rates)}")

        self.arrival_base_rate = arrival_base_rate
        self.arrival_trend = arrival_trend
        self.service_base_rates = np.array(service_base_rates)

        # self.theta_true = np.array([arrival_base_rate, arrival_trend], dtype=FLOAT)
        self.theta_true = np.concatenate([[arrival_base_rate, arrival_trend], service_base_rates], dtype=FLOAT)
        

        self.num_servers = np.array(num_servers)
        self.p = len(self.theta_true)

        self.num_lambda_params = 2

        self.slice_arrival = slice(0,self.num_lambda_params)
        self.slice_service = slice(self.num_lambda_params, None)
        
        A_arr = np.array([[1., TMAX], [1., 0.]])
        A_ser = np.eye(self.D)
        Z1 = np.zeros((2, self.D))
        Z2 = np.zeros((self.D, 2))

        self.constraints = [
            LinearConstraint(
                A=np.block([[A_arr, Z1], [Z2, A_ser]]), 
                lb=np.full(shape=self.p, fill_value=QMIN), 
                ub=np.full(shape=self.p, fill_value=QMAX)
            )]

        # initialize to empty distribution
        if initial_dist is None:
            dist = np.zeros(self._d, dtype=FLOAT)
            dist[0] = 1.
            self.mu = dist
        else:
            self.mu = initial_dist

        self.discontinuity_times = []

        self.sanity_check(self.theta_true)

    # system input rate to tandem queue (input only at first station)
    def get_input_rate(self, theta, t):
        return theta[self.slice_arrival][0] + theta[self.slice_arrival][1] * t
    
    def get_input_rate_time_anti(self, theta, t):
        return t * theta[self.slice_arrival][0] + (t**2 / 2) * theta[self.slice_arrival][1]

    def get_input_rate_grad(self, theta, t):
        return np.concatenate([np.array([1., t], dtype=FLOAT), np.zeros(shape=self.D)])
    
    def get_input_rate_grad_time_anti(self, theta, t):
        return np.concatenate([np.array([t, t**2 / 2], dtype=FLOAT), np.zeros(shape=self.D)])

    # format input rate for lambda representation (requires rates at every station)
    def get_lambda(self, theta, t, left=True):
        in_rate = self.get_input_rate(theta, t)
        return np.concatenate([[in_rate], np.zeros(shape=self.D-1)])
    
    def get_lambda_time_anti(self, theta, t, left=True):
        in_rate_time_anti = self.get_input_rate_time_anti(theta, t)
        return np.concatenate([[in_rate_time_anti], np.zeros(shape=self.D-1)])

    def get_lambda_grad(self, theta, t, left=True):
        in_rate_grad = self.get_input_rate_grad(theta, t) # p dim
        return np.concatenate([[in_rate_grad], np.zeros(shape=(self.D-1, self.p))])
    
    def get_lambda_grad_time_anti(self, theta, t, left=True):
        in_rate_grad_time_anti = self.get_input_rate_grad_time_anti(theta, t)
        return np.concatenate([[in_rate_grad_time_anti], np.zeros(shape=(self.D-1, self.p))])

    def get_service(self, theta, t, left=True) -> np.ndarray:
        """
        Returns:
            Per-server service times. These must be multiplied by the number of active servers to yield the correct rate.
        """
        return theta[self.slice_service]
    
    def get_service_time_anti(self, theta, t, left=True) -> np.ndarray:
        """
        Returns:
            Per-server service time antiderivatives. These must be multiplied by the number of active servers to yield the correct rate.
        """
        return t * self.get_service(theta, t) # note: we can do this because in this model, service rates are constant

    def get_service_grad(self, theta, t, left=True) -> np.ndarray:
        """
        Returns:
            Per-server service time gradients. These must be multiplied by the number of active servers to yield the correct rate.
        """
        return np.concatenate([np.zeros(shape=(self.D, self.num_lambda_params)), np.eye(self.D)], axis=1)
    
    def get_service_grad_time_anti(self, theta, t, left=True) -> np.ndarray:
        """
        Returns:
            Antiderivatives of per-server service times gradinets. These must be multiplied by the number of active servers to yield the correct rate.
        """
        return np.concatenate([np.zeros(shape=(self.D, self.num_lambda_params)), t * np.eye(self.D)], axis=1)

    def get_num_servers(self, t, left=True):
        return self.num_servers
    
    def get_R_t(self, theta, t, left=True):
        return self.R

    def get_R_t_grad(self, theta, t, left=True):
        return np.zeros(shape=(self.D, self.D, self.p), dtype=np.float64) #hardcoded: transition probs are not parameterized

    def get_max_total_rate(self, theta):
        """
        Hardcoded: maximum time t = self.TMAX
        """
        if len(theta) != len(self.theta_true):
            raise ValueError("Dimension mismatch between theta and theta_true")
        exogenous_arrival_max = theta[self.slice_arrival][0] + self.TMAX * theta[self.slice_arrival][1]
        service_max = np.sum(np.abs(theta[self.slice_service] * self.num_servers))
        return exogenous_arrival_max + service_max

    def get_max_total_rate(self, theta):
        if len(theta) != len(self.theta_true):
            raise ValueError("Dimension mismatch between theta and theta_true")
        exogenous_arrival_max = np.sum(np.abs(theta[self.slice_arrival]))
        service_max = np.sum(np.abs(theta[self.slice_service] * self.num_servers))
        return exogenous_arrival_max + service_max

    def get_discontinuity_times(self, t0, t1):
        return super().get_discontinuity_times(t0, t1)

    def save(self, filepath):
        np.savez(
            filepath, 
            class_name=self.__class__.__name__,
            capacities=self.capacities,
            num_servers=self.num_servers,
            arrival_base_rate=self.arrival_base_rate,
            arrival_trend=self.arrival_trend,
            service_base_rates=self.service_base_rates,
            initial_dist=self.mu,
            feedback_p=self.feedback_p
        )
        return filepath
    
    @classmethod
    def load(cls, filepath):
        data = np.load(filepath, allow_pickle=True)

        return cls(
            capacities=data["capacities"],
            num_servers=data["num_servers"],
            arrival_base_rate=float(data["arrival_base_rate"]),
            arrival_trend=data["arrival_trend"],
            service_base_rates=data["service_base_rates"],
            initial_dist=data["initial_dist"],
            feedback_p=float(data["feedback_p"])
        )
    
@register_state_functional
def number_in_system(model: JacksonNetwork) -> np.ndarray:
    v = vec_decode(np.arange(model._d, dtype=INT), model.d_vec)
    return np.sum(v, axis=1)

@register_state_functional
def number_in_system_cross(model: JacksonNetwork) -> np.ndarray:
    v = vec_decode(np.arange(model._d, dtype=INT), model.d_vec)
    return np.sum(v, axis=1) + 0.5 * np.prod(v, axis=1)


@register_state_functional
def product_in_system(model: JacksonNetwork) -> np.ndarray:
    v = vec_decode(np.arange(model._d, dtype=INT), model.d_vec)
    return np.prod(v, axis=1)

@register_state_functional
def sum_square_in_system(model: JacksonNetwork) -> np.ndarray:
    v = vec_decode(np.arange(model._d, dtype=INT), model.d_vec)
    return np.sum(v**2, axis=1)

@register_state_functional
def max_in_system(model: JacksonNetwork) -> np.ndarray:
    v = vec_decode(np.arange(model._d, dtype=INT), model.d_vec)
    return np.max(v, axis=1)

@register_state_functional
def min_in_system(model: JacksonNetwork) -> np.ndarray:
    v = vec_decode(np.arange(model._d, dtype=INT), model.d_vec)
    return np.min(v, axis=1)

@register_state_functional
def pairwise_imbalance(model: JacksonNetwork) -> np.ndarray:
    """
    Sum of absolute pairwise differences across all subsystems.
    Measures total inequality of loads.
    """
    v = vec_decode(np.arange(model._d, dtype=INT), model.d_vec)
    diffs = np.abs(v[:, :, None] - v[:, None, :])
    return np.sum(diffs, axis=(1, 2)) / 2

@register_state_functional
def normalized_imbalance(model: JacksonNetwork) -> np.ndarray:
    """
    Variance normalized by total load — scale-free measure of imbalance.
    """
    v = vec_decode(np.arange(model._d, dtype=INT), model.d_vec)
    total = np.sum(v, axis=1, keepdims=True)
    total[total == 0] = 1  # avoid divide-by-zero
    mean_vals = np.mean(v, axis=1, keepdims=True)
    var_vals = np.sum((v - mean_vals) ** 2, axis=1)
    return var_vals / total.flatten()

@register_state_functional
def load_entropy(model: JacksonNetwork) -> np.ndarray:
    """
    Shannon entropy of the normalized queue length distribution per state.
    High entropy = more uniform load distribution.
    """
    v = vec_decode(np.arange(model._d, dtype=INT), model.d_vec)
    total = np.sum(v, axis=1, keepdims=True)
    total[total == 0] = 1
    p = v / total
    p[p == 0] = 1e-12
    return -np.sum(p * np.log(p), axis=1)

