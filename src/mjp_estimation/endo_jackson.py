from abc import abstractmethod
import itertools
from typing import Sequence
from collections import defaultdict
from functools import partial

import numpy as np
from scipy import sparse
from scipy.optimize import LinearConstraint
from scipy.integrate import quad


from .base_functionality import MJPModel, FLOAT, INT
from .jackson_model import vec_encode, vec_decode, encode, decode, possible_transitions, QMIN


class JacksonNetworkEndogenous(MJPModel):
    def __init__(self, capacities: Sequence[int], observable_transitions: Sequence[Sequence[int]]):
        self.capacities = capacities
        self.d_vec = np.array(capacities, INT) + 1
        self.D = len(self.d_vec)
        self._d = np.prod(self.d_vec)

        self.observable_transitions = set(observable_transitions)
        self.observables_encoded = set((
            (encode(state1, self.d_vec), encode(state2, self.d_vec)) 
            for state1, state2 in observable_transitions
        ))
        self.potentially_observable_states = set([state2 for state1, state2 in observable_transitions])
        self.d_observables = len(self.potentially_observable_states)
        self.map_observables_to_cemetery = {state: self._d + i for i, state in enumerate(self.potentially_observable_states)}
        self.map_encoded_observables_to_cemetery = {state: cemetery for state, cemetery in self.map_observables_to_cemetery.items()}

    def setup_jackson_network(self, routing_matrix, stations_with_arrivals):
        self.R = routing_matrix
        self.L = stations_with_arrivals
        self.generate_transition_structure(self.d_vec, routing_matrix, stations_with_arrivals)

    def generate_transition_structure(self, d_vec, R, L) -> None:
        all_states = np.array(list(itertools.product(*[range(d) for d in d_vec])), dtype=INT)
        all_possible_transitions = []
        for state in all_states:
            all_possible_transitions.append(possible_transitions(state, d_vec, R, L))
        
        # arrivals
        arr_from = []
        arr_to = []
        arr_idx = []

        arr_from_OBS = []
        arr_to_OBS = []
        arr_idx_OBS = []

        arr_from_UNOBS = []
        arr_to_UNOBS = []
        arr_idx_UNOBS = []

        # service (routing)
        route_from = []
        route_to = []
        route_idx_from = []
        route_idx_to = []
        route_idx_number_in_system_at_from_idx = []

        route_from_OBS = []
        route_to_OBS = []
        route_idx_from_OBS = []
        route_idx_to_OBS = []
        route_idx_number_in_system_at_from_idx_OBS = []

        route_from_UNOBS = []
        route_to_UNOBS = []
        route_idx_from_UNOBS = []
        route_idx_to_UNOBS = []
        route_idx_number_in_system_at_from_idx_UNOBS = []

        # service (exit)
        exit_from = []
        exit_to = []
        exit_idx = []
        exit_is_room = []
        exit_idx_number_in_system = []

        exit_from_OBS = []
        exit_to_OBS = []
        exit_idx_OBS = []
        exit_is_room_OBS = []
        exit_idx_number_in_system_OBS = []

        exit_from_UNOBS = []
        exit_to_UNOBS = []
        exit_idx_UNOBS = []
        exit_is_room_UNOBS = []
        exit_idx_number_in_system_UNOBS = []

        enc = partial(encode, d_vec=d_vec)
        dec = partial(decode, d_vec=d_vec)

        xmap = {}

        for pt_state_dict in all_possible_transitions:
            x_enc = encode(pt_state_dict['initial_state'], d_vec)
            xmap[x_enc] = defaultdict(set)
            for pos_arrival, pos_arrival_idx in zip(pt_state_dict['pos_arrival'], pt_state_dict['pos_arrival_idx']):
                arr_from.append(x_enc)
                arr_to.append(enc(pos_arrival))
                arr_idx.append(pos_arrival_idx)
                # if (tuple(pt_state_dict['initial_state']), tuple(pos_arrival)) in self.observable_transitions:
                if (x_enc, enc(pos_arrival)) in self.observable_transitions:
                    arr_from_OBS.append(x_enc)
                    arr_to_OBS.append(self.map_observables_to_cemetery[enc(pos_arrival)])
                    arr_idx_OBS.append(pos_arrival_idx)
                else:
                    arr_from_UNOBS.append(x_enc)
                    arr_to_UNOBS.append(enc(pos_arrival))
                    arr_idx_UNOBS.append(pos_arrival_idx)

                xmap[x_enc]['arr'].add(pos_arrival_idx)
            
            for pos_service_routing, pos_service_routing_from_idx, pos_service_routing_to_idx in zip(
                pt_state_dict['pos_service_routing'], 
                pt_state_dict['pos_service_routing_from_idx'], 
                pt_state_dict['pos_service_routing_to_idx']
            ):
                route_from.append(x_enc)
                route_to.append(enc(pos_service_routing))
                route_idx_from.append(pos_service_routing_from_idx)
                route_idx_to.append(pos_service_routing_to_idx)
                route_idx_number_in_system_at_from_idx.append(pt_state_dict['initial_state'][pos_service_routing_from_idx])

                # if (tuple(pt_state_dict['initial_state']), tuple(pos_service_routing)) in self.observable_transitions:
                if (x_enc, enc(pos_service_routing)) in self.observable_transitions:
                    route_from_OBS.append(x_enc)
                    route_to_OBS.append(self.map_observables_to_cemetery[enc(pos_service_routing)])
                    route_idx_from_OBS.append(pos_service_routing_from_idx)
                    route_idx_to_OBS.append(pos_service_routing_to_idx)
                    route_idx_number_in_system_at_from_idx_OBS.append(pt_state_dict['initial_state'][pos_service_routing_from_idx])
                else:
                    route_from_UNOBS.append(x_enc)
                    route_to_UNOBS.append(enc(pos_service_routing))
                    route_idx_from_UNOBS.append(pos_service_routing_from_idx)
                    route_idx_to_UNOBS.append(pos_service_routing_to_idx)
                    route_idx_number_in_system_at_from_idx_UNOBS.append(pt_state_dict['initial_state'][pos_service_routing_from_idx])

                xmap[x_enc]['serv'].add(pos_service_routing_from_idx) # shared with pure service (because no blocking)
            
            for pos_pure_service, pos_pure_service_idx in zip(
                pt_state_dict['pos_pure_service'],
                pt_state_dict['pos_pure_service_idx']
            ):
                exit_from.append(x_enc)
                exit_to.append(enc(pos_pure_service))
                exit_idx.append(pos_pure_service_idx)
                exit_is_room.append(pt_state_dict['is_room'])
                exit_idx_number_in_system.append(pt_state_dict['initial_state'][pos_pure_service_idx])

                # if (tuple(pt_state_dict['initial_state']), tuple(pos_pure_service)) in self.observable_transitions:
                if (x_enc, enc(pos_pure_service)) in self.observable_transitions:
                    exit_from_OBS.append(x_enc)
                    exit_to_OBS.append(self.map_observables_to_cemetery[enc(pos_pure_service)])
                    exit_idx_OBS.append(pos_pure_service_idx)
                    exit_is_room_OBS.append(pt_state_dict['is_room'])
                    exit_idx_number_in_system_OBS.append(pt_state_dict['initial_state'][pos_pure_service_idx])
                else:
                    exit_from_UNOBS.append(x_enc)
                    exit_to_UNOBS.append(enc(pos_pure_service))
                    exit_idx_UNOBS.append(pos_pure_service_idx)
                    exit_is_room_UNOBS.append(pt_state_dict['is_room'])
                    exit_idx_number_in_system_UNOBS.append(pt_state_dict['initial_state'][pos_pure_service_idx])

                xmap[x_enc]['serv'].add(pos_pure_service_idx) # shared with routing (because no blocking)
        
        self.xmap = xmap

        self.arr_from = np.array(arr_from, INT)
        self.arr_to = np.array(arr_to, INT)
        self.arr_idx = np.array(arr_idx, INT)

        self.route_from = np.array(route_from, INT)
        self.route_to = np.array(route_to, INT)
        self.route_idx_from = np.array(route_idx_from, INT)
        self.route_idx_to = np.array(route_idx_to, INT)
        self.route_idx_number_in_system_at_from_idx = np.array(route_idx_number_in_system_at_from_idx, INT)

        self.exit_from = np.array(exit_from, INT)
        self.exit_to = np.array(exit_to, INT)
        self.exit_idx = np.array(exit_idx, INT)
        self.exit_is_room = np.array(exit_is_room, INT)
        self.exit_idx_number_in_system = np.array(exit_idx_number_in_system, INT)

        # observable information

        self.arr_from_OBS = np.array(arr_from_OBS, INT)
        self.arr_to_OBS = np.array(arr_to_OBS, INT)
        self.arr_idx_OBS = np.array(arr_idx_OBS, INT)

        self.arr_from_UNOBS = np.array(arr_from_UNOBS, INT)
        self.arr_to_UNOBS = np.array(arr_to_UNOBS, INT)
        self.arr_idx_UNOBS = np.array(arr_idx_UNOBS, INT)

        self.route_from_OBS = np.array(route_from_OBS, INT)
        self.route_to_OBS = np.array(route_to_OBS, INT)
        self.route_idx_from_OBS = np.array(route_idx_from_OBS, INT)
        self.route_idx_to_OBS = np.array(route_idx_to_OBS, INT)
        self.route_idx_number_in_system_at_from_idx_OBS = np.array(route_idx_number_in_system_at_from_idx_OBS, INT)

        self.route_from_UNOBS = np.array(route_from_UNOBS, INT)
        self.route_to_UNOBS = np.array(route_to_UNOBS, INT)
        self.route_idx_from_UNOBS = np.array(route_idx_from_UNOBS, INT)
        self.route_idx_to_UNOBS = np.array(route_idx_to_UNOBS, INT)
        self.route_idx_number_in_system_at_from_idx_UNOBS = np.array(route_idx_number_in_system_at_from_idx_UNOBS, INT)

        self.exit_from_OBS = np.array(exit_from_OBS, INT)
        self.exit_to_OBS = np.array(exit_to_OBS, INT)
        self.exit_idx_OBS = np.array(exit_idx_OBS, INT)
        self.exit_is_room_OBS = np.array(exit_is_room_OBS, INT)
        self.exit_idx_number_in_system_OBS = np.array(exit_idx_number_in_system_OBS, INT)

        self.exit_from_UNOBS = np.array(exit_from_UNOBS, INT)
        self.exit_to_UNOBS = np.array(exit_to_UNOBS, INT)
        self.exit_idx_UNOBS = np.array(exit_idx_UNOBS, INT)
        self.exit_is_room_UNOBS = np.array(exit_is_room_UNOBS, INT)
        self.exit_idx_number_in_system_UNOBS = np.array(exit_idx_number_in_system_UNOBS, INT)

        self.row_map = {}
        for x in range(len(all_states)):
            self.row_map[x] = {
                "arr": np.where(self.arr_from == x)[0],
                "route": np.where(self.route_from == x)[0],
                "exit": np.where(self.exit_from == x)[0],
            }

    @property
    def state_space_size(self) -> int:
        return self._d
    
    @abstractmethod
    def get_lambda(self, theta, t): pass
    @abstractmethod 
    def get_service(self, theta, t): pass 
    @abstractmethod 
    def get_num_servers(self, t): pass 
    @abstractmethod 
    def get_R_t(self, theta, t): pass


    def get_Q(self, t, theta, left: bool = True):
        lambda_t = self.get_lambda(theta, t)
        mu_t = self.get_service(theta, t)
        ci_t = self.get_num_servers(t)
        R_t = self.get_R_t(theta, t)

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
        coords = np.concat([
            [self.arr_from, self.arr_to], 
            [self.route_from, self.route_to], 
            [self.exit_from, self.exit_to], 
            [diag_coords, diag_coords]
            ], axis=1)
        data = np.concat([arrival_rate, effective_routing_rate, effective_exit_rate, diag_rates])
        Q = sparse.coo_array((data, coords))
        
        return Q
    
    def get_Q_obs(self, t, theta, left: bool = True):
        # d x d_observables matrix
        lambda_t = self.get_lambda(theta, t)
        mu_t = self.get_service(theta, t)
        ci_t = self.get_num_servers(t)
        R_t = self.get_R_t(theta, t)

        # arrival rate
        arrival_rate = lambda_t[self.arr_idx_OBS]

        # routing rate
        server_cap = ci_t[self.route_idx_from_OBS]
        active_servers = np.minimum(server_cap, self.route_idx_number_in_system_at_from_idx_OBS)
        total_service_rate = active_servers * mu_t[self.route_idx_from_OBS]
        routing_prob = R_t[self.route_idx_from_OBS, self.route_idx_to_OBS]
        effective_routing_rate = total_service_rate * routing_prob

        # pure exit rate
        r_t = 1. - np.sum(R_t[self.exit_idx_OBS] * self.exit_is_room_OBS, axis=1)
        exit_server_cap = ci_t[self.exit_idx_OBS]
        exit_active_servers = np.minimum(exit_server_cap, self.exit_idx_number_in_system_OBS)
        exit_total_service_rate = exit_active_servers * mu_t[self.exit_idx_OBS]

        effective_exit_rate = r_t * exit_total_service_rate
       
        # assemble matrix
        coords = np.concat([
            [self.arr_from_OBS, self.arr_to_OBS - self._d], 
            [self.route_from_OBS, self.route_to_OBS - self._d], 
            [self.exit_from_OBS, self.exit_to_OBS - self._d], 
            ], axis=1)
        data = np.concat([arrival_rate, effective_routing_rate, effective_exit_rate])
        Q_obs = sparse.coo_array((data, coords), shape=(self._d, self.d_observables))

        return Q_obs


    def get_Q_unobs(self, t, theta, left: bool = True):
        # d x d matrix
        lambda_t = self.get_lambda(theta, t)
        mu_t = self.get_service(theta, t)
        ci_t = self.get_num_servers(t)
        R_t = self.get_R_t(theta, t)

        # arrival rate
        arrival_rate = lambda_t[self.arr_idx_UNOBS]

        # routing rate
        server_cap = ci_t[self.route_idx_from_UNOBS]
        active_servers = np.minimum(server_cap, self.route_idx_number_in_system_at_from_idx_UNOBS)
        total_service_rate = active_servers * mu_t[self.route_idx_from_UNOBS]
        routing_prob = R_t[self.route_idx_from_UNOBS, self.route_idx_to_UNOBS]
        effective_routing_rate = total_service_rate * routing_prob

        # pure exit rate
        r_t = 1. - np.sum(R_t[self.exit_idx_UNOBS] * self.exit_is_room_UNOBS, axis=1)
        exit_server_cap = ci_t[self.exit_idx_UNOBS]
        exit_active_servers = np.minimum(exit_server_cap, self.exit_idx_number_in_system_UNOBS)
        exit_total_service_rate = exit_active_servers * mu_t[self.exit_idx_UNOBS]

        effective_exit_rate = r_t * exit_total_service_rate

        # to construct diagonal, we must know the total rates
        # arrival rate
        total_arrival_rate = lambda_t[self.arr_idx]

        # routing rate
        total_server_cap = ci_t[self.route_idx_from]
        total_active_servers = np.minimum(total_server_cap, self.route_idx_number_in_system_at_from_idx)
        total_total_service_rate = total_active_servers * mu_t[self.route_idx_from]
        total_routing_prob = R_t[self.route_idx_from, self.route_idx_to]
        total_effective_routing_rate = total_total_service_rate * total_routing_prob

        # pure exit rate
        total_r_t = 1. - np.sum(R_t[self.exit_idx] * self.exit_is_room, axis=1)
        total_exit_server_cap = ci_t[self.exit_idx]
        total_exit_active_servers = np.minimum(total_exit_server_cap, self.exit_idx_number_in_system)
        total_exit_total_service_rate = total_exit_active_servers * mu_t[self.exit_idx]

        total_effective_exit_rate = total_r_t * total_exit_total_service_rate

        # construct diagonal
        total_arr_rates = np.bincount(self.arr_from, weights=total_arrival_rate, minlength=self._d)
        total_route_rates = np.bincount(self.route_from, weights=total_effective_routing_rate, minlength=self._d)
        total_pure_rates = np.bincount(self.exit_from, weights=total_effective_exit_rate, minlength=self._d)

        diag_rates = -(total_arr_rates + total_route_rates + total_pure_rates)
        diag_coords = np.arange(self._d, dtype=INT)

        # assemble matrix
        coords = np.concat([
            [self.arr_from_UNOBS, self.arr_to_UNOBS], 
            [self.route_from_UNOBS, self.route_to_UNOBS], 
            [self.exit_from_UNOBS, self.exit_to_UNOBS], 
            [diag_coords, diag_coords]
            ], axis=1)
        data = np.concat([arrival_rate, effective_routing_rate, effective_exit_rate, diag_rates])
        Q_obs = sparse.coo_array((data, coords), shape=(self._d, self._d))

        return Q_obs


    def get_Q_T(self, t, theta, left: bool = True):
        """ Full generator matrix including observable cemetery states. """
        # note: there is redundancy here, since both Q_unobs and Q_obs recompute the same quantities
        Q_unobs = self.get_Q_unobs(t, theta, left)
        Q_obs = self.get_Q_obs(t, theta, left)

        Q_T = sparse.hstack([Q_unobs, Q_obs])
        Q_T = sparse.vstack([Q_T, sparse.coo_array((self.d_observables, self._d + self.d_observables))])
        return Q_T



    def get_Q_grad(self, t, theta, left: bool = True):
        lambda_t = self.get_lambda(theta, t) # D 
        mu_t = self.get_service(theta, t) # D 
        R_t = self.get_R_t(theta, t) # D x D 

        lambda_t_grad = self.get_lambda_grad(theta, t) # D x p 
        mu_t_grad = self.get_service_grad(theta, t) # D x p 
        R_t_grad = self.get_R_t_grad(theta, t) # D x D x p

        ci_t = self.get_num_servers(t) # D

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
            coords = np.concat([
                [self.arr_from, self.arr_to], 
                [self.route_from, self.route_to], 
                [self.exit_from, self.exit_to], 
                [diag_coords, diag_coords]
                ], axis=1)
            data = np.concat([arrival_rate_grad, effective_routing_rate_grad, effective_exit_rate_grad, diag_rates])
            Q_grad = sparse.coo_array((data, coords)) 

            gradients.append(Q_grad)

        return gradients

    def get_Q_obs_grad(self, t, theta, left: bool = True):
        # todo: carefully check AI translation work
        lambda_t = self.get_lambda(theta, t)
        mu_t = self.get_service(theta, t)
        R_t = self.get_R_t(theta, t)

        lambda_t_grad = self.get_lambda_grad(theta, t)
        mu_t_grad = self.get_service_grad(theta, t)
        R_t_grad = self.get_R_t_grad(theta, t)

        ci_t = self.get_num_servers(t)

        server_cap = ci_t[self.route_idx_from_OBS]
        active_servers = np.minimum(server_cap, self.route_idx_number_in_system_at_from_idx_OBS)
        routing_prob = R_t[self.route_idx_from_OBS, self.route_idx_to_OBS]

        r_t = 1. - np.sum(R_t[self.exit_idx_OBS] * self.exit_is_room_OBS, axis=1)
        exit_server_cap = ci_t[self.exit_idx_OBS]
        exit_active_servers = np.minimum(exit_server_cap, self.exit_idx_number_in_system_OBS)
        exit_total_service_rate = exit_active_servers * mu_t[self.exit_idx_OBS]

        gradients = []
        for i in range(len(theta)):
            # arrival rate
            arrival_rate_grad = lambda_t_grad[self.arr_idx_OBS, i] 

            # routing rate
            total_service_rate = mu_t[self.route_idx_from_OBS]
            total_service_rate_grad = mu_t_grad[self.route_idx_from_OBS, i]
            routing_prob_grad = R_t_grad[self.route_idx_from_OBS, self.route_idx_to_OBS, i]
            effective_routing_rate_grad = active_servers * (
                total_service_rate_grad * routing_prob + total_service_rate * routing_prob_grad
            )

            # pure exit rate 
            r_t_grad = -1. * np.sum(R_t_grad[self.exit_idx_OBS, :, i] * self.exit_is_room_OBS, axis=1)
            exit_total_service_rate_grad = exit_active_servers * mu_t_grad[self.exit_idx_OBS, i]

            effective_exit_rate_grad = (
                r_t_grad * exit_total_service_rate + r_t * exit_total_service_rate_grad
            )

            # assemble matrix
            coords = np.concat([
                [self.arr_from_OBS, self.arr_to_OBS - self._d], 
                [self.route_from_OBS, self.route_to_OBS - self._d], 
                [self.exit_from_OBS, self.exit_to_OBS - self._d], 
                ], axis=1)
            data = np.concat([arrival_rate_grad, effective_routing_rate_grad, effective_exit_rate_grad])
            Q_grad = sparse.coo_array((data, coords), shape=(self._d, self.d_observables))
            gradients.append(Q_grad)

        return gradients


    def get_Q_unobs_grad(self, t, theta, left: bool = True):
        lambda_t = self.get_lambda(theta, t)
        mu_t = self.get_service(theta, t)
        R_t = self.get_R_t(theta, t)

        lambda_t_grad = self.get_lambda_grad(theta, t)
        mu_t_grad = self.get_service_grad(theta, t)
        R_t_grad = self.get_R_t_grad(theta, t)

        ci_t = self.get_num_servers(t)

        server_cap = ci_t[self.route_idx_from_UNOBS]
        active_servers = np.minimum(server_cap, self.route_idx_number_in_system_at_from_idx_UNOBS)
        routing_prob = R_t[self.route_idx_from_UNOBS, self.route_idx_to_UNOBS]

        r_t = 1. - np.sum(R_t[self.exit_idx_UNOBS] * self.exit_is_room_UNOBS, axis=1)
        exit_server_cap = ci_t[self.exit_idx_UNOBS]
        exit_active_servers = np.minimum(exit_server_cap, self.exit_idx_number_in_system_UNOBS)
        exit_total_service_rate = exit_active_servers * mu_t[self.exit_idx_UNOBS]

        gradients = []
        for i in range(len(theta)):
            # arrival rate
            arrival_rate_grad = lambda_t_grad[self.arr_idx_UNOBS, i] 

            # routing rate
            total_service_rate = mu_t[self.route_idx_from_UNOBS]
            total_service_rate_grad = mu_t_grad[self.route_idx_from_UNOBS, i]
            routing_prob_grad = R_t_grad[self.route_idx_from_UNOBS, self.route_idx_to_UNOBS, i]
            effective_routing_rate_grad = active_servers * (
                total_service_rate_grad * routing_prob + total_service_rate * routing_prob_grad
            )

            # pure exit rate 
            r_t_grad = -1. * np.sum(R_t_grad[self.exit_idx_UNOBS, :, i] * self.exit_is_room_UNOBS, axis=1)
            exit_total_service_rate_grad = exit_active_servers * mu_t_grad[self.exit_idx_UNOBS, i]

            effective_exit_rate_grad = (
                r_t_grad * exit_total_service_rate + r_t * exit_total_service_rate_grad
            )

            # to construct diagonal, we must know the total rates
            # arrival rate
            total_arrival_rate = lambda_t[self.arr_idx]

            # routing rate
            total_server_cap = ci_t[self.route_idx_from]
            total_active_servers = np.minimum(total_server_cap, self.route_idx_number_in_system_at_from_idx)
            total_total_service_rate = total_active_servers * mu_t[self.route_idx_from]
            total_routing_prob = R_t[self.route_idx_from, self.route_idx_to]
            total_effective_routing_rate = total_total_service_rate * total_routing_prob
            # pure exit rate
            total_r_t = 1. - np.sum(R_t[self.exit_idx] * self.exit_is_room, axis=1)
            total_exit_server_cap = ci_t[self.exit_idx]
            total_exit_active_servers = np.minimum(total_exit_server_cap, self.exit_idx_number_in_system)
            total_exit_total_service_rate = total_exit_active_servers * mu_t[self.exit_idx]
            total_effective_exit_rate = total_r_t * total_exit_total_service_rate
            # construct diagonal
            total_arr_rates = np.bincount(self.arr_from, weights=total_arrival_rate, minlength=self._d)
            total_route_rates = np.bincount(self.route_from, weights=total_effective_routing_rate, minlength=self._d)
            total_pure_rates = np.bincount(self.exit_from, weights=total_effective_exit_rate, minlength=self._d)
            diag_rates = -(total_arr_rates + total_route_rates + total_pure_rates)
            diag_coords = np.arange(self._d, dtype=INT) 
            # assemble matrix
            coords = np.concat([
                [self.arr_from_UNOBS, self.arr_to_UNOBS], 
                [self.route_from_UNOBS, self.route_to_UNOBS], 
                [self.exit_from_UNOBS, self.exit_to_UNOBS], 
                [diag_coords, diag_coords]
                ], axis=1)
            data = np.concat([arrival_rate_grad, effective_routing_rate_grad, effective_exit_rate_grad, diag_rates])
            Q_grad = sparse.coo_array((data, coords), shape=(self._d, self._d))
            gradients.append(Q_grad)
        return gradients

    def get_Q_T_grad(self, t, theta, left: bool = True):
        Q_unobs_grad = self.get_Q_unobs_grad(t, theta, left)
        Q_obs_grad = self.get_Q_obs_grad(t, theta, left)

        Q_T_grad = []
        for Q_unobs_g, Q_obs_g in zip(Q_unobs_grad, Q_obs_grad):
            Q_T_g = sparse.hstack([Q_unobs_g, Q_obs_g])
            Q_T_g = sparse.vstack([Q_T_g, sparse.coo_array((self.d_observables, self._d + self.d_observables))])
            Q_T_grad.append(Q_T_g)
        return Q_T_grad

    def get_Q_row(self, x: int, t, theta) -> sparse.coo_array:
        """
        Efficiently compute a single row of the generator matrix Q for state x.
        Returns a sparse 1xd row (in COO format) without building the full Q.
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

        diag_rate = -(np.sum(arr_rate) + np.sum(route_rate) + np.sum(exit_rate))

        row_idx = np.full(len(arr_to) + len(route_to) + len(exit_to) + 1, x, dtype=INT)
        col_idx = np.concatenate([arr_to, route_to, exit_to, [x]])
        data = np.concatenate([arr_rate, route_rate, exit_rate, [diag_rate]])

        return sparse.coo_array((data, (row_idx, col_idx)), shape=(self._d, self._d))
    
    def get_Q_grad_row(self, x: int, t, theta) -> list[sparse.coo_array]:
        """
        Efficiently compute a single-row gradient of Q for state x.
        Returns a list of sparse 1×d rows (one per parameter).
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
            # routing
            arr_rate_grad = lambda_t_grad[self.arr_idx[arr_idx], i]

            # service
            total_service_rate = mu_t[self.route_idx_from[route_idx]]
            total_service_rate_grad = mu_t_grad[self.route_idx_from[route_idx], i]
            routing_prob_grad = R_t_grad[self.route_idx_from[route_idx], self.route_idx_to[route_idx], i]

            route_rate_grad = active_servers * (
                total_service_rate_grad * routing_prob
                + total_service_rate * routing_prob_grad
            )

            # exit
            r_t_grad = -np.sum(
                R_t_grad[self.exit_idx[exit_idx], :, i] * self.exit_is_room[exit_idx], axis=1
            )
            exit_total_service_rate_grad = exit_active_servers * mu_t_grad[self.exit_idx[exit_idx], i]
            exit_rate_grad = r_t_grad * exit_total_service_rate + r_t * exit_total_service_rate_grad

            diag_rate_grad = -(np.sum(arr_rate_grad) + np.sum(route_rate_grad) + np.sum(exit_rate_grad))

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

    def get_Q_row_dense(self, x: int, theta, t):
        """
        Compute a dense Q row for state x (no precomputation).
        Returns:
            q_row      : np.ndarray of shape (self._d,) with transition rates
            state_row  : np.ndarray of shape (self._d,) with 1 where transitions exist
        """
        # Get all instantaneous rates
        lambda_t = self.get_lambda(theta, t)
        mu_t = self.get_service(theta, t)
        R_t = self.get_R_t(theta, t)
        ci_t = self.get_num_servers(t)

        # Lookup pre-indexed transitions
        idx = self.row_map[x]
        arr_idx = idx["arr"]
        route_idx = idx["route"]
        exit_idx = idx["exit"]

        # Prepare output
        q_row = np.zeros(self._d, dtype=float)
        state_row = np.zeros(self._d, dtype=int)

        # --- Arrivals ---
        arr_to = self.arr_to[arr_idx]
        arr_rate = lambda_t[self.arr_idx[arr_idx]]
        q_row[arr_to] += arr_rate
        state_row[arr_to] = 1

        # --- Routing (service + routing) ---
        cap = ci_t[self.route_idx_from[route_idx]]
        active = np.minimum(cap, self.route_idx_number_in_system_at_from_idx[route_idx])
        total_service_rate = active * mu_t[self.route_idx_from[route_idx]]
        routing_prob = R_t[self.route_idx_from[route_idx], self.route_idx_to[route_idx]]
        route_rate = total_service_rate * routing_prob
        route_to = self.route_to[route_idx]
        q_row[route_to] += route_rate
        state_row[route_to] = 1

        # --- Exit (pure service) ---
        r_t = 1.0 - np.sum(R_t[self.exit_idx[exit_idx]] * self.exit_is_room[exit_idx], axis=1)
        exit_cap = ci_t[self.exit_idx[exit_idx]]
        active_exit = np.minimum(exit_cap, self.exit_idx_number_in_system[exit_idx])
        exit_rate = r_t * active_exit * mu_t[self.exit_idx[exit_idx]]
        exit_to = self.exit_to[exit_idx]
        q_row[exit_to] += exit_rate
        state_row[exit_to] = 1

        # --- Diagonal element ---
        diag = -(np.sum(arr_rate) + np.sum(route_rate) + np.sum(exit_rate))
        q_row[x] = diag
        state_row[x] = 1  # mark self-state for completeness

        return q_row, state_row

    def get_Q_grad_row_dense(self, x: int, theta, t):
        """
        Compute dense gradient rows for state x (no precomputation).
        Returns:
            grads : list of np.ndarray, each shape (self._d,)
        """
        # Get instantaneous quantities
        lambda_t = self.get_lambda(theta, t)
        mu_t = self.get_service(theta, t)
        R_t = self.get_R_t(theta, t)
        ci_t = self.get_num_servers(t)

        lambda_t_grad = self.get_lambda_grad(theta, t)
        mu_t_grad = self.get_service_grad(theta, t)
        R_t_grad = self.get_R_t_grad(theta, t)

        # Transition subsets
        idx = self.row_map[x]
        arr_idx = idx["arr"]
        route_idx = idx["route"]
        exit_idx = idx["exit"]

        # Shared components
        cap = ci_t[self.route_idx_from[route_idx]]
        active = np.minimum(cap, self.route_idx_number_in_system_at_from_idx[route_idx])
        routing_prob = R_t[self.route_idx_from[route_idx], self.route_idx_to[route_idx]]

        r_t = 1.0 - np.sum(R_t[self.exit_idx[exit_idx]] * self.exit_is_room[exit_idx], axis=1)
        exit_cap = ci_t[self.exit_idx[exit_idx]]
        active_exit = np.minimum(exit_cap, self.exit_idx_number_in_system[exit_idx])
        exit_total_service_rate = active_exit * mu_t[self.exit_idx[exit_idx]]

        n_params = mu_t_grad.shape[1]
        grads = [np.zeros(self._d, dtype=float) for _ in range(n_params)]

        for i in range(n_params):
            # --- Arrivals ---
            arr_rate_grad = lambda_t_grad[self.arr_idx[arr_idx], i]
            arr_to = self.arr_to[arr_idx]

            # --- Routing ---
            total_service_rate = mu_t[self.route_idx_from[route_idx]]
            total_service_rate_grad = mu_t_grad[self.route_idx_from[route_idx], i]
            routing_prob_grad = R_t_grad[self.route_idx_from[route_idx], self.route_idx_to[route_idx], i]
            route_rate_grad = active * (
                total_service_rate_grad * routing_prob + total_service_rate * routing_prob_grad
            )
            route_to = self.route_to[route_idx]

            # --- Exits ---
            r_t_grad = -np.sum(
                R_t_grad[self.exit_idx[exit_idx], :, i] * self.exit_is_room[exit_idx], axis=1
            )
            exit_total_service_rate_grad = active_exit * mu_t_grad[self.exit_idx[exit_idx], i]
            exit_rate_grad = r_t_grad * exit_total_service_rate + r_t * exit_total_service_rate_grad
            exit_to = self.exit_to[exit_idx]

            # --- Dense gradient row ---
            g = grads[i]
            g[arr_to] += arr_rate_grad
            g[route_to] += route_rate_grad
            g[exit_to] += exit_rate_grad
            g[x] = -(np.sum(arr_rate_grad) + np.sum(route_rate_grad) + np.sum(exit_rate_grad))

        return grads



    def get_exit_rates(self, t, left: bool = True, theta = None):
        """
        Sum over exit rates
        """
        lambda_t = self.get_lambda(theta, t)
        mu_t = self.get_service(theta, t) # D dim
        ci_t = self.get_num_servers(t) # D dim
        R_t = self.get_R_t(theta, t) # D x D dim

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
            np.concat([self.arr_from, self.route_from, self.exit_from]), 
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

    def integrate_exit_rate_at_state(self, theta: np.ndarray, t0: float, t1: float, x: int):
        arridx = list(self.xmap[x]['arr'])
        servidx = list(self.xmap[x]['serv'])
        arrivals = self.get_lambda_time_anti(theta, t1) - self.get_lambda_time_anti(theta, t0)
        service = self.get_service_time_anti(theta, t1) - self.get_service_time_anti(theta, t0)

        return np.sum(arrivals[arridx], axis=0) + np.sum(service[servidx], axis=0) # never forget to sum axis=0
        
    def integrate_grad_exit_rate_at_state(self, theta: np.ndarray, t0: float, t1: float, x: int):
        arridx = list(self.xmap[x]['arr'])
        servidx = list(self.xmap[x]['serv'])
        arrivals_grad = self.get_lambda_grad_time_anti(theta, t1) - self.get_lambda_grad_time_anti(theta, t0)
        service_grad = self.get_service_grad_time_anti(theta, t1) - self.get_service_grad_time_anti(theta, t0)

        return np.sum(arrivals_grad[arridx], axis=0) + np.sum(service_grad[servidx], axis=0) # never forget to sum axis=0

    def get_external_event_intensities(self, t: float, left: bool = True) -> np.ndarray:
        beta = np.zeros(self._d)
        return beta

    def get_initial_dist(self):
        return self.mu

    def get_initial_state(self) -> int: 
        return np.random.choice(self._d, p=self.get_initial_dist())



class TandemQueueEndogenous(JacksonNetworkEndogenous): 
    def __init__(self, capacities: Sequence[int], observable_transitions: set[tuple], feedback_p: float = 0.):
        super().__init__(capacities, observable_transitions)
        routing_matrix = np.diag(np.ones(self.D-1), k=1)
        if feedback_p > 0:
            routing_matrix[self.D-1, 0] = feedback_p
        self.feedback_p = feedback_p
        stations_with_arrivals = np.zeros(shape=self.D, dtype=INT)
        stations_with_arrivals[0] = 1

        self.setup_jackson_network(routing_matrix, stations_with_arrivals)



class TandemQueuePeriodicArrivalsConstantFixedServiceEndogenous(TandemQueueEndogenous):
    def __init__(
            self, 
            capacities: Sequence[int], 
            num_servers: Sequence[int], 
            period: float, 
            arrival_base_rate: float, 
            arrival_amplitude: float, 
            service_base_rates: Sequence[float],
            observable_transitions: set[tuple],
            initial_dist=None,
            feedback_p: float = 0.
        ):
        super().__init__(capacities, observable_transitions, feedback_p=feedback_p)

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
                ub=np.full(shape=self.p, fill_value=np.max(self.theta_true) + 20)
            )]

        # initialize to empty distribution
        if initial_dist is None:
            dist = np.zeros(self._d, dtype=FLOAT)
            dist[0] = 1.
            self.mu = dist
        else:
            self.mu = initial_dist

        self.discontinuity_times = []

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
    def get_lambda(self, theta, t):
        in_rate = self.get_input_rate(theta, t)
        return np.concatenate([[in_rate], np.zeros(shape=self.D-1)])
    
    def get_lambda_time_anti(self, theta, t):
        in_rate_time_anti = self.get_input_rate_time_anti(theta, t)
        return np.concatenate([[in_rate_time_anti], np.zeros(shape=self.D-1)])

    def get_lambda_grad(self, theta, t):
        in_rate_grad = self.get_input_rate_grad(theta, t) # p dim
        return np.concatenate([[in_rate_grad], np.zeros(shape=(self.D-1, self.p))])
    
    def get_lambda_grad_time_anti(self, theta, t):
        in_rate_grad_time_anti = self.get_input_rate_grad_time_anti(theta, t)
        return np.concatenate([[in_rate_grad_time_anti], np.zeros(shape=(self.D-1, self.p))])

    def get_service(self, theta, t):
        return self.service_base_rates
    
    def get_service_time_anti(self, theta, t):
        return t * self.service_base_rates # note: we can do this because in this model, service rates are constant

    def get_service_grad(self, theta, t):
        return np.zeros(shape=(self.D, self.p))
    
    def get_service_grad_time_anti(self, theta, t):
        return np.zeros(shape=(self.D, self.p))
    
    def get_num_servers(self, t):
        return self.num_servers
    
    def get_R_t(self, theta, t):
        return self.R

    def get_R_t_grad(self, theta, t):
        return np.zeros(shape=(self.D, self.D, self.p), dtype=np.float64) #hardcoded: transition probs are not parameterized

    def get_max_total_rate(self, theta):
        return np.sum(np.abs(theta)) + np.sum(self.service_base_rates * self.num_servers) + self.feedback_p * np.max(self.service_base_rates)

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



class JacksonEndogenousSubmodel:
    def __init__(self, base_model: JacksonNetworkEndogenous):
        self.base_model = base_model
        self._d = base_model._d + base_model.d_observables 
        self.constraints = base_model.constraints
    
    @property
    def state_space_size(self):
        return self._d
    
    def get_mock_state(self):
        return self.mock_state

    def get_Q(self, t, theta, left=True):
        return self.base_model.get_Q_T(t, theta, left)

    def get_Q_grad(self, t, theta, left=True):
        return self.base_model.get_Q_T_grad(t, theta, left)
    
    def get_exit_rates(self, t, left):
        return self.base_model.get_exit_rates(t, left)

    def get_external_event_intensities(self, t, left):
        return self.base_model.get_external_event_intensities(t, left)

    def get_max_total_rate(self, theta):
        return self.base_model.get_max_total_rate(theta)

    def get_total_rate_at_state(self, x):
        return self.base_model.get_total_rate_at_state(x)

    def get_initial_state(self):
        return self.base_model.get_initial_state()


def get_all_possible_incoming_transitions(jackson_model, states):
    """
    jackson_model: Jackson model
    states: iterable of states
    """
    if len(states) == 0:
        return set()
    incoming_transitions = set()
    for state in states:
        # Check if state is not an integer type (Python int or numpy integer)
        if not isinstance(state, (int, np.integer)):
            enc_state = encode(state, jackson_model.d_vec)
        else:
            enc_state = state
        arr_idxs = np.nonzero(jackson_model.arr_to == enc_state)[0]
        route_idxs = np.nonzero(jackson_model.route_to == enc_state)[0]
        exit_idxs = np.nonzero(jackson_model.exit_to == enc_state)[0]
        for idx in arr_idxs:
            incoming_transitions.add((jackson_model.arr_from[idx], enc_state))
        for idx in route_idxs:
            incoming_transitions.add((jackson_model.route_from[idx], enc_state))
        for idx in exit_idxs:
            incoming_transitions.add((jackson_model.exit_from[idx], enc_state))
    return incoming_transitions