from abc import ABC, abstractmethod
from typing import Iterable, Sequence, Set, Tuple
import logging

import numpy as np
from scipy.integrate import solve_ivp
import scipy.sparse as sp
from tqdm import tqdm

FLOAT = np.float64
INT = np.int32


class MJPModel(ABC):
    """Abstract Base Class for a time-inhomogeneous Markov Jump Process model."""
    @property
    @abstractmethod
    def state_space_size(self) -> int: pass

    @abstractmethod
    def get_exit_rates(self, t: float, left: bool, theta:np.ndarray=None) -> np.ndarray: pass

    @abstractmethod
    def get_Q(self, t, theta, left: bool) -> sp.sparray: pass

    @abstractmethod
    def get_Q_row_compact(self, x: int, theta: np.ndarray, t: float, left: bool = True): pass

    @abstractmethod
    def get_Q_grad(self, t, theta, left) -> Iterable[sp.sparray]: pass

    @abstractmethod
    def get_Q_grad_row_compact(self, x: int, theta: np.ndarray, t: float, left: bool = True): pass

    @abstractmethod
    def get_external_event_intensities(self, t: float, left: bool) -> np.ndarray: pass

    @abstractmethod
    def get_max_total_rate(self, theta: np.ndarray=None) -> float: pass

    def get_total_rate_at_state(self, x, theta: np.ndarray=None) -> float: pass

    @abstractmethod
    def compute_state_dependent_rate_bound(self, theta: np.ndarray=None) -> np.ndarray : pass

    def get_initial_state(self) -> int: return 0

    @abstractmethod
    def integrate_exit_rate_at_state(self, theta, t0, t1, x) -> float: pass

    @abstractmethod
    def integrate_grad_exit_rate_at_state(self, theta, t0, t1, x) -> np.ndarray: pass

    @abstractmethod
    def get_discontinuity_times(self, t0, t1) -> np.ndarray: return np.array([])

    @abstractmethod
    def save(self, filepath: str) -> str: pass

    @classmethod
    @abstractmethod
    def load(filepath: str) -> str: pass

    def sanity_check(self, theta=None, t_max: float = 10, num_points: int = 20):
        """
        Check that get_max_total_rate is accurate over the interval [0, t_max] by checking
        regularly at num_points intervals
        """
        if theta is None:
            theta = self.theta_true
        max_rate = self.get_max_total_rate(theta)
        for t in np.linspace(0, t_max, num_points):
            max_rate_at_t = np.max(self.get_exit_rates(t, left=True, theta=theta))
            if max_rate_at_t > max_rate:
                raise ValueError(f"Max exit rate at time {t=} is {max_rate_at_t}, exceeding self.get_max_total_rate={max_rate}")


class SamplePath:
    def __init__(self, times: Sequence[float], states: Sequence[int], final_time: float, final_state: int):
        self.times = np.array(times, dtype=FLOAT)
        self.states: Sequence[int] = np.array(states, dtype=INT)
        self.final_time: float = FLOAT(final_time)
        self.final_state: int = INT(final_state)

    def truncate(self, t):
        cutoff_idx = np.searchsorted(self.times, t, side='right')
        return SamplePath(self.times[:cutoff_idx], self.states[:cutoff_idx], t, self.states[cutoff_idx-1])

    def save(self, filename: str):
        np.savez(filename, times=self.times, states=self.states, final_time=self.final_time, final_state=self.final_state)
        return filename

    def print(self):
        for t, x in zip(self.times, self.states):
            print(f"{t:.4f} - {x}")
        print(f"{self.final_time=:.2f} - {self.final_state}")
        
    def __str__(self):
        return f'SamplePath(times={self.times}, states={self.states}, final_time={self.final_time}, final_state={self.final_state})'
            
    def __eq__(self, other):
            if not isinstance(other, SamplePath):
                return NotImplemented
            
            if len(self.times) != len(other.times):
                return False
            
            if len(self.states) != len(other.states):
                return False

            # Use np.allclose for floats to handle tiny rounding differences
            times_equal = np.allclose(self.times, other.times)
            # For integer arrays, array_equal is fine
            states_equal = np.array_equal(self.states, other.states)

            final_time_equal = np.isclose(self.final_time, other.final_time)
            final_state_equal = self.final_state == other.final_state

            return times_equal and states_equal and final_time_equal and final_state_equal

def load_sample_path_from_file(filename: str):
    npzfile = np.load(filename)
    return SamplePath(
        times=npzfile['times'],
        states=npzfile['states'],
        final_time=npzfile['final_time'],
        final_state=npzfile['final_state']
    )
    

class DiscreteSample:
    def __init__(self, path: SamplePath, sample_times):
        self.original_path = path
        self.times = np.sort(sample_times)

        idxs = np.searchsorted(path.times, sample_times, side='right') - 1
        self.states = path.states[idxs]

        self.final_time = sample_times[-1]
        self.final_state = self.states[-1]

    def __str__(self):
        return f'DiscreteSample(times={self.times}, states={self.states}, final_time={self.final_time}, final_state={self.final_state})'
    
    def __len__(self):
        return len(self.states)

    def __eq__(self, other):
        if not isinstance(other, DiscreteSample):
            return NotImplemented

        if len(self.times) != len(other.times):
            return False
        
        if len(self.states) != len(other.states):
            return False


        # Compare arrays with np.array_equal (handles dtype safely)
        times_equal = np.array_equal(self.times, other.times)
        states_equal = np.array_equal(self.states, other.states)

        # Compare scalars directly
        final_time_equal = self.final_time == other.final_time
        final_state_equal = np.array_equal(self.final_state, other.final_state)

        return times_equal and states_equal and final_time_equal and final_state_equal


class EndogenousSample:
    def __init__(self, path: SamplePath, observable_transitions: Set[Tuple[int]]):
        """
        observable_transitions: set of (from_state, to_state) tuples of encoded states that are observable
         """
        self.observable_transitions = observable_transitions
        print(f"Endogenous Samples: {observable_transitions=}")
        endogenous_times, endogenous_states = [], []
        for (s, x), (t, y) in zip(zip(path.times, path.states), zip(path.times[1:], path.states[1:])):
            if (x, y) in observable_transitions:
                endogenous_times.append(t)
                endogenous_states.append(y)
        self.original_path = path
        self.times = endogenous_times
        self.states = endogenous_states


class Simulator:
    """Simulates a time-inhomogeneous MJP with a general reward structure."""
    def __init__(self, model: MJPModel, seed=None, rng=None):
        self.model: MJPModel = model
        # self.rewards = reward_structure
        if rng is None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = rng
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def simulate_single_run(
            self, 
            T_horizon: float, 
            starting_time: float = 0.,
            starting_state=None,
            theta=None,
        ) -> SamplePath:
        rng = self.rng

        total_jumps = 0
        current_time = next_time = starting_time
        if starting_state is None:
            state = self.model.get_initial_state()
        else:
            state = starting_state

        if theta is None:
            print("Warning: did not specify theta for simulation, defaulting to model true parameter")
            theta = self.model.theta_true

        transition_times, states = [starting_time], [state]

        rate_upper_bounds = self.model.compute_state_dependent_rate_bound(theta)
        max_exit_rate_at_state = rate_upper_bounds[state]
        
        dt = rng.exponential(1.0 / max_exit_rate_at_state)
        potential_event_time = current_time + dt

        while current_time < T_horizon:
            next_time = min(potential_event_time, T_horizon)
            current_time = next_time

            compact_states, compact_rates = self.model.get_Q_row_compact(state, theta, current_time)
            total_rate = -compact_rates[-1]
            exit_prob = total_rate / max_exit_rate_at_state

            if rng.random() < exit_prob:
                # event: transition at current_time
                total_jumps += 1

                # --- build transition probabilities ---
                probs = compact_rates[:-1] / total_rate
                try:
                    next_state = rng.choice(compact_states[:-1], p=probs)
                except ValueError as e:
                    print(f"{state=}")
                    print(f"{current_time=}")
                    print(f"{theta=}")
                    print(f"{probs=}")
                    raise ValueError(e)

                state = next_state
                max_exit_rate_at_state = rate_upper_bounds[state]
                transition_times.append(current_time)
                states.append(state)

            dt = rng.exponential(1.0 / max_exit_rate_at_state)
            potential_event_time = current_time + dt

        return SamplePath(
            np.array(transition_times, dtype=FLOAT),
            np.array(states, dtype=np.int32),
            T_horizon,
            state,
        )


    def generate_sample_paths(self, T_horizon: float, scheduled_times: list, num_paths: int):
        paths = []
        for _ in tqdm(range(num_paths), desc="Generating sample paths", ncols=80):
            _, times, states, rewards, jump_points, external_points, scheduled_points = self.simulate_single_run(T_horizon, scheduled_times, record_path=True)
            paths.append({'times': times, 'states': states, 'rewards': rewards,
                          'jump_points': jump_points, 'external_points': external_points, 'scheduled_points': scheduled_points})
        return paths


class JointKolmogorovSolver:
    """ Computes P(theta, Lambda_{i-1}, Lambda_i, Z_{i-1}, *) and grad_theta P(theta, Lambda_{i-1}, Lambda_i, Z_{i-1}, *) simultaneously using a joint ODE """
    def __init__(
        self,
        model: MJPModel,
        theta: np.ndarray,
        fixed_step: float = None,
        method: str='RK45',
        display_tqdm: bool = True,
        rtol: float = 1e-4,
        atol: float = 1e-6
    ):
        self.model = model
        self.d = model.state_space_size
        self.theta = theta
        self.p = len(theta)
        self.vec_dim = self.d * (self.p + 1)
        self.slice_prob = slice(0, self.d)
        self.slice_grad = [slice(i * self.d, (i+1) * self.d) for i in range(1, self.p+1)]
        self.method = method
        self.fixed_step = fixed_step
        self.display_tqdm = display_tqdm
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        self.rtol = rtol
        self.atol = atol

    def forward_solve(
            self, 
            t_left, 
            t_right, 
            initial_state, 
            discontinuity_times,
            max_step=None
            ) -> np.ndarray:
        time_points = sorted(list(set([t_left] + [t for t in discontinuity_times if t_left < t < t_right] + [t_right])))
        time_points = np.array(time_points, dtype=FLOAT)

        def ode_system(t, y, left=True):
            # augmented KFE system
            dy = np.zeros(self.vec_dim)
            Q = self.model.get_Q(t, self.theta, left=left)
            # self.log.info(np.array(Q))
            Q_derivative = self.model.get_Q_grad(t, self.theta, left=left)

            dy[self.slice_prob] = y[self.slice_prob] @ Q
            for i, Qprime in enumerate(Q_derivative):
                # loop over the p derivatives
                dy[self.slice_grad[i]] = y[self.slice_prob] @ Qprime + y[self.slice_grad[i]] @ Q
            return dy

        y = np.zeros(self.vec_dim)
        mu_initial_state = np.zeros(self.d, dtype=FLOAT)
        mu_initial_state[initial_state] = 1.
        y[self.slice_prob] = mu_initial_state

        iteration = zip(time_points[:-1], time_points[1:])
        if self.display_tqdm:
            iteration = tqdm(iteration, desc="Solving joint ODE for moments", ncols=80)

        for sub_t_left, sub_t_right in iteration:
            def ode_system_segment(t, y):
                if t == t_left:
                    return ode_system(t, y, left=False)
                return ode_system(t, y, left=True)
             
            if max_step is not None:
                sol = solve_ivp(
                    fun=ode_system_segment,
                    t_span=[sub_t_left, sub_t_right], # forward
                    y0=y,
                    method=self.method,
                    rtol=self.rtol,
                    atol=self.atol,
                    max_step=max_step
                )
            else:
                sol = solve_ivp(
                    fun=ode_system_segment,
                    t_span=[sub_t_left, sub_t_right], # forward
                    y0=y,
                    method=self.method,
                    rtol=self.rtol,
                    atol=self.atol,
                )
            y = sol.y[:, -1]
        return y

    def forward_solve_at_times(
        self,
        t_left: float,
        t_right_array: np.ndarray,
        initial_state: int,
        discontinuity_times: np.ndarray,
    ):
        """
        Solve the joint KFE for probabilities and gradients at multiple time points.

        Parameters
        ----------
        t_left : float
            Start time.
        t_right_array : np.ndarray
            Sorted array of times at which to return the solution.
        initial_state : int
            Initial discrete state index.
        discontinuity_times : np.ndarray
            Times where Q(t) has discontinuities.

        Returns
        -------
        results : np.ndarray
            Shape (len(t_right_array), d * (p+1)). Row i corresponds to t_right_array[i].
        """
        t_right_array = np.asarray(t_right_array, dtype=FLOAT)
        if t_right_array.size == 0:
            return np.empty((0, self.vec_dim))

        if not np.all(np.diff(t_right_array) >= 0):
            raise ValueError("t_right_array must be sorted in ascending order.")

        # build segments
        last_t = t_right_array[-1]
        crit = sorted([t_left] +
                        [t for t in discontinuity_times if t_left < t < last_t] +
                        [last_t])
        critical_times = np.array(crit, dtype=FLOAT)

        def ode_system(t, y, left=True):
            dy = np.zeros(self.vec_dim)
            Q = self.model.get_Q(t, self.theta, left=left)
            Q_derivative = self.model.get_Q_grad(t, self.theta, left=left)

            dy[self.slice_prob] = y[self.slice_prob] @ Q
            for i, Qprime in enumerate(Q_derivative):
                dy[self.slice_grad[i]] = y[self.slice_prob] @ Qprime + y[self.slice_grad[i]] @ Q
            return dy

        y = np.zeros(self.vec_dim)
        mu_initial_state = np.zeros(self.d, dtype=FLOAT)
        mu_initial_state[initial_state] = 1.0
        y[self.slice_prob] = mu_initial_state

        results = np.zeros((len(t_right_array), self.vec_dim))
        filled = np.zeros(len(t_right_array), dtype=bool)
        at_start_idxs = np.where(np.isclose(t_right_array, t_left))[0]
        if at_start_idxs.size:
            for idx in at_start_idxs:
                results[idx] = y
            filled[at_start_idxs] = True

        iteration = zip(critical_times[:-1], critical_times[1:])
        if self.display_tqdm:
            iteration = tqdm(iteration, desc="Solving joint ODE (multi-time)", ncols=80)

        for sub_t_left, sub_t_right in iteration:
            req_idxs = np.where(
                (t_right_array > sub_t_left) & (t_right_array <= sub_t_right) & (~filled)
            )[0]
            t_eval = t_right_array[req_idxs].tolist()

            def ode_system_segment(t, y_local):
                if np.isclose(t, sub_t_left):
                    return ode_system(t, y_local, left=False)
                return ode_system(t, y_local, left=True)

            sol = solve_ivp(
                fun=ode_system_segment,
                t_span=[sub_t_left, sub_t_right],
                y0=y,
                method=self.method,
                rtol=self.rtol,
                atol=self.atol,
                t_eval=t_eval if len(t_eval) > 0 else None,
            )

            if len(t_eval) > 0:
                # sol.t aligns with t_eval in order
                for j, idx in enumerate(req_idxs):
                    results[idx] = sol.y[:, j]
                    filled[idx] = True
                y = sol.y[:, -1]
            else:
                y = sol.y[:, -1]

            if np.all(filled):
                break

        return results


    def backward_solve(
            self, 
            t_left: float, 
            t_right: float, 
            final_state: int, 
            discontinuity_times: np.ndarray,
            ):
        """
        Solve vectorized backward ODEs for transition probabilities and gradients wrt parameters theta
        over [t_left, t_right], handling discontinuities.

        Parameters:
        ----------
        t_left : float
            Left endpoint of time interval.
        t_right : float
            Right endpoint.
        final_state : int
            Terminal state at time t_right.
        discontinuity_times : list of floats
            Times at which Q(t) has discontinuities.
        
        Returns
        ----------
        y : 
        """
        time_points = sorted(list(set([t_left] + [t for t in discontinuity_times if t_left < t < t_right] + [t_right])))
        time_points = np.array(time_points, dtype=FLOAT)

        def ode_system(t, y, left=True):
            # augmented KBE ODE system
            dy = np.zeros(self.vec_dim)
            Q = self.model.get_Q(t, self.theta, left=left)
            Q_derivative = self.model.get_Q_grad(t, self.theta, left=left)

            dy[self.slice_prob] = Q @ y[self.slice_prob]
            for i, Qprime in enumerate(Q_derivative):
                dy[self.slice_grad[i]] = Qprime @ y[self.slice_prob] + Q @ y[self.slice_grad[i]]
            return -dy

        y = np.zeros(self.vec_dim)
        mu_final_state = np.zeros(self.d, dtype=FLOAT)
        mu_final_state[final_state] = 1.
        y[self.slice_prob] = mu_final_state

        iteration = zip(reversed(time_points[:-1]), reversed(time_points[1:]))
        if self.display_tqdm:
            iteration = tqdm(iteration, desc="Solving joint ODE for moments\n", ncols=80)
        for sub_t_left, sub_t_right in iteration:
            def ode_system_segment(t, y):
                if t == t_left:
                    return ode_system(t, y, left=False)
                return ode_system(t, y, left=True)
             
            sol = solve_ivp(
                fun=ode_system_segment,
                t_span=[sub_t_right, sub_t_left], # backward
                y0=y,
                method=self.method,
                rtol=self.rtol,
                atol=self.atol
            )
            y = sol.y[:, -1]
        return y

    def backward_solve_h(
            self,
            t_left,
            t_right,
            h_final: np.ndarray,
            discontinuity_times,
        ):
        """
        Solve backward ODEs for vector-valued function h(t) and its gradients
        wrt parameters theta, over [t_left, t_right], handling discontinuities.

        Parameters
        ----------
        t_left : float
            Left endpoint of time interval.
        t_right : float
            Right endpoint
        h_final : np.ndarray, shape (d, r)
            Terminal condition for h(t), i.e. h(t_right) = h_final.
        discontinuity_times : list of floats
            Times at which Q(t) has discontinuities.
        
        Returns
        ----------
        h_initial
        h_grad_initial - shape (p, d, r)
        """

        time_points = sorted(
            list(set([t_left] + [t for t in discontinuity_times if t_left < t < t_right] + [t_right]))
        )
        time_points = np.array(time_points, dtype=np.float64)

        d, r = h_final.shape
        p = self.p
        vec_dim = d * r * (1 + p)
        slice_h = slice(0, d * r)
        slice_grad = [slice(d * r * (i + 1), d * r * (i + 2)) for i in range(p)]

        def ode_system(t, y, left=True):
            dy = np.zeros_like(y)
            Q = self.model.get_Q(t, self.theta, left=left)
            Q_grad = self.model.get_Q_grad(t, self.theta, left=left)

            h = y[slice_h].reshape(d, r)
            dy[slice_h] = (Q @ h).reshape(-1)

            for i in range(p):
                h_grad_i = y[slice_grad[i]].reshape(d, r)
                Qprime = Q_grad[i]
                dy[slice_grad[i]] = (Qprime @ h + Q @ h_grad_i).reshape(-1)

            return -dy  # backward

        # initialize final condition
        y = np.zeros(vec_dim)
        y[slice_h] = h_final.reshape(-1)

        # backward integration over discontinuity intervals
        iteration = zip(reversed(time_points[:-1]), reversed(time_points[1:]))
        if self.display_tqdm:
            iteration = tqdm(iteration, desc="Solving backward ODE for h and gradients\n", ncols=80)

        for sub_t_left, sub_t_right in iteration:

            def ode_system_segment(t, y):
                if t == t_left:
                    return ode_system(t, y, left=False)
                return ode_system(t, y, left=True)

            sol = solve_ivp(
                fun=ode_system_segment,
                t_span=[sub_t_right, sub_t_left],  # backward
                y0=y,
                method=self.method,
                rtol=self.rtol,
                atol=self.atol,
            )
            y = sol.y[:, -1]

        h_initial = y[slice_h].reshape(d, r)
        h_grad_initial = np.stack([y[s].reshape(d, r) for s in slice_grad], axis=0)  # shape (p, d, r)

        return h_initial, h_grad_initial

