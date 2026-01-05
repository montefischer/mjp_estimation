import time
import logging

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from tqdm import tqdm

from .base_functionality import MJPModel, Simulator, FLOAT, SamplePath, DiscreteSample, JointKolmogorovSolver
from .amortization import construct_amortization_groups_vectorized, process_all_groups
from .endo_jackson import JacksonEndogenousSubmodel


OBJECTIVE_REGISTRY = {}
def register_objective(cls):
    """Decorator that registers the class by its name in CLASS_REGISTRY."""
    OBJECTIVE_REGISTRY[cls.__name__] = cls
    return cls


@register_objective
class ContinuousTimeLikelihood:
    def __init__(self, model: MJPModel, path: SamplePath, fixed_step: float = 1e-2):
        self.model = model
        self.path = path
        self.fixed_step = fixed_step
    
    def _eval(self, theta, include_scores=True):
        p = len(theta)
        num_observations = len(self.path.times)
        logL = 0.
        grad = np.zeros(p, dtype=FLOAT)
        if include_scores:
            scores = np.zeros((num_observations - 1, p), dtype=FLOAT)

        for k in range(1, num_observations):
            exit_rate_val, exit_rate_grad = self.model.integrate_exit_rate_and_grad_at_state(
                theta, self.path.times[k-1], self.path.times[k], self.path.states[k-1]
            )
            logL += (-1.) * exit_rate_val
            grad += (-1.) * exit_rate_grad
            col_indices, rates = self.model.get_Q_row_compact(self.path.states[k-1], theta, self.path.times[k])
            # Find the rate for transition to states[k]
            transition_mask = col_indices == self.path.states[k]
            if np.any(transition_mask):
                Q_at_transition = rates[transition_mask][0]
            else:
                Q_at_transition = 0.0  # No transition exists

            if Q_at_transition < 0:
                print("NEGATIVE Q AT TRANSITION")
                print(f"{theta=}")
                print(f"{self.path.times[k]=}")
                print(f"{self.path.states[k-1]=}")
                print(f"{self.path.states[k]=}")
                raise ValueError("Negative value of Q observed at transition time, indicating a problem with the Q matrix.")
            logL += np.log(Q_at_transition)
            col_indices, gradient_rates = self.model.get_Q_grad_row_compact(
                self.path.states[k-1], theta, self.path.times[k]
            )
            # Find which transition corresponds to states[k]
            transition_idx = np.where(col_indices == self.path.states[k])[0]
            if len(transition_idx) > 0:
                s = gradient_rates[transition_idx[0], :] / Q_at_transition
            else:
                s = np.zeros(len(theta))  # No transition exists
            grad += s
            if include_scores:
                scores[k-1] = s

        # integrate final time
        if self.path.times[-1] < self.path.final_time:
            final_exit_val, final_exit_grad = self.model.integrate_exit_rate_and_grad_at_state(
                theta, self.path.times[-1], self.path.final_time, self.path.states[-1]
            )
            logL += (-1.) * final_exit_val
            grad += (-1.) * final_exit_grad
        
        if include_scores:
            return -logL, -grad, scores
        else:
            return -logL, -grad

    def eval_log_likelihood(self, theta):
        return self._eval(theta, include_scores=False)

    def eval_log_likelihood_scores(self, theta):
        return self._eval(theta, include_scores=True)

@register_objective
class DiscreteLikelihood:
    def __init__(self, model: MJPModel, discrete_path: DiscreteSample, fixed_step=1e-2, rtol=1e-4, atol=1e-6):
        self.model = model
        self.path = discrete_path
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.fixed_step = fixed_step
        self.rtol = rtol
        self.atol = atol

        self.evaluation_method = self.eval_log_likelihood_scores
    
    def eval_log_likelihood(self, theta):
        p = len(theta)
        solver = JointKolmogorovSolver(
            self.model, 
            theta, 
            self.fixed_step,
            rtol=self.rtol,
            atol=self.atol
        )

        logL = 0.
        gradLogL = np.zeros(p)

        for k in range(1, len(self.path.times)):
            kfe = solver.forward_solve(self.path.times[k-1], self.path.times[k], self.path.states[k-1], self.model.get_discontinuity_times(self.path.times[k-1], self.path.times[k])) 
            strided = kfe[self.path.states[k]::self.model.state_space_size]
            logL += np.log(strided[0])
            gradLogL += strided[1:] / strided[0]
        
        return -logL, -gradLogL
    
    def eval_log_likelihood_scores(self, theta, fixed_step=1e-2):
        p = len(theta)
        solver = JointKolmogorovSolver(
            self.model, 
            theta,
            fixed_step, 
            display_tqdm=False,
            rtol=self.rtol,
            atol=self.atol
        )

        logL = 0.
        gradLogL = np.zeros(p)

        scores = np.empty(((len(self.path.times) - 1), p))

        for k in range(1, len(self.path.times)):
            kfe = solver.forward_solve(self.path.times[k-1], self.path.times[k], self.path.states[k-1], self.model.get_discontinuity_times(self.path.times[k-1], self.path.times[k]))
            self.log.info(f"{kfe=}")
            self.log.info(f"{self.path.states[k]=}")
            strided = kfe[self.path.states[k]::self.model.state_space_size]
            if strided[0] < 0:
                print(f"Negative probability at {k=}, {strided[0]=}")
                print(k, self.path.times[k-1], self.path.times[k], self.path.states[k-1])
                print('probs', kfe[solver.slice_prob])
                print(strided)
                print("Attempting solve with high precision")
                kfe = solver.forward_solve(self.path.times[k-1], self.path.times[k], self.path.states[k-1], self.model.get_discontinuity_times(self.path.times[k-1], self.path.times[k]), max_step=1e-4)
                strided = kfe[self.path.states[k]::self.model.state_space_size]
                if strided[0] < 0:
                    print(f"High precision solve failed: {strided[0]=}")

            self.log.info(f"{strided=}")
            logL += np.log(strided[0])
            gradLogL += strided[1:] / strided[0]
            scores[k-1] = strided[1:] / strided[0]
        
        self.log.info(f"{logL=}, {gradLogL=}")
        return -logL, -gradLogL, scores

    def f(self, theta):
        f, grad, scores = self.eval_log_likelihood_scores(theta)
        return f

    def grad(self, theta):
        f, grad, scores = self.eval_log_likelihood_scores(theta)
        return grad

    def scores(self, theta):
        f, grad, scores = self.eval_log_likelihood_scores(theta)
        return scores

@register_objective
class DiscreteLikelihoodPeriodicAmortization:
    """
    Solve full-batch ODEs using cost-amortization based on model periodicities.
    """
    def __init__(
            self,
            model: MJPModel,
            discrete_path: DiscreteSample,
            period: float,
            fixed_step=1e-2,
            rtol=1e-4,
            atol=1e-6,
            verbose=False
            ):
        self.model = model
        self.path = discrete_path
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.fixed_step = fixed_step
        self.rtol = rtol
        self.atol = atol

        self.period = period

        tstart = time.perf_counter()
        self.amortization_groups = process_all_groups(self.path.states, self.path.times, self.period)
        tend = time.perf_counter()
        self.optimal_schedule_compute_time = tend - tstart
        naive_kfe = 0
        naive_kbe = 0
        hk_optimal = 0
        for (s,t), (A, forward_indices, backward_indices) in self.amortization_groups.items():
            hk_optimal += len(forward_indices) + len(backward_indices)
            naive_kfe += np.sum(A.sum(axis=0) != 0) # number of naive backward solves
            naive_kbe += (np.sum(A.sum(axis=1) != 0)) # number of naive forward solves
        self.hk_optimal = hk_optimal
        self.naive_kfe = naive_kfe 
        self.naive_kbe = naive_kbe
        self.efficiency_gain = 1 - hk_optimal / min(naive_kfe, naive_kbe)
        print(f"efficiency gain: {self.efficiency_gain*100:.2f}%; # Optimal Solves: {hk_optimal}; # KFE-Only Solves: {naive_kfe}, # KBE-Only Solves: {naive_kbe}, # Samples: {len(self.path.times)}")


        self.evaluation_method = self.eval_log_likelihood_scores

        self.cached_theta = None
        self.cached_result = None

        self.verbose = verbose

    
    def eval_log_likelihood(self, theta):
        neg_logL, neg_gradLogL, scores = self.eval_log_likelihood_scores(theta) 
        return neg_logL, neg_gradLogL
    
    def eval_log_likelihood_scores(self, theta, fixed_step=1e-2):
        if np.all(np.equal(theta, self.cached_theta)):
            return self.cached_result
        p = len(theta)
        solver = JointKolmogorovSolver(
            self.model, 
            theta,
            fixed_step, 
            display_tqdm=False,
            rtol=self.rtol,
            atol=self.atol
        )

        logL = 0.
        gradLogL = np.zeros(p)

        scores = np.empty(((len(self.path.times) - 1), p))
        scores_idx = 0

        if self.verbose:
            iterator = tqdm(self.amortization_groups.items())
        else:
            iterator = self.amortization_groups.items()

        for (s,t), (A, forward_indices, backward_indices) in iterator:
            Acopy = A.copy()
            if self.verbose:
                forward_indices_iter = tqdm(forward_indices)
            else:
                forward_indices_iter = forward_indices
            for i in forward_indices_iter:
                kfe = solver.forward_solve(s, t, i, self.model.get_discontinuity_times(s, t)) 
                # solve KFE(s,t,i)
                row_start = Acopy.indptr[i]
                row_end = Acopy.indptr[i+1]
                for idx in range(row_start, row_end):
                    row_idx = Acopy.indices[idx]
                    count = Acopy.data[idx]
                    # process
                    strided = kfe[row_idx::self.model.state_space_size]
                    if strided[0] < 0:
                        print(f"Negative probability at {row_idx=}")
                    logL += count * np.log(strided[0])
                    score = strided[1:] / strided[0]
                    gradLogL += count * score
                    for _ in range(int(count)):
                        scores[scores_idx] = score
                        scores_idx +=1 
                    Acopy.data[idx] = 0 # avoid double-counting

            Acopy = Acopy.tocsc()
            for j in backward_indices:
                # solve KBE(s,t,j)
                kbe = solver.backward_solve(s, t, j, [])
                col_start = Acopy.indptr[j]
                col_end = Acopy.indptr[j+1]
                for idx in range(col_start, col_end):
                    col_idx = Acopy.indices[idx]
                    count = Acopy.data[idx]
                    if count > 0:
                        # process
                        strided = kbe[col_idx::self.model.state_space_size]
                        logL += count * np.log(strided[0])
                        score = strided[1:] / strided[0]
                        gradLogL += count * score
                        for _ in range(int(count)):
                            scores[scores_idx] = score
                            scores_idx +=1 

        self.log.info(f"{logL=}, {gradLogL=}")
        self.cached_theta = theta 
        self.cached_result = -logL, -gradLogL, scores
        return -logL, -gradLogL, scores
    
    def f(self, theta):
        f, grad, scores = self.eval_log_likelihood_scores(theta)
        return f

    def grad(self, theta):
        f, grad, scores = self.eval_log_likelihood_scores(theta)
        return grad

    def scores(self, theta):
        f, grad, scores = self.eval_log_likelihood_scores(theta)
        return scores




@register_objective
class DiscreteLikelihoodSimulationFullBatch:

    def __init__(self, model: MJPModel, discrete_path: DiscreteSample, fixed_step=1e-2, rtol=1e-4, atol=1e-6):
        self.model = model
        self.path = discrete_path
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.fixed_step = fixed_step
        self.rtol = rtol
        self.atol = atol
        self.simulator = Simulator(self.model)

        self.evaluation_method = self.eval_log_likelihood_scores

        self.name = "Discrete (Simulation)"
    

    def eval_log_likelihood_scores(self, theta, num_conditional_samples=1):
        p = len(theta)

        scores = []

        logp = 0
        scores = np.zeros((len(self.path.times)-1, p))

        score_total = np.zeros_like(theta)
        for k in (range(0, len(self.path.times)-1)):
            hits = 0
            n = 0
            is_hit = False
            score_k = np.zeros_like(theta)
            while hits < num_conditional_samples:
                Lambda_k = self.path.times[k]
                Lambda_kp1 = self.path.times[k+1]
                X_k = self.path.states[k]
                X_kp1 = self.path.states[k+1]
                path = self.simulator.simulate_single_run(
                    Lambda_kp1, 
                    starting_time=Lambda_k, 
                    starting_state=X_k, 
                    theta=theta
                )

                indicator = 1. * (path.final_state == X_kp1)
                is_hit = is_hit or (indicator == 1)
                if indicator == 1:
                    cl = ContinuousTimeLikelihood(self.model, path)
                    _, score = cl.eval_log_likelihood(theta)
                    score_k += -score
                    hits += 1
                n += 1
                
            prob = hits / n

            logp += np.log(prob)
            score_total += score_k / hits
            scores[k] = score_k / hits

        return -logp, -score_total, scores


class DiscreteLikelihoodSimulationSingle:
    def __init__(self, model: MJPModel, discrete_path: DiscreteSample, fixed_step=1e-2, rtol=1e-4, atol=1e-6):
        self.model = model
        self.path = discrete_path
        self.log = logging.getLogger("DiscreteLikelihood")
        self.fixed_step = fixed_step
        self.rtol = rtol
        self.atol = atol
        self.simulator = Simulator(self.model)
        self.evaluation_method = self.eval_log_likelihood_scores
        self.name = "Discrete (Simulation)"

    def eval_log_likelihood_scores(self, theta):
        p = len(theta)

        logp = 0
        score_total = np.zeros_like(theta)
        scores = np.zeros((len(self.path.times)-1, p))
        k = np.random.randint(0, len(self.path.times)-1)
        hits = 0.
        n = 0
        is_hit = False
        while (not is_hit):
            Lambda_k = self.path.times[k]
            Lambda_kp1 = self.path.times[k+1]
            X_k = self.path.states[k]
            X_kp1 = self.path.states[k+1]
            path = self.simulator.simulate_single_run(
                Lambda_kp1, 
                starting_time=Lambda_k, 
                starting_state=X_k, 
                theta=theta
            )
            indicator = 1. * (path.final_state == X_kp1)
            if indicator == 1:
                is_hit = True
                cl = ContinuousTimeLikelihood(self.model, path)
                _, score = cl.eval_log_likelihood(theta)
                score = -score
                hits += 1
            n += 1
            

        prob = hits / n
        logp += np.log(prob)
        score_total += score
        scores[k] = score

        return -logp, -score_total, scores


@register_objective
class DiscreteLikelihoodMiniBatchODE:
    def __init__(self, model: MJPModel, discrete_path: DiscreteSample, batch_size: int = 10,):
        self.model = model
        self.path = discrete_path
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.simulator = Simulator(self.model)
        
        self.batch_size = batch_size

        self.evaluation_method = self.eval_log_likelihood_scores

        self.name = "Discrete (Simulation)"


    def eval_log_likelihood_scores(self, theta, num_samples=1):
        p = len(theta)
        logp = 0

        scores = np.zeros((self.batch_size, p))
        for i in range(self.batch_size):
            k = np.random.randint(0, len(self.path.times)-1)
            Lambda_k = self.path.times[k]
            Lambda_kp1 = self.path.times[k+1]
            X_k = self.path.states[k]
            X_kp1 = self.path.states[k+1]

            solver = JointKolmogorovSolver(
                self.model, 
                theta,
                fixed_step=1e-2,
                display_tqdm=False,
            )

            kfe = solver.forward_solve(Lambda_k, Lambda_kp1, X_k, self.model.get_discontinuity_times(Lambda_k, Lambda_kp1)) 
            strided = kfe[X_kp1::self.model.state_space_size]
            prob = strided[0]
            gradLogL = strided[1:] / strided[0]
            logp += np.log(prob)
            if np.isnan(np.log(prob)):
                print(f"Batch: {i+1}/{self.batch_size}. Index {k}. {Lambda_k=}, {Lambda_kp1=}, {X_k=}, {X_kp1=}")
                print(f"{np.log(prob)=}, {strided=}")
            scores[i] = gradLogL


        return -logp / self.batch_size, -scores.mean(axis=0), scores


@register_objective
class DiscreteLikelihoodMiniBatchODEWithoutReplacement:
    def __init__(self, model: MJPModel, discrete_path: DiscreteSample, batch_size: int = 10):
        self.model = model
        self.path = discrete_path
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.simulator = Simulator(self.model)
        
        self.batch_size = batch_size
        self.evaluation_method = self.eval_log_likelihood_scores
        self.name = "Discrete (Simulation)"

        # --- New internal minibatch attributes ---
        self._all_indices = np.arange(len(self.path.times) - 1)  # indices for transitions
        self._current_pos = 0
        self._shuffled_indices = None
        self._reshuffle_indices()

    def _reshuffle_indices(self):
        """Shuffle the indices at the beginning of each epoch."""
        self._shuffled_indices = np.random.permutation(self._all_indices)
        self._current_pos = 0

    def _get_next_batch_indices(self):
        """Return exactly `batch_size` indices, wrapping across epoch boundaries without dropping."""
        N = len(self._shuffled_indices)
        start = self._current_pos
        end = start + self.batch_size

        if end <= N:
            batch = self._shuffled_indices[start:end]
            self._current_pos = end
            if self._current_pos == N:
                # reached end; reshuffle for next call
                self._reshuffle_indices()
            return batch

        # Need to wrap: take the tail of current epoch, then head of next epoch
        tail = self._shuffled_indices[start:N]
        self._reshuffle_indices()  # new permutation for the next epoch
        head_needed = self.batch_size - len(tail)
        head = self._shuffled_indices[0:head_needed]
        self._current_pos = head_needed
        return np.concatenate([tail, head])


    def eval_log_likelihood_scores(self, theta, num_samples=1):
        p = len(theta)
        logp = 0.0
        scores = np.zeros((self.batch_size, p))
        batch_indices = self._get_next_batch_indices()

        for i, k in enumerate(batch_indices):
            Lambda_k = self.path.times[k]
            Lambda_kp1 = self.path.times[k + 1]
            X_k = self.path.states[k]
            X_kp1 = self.path.states[k + 1]

            solver = JointKolmogorovSolver(
                self.model,
                theta,
                fixed_step=1e-2,
                display_tqdm=False,
            )

            kfe = solver.forward_solve(Lambda_k, Lambda_kp1, X_k, [])
            strided = kfe[X_kp1::self.model.state_space_size]
            prob = strided[0]
            gradLogL = strided[1:] / strided[0]
            logp += np.log(prob)
            if np.isnan(np.log(prob)):
                print(f"Batch: {i+1}/{self.batch_size}. Index {k}. {Lambda_k=}, {Lambda_kp1=}, {X_k=}, {X_kp1=}")
                print(f"{np.log(prob)=}, {strided=}")
            scores[i] = gradLogL

        return -logp / self.batch_size, -scores.mean(axis=0), scores


@register_objective
class DiscreteLikelihoodSimulationMiniBatch:
    def __init__(self, model: MJPModel, discrete_path: DiscreteSample, batch_size: int = 10, max_tries: int = 10_000):
        self.model = model
        self.path = discrete_path
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.simulator = Simulator(self.model)
        self.batch_size = batch_size
        self.evaluation_method = self.eval_log_likelihood_scores
        self.name = "Discrete (Simulation)"
        self.max_tries = max_tries
    

    def eval_log_likelihood_scores(self, theta, num_samples=1):
        p = len(theta)
        logp = 0

        scores = np.zeros((self.batch_size, p))
        for i in range(self.batch_size):
            k = np.random.randint(0, len(self.path.times)-1)
            hits = 0
            n = 0
            is_hit = False
            score_i = np.zeros_like(theta)
            Lambda_k = self.path.times[k]
            Lambda_kp1 = self.path.times[k+1]
            X_k = self.path.states[k]
            X_kp1 = self.path.states[k+1]

            while hits < num_samples:
                if n > self.max_tries:
                    k = np.random.randint(0, len(self.path.times) - 1)
                    Lambda_k = self.path.times[k]
                    Lambda_kp1 = self.path.times[k + 1]
                    X_k = self.path.states[k]
                    X_kp1 = self.path.states[k + 1]
                    n = 0  # reset counter
                    print(f"Warning: Re-drawing k after {self.max_tries} failed attempts. New index {k}. {Lambda_k=}, {Lambda_kp1=}, {X_k=}, {X_kp1=}")

                sim_path = self.simulator.simulate_single_run(
                    Lambda_kp1, 
                    starting_time=Lambda_k, 
                    starting_state=X_k, 
                    theta=theta
                )

                indicator = 1. * (sim_path.final_state == X_kp1)
                is_hit = is_hit or (indicator == 1)
                if indicator == 1:
                    cl = ContinuousTimeLikelihood(self.model, sim_path)
                    _, score = cl.eval_log_likelihood(theta)
                    score_i += -score
                    hits += 1
                n += 1
                
            prob = hits / n

            logp += np.log(prob)
            scores[i] = score_i / hits

        return -logp, -scores.sum(axis=0), scores



@register_objective
class DiscreteLikelihoodSimulationMiniBatchWithoutReplacement:
    def __init__(self, model: MJPModel, discrete_path: DiscreteSample, batch_size: int = 10, max_tries: int = 10_000):
        self.model = model
        self.path = discrete_path
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.simulator = Simulator(self.model)

        self.max_tries = max_tries
        
        self.batch_size = batch_size
        self.evaluation_method = self.eval_log_likelihood_scores
        self.name = "Discrete (Simulation)"

        self._all_indices = np.arange(len(self.path.times) - 1)  # indices for transitions
        self._current_pos = 0
        self._shuffled_indices = None
        self._reshuffle_indices()

    def _reshuffle_indices(self):
        """Shuffle the indices at the beginning of each epoch."""
        self._shuffled_indices = np.random.permutation(self._all_indices)
        self._current_pos = 0

    def _get_next_batch_indices(self):
        """Return exactly `batch_size` indices, wrapping across epoch boundaries without dropping."""
        N = len(self._shuffled_indices)
        start = self._current_pos
        end = start + self.batch_size

        if end <= N:
            batch = self._shuffled_indices[start:end]
            self._current_pos = end
            if self._current_pos == N:
                # reached end cleanly → reshuffle for next call
                self._reshuffle_indices()
            return batch

        # Need to wrap: take the tail of current epoch, then head of next epoch
        tail = self._shuffled_indices[start:N]
        self._reshuffle_indices()  # new permutation for the next epoch
        head_needed = self.batch_size - len(tail)
        head = self._shuffled_indices[0:head_needed]
        self._current_pos = head_needed
        return np.concatenate([tail, head])


    def eval_log_likelihood_scores(self, theta, num_samples=1):
        p = len(theta)
        logp = 0.0
        scores = np.zeros((self.batch_size, p))

        batch_indices = self._get_next_batch_indices()

        for i, k in enumerate(batch_indices):
            Lambda_k = self.path.times[k]
            Lambda_kp1 = self.path.times[k + 1]
            X_k = self.path.states[k]
            X_kp1 = self.path.states[k + 1]

            
            hits = 0
            n = 0
            is_hit = False
            score_i = np.zeros_like(theta)
            Lambda_k = self.path.times[k]
            Lambda_kp1 = self.path.times[k+1]
            X_k = self.path.states[k]
            X_kp1 = self.path.states[k+1]
            print(f"Batch: {i+1}/{self.batch_size}. Index {k}. {Lambda_k=}, {Lambda_kp1=}, {X_k=}, {X_kp1=}")

            while hits < num_samples:
                if n > self.max_tries:
                    k = np.random.randint(0, len(self.path.times) - 1)
                    Lambda_k = self.path.times[k]
                    Lambda_kp1 = self.path.times[k + 1]
                    X_k = self.path.states[k]
                    X_kp1 = self.path.states[k + 1]
                    n = 0  # reset counter
                    print(f"️Warning: Re-drawing k after {self.max_tries} failed attempts. New index {k}. {Lambda_k=}, {Lambda_kp1=}, {X_k=}, {X_kp1=}")

                sim_path = self.simulator.simulate_single_run(
                    Lambda_kp1, 
                    starting_time=Lambda_k, 
                    starting_state=X_k, 
                    theta=theta
                )

                indicator = 1. * (sim_path.final_state == X_kp1)
                is_hit = is_hit or (indicator == 1)
                if indicator == 1:
                    cl = ContinuousTimeLikelihood(self.model, sim_path)
                    _, score = cl.eval_log_likelihood(theta)
                    score_i += -score
                    hits += 1
                n += 1
                
            prob = hits / n

            logp += np.log(prob)
            scores[i] = score_i / hits

        return -logp, -scores.sum(axis=0), scores

@register_objective
class DiscreteLikelihoodSGD:
    def __init__(self, model: MJPModel, discrete_path: DiscreteSample, fixed_step=1e-2, rtol=1e-4, atol=1e-6):
        self.model = model
        self.path = discrete_path
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.fixed_step = fixed_step
        self.rtol = rtol
        self.atol = atol

        self.name = "Discrete (ODE-based SGD)"
        self.evaluation_method = self.eval_log_likelihood_scores
    
    def eval_log_likelihood(self, theta):
        logL, gradLogL, score = self.eval_log_likelihood_scores(theta)
        return logL, gradLogL
    
    def eval_log_likelihood_scores(self, theta, fixed_step=1e-2, num_samples=None):
        p = len(theta)
        solver = JointKolmogorovSolver(
            self.model, 
            theta,
            fixed_step, 
            display_tqdm=False,
            rtol=self.rtol,
            atol=self.atol
        )

        logL = 0.
        gradLogL = np.zeros(p)

        k = np.random.randint(0, len(self.path.times))
        kfe = solver.forward_solve(self.path.times[k-1], self.path.times[k], self.path.states[k-1], self.model.get_discontinuity_times(self.path.times[k-1], self.path.times[k]))
        strided = kfe[self.path.states[k]::self.model.state_space_size]
        logL += np.log(strided[0])
        gradLogL += strided[1:] / strided[0]
        
        self.log.info(f"{strided=}")
        self.log.info(f"{kfe=}")
        self.log.info(f"{self.path.states[k]=}")
        self.log.info(f"{logL=}, {gradLogL=}")
        return -logL, -gradLogL, None

@register_objective
class ConditionalLeastSquares:
    def __init__(self, model: MJPModel, discrete_path: DiscreteSample, h: np.ndarray, fixed_step=1e-2):
        self.model = model
        self.path = discrete_path
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.h = h # (d x p) matrix
        self.fixed_step = fixed_step

        self.name = "CLS (ODE)"

        self.cached_theta = None
        self.cached_residuals = None
        self.cached_jacobian = None

        self.nobs = len(discrete_path) - 1


    def eval_residuals_and_jacobian(self, theta):
        if np.all(np.equal(theta, self.cached_theta)):
            return self.cached_residuals, self.cached_jacobian

        p = len(theta)
        solver = JointKolmogorovSolver(
            self.model,
            theta,
            display_tqdm=False
        )
        n = len(self.path.times)-1
        residuals = np.zeros(n)
        J = np.zeros((n, p))
        for k in range(n):
            kfe = solver.forward_solve(self.path.times[k], self.path.times[k+1], self.path.states[k], self.model.get_discontinuity_times(self.path.times[k], self.path.times[k+1]))
            Eh = kfe[solver.slice_prob] @ self.h # produces p dim vector 
            residuals[k] = self.h[self.path.states[k+1]] - Eh # p dim
            grad = np.stack([kfe[grad_i] for grad_i in solver.slice_grad]) # produces p x d matrix  
            J[k] = (grad @ self.h).T # results in m x p matrix (h is d x m, i.e. maps each state to m dimensional vector)

        self.cached_theta = theta
        self.cached_residuals = residuals
        self.cached_jacobian = J
        return residuals, J
    
    def eval_ssr(self, theta, fixed_step=None):
        residuals, J = self.eval_residuals_and_jacobian(theta)
        return np.sum(np.power(residuals, 2)) / self.nobs

    def eval_ssr_grad_ssr(self, theta):
        residuals, J = self.eval_residuals_and_jacobian(theta)
        print(f"{residuals=}")
        # ssr = np.sum(np.power(residuals, 2))
        ssr = residuals.T @ residuals
        grad_ssr = -2 * J.T @ residuals 
        return ssr / self.nobs, grad_ssr / self.nobs, None

    def residuals(self, theta):
        residuals, J = self.eval_residuals_and_jacobian(theta)
        return residuals
 
    def jac(self, theta):
        residuals, J = self.eval_residuals_and_jacobian(theta)
        return J

    def clear_cache(self):
        del self.cached_theta
        del self.cached_residuals
        del self.cached_jacobian

    def A_matrix(self, theta):
        J = self.jac(theta)
        return J.T @ J

    def B_matrix(self, theta):
        """matrix associated with predictable qv (angle brackets process)"""
        p = len(theta)
        solver = JointKolmogorovSolver(
            self.model,
            theta,
            display_tqdm=False
        )
        n = len(self.path.times)-1

        Bmat = np.zeros((p,p))
        for k in range(n):
            Lambda_k, Lambda_kp1, X_k, X_kp1 = self.path.times[k], self.path.times[k+1], self.path.states[k], self.path.states[k+1]
            disc_times_k = self.model.get_discontinuity_times(Lambda_k, Lambda_kp1)
            kbe_h, kbe_h_grad = solver.backward_solve_h(Lambda_k, Lambda_kp1, self.h.reshape(-1,1), disc_times_k)
            kbe_h2, _ = solver.backward_solve_h(
                Lambda_k, Lambda_kp1, (self.h.reshape(-1, 1))**2, disc_times_k
            )
            kbe_var = kbe_h2 - kbe_h**2
            grad = kbe_h_grad[:, X_k, 0] 
            Bmat += kbe_var[X_k] * np.outer(grad, grad)

        return Bmat

    def C2(self, theta):
        # Computes: 4 * sum_{k=1}^n D_k(theta)' (r(theta, k))^2 D_k(theta)
        res, J = self.eval_residuals_and_jacobian(theta)
        # J: n x p matrix
        # res: n dimensional vector
        n = len(self.path.times)-1
        p = len(theta)
        C2 = np.zeros((p,p))
        for k in range(n):
            C2 += np.outer(J[k], J[k]) * np.pow(res[k], 2)
        return C2 # constant factor 4 omitted, balances omission for A.

    def prec_matrix(self, theta, qv_type='square'):
        # B A^{-1} B
        A = self.A_matrix(theta)
        if qv_type == 'angle':
            B = self.B_matrix(theta)
        elif qv_type == 'square':
            B = self.C2(theta)
        else:
            raise NotImplementedError(f"{qv_type=} not implemented in ConditionalLeastSquares.prec_matrix")

        # Regularize if B is near-singular
        eps=1e-8
        B_reg = B + eps * np.eye(B.shape[0])

        try:
            # Cholesky factorization (fastest and most stable if SPD)
            L = cholesky(B_reg, lower=True)
            # Solve L Y = A.T
            Y = solve_triangular(L, A.T, lower=True)
            # Solve L^T X = Y
            X = solve_triangular(L.T, Y, lower=False)
        except np.linalg.LinAlgError:
            # If B isn't SPD (e.g., numerical issues), fall back to general solver
            X = np.linalg.solve(B_reg, A.T)

        return A @ X


@register_objective
class ConditionalLeastSquaresPeriodicAmortization:
    def __init__(self, model: MJPModel, discrete_path: DiscreteSample, h: np.ndarray, fixed_step=1e-2):
        self.model = model
        self.path = discrete_path
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.h = h # d-dim vector matrix
        self.fixed_step = fixed_step
        self.discontinuity_times = []

        self.name = "CLS (ODE) Periodic Amortization"

        self.cached_theta = None
        self.cached_residuals = None
        self.cached_jacobian = None

        time_start = time.perf_counter()
        self.agroups = construct_amortization_groups_vectorized(discrete_path.states, discrete_path.times, self.model.period)
        time_end = time.perf_counter()
        self.agroups_computation_time = time_end - time_start

        amortization_groups = process_all_groups(self.path.states, self.path.times, self.model.period)
        hk_optimal = 0
        for (s,t), (A, forward_indices, backward_indices) in amortization_groups.items():
            hk_optimal += len(forward_indices) + len(backward_indices)

        self.efficiency_factor = 1 - len(self.agroups.keys()) / hk_optimal

        self.nobs = len(discrete_path) - 1

    def eval_ssr(self, theta, fixed_step=None):
        residuals, J = self.eval_residuals_and_jacobian(theta)
        return np.sum(np.power(residuals, 2)) / self.nobs

    def eval_ssr_grad_ssr(self, theta):
        residuals, J = self.eval_residuals_and_jacobian(theta)
        ssr = np.sum(np.power(residuals, 2))
        grad_ssr = -2 * J.T @ residuals 
        return ssr / self.nobs, grad_ssr / self.nobs, None 

    def eval_residuals_and_jacobian(self, theta):
        if np.all(np.equal(theta, self.cached_theta)):
            return self.cached_residuals, self.cached_jacobian

        p = len(theta)
        solver = JointKolmogorovSolver(
            self.model,
            theta,
            display_tqdm=False
        )
        n = len(self.path.times)-1
        residuals = np.zeros(n)
        J = np.zeros((n, p))
        
        k = 0
        for (s,t), (Xs, Ys) in self.agroups.items():
            kbe_h, kbe_h_grad = solver.backward_solve_h(s, t, self.h.reshape(-1,1), self.discontinuity_times)
            for (x, y) in zip(Xs, Ys):
                Eh = kbe_h[x, 0] # slice_prob here is actually the E h slice (i.e. NOT sensitivities)
                frak_h = self.h[y] - Eh
                grad = kbe_h_grad[:, x, 0] 
                
                residuals[k] = frak_h 
                J[k] = grad
                k += 1

        self.cached_theta = theta
        self.cached_residuals = residuals
        self.cached_jacobian = J
        return residuals, J

    def residuals(self, theta):
        residuals, J = self.eval_residuals_and_jacobian(theta)
        return residuals
 
    def jac(self, theta):
        residuals, J = self.eval_residuals_and_jacobian(theta)
        return J
    
    def A_matrix(self, theta):
        J = self.jac(theta)
        return J.T @ J

    def B_matrix(self, theta):
        p = len(theta)
        solver = JointKolmogorovSolver(
            self.model,
            theta,
            display_tqdm=False
        )
        n = len(self.path.times)-1

        Bmat = np.zeros((p,p))
        for (s,t), (Xs, Ys) in self.agroups.items():
            kbe_h, kbe_h_grad = solver.backward_solve_h(s, t, self.h.reshape(-1,1), self.discontinuity_times)
            kbe_h2, _ = solver.backward_solve_h(
                s, t, (self.h.reshape(-1, 1))**2, self.discontinuity_times
            )
            kbe_var = kbe_h2 - kbe_h**2
            for x in Xs:
                grad = kbe_h_grad[:, x, 0] 
                Bmat += kbe_var[x] * np.outer(grad, grad)

        return Bmat

    def C2(self, theta):
        # 4 * sum_{k=1}^n D_k(theta)' (r(theta, k))^2 D_k(theta)
        res, J = self.eval_residuals_and_jacobian(theta)
        # J: n x p matrix
        # res: n dimensional vector
        n = len(self.path.times)-1
        p = len(theta)
        C2 = np.zeros((p,p))
        for k in range(n):
            C2 += np.outer(J[k], J[k]) * np.pow(res[k], 2)
        return C2 # constant factor 4 omitted, balances omission for A.


    def prec_matrix(self, theta, qv_type='square'):
        # B A^{-1} B
        A = self.A_matrix(theta)
        if qv_type == 'angle':
            B = self.B_matrix(theta)
        elif qv_type == 'square':
            B = self.C2(theta)
        else:
            raise NotImplementedError(f"{qv_type=} not implemented in ConditionalLeastSquaresPeriodicAmortization.prec_matrix")
        
        # Regularizeif B is near-singular
        eps=1e-8
        B_reg = B + eps * np.eye(B.shape[0])

        try:
            # Cholesky factorization (fastest and most stable if SPD)
            L = cholesky(B_reg, lower=True)
            # Solve L Y = A.T
            Y = solve_triangular(L, A.T, lower=True)
            # Solve L^T X = Y
            X = solve_triangular(L.T, Y, lower=False)
        except np.linalg.LinAlgError:
            # If B isn't SPD (e.g., numerical issues), fall back to general solver
            X = np.linalg.solve(B_reg, A.T)

        return A @ X


class SimulatedConditionalLeastSquares:
    def __init__(self, model: MJPModel, discrete_path: DiscreteSample, h: np.ndarray):
        self.model = model
        self.path = discrete_path
        self.log = logging.getLogger("DiscreteLikelihood")
        self.h = h # d-dimensional array, map from states to R+
        self.discontinuity_times = []
        self.evaluation_method = self.eval_sum_least_squares

        self.simulator = Simulator(model)
        self.name='CLS (Simulated)'

        self.nobs = len(discrete_path) - 1

    
    def eval_sum_least_squares(self, theta, num_samples=20):
        target = 0.
        grad = np.zeros_like(theta)

        for k in range(0, len(self.path.times) - 1):
            h_sample_conditional_sum = np.zeros(self.h.shape[1])
            lr_grad_sum = np.zeros((self.h.shape[1], theta.shape[0]))
            for _ in (range(num_samples)):
                Lambda_k = self.path.times[k]
                Lambda_kp1 = self.path.times[k+1]
                X_k = self.path.states[k]
                X_kp1 = self.path.states[k+1]
                simulated_path = self.simulator.simulate_single_run(
                    Lambda_kp1, 
                    starting_time=Lambda_k, 
                    starting_state=X_k, 
                    theta=theta
                )
                h_sample_conditional_sum += self.h[simulated_path.final_state]

                simulated_path_2 = self.simulator.simulate_single_run(
                    Lambda_kp1, 
                    starting_time=Lambda_k, 
                    starting_state=X_k, 
                    theta=theta
                )

                cl = ContinuousTimeLikelihood(self.model, simulated_path_2)
                _, nabla = cl.eval_log_likelihood(theta)
                nabla = -nabla # dimension p

                lr_grad_sum += np.outer(self.h[simulated_path_2.final_state], nabla)

            difference = self.h[X_kp1] - h_sample_conditional_sum / num_samples
            incremental_sum_squares = difference.T @ difference

            target += incremental_sum_squares
            grad += -2 * (lr_grad_sum / num_samples).T @ difference

        return target / self.nobs, grad / self.nobs, None


class SimulatedConditionalLeastSquaresODE:
    def __init__(
            self, 
            model: MJPModel, 
            discrete_path: DiscreteSample, 
            h: np.ndarray,
            batch_size: int = 1,
        ):
        self.model = model
        self.path = discrete_path
        self.log = logging.getLogger("DiscreteLikelihood")
        self.h = h # d-dimensional array, map from states to R+
        self.discontinuity_times = []
        # self.evaluation_method = self.eval_sum_least_squares

        self.simulator = Simulator(model)
        self.name='CLS (Simulated)'

        self.batch_size = batch_size

    def eval_residuals_and_jacobian(self, theta, mc_samples_per_observation=1):
        p = len(theta)

        residuals = np.zeros(self.batch_size)
        J = np.zeros((self.batch_size, p))

        for i in range(self.batch_size):
            k = np.random.randint(0, len(self.path.times) - 1)
            Lambda_k = self.path.times[k]
            Lambda_kp1 = self.path.times[k+1]
            Zk = self.path.states[k]
            Zkp1 = self.path.states[k+1]
            
            solver = JointKolmogorovSolver(
                self.model,
                theta,
                display_tqdm=False
            )
            
            kfe = solver.forward_solve(Lambda_k, Lambda_kp1, Zk, self.discontinuity_times)
            Eh = kfe[solver.slice_prob] @ self.h # produces p dim vector 
            resid_i = self.h[Zkp1] - Eh # p dim
            grad = np.stack([kfe[grad_i] for grad_i in solver.slice_grad]) # produces p x d matrix  
            D = grad @ self.h # results in p x r matrix (second dimension is dimension of return value of h)
            
            residuals[i] = resid_i
            J[i] = D
        
        return residuals, J


    def eval_ssr_grad_ssr(self, theta, mc_samples_per_observation=1):
        residuals, J = self.eval_residuals_and_jacobian(theta, mc_samples_per_observation)
        ssr = np.sum(np.power(residuals, 2)) / self.batch_size
        grad_ssr = -2 * np.sum(residuals.reshape(-1,1) * J, axis=0) / self.batch_size
        return ssr, grad_ssr, None 


class SimulatedConditionalLeastSquaresODEFullBatch:
    def __init__(
            self,
            model: MJPModel,
            discrete_path: DiscreteSample,
            h: np.ndarray,
        ):
        self.model = model
        self.path = discrete_path
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.h = h # d-dimensional array, map from states to R+
        self.discontinuity_times = []

        self.simulator = Simulator(model)
        self.name='CLS (Simulated)'

        self.batch_size = len(discrete_path.states)-1

    def eval_residuals_and_jacobian(self, theta, mc_samples_per_observation=1):
        p = len(theta)

        residuals = np.zeros(self.batch_size)
        J = np.zeros((self.batch_size, p))

        solver = JointKolmogorovSolver(
            self.model,
            theta,
            display_tqdm=False
        )

        for k in range(self.batch_size):
            Lambda_k = self.path.times[k]
            Lambda_kp1 = self.path.times[k+1]
            Zk = self.path.states[k]
            Zkp1 = self.path.states[k+1]
            
            
            kfe = solver.forward_solve(Lambda_k, Lambda_kp1, Zk, self.discontinuity_times)
            Eh = kfe[solver.slice_prob] @ self.h # produces p dim vector 
            resid_i = self.h[Zkp1] - Eh # p dim
            grad = np.stack([kfe[grad_i] for grad_i in solver.slice_grad]) # produces p x d matrix  
            D = grad @ self.h # results in p x r matrix (second dimension is dimension of return value of h)
            
            residuals[k] = resid_i
            J[k] = D
        
        return residuals, J


    def eval_ssr_grad_ssr(self, theta, mc_samples_per_observation=1):
        residuals, J = self.eval_residuals_and_jacobian(theta, mc_samples_per_observation)
        ssr = np.sum(np.power(residuals, 2)) / self.batch_size
        grad_ssr = -2 * np.sum(residuals.reshape(-1,1) * J, axis=0) / self.batch_size
        return ssr, grad_ssr, None 


def _cls_gradient_core(theta, allocation, path, model, simulator, h):
    p = len(theta)
    count_nonzero_allocation = np.count_nonzero(allocation)

    residuals = np.zeros(len(allocation))
    J = np.zeros((len(allocation), p))

    i = 0
    for k, num_mc_samples in enumerate(allocation):
        if num_mc_samples == 0:
            i += 1
            continue
        Lambda_k = path.times[k]
        Lambda_kp1 = path.times[k+1]
        Zk = path.states[k]
        Zkp1 = path.states[k+1]
        
        Eh_mc = 0.
        grad_Eh_mc = np.zeros(p)
        for _ in range(num_mc_samples):
            # compute LR gradient along a simulated path
            simulated_path_1 = simulator.simulate_single_run(
                Lambda_kp1, 
                starting_time=Lambda_k, 
                starting_state=Zk, 
                theta=theta,
            )
            Eh_mc += h[simulated_path_1.final_state]

            simulated_path_2 = simulator.simulate_single_run(
                Lambda_kp1, 
                starting_time=Lambda_k, 
                starting_state=Zk, 
                theta=theta,
            )

            cl = ContinuousTimeLikelihood(model, simulated_path_2)
            _, nabla = cl.eval_log_likelihood(theta)
            nabla = -nabla # dimension p 

            grad_Eh_mc += nabla * h[simulated_path_2.final_state]
        
        Eh_mc /= num_mc_samples
        grad_Eh_mc /= num_mc_samples

        residuals[i] = h[Zkp1] - Eh_mc
        J[i] = grad_Eh_mc
        i += 1
    
    return residuals, J 


def _cls_core(theta, allocation, path, model, simulator, h):
    p = len(theta)
    n = len(allocation)
    max_b = int(np.max(allocation))

    resids = np.zeros((n, max_b))
    grads = np.zeros((n, max_b, p))

    for k, b in enumerate(allocation):
        if b == 0:
            continue 

        Lambda_k = path.times[k]
        Lambda_kp1 = path.times[k+1]
        Zk = path.states[k]
        Zkp1 = path.states[k+1]

        for j in range(b):
            # compute LR gradient along a simulated path
            simulated_path_1 = simulator.simulate_single_run(
                Lambda_kp1, 
                starting_time=Lambda_k, 
                starting_state=Zk, 
                theta=theta,
            )
            resids[k, j] = h[Zkp1] - h[simulated_path_1.final_state]

            simulated_path_2 = simulator.simulate_single_run(
                Lambda_kp1, 
                starting_time=Lambda_k, 
                starting_state=Zk, 
                theta=theta,
            )

            cl = ContinuousTimeLikelihood(model, simulated_path_2)
            _, nabla = cl.eval_log_likelihood(theta)
            nabla = -nabla # dimension p 

            grads[k, j] = nabla * h[simulated_path_2.final_state]
        
    return resids, grads


def _rss_from_core(core_resids, allocation):
    # average squared residuals at each index k, and then sum over indices
    return np.sum(np.sum(np.power(core_resids, 2), axis=1) / allocation)

def _J_from_core(core_grads, allocation):
    return np.sum(core_grads, axis=1) / allocation

def _resids_from_core_inner(core_resids, allocation):
    return np.sum(core_resids, axis=1) / allocation

def _grad_rss_from_core_inner(core_resids, core_grads, allocation):
    J = _J_from_core(core_grads, allocation)
    res = _resids_from_core_inner(core_resids, core_grads, allocation)
    return -2 * J.T @ res


@register_objective
class SimulatedConditionalLeastSquaresFullBatch:
    def __init__(
            self,
            model: MJPModel,
            discrete_path: DiscreteSample,
            h: np.ndarray,
            seed=None,
            mc_samples_per_observation=10
        ):
        self.model = model
        self.path = discrete_path
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.h = h # d-dimensional array, map from states to R+
        self.discontinuity_times = []

        self.simulator = Simulator(model, seed)
        self.name='CLS (Simulated)'

        self.nobs = len(discrete_path.states) - 1
        self.mc_samples_per_observation = mc_samples_per_observation

    def eval_residuals_and_jacobian(self, theta, mc_samples_per_observation=None):
        # override mc samples per observation 
        if mc_samples_per_observation is None:
            mc_samples_per_observation = self.mc_samples_per_observation
        allocation = np.full(self.nobs, mc_samples_per_observation)
        residuals, J = _cls_gradient_core(theta, allocation, self.path, self.model, self.simulator, self.h)
        return residuals, J

    def eval_ssr_grad_ssr(self, theta, mc_samples_per_observation=None):
        # override mc samples per observation 
        if mc_samples_per_observation is None:
            mc_samples_per_observation = self.mc_samples_per_observation
        residuals, J = self.eval_residuals_and_jacobian(theta, mc_samples_per_observation)

        allocation = np.full(self.nobs, mc_samples_per_observation)
        core_resids, core_J = _cls_core(theta, allocation, self.path, self.model, self.simulator, self.h)
        ssr = _rss_from_core(core_resids, allocation)
        grad_ssr = -2 * J.T @ residuals
        return ssr / self.nobs, grad_ssr / self.nobs, None

@register_objective
class SimulatedConditionalLeastSquaresMiniBatch:
    def __init__(
            self,
            model: MJPModel,
            discrete_path: DiscreteSample,
            h: np.ndarray,
            batch_size: int = 1,
            seed: int = None
        ):
        self.model = model
        self.path = discrete_path
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.h = h # d-dimensional array, map from states to R+
        self.discontinuity_times = []
        # self.evaluation_method = self.eval_sum_least_squares

        self.simulator = Simulator(model)
        self.name='CLS (Simulated)'

        self.batch_size = batch_size
        self.n_obs = len(self.path) - 1
        self.factor = 1/(self.n_obs * (1 - (1 - 1/self.n_obs)**self.batch_size)) # 1/(nobs * (probability that a multinomial entry is > 0))

        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def eval_residuals_and_jacobian(self, theta):
        """
        Sampling without replacement to produce unbiased gradients
        """
        p = len(theta)

        rng = self.rng
        allocation = rng.multinomial(self.batch_size, np.full(self.n_obs, 1/self.n_obs))
        # print(allocation)
        residuals, J = _cls_gradient_core(theta, allocation, self.path, self.model, self.simulator, self.h)
        return residuals, J


    def eval_ssr_grad_ssr(self, theta):
        residuals, J = self.eval_residuals_and_jacobian(theta)
        ssr = np.sum(np.power(residuals, 2)) * self.factor
        grad_ssr = -2 * J.T @ residuals * self.factor
        return ssr, grad_ssr, None




@register_objective
class SimulatedConditionalLeastSquaresMiniBatchWithReplacement:
    def __init__(
            self,
            model: MJPModel,
            discrete_path: DiscreteSample,
            h: np.ndarray,
            batch_size: int = 1,
        ):
        self.model = model
        self.path = discrete_path
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.h = h # d-dimensional array, map from states to R+
        self.discontinuity_times = []

        self.simulator = Simulator(model)
        self.name='CLS (Simulated)'

        self.batch_size = batch_size
        self.n_obs = len(self.path) - 1
        self.factor = self.n_obs / self.batch_size
        print(f"factor = {self.factor}")

        self.rng = np.random.default_rng()

    def eval_residuals_and_jacobian(self, theta):
        """
        Sampling without replacement to produce unbiased gradients
        """
        p = len(theta)

        rng = self.rng
        residuals = np.zeros(self.batch_size)
        J = np.zeros((self.batch_size, p))

        for i in range(self.batch_size):
            k = rng.integers(0, len(self.path) - 1)
            Lambda_k = self.path.times[k]
            Lambda_kp1 = self.path.times[k+1]
            Zk = self.path.states[k]
            Zkp1 = self.path.states[k+1]

            # compute LR gradient along a simulated path
            simulated_path_1 = self.simulator.simulate_single_run(
                Lambda_kp1, 
                starting_time=Lambda_k, 
                starting_state=Zk, 
                theta=theta,
            )
            Eh_mc = self.h[simulated_path_1.final_state]

            simulated_path_2 = self.simulator.simulate_single_run(
                Lambda_kp1, 
                starting_time=Lambda_k, 
                starting_state=Zk, 
                theta=theta,
            )

            cl = ContinuousTimeLikelihood(self.model, simulated_path_2)
            _, nabla = cl.eval_log_likelihood(theta)
            nabla = -nabla # dimension p 

            grad_Eh_mc = nabla * self.h[simulated_path_2.final_state]
            
            residuals[i] = self.h[Zkp1] - Eh_mc
            J[i] = grad_Eh_mc
        
        return residuals, J


    def eval_ssr_grad_ssr(self, theta):
        residuals, J = self.eval_residuals_and_jacobian(theta)
        ssr = np.sum(np.power(residuals, 2)) * self.factor
        grad_ssr = -2 * np.sum(residuals.reshape(-1,1) * J, axis=0) * self.factor
        return ssr, grad_ssr, None


@register_objective
class SimulatedConditionalLeastSquaresMiniBatchPeter:
    def __init__(
            self,
            model: MJPModel,
            discrete_path: DiscreteSample,
            h: np.ndarray,
            batch_size: int = 1,
        ):
        self.model = model
        self.path = discrete_path
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.h = h # d-dimensional array, map from states to R+
        self.discontinuity_times = []

        self.name='CLS (Simulated)'

        self.batch_size = batch_size
        self.n_obs = len(self.path) - 1
        self.factor = self.n_obs / self.batch_size

        self.rng = np.random.default_rng()
        self.simulator = Simulator(model, rng=self.rng)

        self.b = np.mean([h[x] for x in self.path.states])

    def eval_residuals_and_jacobian(self, theta, k=None):
        """
        Sampling the same index repeatedly to compute sample average estimates of function value and gradient
        """
        p = len(theta)

        rng = self.rng
        residuals = np.zeros(self.batch_size)
        J = np.zeros((self.batch_size, p))
        nablas = np.zeros((self.batch_size, p))
        Ls = np.zeros(self.batch_size)

        if k is None:
            k = rng.integers(0, len(self.path) - 1)


        Lambda_k = self.path.times[k]
        Lambda_kp1 = self.path.times[k+1]
        Zk = self.path.states[k]
        Zkp1 = self.path.states[k+1]

        for i in range(self.batch_size):
            # compute LR gradient along a simulated path
            simulated_path_1 = self.simulator.simulate_single_run(
                Lambda_kp1, 
                starting_time=Lambda_k, 
                starting_state=Zk, 
                theta=theta,
            )
            Eh_mc = self.h[simulated_path_1.final_state]

            simulated_path_2 = self.simulator.simulate_single_run(
                Lambda_kp1, 
                starting_time=Lambda_k, 
                starting_state=Zk, 
                theta=theta,
            )

            cl = ContinuousTimeLikelihood(self.model, simulated_path_2)
            logL, nabla = cl.eval_log_likelihood(theta)
            nabla = -nabla # dimension p 

            grad_Eh_mc = nabla * (self.h[simulated_path_2.final_state])
            
            residuals[i] = self.h[Zkp1] - Eh_mc
            J[i] = grad_Eh_mc
            nablas[i] = nabla
            Ls[i] = logL
        
        return residuals, J, nablas, k, Ls


    def eval_ssr_grad_ssr(self, theta):
        residuals, J = self.eval_residuals_and_jacobian(theta)
        ssr = np.sum(np.power(residuals, 2)) * self.factor
        grad_ssr = -2 * np.sum(residuals.reshape(-1,1) * J, axis=0) * self.factor
        return ssr, grad_ssr, None
 

class EndogenousLikelihoodODE:
    def __init__(self, endo_submodel: JacksonEndogenousSubmodel, path, fixed_step=1e-3, rtol=1e-3, atol=1e-5):
        self.endo_submodel = endo_submodel
        self.path = path
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.fixed_step = fixed_step
        self.rtol = rtol
        self.atol = atol

        self.name = "Endogenous (ODE)"
    
    def eval_log_likelihood_scores(self, theta, fixed_step=None):
        p = len(theta)
        S = [0]

        logL = 0.
        gradLogL = np.zeros(p)
        scores = np.empty(((len(self.path.times) - 1), p))

        if fixed_step is None:
            fixed_step = self.fixed_step

        for k in range(1, len(self.path.times)):
            self.log.info(f"in path: {k=}")
            t_prev, t_curr = self.path.times[k-1], self.path.times[k]
            x_prev, x_curr = self.path.states[k-1], self.path.states[k]
            solver = JointKolmogorovSolver(
                self.endo_submodel, 
                theta, 
                fixed_step, 
                display_tqdm=False,
                rtol=self.rtol,
                atol=self.atol
            )


            adj_curr = self.endo_submodel.base_model.map_encoded_observables_to_cemetery[x_curr]
            adj_curr -= self.endo_submodel.base_model._d  # adjust for cemetery indexing
            # print(f"{adj_curr=}")

            self.log.info(f"{self.path.times[k-1]=}")
            self.log.info(f"{self.path.times[k]=}")
            self.log.info(f"{self.path.states[k-1]=}")
            self.log.info(f"{self.path.states[k]=}")
            self.log.info(f"{adj_curr=}")

            disc_times_k = self.endo_submodel.base_model.get_discontinuity_times(t_prev, t_curr)

            kfe = solver.forward_solve(t_prev, t_curr, x_prev, disc_times_k, max_step=1/self.endo_submodel.get_max_total_rate(theta))
            
            R_x_S = self.endo_submodel.base_model.get_Q_obs(t_curr, theta)
            R_x_S_grad = self.endo_submodel.base_model.get_Q_obs_grad(t_curr, theta)

            self.log.info(f'{R_x_S=}')
            self.log.info(f"{kfe[solver.slice_prob]=}")

            if kfe[solver.slice_prob][x_curr] < 0:
                print(f"WARNING: NEGATIVE PROBABILITIES at index {k}! Val: {kfe[solver.slice_prob][x_curr]}")
                print(kfe[solver.slice_prob])
                print(k, t_prev, t_curr, x_prev, x_curr)
                print("Attempting solve with high precision")
                kfe = solver.forward_solve(t_prev, t_curr, x_prev, disc_times_k, max_step=1e-4) 
                if kfe[solver.slice_prob][x_curr] < 0:
                    print(f"High precision solve failed: {kfe[solver.slice_prob][x_curr]=}")


            non_absorbing_slice = slice(0, self.endo_submodel.base_model._d)
            prob = (np.atleast_2d(kfe[solver.slice_prob][non_absorbing_slice]) @ R_x_S)[0, adj_curr]
            self.log.info(f"{prob=}")
            if prob <= 0:
                print(f"Negative Prob: {prob=}, at index {k}")
                print(f"WARNING: NEGATIVE PROBABILITIES at index {k}! Val: {kfe[solver.slice_prob][x_curr]}")
                print(kfe[solver.slice_prob])
                print(k, t_prev, t_curr, x_prev, x_curr)
 
                print(f"{kfe[solver.slice_prob][non_absorbing_slice]=}")
            logL += np.log(prob)

            gradLogLincrement = np.zeros(p)
            for i in range(p):
                gradLogLincrement[i] = (np.atleast_2d(kfe[solver.slice_grad[i]][non_absorbing_slice]) @ R_x_S + np.atleast_2d(kfe[solver.slice_prob][non_absorbing_slice]) @ R_x_S_grad[i])[0, adj_curr]
            gradLogL += gradLogLincrement / prob
            scores[k-1] = gradLogLincrement / prob
        
        self.log.info(f"{logL=}, {gradLogL=}")
        return -logL, -gradLogL, scores    
        