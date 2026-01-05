from abc import ABC, abstractmethod
from functools import lru_cache
import time
import collections


import numpy as np
from scipy import sparse as sp
from scipy.optimize import minimize
from tqdm import tqdm

import logging

log = logging.getLogger(__name__)

OPTIMIZER_REGISTRY = {}
def register_optimizer(cls):
    """Decorator that registers the class by its name in CLASS_REGISTRY."""
    OPTIMIZER_REGISTRY[cls.__name__] = cls
    return cls


class SGD(ABC):
    def __init__(self, learning_rate=1e-1, seed=None, output_field="theta", verbose=False, **kwargs):
        self.learning_rate = learning_rate
        self.method_kwargs = kwargs  # pass method-specific parameters
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.seed = seed
        self.output_field = output_field
        self.verbose = verbose

    @abstractmethod
    def _schedule(self, k): pass

    def _constrained_step(self, theta, step, eta_t, k):
        theta_unconstrained = theta - step
        violation = True
        consecutive_violations = 0

        while violation and len(self.constraints) > 0:
            violation = False
            for c in self.constraints:
                Ax = c.A @ theta_unconstrained
                if not (np.all(c.lb <= Ax) and np.all(Ax <= c.ub)):
                    violation = True
                    break
            if violation:
                eta_t *= 0.5
                consecutive_violations += 1

                if consecutive_violations > 10:
                    self.log.warning(f"Iter {k:5d}: Line search failed.")
                    print(f"Iter {k:5d}: Line search failed.")
                    return theta
                step = eta_t * self._direction(theta, self.last_grad, k)
                theta_unconstrained = theta - step
            else:
                consecutive_violations = 0
        return theta_unconstrained

    def post_step_hook(self, theta, k, history):
        """Optional hook for subclasses to record additional outputs (e.g. avg_theta)."""
        return


    @abstractmethod
    def reset_state(self, theta0):
        """Initialize any optimizer state (e.g., moments)."""
        pass

    @abstractmethod
    def _direction(self, theta, grad_t, k):
        """Return direction for parameter update (excluding step-size scaling)."""
        pass

    def optimize(self, objective, theta0, n_iter=1000, constraints=None):
        self.objective = objective
        self.constraints = constraints or []

        theta = np.array(theta0, dtype=float)
        self.reset_state(theta0)
        history = collections.defaultdict(list)

        iteration_range = range(1, n_iter + 1)
        if self.verbose:
            iteration_range = tqdm(iteration_range, desc="SGD Iteration", ncols=130)

        try:
            for k in iteration_range:
                f_t, grad_t, _ = self.objective(theta)
                self.last_grad = grad_t

                eta_t = self._schedule(k)
                step_dir = self._direction(theta, grad_t, k)
                step = eta_t * step_dir

                theta = self._constrained_step(theta, step, eta_t, k)
            
                if self.verbose:
                    iteration_range.set_postfix({'theta': f'{theta}'})

                grad_norm = np.linalg.norm(grad_t)
                history['theta'].append(theta.copy())
                history['f'].append(f_t)
                history['grad'].append(grad_t)
                history['grad_norm'].append(grad_norm)
                history['eta_t'].append(eta_t)

                self.post_step_hook(theta, k, history)
                self.log.info(f"Iter {k:5d}: f={f_t:.4f}, ||grad||={grad_norm:.3e}, η_t={eta_t:.3e}")

        except KeyboardInterrupt:
            self.log.warning(f"\nOptimization interrupted at iteration {k}. Returning last computed result.")

        # Return final "output_field" if available, otherwise last theta
        if self.output_field in history:
            return history[self.output_field][-1], history
        else:
            return history['theta'][-1], history


@register_optimizer
class PolyakRuppertSGD(SGD):
    def __init__(self, learning_rate=1e-2, decay=0.75, scale=0.02,
                 seed=None, log_every=100, output_field="avg_theta", **kwargs):
        super().__init__(learning_rate=learning_rate, seed=seed,
                         log_every=log_every, output_field=output_field, **kwargs)
        self.decay = decay
        self.scale = scale

    def _schedule(self, k):
        return self.learning_rate / (1.0 + self.scale * k) ** self.decay

    def reset_state(self, theta0):
        self.avg_theta = np.zeros_like(theta0)

    def _direction(self, theta, grad_t, k):
        return grad_t

    def post_step_hook(self, theta, k, history):
        self.avg_theta = ((k - 1) * self.avg_theta + theta) / k
        history['avg_theta'].append(self.avg_theta.copy())


@register_optimizer
class AdamW(SGD):
    def __init__(self, learning_rate=1e-3, decay=0.75, scale=0.02,
                 beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=1e-2,
                 seed=None, log_every=100, output_field="theta", **kwargs):
        super().__init__(learning_rate=learning_rate, seed=seed,
                         log_every=log_every, output_field=output_field, **kwargs)
        self.decay = decay
        self.scale = scale
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay

    def _schedule(self, k):
        return self.learning_rate / (1.0 + self.scale * k) ** self.decay

    def reset_state(self, theta0):
        self.m = np.zeros_like(theta0)
        self.v = np.zeros_like(theta0)

    def _direction(self, theta, grad_t, k):
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad_t
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad_t ** 2)
        m_hat = self.m / (1 - self.beta1 ** k)
        v_hat = self.v / (1 - self.beta2 ** k)
        adam_step = m_hat / (np.sqrt(v_hat) + self.eps)
        weight_decay_step = self.weight_decay * theta
        return adam_step + weight_decay_step


@register_optimizer
class AdamWAveraged(AdamW):
    def __init__(self, learning_rate=0.001, decay=0.75, scale=0.02, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01, seed=None, log_every=100, output_field="avg_theta", **kwargs):
        output_field = "avg_theta"
        super().__init__(learning_rate, decay, scale, beta1, beta2, eps, weight_decay, seed, log_every, output_field, **kwargs)
    
    def reset_state(self, theta0):  
        self.m = np.zeros_like(theta0)
        self.v = np.zeros_like(theta0)
        self.avg_theta = np.zeros_like(theta0)

    def post_step_hook(self, theta, k, history):
        self.avg_theta = ((k - 1) * self.avg_theta + theta) / k
        history['avg_theta'].append(self.avg_theta.copy())


@register_optimizer
class BHHH:
    def __init__(self, f_grad_method, tol=1e-2, constraints=[]):
        self.f_grad_method = f_grad_method
        self.tol = tol 
        self.constraints =constraints

    def optimize(self, theta_initial, max_iter=50):
        p = theta_initial.shape[0]
        theta = theta_initial
        is_converged = False
        is_failed = False
        history = [theta_initial]
        history = {
            'x': [],
            'f': [],
            'grad': [],
            'scores': [],
            'eta': [],
            'inc': [],
            'iteration': [],
        }
        it = 0
        f, grad, scores = self.f_grad_method(theta)
        while not (is_converged or is_failed):
            
            print(f"{it=}, {theta=}, {f=}")

            history['x'].append(theta)
            history['f'].append(f)
            history['grad'].append(grad)
            history['scores'].append(scores)
            history['iteration'].append(it)

            H = scores.T @ scores  # outer product of scores 
            try:
                inc = np.linalg.solve(H, grad)
            except np.linalg.LinAlgError:
                inc, *_ = np.linalg.lstsq(H, grad, rcond=None) # fallback if singular
                print(f"np.linalg.solve(H, grad) unsuccessful. {inc=}")
 
            history['inc'].append(inc)

            print(f"{grad=}")
            print(f"{inc=}")

            # --- Convergence checks ---
            grad_norm = np.linalg.norm(grad, ord=np.inf)
            step_norm = np.linalg.norm(inc)

            grad_tol = self.tol
            step_tol = self.tol * (1 + np.linalg.norm(theta))
            # f_tol = self.tol * (1 + abs(f))

            if grad_norm < grad_tol:
                print(f"Converged: gradient norm {grad_norm:.2e} below tolerance {grad_tol:.2e}")
                is_converged = True
            elif step_norm < step_tol:
                print(f"Converged: step norm {step_norm:.2e} below tolerance {step_tol:.2e}")
                is_converged = True
            # elif f_diff < f_tol:
            #     print(f"Converged: objective improvement {f_diff:.2e} below tolerance {f_tol:.2e}")
            #     is_converged = True
            
            # --- Damping / Line Search Logic ---
            eta = 1.0
            old_f = f.copy()
            print(f"Proposed update: {theta - eta * inc}")
            while not is_converged: # will loop infinitely until broken, unless convergence criteria above was reached
                theta_new = theta - eta * inc
                violation = False
                for c in self.constraints:
                    if not (np.all(c.lb <= c.A @ theta_new) and np.all(c.A @ theta_new <= c.ub)):
                        violation = True
                        reason = f"constraint violation at {theta_new=}"

                if not violation:
                    f, grad, scores = self.f_grad_method(theta_new)
                    theta = theta_new
                    history['eta'].append(eta)
                    # break # let's 
                    if f < old_f:
                        # Good step, accept it and break the inner loop
                        
                        theta = theta_new
                        history['eta'].append(eta)
                        break
                    # otherwise, continue to shrink
                    reason = f"{old_f=} < {f=}"
                
                eta *= 0.5 # Step was too big, shrink it
                print(f"Shrinking step size: {eta=}. Reason: {reason}")
                
                if eta < 2**-5: # Failsafe to prevent infinite loop
                    print("Warning: Line search failed.")
                    is_failed = True # Could not find a better point
                    history['eta'].append(0.)
                    break
            # --- End of Damping Logic ---
            if it >= max_iter:
                print("Maximum iterations exceeded")
                is_failed = True
            violation = False
            for c in self.constraints:
                if not (np.all(c.lb <= c.A @ theta) and np.all(c.A @ theta <= c.ub)):
                    violation = True
            if violation:
                print("Could not find valid parameter")
                is_failed = True

            it += 1
        # final iteration append
        
        self.theta_hat = theta 
        self.num_iterations = it

        self.is_converged = is_converged
        return theta, history


@register_optimizer
class GaussNewton:
    def __init__(self, residuals_and_jacobian_method, tol=1e-2, constraints=[]):
        self.get_residuals_and_jacobian = residuals_and_jacobian_method
        self.is_converged = None
        self.constraints = constraints
        self.tol = tol


    def optimize(self, theta_initial, max_iter=50):
        p = theta_initial.shape[0]
        theta = theta_initial
        is_converged = False
        is_failed = False
        history = [theta_initial]
        history = {
            'x': [],
            'f': [],
            'eta': [],
            'inc': [],
            'jacobian': [],
            'iteration': [],
            'time': []
        }
        it = 0
        iteration_start_time = time.perf_counter()
        res, Jh = self.get_residuals_and_jacobian(theta)
        iteration_end_time = time.perf_counter()
        while not (is_converged or is_failed):
            rss = np.sum(res**2)
            print(f"{it=}, {theta=}, {rss=}")

            history['x'].append(theta)
            history['f'].append(rss)
            history['jacobian'].append(Jh)
            history['iteration'].append(it)
            history['time'].append(iteration_end_time - iteration_start_time)
            
            iteration_start_time = time.perf_counter()
 
            inc, _, _, _ = np.linalg.lstsq(Jh, res, rcond=None)
            history['inc'].append(inc)
            

            if np.linalg.norm(inc) / np.linalg.norm(theta) < self.tol:
                is_converged = True
            
            # --- Damping / Line Search Logic ---
            eta = 1.0
            while True:
                theta_new = theta + eta * inc
                violation = False
                for c in self.constraints:
                    if not (np.all(c.lb <= c.A @ theta_new) and np.all(c.A @ theta_new <= c.ub)):
                        violation = True
                        reason = "violated constraints"

                if not violation:
                    res, Jh = self.get_residuals_and_jacobian(theta_new) # strictly speaking, only need residuals here
                    new_cost = np.sum(res**2)
                    
                    if new_cost < rss:
                        # Good step, accept it and break the inner loop
                        theta = theta_new
                        history['eta'].append(eta)
                        break
                    else:
                        reason = "had higher rss"
                
                eta *= 0.5 # Step was too big, shrink it
                print(f"Shrinking step size: {eta=}, because {theta_new} {reason}")
                
                if eta < 1e-4: # Failsafe to prevent infinite loop
                    print("Warning: Line search failed.")
                    is_failed = True # Could not find a better point
                    history['eta'].append(0.)
                    break
            # --- End of Damping Logic ---
            if it >= max_iter:
                print("Maximum iterations exceeded")
                is_failed = True
            violation = False
            for c in self.constraints:
                if not (np.all(c.lb <= c.A @ theta) and np.all(c.A @ theta <= c.ub)):
                    violation = True
            if violation:
                print(f"Constraint violation at it={it}")
                print("Could not find valid parameter")
                is_failed = True

            iteration_end_time = time.perf_counter()

            it += 1
        # final iteration append
        
        self.theta_hat = theta 
        self.num_iterations = it

        self.is_converged = is_converged
        return theta, history
