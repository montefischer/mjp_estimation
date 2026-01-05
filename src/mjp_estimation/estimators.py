import time
from abc import ABC, abstractmethod
import logging

import numpy as np
from scipy.linalg import sqrtm
from scipy.optimize import minimize, BFGS

from .base_functionality import MJPModel, DiscreteSample, EndogenousSample
from .confidence import in_confidence_region
from . import optimizers
from . import objectives 

ESTIMATOR_REGISTRY = {}
def register_estimator(cls):
    """Decorator that registers the class by its name in CLASS_REGISTRY."""
    ESTIMATOR_REGISTRY[cls.__name__] = cls
    return cls

class Estimator(ABC):
    """Base class for MJP parameter estimators."""
    def __init__(self, model: MJPModel):
        self.model = model
        self.theta_hat = None
        self.theta_initial = None 
        self.elapsed_time = None 
        self.history = None
        self.periodic = None

    @abstractmethod
    def fit(self, path, theta_initial: np.ndarray, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, theta_true: np.ndarray, alpha: float = 0.05, **kwargs):
        pass

    def save(self, fname):
        """
        Write estimation results to disk
        """
        if self.results is None:
            raise ValueError("No results to save; call evaluate() to generate results.")
        np.savez(fname, **self.results)


@register_estimator
class MLEDiscreteODE(Estimator):
    def __init__(
            self,
            model: MJPModel,
            ):
        super().__init__(model)
        self.dl = None
        self.res = None
    
    def fit(
            self,
            path: DiscreteSample,
            theta_initial: np.ndarray,
            xtol=1e-2,
            gtol=1e-2,
            verbose=False,
            method='bhhh',
    ):
        """Fit the MLE estimator.

        Parameters
        ----------
        path : DiscreteSample
            Observed sample path
        theta_initial : np.ndarray
            Initial parameter values
        xtol : float, default=1e-2
            Tolerance for parameter convergence
        gtol : float, default=1e-2
            Tolerance for gradient convergence
        verbose : bool, default=False
            Print progress information
        method : str, default='bhhh'
            Optimization method: 'bhhh' or 'trust-constr'
        """
        if type(path) is not DiscreteSample:
            print("Warning: path is not DiscreteSample type")
        model = self.model
        self.theta_initial = theta_initial

        if hasattr(model, "period"):
            self.dl = objectives.DiscreteLikelihoodPeriodicAmortization(model, path, model.period, verbose=verbose)
            self.periodic = True
            if verbose:
                print("Fitting periodic model")
        else:
            self.dl = objectives.DiscreteLikelihood(model, path)
            self.periodic = False
            if verbose:
                print("Fitting non-periodic model")

        self.xtol = xtol
        self.gtol = gtol

        start_time = time.perf_counter()

        if method.lower() == 'bhhh':
            # BHHH optimizer (default)
            bhhh = optimizers.BHHH(self.dl.eval_log_likelihood_scores, tol=xtol, constraints=model.constraints)
            theta_hat, history = bhhh.optimize(theta_initial)

            self.theta_hat = theta_hat
            self.history = history
            self.is_converged = bhhh.is_converged
            self.optimizer_iterations = len(self.history['x'])
            self.func_evaluations = len(self.history['x'])
            self.grad_evaluations = len(self.history['x'])
            self.optimal_objective = self.history['f'][-1]

        elif method.lower() == 'trust-constr':
            # scipy trust-constr with BFGS Hessian approximation
            history = {
                'x': [],
                'f': [],
                'grad': [],
                'iteration': [],
                'state': []
            }

            def callback(xk, state=None):
                """Callback to record optimization progress."""
                if verbose:
                    print(f"{xk=}")
                history['x'].append(xk.copy())
                history['f'].append(state.fun)
                history['grad'].append(state.grad)
                history['iteration'].append(state.niter)
                history['state'].append(state.copy())

            res = minimize(
                self.dl.f,
                theta_initial,
                method='trust-constr',
                hess=BFGS(),
                jac=self.dl.grad,
                options={'disp': verbose, 'xtol': xtol, 'gtol': gtol},
                constraints=model.constraints,
                callback=callback
            )

            self.res = res
            self.theta_hat = res.x
            self.history = history
            self.is_converged = res.success
            self.optimizer_iterations = res.nit
            self.func_evaluations = res.nfev
            self.grad_evaluations = res.njev
            self.optimal_objective = res.fun

        else:
            raise ValueError(f"Unknown method: {method}. Choose 'bhhh' or 'trust-constr'.")

        end_time = time.perf_counter()
        self.elapsed_time = end_time - start_time

    def evaluate(self, theta_true, alpha=0.05):
        theta_hat = self.theta_hat
        p = len(theta_hat) 
        
        # get (n x p) scores matrix at the mle
        scores = self.dl.scores(self.theta_hat)

        fisher_information_approx = scores.T @ scores
        C_hat = sqrtm(fisher_information_approx)
        self.prec = fisher_information_approx
        is_good_CI, _, _, volume_ci = in_confidence_region(theta_true, theta_hat, fisher_information_approx, alpha)
        euclidean_distance = np.linalg.norm(theta_hat - theta_true)

        optimization_runtime = self.elapsed_time
        time_to_compute_optimal_schedule = self.dl.optimal_schedule_compute_time if self.periodic else None
        efficiency_gain_from_optimal_schedule = self.dl.efficiency_gain if self.periodic else None

        self.results = {
            # scalars
            'converged': self.is_converged,
            'objective': self.optimal_objective,
            'is_good_CI': is_good_CI,
            'CI_volume': volume_ci,
            'euclidean_distance': euclidean_distance,
            'optimizer_iterations': self.optimizer_iterations,
            'func_evaluations': self.func_evaluations,
            'grad_evaluations': self.grad_evaluations,
            'optimization_runtime': optimization_runtime,
            'time_to_compute_optimal_schedule': time_to_compute_optimal_schedule,
            'efficiency_gain_from_optimal_schedule': efficiency_gain_from_optimal_schedule,

            'optimization_algo': 'trust-constr',
            'xtol': self.xtol,
            'gtol': self.gtol,

            # non-scalars
            'theta_hat': theta_hat,
            'theta_star': theta_true,
            'theta_initial': self.theta_initial,
            'scores': scores,
            'fisher_information': fisher_information_approx,
            'C_hat': C_hat,
            'history': self.history,
        }



@register_estimator
class CLSDiscreteODE(Estimator):
    def __init__(
            self,
            model: MJPModel,
            h: np.ndarray,
            ):
        super().__init__(model)
        self.h = h
        self.ls = None
        self.res = None
    
    def fit(
            self,
            path: DiscreteSample,
            theta_initial: np.ndarray,
            tol=1e-4
    ):
        model = self.model
        self.theta_initial = theta_initial
        assert len(theta_initial) == len(model.theta_true)

        if hasattr(model, "period"):
            self.ls = objectives.ConditionalLeastSquaresPeriodicAmortization(model, path, self.h)
            self.periodic = True
        else:
            self.ls = objectives.ConditionalLeastSquares(model, path, self.h)
            self.periodic = False
        
        self.tol = tol

        gn = optimizers.GaussNewton(self.ls.eval_residuals_and_jacobian, tol=tol, constraints=model.constraints)
        start_time = time.perf_counter()
        theta_hat, history = gn.optimize(theta_initial) 
        end_time = time.perf_counter()
        self.elapsed_time = end_time - start_time
        self.theta_hat = theta_hat

        self.is_converged = gn.is_converged

        self.history = history

    def evaluate(self, theta_true, alpha=0.05, qv_type='square'):
        theta_hat = self.theta_hat
        p = len(theta_hat) 

        prec = self.ls.prec_matrix(self.theta_hat, qv_type=qv_type)
        is_good_CI, _, _, volume_ci = in_confidence_region(theta_true, theta_hat, prec, alpha)
        euclidean_distance = np.linalg.norm(theta_hat - theta_true)
        
        optimizer_iterations = len(self.history['x'])
        optimal_objective = self.ls.eval_ssr(theta_hat)

        optimization_runtime = self.elapsed_time
        time_to_compute_optimal_schedule = self.ls.agroups_computation_time if self.periodic else None
        efficiency_gain_from_optimal_schedule = self.ls.efficiency_factor if self.periodic else None

        self.results = {
            # scalars
            'converged': self.is_converged,
            'objective': optimal_objective,
            'is_good_CI': is_good_CI,
            'CI_volume': volume_ci,
            'euclidean_distance': euclidean_distance,

            'optimizer_iterations': optimizer_iterations,
            'optimization_runtime': optimization_runtime,
            'time_to_compute_optimal_schedule': time_to_compute_optimal_schedule,
            'efficiency_gain_from_optimal_schedule': efficiency_gain_from_optimal_schedule,

            'optimization_algo': 'trust-constr',
            'xtol': self.tol,

            # non-scalars
            'theta_hat': theta_hat,
            'theta_star': theta_true,
            'theta_initial': self.theta_initial,
            'prec': prec,
            'history': self.history,
        }


@register_estimator
class MLESGD(Estimator):
    def __init__(
            self,
            model: MJPModel,
            batch_size: int,
            checkpoint_multiples,
            sgd_method: optimizers.SGD=None,
            likelihood_method=None,
        ):
        super().__init__(model)
        self.dl = None
        self.res = None

        # default SGD to PR averaging vanilla
        if sgd_method is None:
            print("No SGD method provided, defaulting to Polyak Ruppert averaging")
            pr_lr=1.
            pr_decay=0.75
            pr_scale=0.02
            self.sgd_method = optimizers.PolyakRuppertSGD(
                pr_lr,
                pr_decay,
                pr_scale
            )
        else:
            self.sgd_method = sgd_method

        self.likelihood_method = likelihood_method or objectives.DiscreteLikelihoodMiniBatchODE
        self.batch_size = batch_size
        self.checkpoint_multiples = np.array(checkpoint_multiples)


    def fit(self, path: DiscreteSample, theta_initial: np.ndarray):
        if type(path) is not DiscreteSample:
            print("Warning: path is not DiscreteSample type")
        model = self.model
        self.path = path
        self.theta_initial = theta_initial

        history = {
                'x': [],
                'f': [],
                'grad': [],
                'iteration': [],
                'state': []
            }
        
        dl = self.likelihood_method(model, path, batch_size=self.batch_size)
        optimizer = self.sgd_method

        dataset_size_multiplier = np.max(self.checkpoint_multiples)
        n_iter = int(np.ceil(len(path.states) / self.batch_size * dataset_size_multiplier))
        print(f"{n_iter=}")
        start_time = time.perf_counter()
        theta_hat, history = optimizer.optimize( 
            objective=dl.eval_log_likelihood_scores,
            theta0=theta_initial,
            n_iter=n_iter,
            constraints=model.constraints
        )
        end_time = time.perf_counter()

        self.elapsed_time = end_time - start_time
        self.theta_hat = theta_hat
        self.history = history
        

    def evaluate(self, theta_true, alpha=0.05):
        model = self.model
        path = self.path
        # may be convenient to evaluate using ODE solvers
        if hasattr(model, "period"):
            self.dl = objectives.DiscreteLikelihoodPeriodicAmortization(model, path, model.period)
            self.periodic = True
        else:
            self.dl = objectives.DiscreteLikelihood(model, path)
            self.periodic = False
        
        checkpoint_idx = []
        checkpoint_thetas = []
        checkpoint_inside = []
        checkpoint_CI_volumes = []
        checkpoint_objectives = []
        checkpoint_scores = []
        checkpoint_euclidean = []

        for k in self.checkpoint_multiples:
            idx = int(np.ceil(k * len(path.states) / self.batch_size)) - 1
            print(idx)
            checkpoint_idx.append(idx)
            checkpoint_theta = self.history[self.sgd_method.output_field][idx]
            f, grad, scores = self.dl.eval_log_likelihood_scores(checkpoint_theta)

            sample_size = scores.shape[0]
            effective_sgd_iterations = idx * self.batch_size
            C_hat = scores.T @ scores
            # Warning: this choice of prec assumes that SGD estimates the exact CLS estimator, which is only true in the limit.
            prec = C_hat

            is_good_CI, distance_sq, threshold, volume_ci = in_confidence_region(theta_true, checkpoint_theta, prec, 0.05)
            checkpoint_inside.append(is_good_CI)
            checkpoint_CI_volumes.append(volume_ci)

            print(f"{k=}, {idx=}, {is_good_CI=}, {checkpoint_theta=}, {distance_sq=}, {threshold=}, {volume_ci=}")

            euclidean_distance = np.linalg.norm(checkpoint_theta - theta_true)
            checkpoint_euclidean.append(euclidean_distance)
            
            checkpoint_thetas.append(checkpoint_theta)
            checkpoint_objectives.append(f)
            checkpoint_scores.append(scores)
        
        # now eval for theta hat
        f, grad, scores = self.dl.eval_log_likelihood_scores(self.theta_hat)

        sample_size = scores.shape[0]
        effective_sgd_iterations = idx * self.batch_size
        C_hat = scores.T @ scores
        prec = C_hat / (1 + sample_size / effective_sgd_iterations) #TODO: figure this out
        self.prec = prec

        is_good_CI, distance_sq, threshold, volume_ci = in_confidence_region(theta_true, self.theta_hat, prec, 0.05)

        euclidean_distance = np.linalg.norm(self.theta_hat - theta_true)

        self.results = {
            # scalars
            'converged': np.linalg.norm(grad) < 1e-1, # this is ad-hoc
            'objective': f,
            'is_good_CI': is_good_CI,
            'CI_volume': volume_ci,
            'euclidean_distance': euclidean_distance,

            'optimizer_iterations': len(self.history['theta']),
            'optimization_runtime': self.elapsed_time,

            # non-scalars
            'theta_hat': self.theta_hat,
            'theta_star': theta_true,
            'theta_initial': self.theta_initial,
            'prec': prec,
            'history': self.history,

            # checkpoints
            'checkpoint': {
                'multiple': self.checkpoint_multiples,
                'idx': checkpoint_idx,
                'theta': checkpoint_thetas,
                'inside': checkpoint_inside,
                'CI_volume': checkpoint_CI_volumes,
                'objective': checkpoint_objectives,
                'scores': checkpoint_scores,
                'euclidean_distance': checkpoint_euclidean,
            }
        }
        
    

    def evaluate_coverage(self, theta_true, alpha=0.05):
        model = self.model
        path = self.path
        # may be convenient to evaluate using ODE solvers
        if hasattr(model, "period"):
            self.dl = objectives.DiscreteLikelihoodPeriodicAmortization(model, path, model.period)
            self.periodic = True
        else:
            self.dl = objectives.DiscreteLikelihood(model, path)
            self.periodic = False
        
        euclidean_distance = np.linalg.norm(self.theta_hat - theta_true)

        self.results = {
            # scalars
            'euclidean_distance_true': euclidean_distance,

            'optimizer_iterations': len(self.history['theta']),
            'optimization_runtime': self.elapsed_time,

            # non-scalars
            'theta_hat': self.theta_hat,
            'theta_star': theta_true,
            'theta_initial': self.theta_initial,
            'history': self.history,
        }


@register_estimator
class CLSSGD(Estimator):
    def __init__(
            self,
            model: MJPModel,
            h,
            batch_size: int,
            checkpoint_multiples,
            sgd_method: optimizers.SGD,
            ls_method,
            ls_params = {}
        ):
        super().__init__(model)
        self.dl = None
        self.res = None
        self.h = h
        self.sgd_method = sgd_method
        self.ls_method = ls_method
        self.ls_params = ls_params
        self.batch_size = batch_size
        self.checkpoint_multiples = np.array(checkpoint_multiples)

    def fit(self, path: DiscreteSample, theta_initial: np.ndarray):
        if not isinstance(path, DiscreteSample):
            print("Warning: path is not DiscreteSample type")
        model = self.model
        self.path = path
        self.theta_initial = theta_initial

        try:
            ls = self.ls_method(model, path, self.h, self.batch_size, **self.ls_params)
        except:
            ls = self.ls_method(model, path, self.h, **self.ls_params)
        self.ls = ls

        optimizer = self.sgd_method

        dataset_size_multiplier = np.max(self.checkpoint_multiples)
        n_iter = int(np.ceil(len(path.states) / self.batch_size * dataset_size_multiplier))
        # print(f"{n_iter=}")
        start_time = time.perf_counter()
        theta_hat, history = optimizer.optimize(
            objective=ls.eval_ssr_grad_ssr,
            theta0=theta_initial,
            n_iter=n_iter,
            constraints=model.constraints
        )
        end_time = time.perf_counter()

        self.elapsed_time = end_time - start_time
        self.theta_hat = theta_hat
        self.history = history
        

    def evaluate(self, theta_true, alpha=0.05):
        model = self.model
        path = self.path
        # may be convenient to evaluate using ODE solvers
        # if hasattr(model, "period"):
        #     self.dl = DiscreteLikelihoodPeriodicAmortization(model, path, model.period)
        #     self.periodic = True
        # else:
        #     self.dl = DiscreteLikelihood(model, path)
        #     self.periodic = False
    

        if hasattr(model, "period"):
            self.ls_eval = objectives.ConditionalLeastSquaresPeriodicAmortization(model, path, self.h)
            self.periodic = True
        else:
            self.ls_eval = objectives.ConditionalLeastSquares(model, path, self.h)
            self.periodic = False
        
        checkpoint_thetas = []
        checkpoint_inside = []
        checkpoint_CI_volumes = []
        checkpoint_objectives = []
        checkpoint_euclidean = []

        for k in self.checkpoint_multiples:
            idx = int(np.ceil(k * len(path.states) / self.batch_size)) - 1
            checkpoint_theta = self.history[self.sgd_method.output_field][idx]

            f, grad, _ = self.ls_eval.eval_ssr_grad_ssr(checkpoint_theta)
            prec = self.ls_eval.prec_matrix(checkpoint_theta)

            is_good_CI, distance_sq, threshold, volume_ci = in_confidence_region(theta_true, checkpoint_theta, prec, 0.05)
            checkpoint_inside.append(is_good_CI)
            checkpoint_CI_volumes.append(volume_ci)

            print(f"{k=}, {idx=}, {is_good_CI=}, {checkpoint_theta=}, {distance_sq=}, {threshold=}, {volume_ci=}")

            euclidean_distance = np.linalg.norm(checkpoint_theta - theta_true)
            checkpoint_euclidean.append(euclidean_distance)
            
            checkpoint_thetas.append(checkpoint_theta)
            checkpoint_objectives.append(f)
        
        # now eval for theta hat
        f, grad, _ = self.ls.eval_ssr_grad_ssr(self.theta_hat)

        prec = self.ls_eval.prec_matrix(self.theta_hat)

        is_good_CI, distance_sq, threshold, volume_ci = in_confidence_region(theta_true, self.theta_hat, prec, 0.05)

        euclidean_distance = np.linalg.norm(self.theta_hat - theta_true)

        self.results = {
            # scalars
            'converged': np.linalg.norm(grad) < 1e-2, # this is ad-hoc
            'objective': f,
            'is_good_CI': is_good_CI,
            'CI_volume': volume_ci,
            'euclidean_distance': euclidean_distance,

            'optimizer_iterations': len(self.history['theta']),
            'optimization_runtime': self.elapsed_time,

            # non-scalars
            'theta_hat': self.theta_hat,
            'theta_star': theta_true,
            'theta_initial': self.theta_initial,
            'prec': prec,

            # checkpoints
            'checkpoint': {
                'multiple': self.checkpoint_multiples,
                'theta': checkpoint_thetas,
                'inside': checkpoint_inside,
                'CI_volume': checkpoint_CI_volumes,
                'objective': checkpoint_objectives,
                # 'scores': checkpoint_scores,
                'euclidean_distance': checkpoint_euclidean,
            },

            'history': self.history,
        }

    def evaluate_detailed(self, theta_true: np.ndarray, cls_ode_theta_hat: np.ndarray, cls_ode_prec: np.ndarray, alpha=0.05):
        model = self.model
        path = self.path

        if hasattr(model, "period"):
            self.ls_eval = objectives.ConditionalLeastSquaresPeriodicAmortization(model, path, self.h)
            self.periodic = True
        else:
            self.ls_eval = objectives.ConditionalLeastSquares(model, path, self.h)
            self.periodic = False

        
        checkpoint_theta_last_iterate = []
        checkpoint_inside_last = []
        checkpoint_CI_volume_last = []
        checkpoint_objective_last  = []
        checkpoint_euclidean_to_true_last = []
        checkpoint_euclidean_to_ode_last = []
        checkpoint_ode_in_sgd_CR_last = []
        checkpoint_sgd_in_ode_CR_last = []
        checkpoint_prec_last = []

        checkpoint_theta_pr_iterate = []
        checkpoint_inside_pr = []
        checkpoint_CI_volume_pr = []
        checkpoint_objective_pr  = []
        checkpoint_euclidean_to_true_pr = []
        checkpoint_euclidean_to_ode_pr = []
        checkpoint_ode_in_sgd_CR_pr = []
        checkpoint_sgd_in_ode_CR_pr = []
        checkpoint_prec_pr = []

        for k in self.checkpoint_multiples:
            idx = int(np.ceil(k * len(path.states) / self.batch_size)) - 1

            ### last-iterate ###
            checkpoint_theta_last = self.history['theta'][idx]
            f, grad, _ = self.ls_eval.eval_ssr_grad_ssr(checkpoint_theta_last)

            # is theta star within last iterate confidence region?
            prec = self.ls_eval.prec_matrix(checkpoint_theta_last)
            checkpoint_prec_last.append(prec)

            is_good_CI, distance_sq, threshold, volume_ci = in_confidence_region(theta_true, checkpoint_theta_last, prec, alpha)
            checkpoint_inside_last.append(is_good_CI)
            checkpoint_CI_volume_last.append(volume_ci)

            print(f"{k=}, {idx=}, {is_good_CI=}, {checkpoint_theta_last=}, {distance_sq=}, {threshold=}, {volume_ci=}")

            euclidean_distance = np.linalg.norm(checkpoint_theta_last - theta_true)
            checkpoint_euclidean_to_true_last.append(euclidean_distance)

            euclidean_distance = np.linalg.norm(checkpoint_theta_last - cls_ode_theta_hat)
            checkpoint_euclidean_to_ode_last.append(euclidean_distance)
            
            checkpoint_theta_last_iterate.append(checkpoint_theta_last)
            checkpoint_objective_last.append(f)

            # is theta hat (ODE) within last iterate confidence region (SGD)?
            is_good_CI, distance_sq, threshold, volume_ci = in_confidence_region(cls_ode_theta_hat, checkpoint_theta_last, prec, alpha)
            checkpoint_ode_in_sgd_CR_last.append(is_good_CI)
            
            # is last iterate (SGD) within ODE confidence region of theta hat (ODE)?
            is_good_CI, distance_sq, threshold, volume_ci = in_confidence_region(checkpoint_theta_last, cls_ode_theta_hat, cls_ode_prec, alpha)
            checkpoint_sgd_in_ode_CR_last.append(is_good_CI)

            ### PR-averaged iterate ###
            checkpoint_theta_pr = self.history['avg_theta'][idx]
            f, grad, _ = self.ls_eval.eval_ssr_grad_ssr(checkpoint_theta_pr)

            # is theta star within pr iterate confidence region?
            prec = self.ls_eval.prec_matrix(checkpoint_theta_pr)
            checkpoint_prec_pr.append(prec)

            is_good_CI, distance_sq, threshold, volume_ci = in_confidence_region(theta_true, checkpoint_theta_pr, prec, alpha)
            checkpoint_inside_pr.append(is_good_CI)
            checkpoint_CI_volume_pr.append(volume_ci)

            print(f"{k=}, {idx=}, {is_good_CI=}, {checkpoint_theta_pr=}, {distance_sq=}, {threshold=}, {volume_ci=}")

            euclidean_distance = np.linalg.norm(checkpoint_theta_pr - theta_true)
            checkpoint_euclidean_to_true_pr.append(euclidean_distance)

            euclidean_distance = np.linalg.norm(checkpoint_theta_pr - cls_ode_theta_hat)
            checkpoint_euclidean_to_ode_pr.append(euclidean_distance)
            
            checkpoint_theta_pr_iterate.append(checkpoint_theta_pr)
            checkpoint_objective_pr.append(f)

            # is theta hat (ODE) within pr iterate confidence region (SGD)?
            is_good_CI, distance_sq, threshold, volume_ci = in_confidence_region(cls_ode_theta_hat, checkpoint_theta_pr, prec, alpha)
            checkpoint_ode_in_sgd_CR_pr.append(is_good_CI)
            
            # is pr iterate (SGD) within ODE confidence region of theta hat (ODE)?
            is_good_CI, distance_sq, threshold, volume_ci = in_confidence_region(checkpoint_theta_pr, cls_ode_theta_hat, cls_ode_prec, alpha)
            checkpoint_sgd_in_ode_CR_pr.append(is_good_CI)

        prec = self.ls_eval.prec_matrix(self.theta_hat)

        self.thesis = {
            # scalars
            'objective_last': checkpoint_objective_last[-1],
            'is_good_CI_last': checkpoint_inside_last[-1],
            'CI_volume_last': checkpoint_CI_volume_last[-1],
            'euclidean_distance_true_last': checkpoint_euclidean_to_true_last[-1],
            'euclidean_distance_ode_last': checkpoint_euclidean_to_ode_last[-1],

            'objective_pr': checkpoint_objective_pr[-1],
            'is_good_CI_pr': checkpoint_inside_pr[-1],
            'CI_volume_pr': checkpoint_CI_volume_pr[-1],
            'euclidean_distance_true_pr': checkpoint_euclidean_to_true_pr[-1],
            'euclidean_distance_ode_pr': checkpoint_euclidean_to_ode_pr[-1],

            'optimizer_iterations': len(self.history['theta']),
            'optimization_runtime': self.elapsed_time,

            # non-scalars
            'theta_hat_last': checkpoint_theta_last_iterate[-1],
            'theta_hat_pr': checkpoint_theta_pr_iterate[-1],
            'theta_star': theta_true,
            'theta_initial': self.theta_initial,
            'prec_last': checkpoint_prec_last[-1],
            'prec_pr': checkpoint_prec_pr[-1],
            'history': self.history,

            # checkpoints
            'checkpoint': {
                'checkpoint_multiples': self.checkpoint_multiples,

                'checkpoint_theta_last': checkpoint_theta_last,
                'checkpoint_inside_last': checkpoint_inside_last,
                'checkpoint_CI_volume_last': checkpoint_CI_volume_last,
                'checkpoint_objective_last': checkpoint_objective_last ,
                'checkpoint_euclidean_to_true_last': checkpoint_euclidean_to_true_last,
                'checkpoint_euclidean_to_ode_last': checkpoint_euclidean_to_ode_last,
                'checkpoint_ode_in_sgd_CR_last': checkpoint_ode_in_sgd_CR_last,
                'checkpoint_sgd_in_ode_CR_last': checkpoint_sgd_in_ode_CR_last,
                'checkpoint_prec_last': checkpoint_prec_last,

                'checkpoint_theta_pr': checkpoint_theta_pr,
                'checkpoint_inside_pr': checkpoint_inside_pr,
                'checkpoint_CI_volume_pr': checkpoint_CI_volume_pr,
                'checkpoint_objective_pr': checkpoint_objective_pr ,
                'checkpoint_euclidean_to_true_pr': checkpoint_euclidean_to_true_pr,
                'checkpoint_euclidean_to_ode_pr': checkpoint_euclidean_to_ode_pr,
                'checkpoint_ode_in_sgd_CR_pr': checkpoint_ode_in_sgd_CR_pr,
                'checkpoint_sgd_in_ode_CR_pr': checkpoint_sgd_in_ode_CR_pr,
                'checkpoint_prec_pr': checkpoint_prec_pr,
            }
        }

    def save_thesis(self, fname):
        np.savez(fname, **self.thesis)


@register_estimator
class MLEEndogenousODE(Estimator):
    def __init__(
            self,
            model: MJPModel,
            ):
        super().__init__(model)
        self.dl = None
        self.res = None

    def fit(
            self,
            path: EndogenousSample,
            theta_initial: np.ndarray,
            xtol=1e-1,
            gtol=1e-1,
    ):
        if type(path) is not EndogenousSample:
            print("Warning: path is not EndogenousSample type")
        model = self.model
        self.theta_initial = theta_initial

        self.periodic = False
        self.endo_likelihood = objectives.EndogenousLikelihoodODE(model, path)
        
        self.xtol = xtol 
        self.gtol = gtol

        bhhh = optimizers.BHHH(self.endo_likelihood.eval_log_likelihood_scores, tol=xtol, constraints=model.constraints)
        start_time = time.perf_counter()
        theta_hat, history = bhhh.optimize(theta_initial) 
        end_time = time.perf_counter()
        self.elapsed_time = end_time - start_time
        self.theta_hat = theta_hat

        self.history = history

        self.is_converged = bhhh.is_converged
        self.optimizer_iterations = len(self.history['x'])
        self.func_evaluations =len(self.history['x'])
        self.grad_evaluations =len(self.history['x'])
        self.optimal_objective = self.history['f'][-1]




    def evaluate(self, theta_true, alpha=0.05):
        theta_hat = self.theta_hat
        p = len(theta_hat) 
        
        # get (n x p) scores matrix at the mle
        _, _, scores = self.endo_likelihood.eval_log_likelihood_scores(self.theta_hat)

        fisher_information_approx = scores.T @ scores
        C_hat = sqrtm(fisher_information_approx)
        self.prec = fisher_information_approx
        is_good_CI, _, _, volume_ci = in_confidence_region(theta_true, theta_hat, fisher_information_approx, alpha)
        euclidean_distance = np.linalg.norm(theta_hat - theta_true)
        optimization_runtime = self.elapsed_time
        time_to_compute_optimal_schedule = None
        efficiency_gain_from_optimal_schedule =  None

        self.results = {
            # scalars
            'converged': self.is_converged,
            'objective': self.optimal_objective,
            'is_good_CI': is_good_CI,
            'CI_volume': volume_ci,
            'euclidean_distance': euclidean_distance,
            'optimizer_iterations': self.optimizer_iterations,
            'func_evaluations': self.func_evaluations,
            'grad_evaluations': self.grad_evaluations,
            'optimization_runtime': optimization_runtime,
            'time_to_compute_optimal_schedule': time_to_compute_optimal_schedule,
            'efficiency_gain_from_optimal_schedule': efficiency_gain_from_optimal_schedule,

            'optimization_algo': 'trust-constr',
            'xtol': self.xtol,
            'gtol': self.gtol,

            # non-scalars
            'theta_hat': theta_hat,
            'theta_star': theta_true,
            'theta_initial': self.theta_initial,
            'scores': scores,
            'fisher_information': fisher_information_approx,
            'C_hat': C_hat,
            'history': self.history,
        }