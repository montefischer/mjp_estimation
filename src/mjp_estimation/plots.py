from typing import Iterable, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from .base_functionality import SamplePath
from .jackson_model import vec_decode

def plot_sample_path(path: SamplePath, num_states: int = None):
    return plot_sample_paths([path], num_states)
    
def plot_sample_paths(paths: Iterable[SamplePath], num_states: int = None, labels: Iterable[str] = None, filepath=None):
    if num_states is None:
        num_states = np.max(np.concatenate([np.unique_values(path.states) for path in paths]))
    
    plt.figure(figsize=(5, 3))
    plt.grid(True, which='both', linestyle='--')
    for n,path in enumerate(paths):
        plt.step(path.times, path.states, where='post', marker='', label=labels[n] if labels is not None else None)
    plt.ylim((0, num_states))
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlim((0, np.max([path.final_time for path in paths])))
    if labels is not None:
        plt.legend()

    if filepath is not None:
        plt.savefig(filepath)
    plt.show()

def plot_jackson_paths(path: SamplePath, d_vec: Sequence[int], filepath=None):
    d = len(d_vec)
    v = vec_decode(path.states, d_vec)

    system_paths = []
    for i in range(d):
        sys_i_states = v[:,i]
        system_paths.append(SamplePath(path.times, sys_i_states, path.times[-1], path.states[-1]))

    plot_sample_paths(system_paths, labels=np.arange(1, d+1), filepath=filepath)

def plot_parameter_convergence(estimator, theta_target):
    plt.plot(estimator.history['theta'])
    if 'avg_theta' in estimator.history.keys():
        plt.plot(estimator.history['avg_theta'])
    plt.hlines(theta_target, xmin=0, xmax=len(estimator.history['theta']), linestyle='--', color='black')
    plt.show()

def plot_stepsize_decay(estimator):
    plt.plot(estimator.history['eta_t'])
    plt.ylim(bottom=0)
    plt.show()

def plot_planned_stepsize_decay(sgd_method, n=1000):
    plt.plot(sgd_method._schedule(np.arange(n)))
    plt.ylim(bottom=0)
    plt.show()