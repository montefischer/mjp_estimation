import numpy as np
from scipy.stats import chi2
from scipy.special import gamma
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt

def in_confidence_region(y, x, A, alpha):
    """
    Check if a point y lies within the (1 - alpha) confidence region 
    of a multivariate normal with mean x and precision matrix A.
    
    Parameters
    ----------
    y : array-like, shape (d,)
        Candidate point to test.
    x : array-like, shape (d,)
        Mean vector of the multivariate normal.
    A : array-like, shape (d, d)
        Precision matrix (inverse of covariance matrix) of multivariate normal.
    alpha : float, optional (default=0.05)
        Significance level. Confidence level = 1 - alpha.
    
    Returns
    -------
    inside : bool
        True if y is inside the (1 - alpha) confidence region.
    distance_sq : float
        The squared Mahalanobis distance.
    threshold : float
        The chi-squared threshold.
    volume: float
        Volume of the confidence region
    """
    y = np.asarray(y)
    x = np.asarray(x)
    A = np.asarray(A)
    
    diff = y - x
    distance_sq = diff.T @ A @ diff
    
    d = len(x)
    threshold = chi2.ppf(1 - alpha, df=d)

    det_A = np.linalg.det(A)
    det_Sigma = 1.0 / det_A

    unit_ball_vol = (np.pi ** (d / 2)) / gamma((d / 2) + 1)
    volume = unit_ball_vol * (threshold ** (d / 2)) * np.sqrt(det_Sigma)
    
    return distance_sq <= threshold, distance_sq, threshold, volume


def in_confidence_region_cov(y, x, Sigma, alpha=0.05):
    """
    Confidence region membership test using covariance matrix Σ directly.
    """
    y = np.asarray(y)
    x = np.asarray(x)
    diff = y - x
    d = len(x)

    L, lower = cho_factor(Sigma, lower=True, check_finite=False)
    z = cho_solve((L, lower), diff)  # solves Σ z = diff
    distance_sq = diff @ z
    
    # chi-square threshold
    threshold = chi2.ppf(1 - alpha, df=d)
    
    # determinant using Cholesky (det Σ = (prod diag(L))^2)
    logdetSigma = 2 * np.sum(np.log(np.diag(L)))
    
    # unit ball volume
    unit_ball_vol = (np.pi ** (d/2)) / gamma((d/2) + 1)
    
    # volume of ellipsoid
    volume = unit_ball_vol * (threshold ** (d/2)) * np.exp(0.5 * logdetSigma)

    inside = distance_sq <= threshold
    return inside, distance_sq, threshold, volume

def plot_confidence_ellipse(x, prec, alpha=0.05, y_points=None, ax=None, **kwargs):
    """
    Plot the (1 - alpha) confidence ellipse of a 2D multivariate normal 
    with mean x and precision matrix A.
    
    Parameters
    ----------
    x : array-like, shape (2,)
        Mean vector of the multivariate normal.
    A : array-like, shape (2, 2)
        Precision matrix (inverse of covariance matrix).
    alpha : float, optional
        Significance level. Confidence level = 1 - alpha.
    y_points : array-like, shape (n_points, 2), optional
        Points to overlay on the plot.
    ax : matplotlib.axes.Axes, optional
        Existing matplotlib axis to plot on. If None, creates a new figure.
    **kwargs :
        Additional plotting arguments for the ellipse boundary (e.g., color, lw).
    """
    x = np.asarray(x)
    prec = np.asarray(prec)

    if x.shape[0] != 2 or prec.shape != (2, 2):
        raise ValueError("This function only supports 2D data.")
    Sigma = np.linalg.inv(prec)
    threshold = chi2.ppf(1 - alpha, df=2)
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    axis_lengths = np.sqrt(eigvals * threshold)
    # Parameterize ellipse
    theta = np.linspace(0, 2 * np.pi, 300)
    circle = np.array([np.cos(theta), np.sin(theta)])  # unit circle
    ellipse = (eigvecs @ np.diag(axis_lengths) @ circle) + x[:, np.newaxis]

    # Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot(ellipse[0, :], ellipse[1, :], **kwargs)
    ax.scatter(x[0], x[1], c='red', label='Mean', zorder=5)

    if y_points is not None:
        y_points = np.atleast_2d(y_points)
        ax.scatter(y_points[:, 0], y_points[:, 1], c='blue', label='Points', alpha=0.7)

    ax.set_aspect('equal')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.legend()
    ax.grid(True)
    ax.set_title(f'{100*(1 - alpha):.1f}% Confidence Ellipse')

    plt.show()