"""EECS545 HW5 Q1. K-means"""

import numpy as np
from typing import NamedTuple, Union, Literal
from scipy.stats import multivariate_normal

def e_step(train_data, pi, mu, sigma, K, N):
    # Vectorized e-step
    gamma = np.zeros((N, K))
    for k in range(K):
        rv = multivariate_normal(mean=mu[k], cov=sigma[k], allow_singular=True)
        gamma[:, k] = rv.pdf(train_data) * pi[k]
    gamma /= np.sum(gamma, axis=1, keepdims=True)
    return gamma

def m_step(train_data, gamma, K, N, d):
    Nk = np.sum(gamma, axis=0) + 1e-6  # prevent division by zero
    pi = Nk / N
    mu = np.dot(gamma.T, train_data) / Nk[:, None]
    
    sigma = np.zeros((K, d, d))
    for k in range(K):
        X_centered = train_data - mu[k]
        # Create a diagonal covariance matrix for numerical stability
        sigma[k] = np.dot(X_centered.T * gamma[:, k], X_centered) / Nk[k]
        sigma[k] += np.eye(d) * 1e-6  # Regularization to prevent singular matrices
    return pi, mu, sigma

def hello():
    print('Hello from gmm.py!')


class GMMState(NamedTuple):
    """Parameters to a GMM Model."""
    pi: np.ndarray  # [K]
    mu: np.ndarray  # [K, d]
    sigma: np.ndarray  # [K, d, d]


def train_gmm(train_data: np.ndarray,
              init_pi: np.ndarray,
              init_mu: np.ndarray,
              init_sigma: np.ndarray,
              *,
              num_iterations: int = 50,
              ) -> GMMState:
    """Fit a GMM model.

    Arguments:
        train_data: A numpy array of shape (N, d), where
            N is the number of data points
            d is the dimension of each data point. Note: you should NOT assume
              d is always 3; rather, try to implement a general K-means.
        init_pi: The initial value of pi. Shape (K, )
        init_mu: The initial value of mu. Shape (K, d)
        init_sigma: The initial value of sigma. Shape (K, d, d)
        num_iterations: Run EM (E-steps and M-steps) for this number of
            iterations.

    Returns:
        A GMM parameter after running `num_iterations` number of EM steps.
    """
    # Sanity check
    N, d = train_data.shape
    K, = init_pi.shape
    assert init_mu.shape == (K, d)
    assert init_sigma.shape == (K, d, d)

    ###########################################################################
    # Implement EM algorithm for learning GMM.
    ###########################################################################
    # TODO: Add your implementation.
    # Feel free to add helper functions as much as needed.
    pi, mu, sigma = init_pi.copy(), init_mu.copy(), init_sigma.copy()
    for _ in range(num_iterations):
        gamma = e_step(train_data, pi, mu, sigma, K, N)
        pi, mu, sigma = m_step(train_data, gamma, K, N, d)
        if np.isnan(sigma).any():
            raise ValueError("Sigma contains NaNs, possibly due to numerical instability.")
    #######################################################################
    # raise NotImplementedError("TODO: Add your implementation here.")
    #######################################################################

    return GMMState(pi, mu, sigma)


# # Helper function for image compression
# def assign_pixels_to_components(image_data, mu):
#     N, d = image_data.shape
#     K, _ = mu.shape
#     distances = np.linalg.norm(image_data[:, None] - mu, axis=2)
#     closest_components = np.argmin(distances, axis=1)
#     compressed_data = mu[closest_components]
#     return compressed_data


def compress_image(image: np.ndarray, gmm_model: GMMState) -> np.ndarray:
    """Compress image by mapping each pixel to the mean value of a
    Gaussian component (hard assignment).

    Arguments:
        image: A numpy array of shape (H, W, 3) and dtype uint8.
        gmm_model: type GMMState. A GMM model parameters.
    Returns:
        compressed_image: A numpy array of (H, W, 3) and dtype uint8.
            Be sure to round off to the nearest integer.
    """
    H, W, C = image.shape
    K = gmm_model.mu.shape[0]

    ##########################################################################
    # # raise NotImplementedError("TODO: Add your implementation here.")
    # image_data = image.reshape(-1, C)
    # likelihoods = np.zeros((len(image_data), K))

    # for k in range(K):
    #     distribution = multivariate_normal(mean=gmm_model.mu[k], cov=gmm_model.sigma[k], allow_singular=True)
    #     likelihoods[:, k] = distribution.pdf(image_data)

    # best_indices = np.argmax(likelihoods, axis=1)
    # compressed_data = gmm_model.mu[best_indices]
    # compressed_data = np.clip(compressed_data, 0, 255)
    # compressed_image = compressed_data.reshape(H, W, C).astype(np.uint8)

    pixels = image.reshape(-1, C)
    posteriors = np.zeros((len(pixels), K))
    for k in range(K):
        distribution = multivariate_normal(mean=gmm_model.mu[k], cov=gmm_model.sigma[k], allow_singular=True)
        posteriors[:, k] = distribution.pdf(pixels) * gmm_model.pi[k]
    best_component_indices = np.argmax(posteriors, axis=1)
    compressed_pixels = gmm_model.mu[best_component_indices]
    compressed_pixels = np.clip(compressed_pixels, 0, 255)
    compressed_image = compressed_pixels.reshape(H, W, C).astype(np.uint8)

    ##########################################################################

    assert compressed_image.dtype == np.uint8
    assert compressed_image.shape == (H, W, C)
    return compressed_image
