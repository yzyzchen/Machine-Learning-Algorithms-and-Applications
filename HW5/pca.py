"""EECS545 HW5: PCA."""

from typing import Tuple
import numpy as np


def hello_world():
    print("Hello world from EECS 545 PCA!")


def train_PCA(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Run PCA on the data.

    Input:
        data: A numpy array of shape [N, d], where N is the number of data
            points and d is the dimension of each data point.
            We assume the data has full rank.

    Returns: A tuple of (U, eigenvalues)
        U: The U matrix, whose column vectors are principal components
            (i.e., eigenvectors) in the order of decreasing variance.
        eigenvalues:
            An array (or list) of all eigenvalues sorted in a decreasing order.
    """
    if len(data.shape) != 2:
        raise ValueError("Invalid shape of data; did you forget flattening?")
    N, d = data.shape

    #######################################################
    ###              START OF YOUR CODE                 ###
    #######################################################
    # raise NotImplementedError("TODO: Add your implementation here.")
    # Subtract the mean from the data
    data_mean = np.mean(data, axis=0)
    centered_data = data - data_mean

    # Compute the covariance matrix
    covariance_matrix = np.cov(centered_data.T)

    # Compute the eigenvectors and eigenvalues
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Sort the eigenvectors and eigenvalues in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    U = eigenvectors  # The principal components are the eigenvectors
    #######################################################
    ###                END OF YOUR CODE                 ###
    #######################################################

    return U, eigenvalues
