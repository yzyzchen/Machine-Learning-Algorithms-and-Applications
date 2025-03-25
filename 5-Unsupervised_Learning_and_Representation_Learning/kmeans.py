"""EECS545 HW5 Q1. K-means"""

import numpy as np
import sklearn.metrics


def hello():
    print('Hello from kmeans.py!')


def euclidean_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute the pixel error between the data and compressed data.

    Please do not change this function!

    Arguments:
        x: A numpy array of shape (N*, d), where d is the data dimension.
        y: A numpy array of shape (N*, d), where d is the data dimension.
    Return:
        errors: A numpy array of shape (N*). Euclidean distances.
    """
    assert x.shape == y.shape
    error = np.sqrt(np.sum(np.power(x - y, 2), axis=-1))
    return error


def train_kmeans(train_data: np.ndarray, initial_centroids, *,
                 num_iterations: int = 50):
    """K-means clustering.

    Arguments:
        train_data: A numpy array of shape (N, d), where
            N is the number of data points
            d is the dimension of each data point. Note: you should NOT assume
              d is always 3; rather, try to implement a general K-means.
        initial_centroids: A numpy array of shape (K, d), where
            K is the number of clusters. Each data point means the initial
            centroid of cluster. You should NOT assume K = 16.
        num_iterations: Run K-means algorithm for this number of iterations.

    Returns:
        centroids: A numpy array of (K, d), the centroid of K-means clusters
            after convergence.
    """
    # Sanity check
    N, d = train_data.shape
    K, d2 = initial_centroids.shape
    if d != d2:
        raise ValueError(f"Invalid dimension: {d} != {d2}")

    assert train_data.dtype.kind == 'f'

    ###########################################################################
    # Implement K-means algorithm.
    ###########################################################################
    centroids = initial_centroids.copy()
    for i in range(num_iterations):

        #######################################################################
        # raise NotImplementedError("TODO: Add your implementation here.")
        #######################################################################
        distances = np.linalg.norm(train_data[:, np.newaxis] - centroids, axis=2)
        closest_centroids = np.argmin(distances, axis=1)
        new_centroids = np.array([train_data[closest_centroids == k].mean(axis=0) if k in closest_centroids else centroids[k] for k in range(K)])
        
        # Check if centroids have changed, if not, break from the loop
        if np.allclose(centroids, new_centroids, rtol=1e-6):
            break
        
        centroids = new_centroids

    assert centroids.shape == (K, d)
    return centroids


def compress_image(image: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Compress image by mapping each pixel to the closest centroid.

    Arguments:
        image: A numpy array of shape (H, W, 3) and dtype uint8.
        centroids: A numpy array of shape (K, 3), each row being the centroid
            of a cluster.
    Returns:
        compressed_image: A numpy array of (H, W, 3) and dtype uint8.
            Be sure to round off to the nearest integer.
    """
    H, W, C = image.shape
    K, C2 = centroids.shape
    assert C == C2 == 3, "Invalid number of channels."
    assert image.dtype == np.uint8

    reshaped_image = image.reshape((-1, 3))
    distances = np.sqrt(np.sum((reshaped_image[:, np.newaxis, :] - centroids) ** 2, axis=2))
    closest_centroids = np.argmin(distances, axis=1)
    compressed_image = centroids[closest_centroids]
    compressed_image = compressed_image.reshape((H, W, C)).astype(np.uint8)

    # raise NotImplementedError("TODO: Add your implementation here.")
    assert compressed_image.dtype == np.uint8
    assert compressed_image.shape == (H, W, C)
    return compressed_image
