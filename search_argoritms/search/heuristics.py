import numpy as np


def l1_norm(p1, p2):
    """
    Computes the L1 norm distance between two n-dimensional points
    :param p1: first point
    :param p2: second point
    :return: L1 norm distance value
    """
    return np.sum(np.abs(np.asarray(p1) - np.asarray(p2)))


def l2_norm(p1, p2):
    """
    Computes the L1 norm distance between two n-dimensional points
    :param p1: first point
    :param p2: second point
    :return: L1 norm distance value
    """
    return np.linalg.norm((np.asarray(p1), np.asarray(p2)))


def chebyshev(p1, p2):
    """
    Computes the Chebyshev distance, between two n-dimensional points
    :param p1: first point
    :param p2: second point
    :return: Chebyshev distance value
    """
    return np.max(np.abs(np.asarray(p1) - np.asarray(p2)))
