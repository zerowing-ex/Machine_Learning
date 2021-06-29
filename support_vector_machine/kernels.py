import numpy as np


def linear(xi, xj):
    return np.dot(xi.T, xj)


def poly(xi, xj, beta=0.5, theta=0.5, d=2):
    if d < 1:
        raise ValueError(f"The order d of the polynomial is not correct, got {d}, expected d >= 1")
    return (beta * np.dot(xi.T, xj) + theta) ** d


def gauss(xi, xj, sigma=0.5):
    if sigma <= 0:
        raise ValueError(f"The width sigma of the gauss is not correct, got {sigma}, expected sigma > 0")
    return np.exp(-np.square(np.linalg.norm(xi - xj)) / 2 / sigma / sigma)


def laplace(xi, xj, sigma=0.5):
    if sigma <= 0:
        raise ValueError(f"The param sigma of the laplace is not correct, got {sigma}, expected sigma > 0")
    return np.exp(-np.linalg.norm(xi - xj) / sigma)


def Sigmoid(xi, xj, beta=0.5, theta=0.5):
    if beta <= 0:
        raise ValueError(f"The param beta of the laplace is not correct, got {beta}, expected beta > 0")
    if theta <= 0:
        raise ValueError(f"The param theta of the laplace is not correct, got {theta}, expected theta > 0")
    np.tanh(beta * np.dot(xi.T, xj) + theta)
