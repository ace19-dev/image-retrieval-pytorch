import numpy as np

import torch.nn.functional as F


def _pdist(a, b):
    """Compute pair-wise squared distance between points in `a` and `b`.

    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.

    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that element (i, j)
        contains the squared distance between `a[i]` and `b[j]`.

    """
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a2, b2 = np.square(a).sum(axis=1), np.square(b).sum(axis=1)
    r2 = -2. * np.dot(a, b.T) + a2[:, None] + b2[None, :]
    r2 = np.clip(r2, 0., float(np.inf))
    return r2


# TODO: modify
def _cosine_distance(a, b, data_is_normalized=False):
    """Compute pair-wise cosine distance between points in `a` and `b`.

    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.
    data_is_normalized : Optional[bool]
        If True, assumes rows in a and b are unit length vectors.
        Otherwise, a and b are explicitly normalized to lenght 1.

    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that element (i, j)
        contains the squared distance between `a[i]` and `b[j]`.

    """
    a = a.numpy()
    b = b.numpy()

    if not data_is_normalized:
        # To avoid RuntimeWarning: invalid value encountered in true_divide import numpy as np
        # a_normed = F.normalize(a, p=2, dim=1, eps=1e-8)
        a_normed = np.linalg.norm(a, axis=1, keepdims=True)
        a = np.asarray(a) / np.where(a_normed==0, 1, a_normed)
        # b_normed = F.normalize(a, p=2, dim=1, eps=1e-8)
        b_normed = np.linalg.norm(b, axis=1, keepdims=True)
        b = np.asarray(b) / np.where(b_normed==0, 1, b_normed)
    else:
        a = np.asarray(a)
        b = np.asarray(b)

    return 1. - np.dot(a, b.T)


def _nn_euclidean_distance(x, y):
    """ Helper function for nearest neighbor distance metric (Euclidean).

    Parameters
    ----------
    x : ndarray
        A matrix of N row-vectors (sample points).
    y : ndarray
        A matrix of M row-vectors (query points).

    Returns
    -------
    ndarray
        A vector of length M that contains for each entry in `y` the
        smallest Euclidean distance to a sample in `x`.

    """
    distances = _pdist(x, y)
    return np.maximum(0.0, distances.min(axis=0))


def _nn_cosine_distance(x, y):
    """ Helper function for nearest neighbor distance metric (cosine).

    Parameters
    ----------
    x : ndarray
        A matrix of M row-vectors (query points).
    y : ndarray
        A matrix of N row-vectors (gallery points).

    Returns
    -------
    # ndarray
    #     A vector of length M that contains for each entry in `y` the
    #     smallest cosine distance to a sample in `x`.

    """
    distances = _cosine_distance(x, y)
    return distances


class NearestNeighborDistanceMetric(object):
    """
    A nearest neighbor distance metric that, for each target, returns
    the closest distance to any sample that has been observed so far.

    Parameters
    ----------
    metric : str
        Either "euclidean" or "cosine".
    matching_threshold: float
        The matching threshold. Samples with larger distance are considered an
        invalid match.
    budget : Optional[int]
        If not None, fix samples per class to at most this number. Removes
        the oldest samples when the budget is reached.

    Attributes
    ----------
    samples : Dict[int -> List[ndarray]]
        A dictionary that maps from target identities to the list of samples
        that have been observed so far.

    """

    def __init__(self, metric, matching_threshold=None, budget=None):
        if metric == "euclidean":
            self._metric = _nn_euclidean_distance
        elif metric == "cosine":
            self._metric = _nn_cosine_distance
        else:
            raise ValueError(
                "Invalid metric; must be either 'euclidean' or 'cosine'")
        self.matching_threshold = matching_threshold
        self.budget = budget    # Gating threshold for cosine distance
        self.samples = {}

    def distance(self, queries, galleries):
        """Compute distance between galleries and queries.

        Parameters
        ----------
        queries : ndarray
            An LxM matrix of L features of dimensionality M to match the given `galleries` against.
        galleries : ndarray
            An NxM matrix of N features of dimensionality M.

        Returns
        -------
        ndarray
            Returns a cost matrix of shape LxN

        """
        return self._metric(queries, galleries)
