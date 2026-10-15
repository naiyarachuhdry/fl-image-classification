"""FedPFT helpers: fit per-class GMMs on features, pack/unpack them, sample from them.

Ported from the Flower FedPFT baseline (utils.py / strategy.py).
"""
from dataclasses import dataclass

import numpy as np
from sklearn.mixture import GaussianMixture


@dataclass
class GMMParameters:
    """Parameters of one class's GMM (what a client sends to the server)."""

    label: np.ndarray        # shape (1,)   class id
    means: np.ndarray        # (n_mixtures, feat_dim)
    weights: np.ndarray      # (n_mixtures,)
    covariances: np.ndarray  # spherical: (n_mixtures,)
    num_samples: np.ndarray  # shape (1,)   number of features of this class


def fit_class_gmms(features, labels, n_mixtures=1, cov_type="spherical",
                   seed=0, tol=1e-12, max_iter=10000):
    """Fit one GMM per class present in `labels`. Returns list[GMMParameters]."""
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels)
    out = []
    for cls in np.unique(labels):
        x = features[labels == cls]
        if len(x) < 2:  # sklearn GMM needs >= 2 samples: use the point itself
            out.append(GMMParameters(
                label=np.array([cls], dtype=np.int64),
                means=x.astype(np.float16).reshape(1, -1),
                weights=np.ones(1, dtype=np.float16),
                covariances=np.full(1, 1e-3, dtype=np.float16),
                num_samples=np.array([len(x)], dtype=np.int64),
            ))
            continue
        k = max(1, min(n_mixtures, len(x)))  # cannot have more mixtures than samples
        gmm = GaussianMixture(n_components=k, covariance_type=cov_type,
                              random_state=seed, tol=tol, max_iter=max_iter)
        gmm.fit(x)
        out.append(GMMParameters(
            label=np.array([cls], dtype=np.int64),
            means=gmm.means_.astype(np.float16),
            weights=gmm.weights_.astype(np.float16),
            covariances=gmm.covariances_.astype(np.float16),
            num_samples=np.array([len(x)], dtype=np.int64),
        ))
    return out


def pack(params):
    """list[GMMParameters] -> flat list of 5 ndarrays per class (for ArrayRecord)."""
    flat = []
    for p in params:
        flat += [p.label, p.means, p.weights, p.covariances, p.num_samples]
    return flat


def unpack(arrays):
    """Inverse of pack()."""
    arrays = list(arrays)
    assert len(arrays) % 5 == 0, "GMM arrays must come in groups of 5"
    return [GMMParameters(*arrays[i:i + 5]) for i in range(0, len(arrays), 5)]


def sample_features(p, cov_type="spherical", seed=0):
    """Draw p.num_samples synthetic features from one class's GMM.

    Returns (features float32 [n, d], labels int64 [n]).
    """
    means = p.means.astype(np.float32)
    weights = p.weights.astype(np.float32)
    weights = weights / weights.sum()  # float16 rounding: renormalise
    cov = p.covariances.astype(np.float32)
    gmm = GaussianMixture(n_components=len(weights), covariance_type=cov_type)
    gmm.weights_ = weights
    gmm.means_ = means
    gmm.covariances_ = cov
    gmm.precisions_cholesky_ = _precisions_cholesky(cov, cov_type)
    gmm.random_state = seed
    n = int(p.num_samples[0])
    x, _ = gmm.sample(n)
    return x.astype(np.float32), np.full(n, int(p.label[0]), dtype=np.int64)


def _precisions_cholesky(cov, cov_type):
    from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky
    return _compute_precision_cholesky(cov, cov_type)

def build_synthetic_dataset(params, cov_type="spherical", seed=0):
    """Sample every GMM in `params` and stack the results.

    Returns (features float32 [N, d], labels int64 [N]).
    Each GMM gets its own seed (seed + i) so classes/clients don't share noise.
    """
    xs, ys = [], []
    for i, p in enumerate(params):
        x, y = sample_features(p, cov_type=cov_type, seed=seed + i)
        xs.append(x)
        ys.append(y)
    return np.concatenate(xs), np.concatenate(ys)