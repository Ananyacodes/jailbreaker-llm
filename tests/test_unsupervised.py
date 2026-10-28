import numpy as np
import pytest

from src.analysis.unsupervised import adjusted_rand, agglomerative, gmm_em, kmeans, pca


@pytest.fixture
def blobs():
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(c, 0.3, size=(60, 2)) for c in [(0, 0), (3, 0), (0, 3)]])
    return X, np.repeat([0, 1, 2], 60)


def test_kmeans_gmm_and_hierarchical_recover_blobs(blobs):
    X, y = blobs
    assert adjusted_rand(y, kmeans(X, 3)[0]) == pytest.approx(1.0)
    assert adjusted_rand(y, gmm_em(X, 3)[0]) == pytest.approx(1.0)
    for linkage in ('single', 'complete', 'average', 'ward'):
        assert adjusted_rand(y, agglomerative(X, 3, linkage)) == pytest.approx(1.0)


def test_em_log_likelihood_never_decreases(blobs):
    X, _ = blobs
    _, lls, _ = gmm_em(X, 3)
    assert np.all(np.diff(lls) > -1e-8)


def test_pca_sorts_variance_and_spots_simplex_constraint():
    rng = np.random.default_rng(2)
    P = rng.dirichlet([2.0, 2.0, 2.0], size=300)  # rows sum to 1, like judge scores
    _, _, ratio = pca(P)
    assert np.all(np.diff(ratio) <= 1e-12)
    assert ratio[-1] < 1e-10
