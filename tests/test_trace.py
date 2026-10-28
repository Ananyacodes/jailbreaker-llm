import numpy as np
import pytest

from src.analysis import trace
from src.analysis.unsupervised import adjusted_rand, agglomerative, gmm_em, kmeans, pca


@pytest.fixture
def blobs():
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(c, 0.6, size=(50, 2)) for c in [(0, 0), (3, 0), (0, 3)]])
    return X, np.repeat([0, 1, 2], 50)


def test_kmeans_trace_matches_original(blobs):
    X, _ = blobs
    labels, centers, J = kmeans(X, 3)
    best, runs = trace.kmeans_trace(X, 3)
    assert len(runs) == 10
    assert np.array_equal(best['labels'], labels)
    assert best['J'] == pytest.approx(J)
    assert best['history'][-1]['iter'] == len(best['history'])


def test_kmeans_distortion_never_increases(blobs):
    X, _ = blobs
    best, _ = trace.kmeans_trace(X, 3)
    Js = [h['J'] for h in best['history']]
    assert all(b <= a + 1e-9 for a, b in zip(Js, Js[1:]))


def test_gmm_trace_matches_original(blobs):
    X, _ = blobs
    labels, lls, bic = gmm_em(X, 3)
    out = trace.gmm_trace(X, 3)
    assert np.array_equal(out['labels'], labels)
    assert out['lls'] == pytest.approx(lls)
    assert out['bic'] == pytest.approx(bic)


def test_agglo_trace_matches_original_and_scipy(blobs):
    from scipy.cluster.hierarchy import fcluster, linkage as sp_linkage
    X, _ = blobs
    for method in ('single', 'complete', 'average', 'ward'):
        merges, Z = trace.agglo_trace(X, method)
        ours = trace.labels_at(merges, len(X), 3)
        assert adjusted_rand(agglomerative(X, 3, method), ours) == pytest.approx(1.0)
        ref = fcluster(sp_linkage(X, method), 3, 'maxclust')
        assert adjusted_rand(ref, ours) == pytest.approx(1.0)
        assert np.allclose(np.sort(Z[:, 2]), np.sort(sp_linkage(X, method)[:, 2]))


def test_pca_trace_matches_original(blobs):
    X, _ = blobs
    mean, U, ratio = pca(X)
    out = trace.pca_trace(X)
    assert np.allclose(out['ratio'], ratio)
    assert np.allclose(out['mean'], mean)


def test_match_labels_is_permutation_invariant():
    y = np.array([0, 0, 1, 1, 2, 2])
    labels = np.array([2, 2, 0, 0, 1, 1])
    assert trace.match_labels(y, labels)['accuracy'] == pytest.approx(1.0)
    assert trace.score(np.eye(6), labels, y)['ari'] == pytest.approx(1.0)


def test_match_labels_handles_more_clusters_than_classes():
    y = np.array([0, 0, 0, 1, 1, 1])
    labels = np.array([0, 0, 1, 2, 2, 3])
    out = trace.match_labels(y, labels)
    assert 0 < out['accuracy'] < 1


def test_hmm_trace_matches_original_pipeline():
    pytest.importorskip('hmmlearn')
    from src.analysis.compliance_hmm import build_sequences, match_states_to_truth
    from src.analysis.unsupervised import SCORE_COLUMNS
    from src.analysis.utils import COMPLIANCE_STATES, load_transcript_rows
    from hmmlearn.hmm import GaussianHMM
    df = load_transcript_rows().dropna(subset=['true_compliance_state'])
    y = df['true_compliance_state'].map(COMPLIANCE_STATES.index).values
    out = trace.hmm_trace(df[SCORE_COLUMNS].values, df['conversation_id'].values, df['timestep_in_conversation'].values, y)
    X, lengths, codes = build_sequences(df)
    model = GaussianHMM(n_components=3, covariance_type='diag', n_iter=200, random_state=42).fit(X, lengths)
    ref_acc = float(np.mean(match_states_to_truth(codes, model.predict(X, lengths), 3) == codes))
    assert out['accuracy'] == pytest.approx(ref_acc)
    assert out['log_likelihood'] == pytest.approx(float(model.score(X, lengths)))
    assert np.allclose(out['transmat'].sum(axis=1), 1.0)
