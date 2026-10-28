"""Step-recording versions of the Review 2 algorithms, used by the transparency UI (app.py).

Each *_trace function follows src/analysis/unsupervised.py line for line (same
random-number use, same update order, same stopping rule) but keeps every
intermediate state so the UI can replay it. tests/test_trace.py checks that the
final answers are identical to the originals, so what the UI shows is what the
pipeline computes.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.analysis.unsupervised import adjusted_rand, kmeans, log_gauss, silhouette, sq_dists


def zscore(X):
    mean, sd = X.mean(axis=0), X.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    return (X - mean) / sd, mean, sd


# ---------- PCA ----------

def pca_trace(X):
    mean = X.mean(axis=0)
    Xc = X - mean
    S = Xc.T @ Xc / len(X)
    eigvals, eigvecs = np.linalg.eigh(S)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = np.maximum(eigvals[order], 0.0), eigvecs[:, order]
    return {'mean': mean, 'cov': S, 'eigvals': eigvals, 'eigvecs': eigvecs, 'ratio': eigvals / eigvals.sum()}


# ---------- K-means ----------

def kmeans_trace(X, k, n_init=10, max_iter=300, seed=42):
    """Same procedure as unsupervised.kmeans, recording every iteration of every restart.

    Returns (best_run, all_runs). A run is a dict with the k-means++ start, a
    history of iterations and the final labels / centres / distortion J.
    """
    rng = np.random.default_rng(seed)
    runs, best = [], None
    for r in range(n_init):
        mu = [X[rng.integers(len(X))]]
        for _ in range(1, k):  # k-means++ seeding
            d2 = sq_dists(X, np.array(mu)).min(axis=1)
            mu.append(X[rng.choice(len(X), p=d2 / d2.sum())])
        mu = np.array(mu)
        init = mu.copy()
        hist = []
        for it in range(max_iter):
            d2_all = sq_dists(X, mu)
            labels = d2_all.argmin(axis=1)  # assignment step
            new_mu = np.array([X[labels == j].mean(axis=0) if np.any(labels == j) else mu[j] for j in range(k)])  # update step
            hist.append({'iter': it + 1, 'centers_before': mu.copy(), 'centers_after': new_mu.copy(), 'labels': labels, 'J': float(d2_all[np.arange(len(X)), labels].sum())})
            done = np.allclose(new_mu, mu)
            mu = new_mu
            if done:
                break
        d2 = sq_dists(X, mu)
        labels = d2.argmin(axis=1)
        J = float(d2[np.arange(len(X)), labels].sum())
        run = {'run': r, 'init': init, 'history': hist, 'labels': labels, 'centers': mu, 'J': J}
        runs.append(run)
        if best is None or J < best['J']:
            best = run
    return best, runs


# ---------- Gaussian mixture / EM ----------

def _mstep(X, R, reg):
    N, D = X.shape
    k = R.shape[1]
    Nk = R.sum(axis=0) + 1e-12
    pi, mu = Nk / N, (R.T @ X) / Nk[:, None]
    cov = np.array([(R[:, j, None] * (X - mu[j])).T @ (X - mu[j]) / Nk[j] + reg * np.eye(D) for j in range(k)])
    return Nk, pi, mu, cov


def _estep(X, pi, mu, cov):
    k = len(pi)
    log_joint = np.column_stack([np.log(pi[j]) + log_gauss(X, mu[j], cov[j]) for j in range(k)])
    m = log_joint.max(axis=1, keepdims=True)
    log_norm = m[:, 0] + np.log(np.exp(log_joint - m).sum(axis=1))
    return log_joint, log_norm, np.exp(log_joint - log_norm[:, None])


def gmm_trace(X, k, max_iter=500, tol=1e-6, reg=1e-6, seed=42):
    """Same loop as unsupervised.gmm_em, recording parameters and responsibilities each iteration."""
    N, D = X.shape
    R = np.eye(k)[kmeans(X, k, seed=seed)[0]]
    _, pi0, mu0, cov0 = _mstep(X, R, reg)
    hist = [{'iter': 0, 'R': R.copy(), 'pi': pi0, 'mu': mu0, 'cov': cov0, 'll': None}]
    lls = []
    for it in range(1, max_iter + 1):
        Nk, pi, mu, cov = _mstep(X, R, reg)  # M-step
        log_joint, log_norm, R = _estep(X, pi, mu, cov)  # E-step
        lls.append(float(log_norm.sum()))
        hist.append({'iter': it, 'R': R.copy(), 'pi': pi, 'mu': mu, 'cov': cov, 'll': lls[-1], 'Nk': Nk})
        if len(lls) > 1 and abs(lls[-1] - lls[-2]) < tol:
            break
    n_params = (k - 1) + k * D + k * D * (D + 1) // 2
    bic = -2 * lls[-1] + n_params * np.log(N)
    return {'history': hist, 'labels': R.argmax(axis=1), 'lls': lls, 'bic': float(bic)}


def point_breakdown(x, pi, mu, cov):
    """E-step arithmetic for a single point: log pi_j, log N(x|mu_j,Sigma_j), their sum, and the responsibility."""
    x = np.asarray(x, dtype=float).reshape(1, -1)
    log_pi = np.log(pi)
    log_dens = np.array([log_gauss(x, mu[j], cov[j])[0] for j in range(len(pi))])
    log_joint = log_pi + log_dens
    resp = np.exp(log_joint - log_joint.max())
    return log_pi, log_dens, log_joint, resp / resp.sum()


# ---------- Agglomerative clustering ----------

def agglo_trace(X, linkage='complete'):
    """Full merge tree (down to one cluster), using the same Lance-Williams update as unsupervised.agglomerative.

    Returns merges [(slot_a, slot_b, distance, new_size)] and a scipy-style linkage matrix Z.
    """
    N = len(X)
    D = np.sqrt(sq_dists(X, X))
    np.fill_diagonal(D, np.inf)
    size, alive, node = np.ones(N), np.ones(N, dtype=bool), np.arange(N)
    merges, Z = [], []
    for step in range(N - 1):
        a, b = np.unravel_index(np.argmin(D), D.shape)
        na, nb, dab = size[a], size[b], D[a, b]
        o = alive.copy()
        o[[a, b]] = False
        dac, dbc, nc = D[a, o], D[b, o], size[o]
        if linkage == 'single':
            new = np.minimum(dac, dbc)
        elif linkage == 'complete':
            new = np.maximum(dac, dbc)
        elif linkage == 'average':
            new = (na * dac + nb * dbc) / (na + nb)
        else:  # ward
            new = np.sqrt(((na + nc) * dac ** 2 + (nb + nc) * dbc ** 2 - nc * dab ** 2) / (na + nb + nc))
        D[a, o] = D[o, a] = new
        D[b, :] = D[:, b] = np.inf
        alive[b], size[a] = False, na + nb
        merges.append((int(a), int(b), float(dab), int(na + nb)))
        Z.append([node[a], node[b], float(dab), na + nb])
        node[a] = N + step
    return merges, np.array(Z, dtype=float)


def labels_at(merges, N, n_clusters):
    """Cluster labels after performing merges until n_clusters remain."""
    lab = np.arange(N)
    for a, b, _, _ in merges[: N - n_clusters]:
        lab[lab == b] = a
    return np.unique(lab, return_inverse=True)[1]


# ---------- scoring without cheating ----------

def match_labels(y, labels):
    """Name clusters by the one-to-one assignment (Hungarian) that agrees best with the true classes."""
    classes, ids = np.unique(y), np.unique(labels)
    cm = np.array([[np.sum((y == c) & (labels == g)) for g in ids] for c in classes])
    rows, cols = linear_sum_assignment(-cm)
    mapping = {int(ids[ci]): int(classes[ri]) for ri, ci in zip(rows, cols)}
    pred = np.array([mapping.get(int(l), -1) for l in labels])
    return {'classes': classes, 'ids': ids, 'cm': cm, 'mapping': mapping, 'pred': pred, 'accuracy': float(np.mean(pred == y))}


def score(X, labels, y):
    n_clusters = len(np.unique(labels))
    return {
        'ari': adjusted_rand(y, labels),
        'accuracy': match_labels(y, labels)['accuracy'],
        'silhouette': silhouette(X, labels) if n_clusters > 1 else float('nan'),
        'n_clusters': n_clusters,
    }


# ---------- HMM (hmmlearn does the EM; this wrapper adds alignment and reporting) ----------

def hmm_trace(X, conv, t, y, n_states=3, n_iter=200, seed=42):
    """Fit the same GaussianHMM as src/analysis/compliance_hmm.py on ordered conversations and report everything.

    X: judge scores in the original row order; conv / t: conversation id and turn index of each row; y: true state codes.
    Returns decoded labels in the original row order plus the learned parameters re-ordered by matched class.
    """
    from hmmlearn.hmm import GaussianHMM
    conv = np.asarray(conv).astype(str)
    order = np.lexsort((np.asarray(t), conv))
    Xs, ys, cs = X[order], np.asarray(y)[order], conv[order]
    ids, counts = np.unique(cs, return_counts=True)
    lengths = [int(c) for c in counts]
    model = GaussianHMM(n_components=n_states, covariance_type='diag', n_iter=n_iter, random_state=seed)
    model.fit(Xs, lengths)
    decoded = model.predict(Xs, lengths)
    m = match_labels(ys, decoded)
    inv = {c: s for s, c in m['mapping'].items()}
    perm = [inv[c] for c in sorted(inv)]
    labels_orig = np.empty(len(X), dtype=int)
    labels_orig[order] = decoded
    return {
        'order': order, 'conv_sorted': cs, 'lengths': lengths, 'conv_ids': ids,
        'decoded_sorted': decoded, 'labels': labels_orig, 'match': m, 'perm': perm,
        'transmat': model.transmat_[np.ix_(perm, perm)], 'startprob': model.startprob_[perm],
        'means': model.means_[perm], 'vars': np.array([np.diag(c) if c.ndim == 2 else c for c in model.covars_])[perm] if model.covars_.ndim == 3 else model.covars_[perm],
        'log_likelihood': float(model.score(Xs, lengths)), 'accuracy': m['accuracy'],
        'ari': adjusted_rand(ys, decoded), 'y_sorted': ys, 'X_sorted': Xs,
    }
