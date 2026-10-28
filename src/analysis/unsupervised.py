"""PBL Review 2 (Unit 3): PCA, K-means, Gaussian mixture (EM) and hierarchical
clustering, written from their equations in NumPy -- no sklearn/scipy.

Question: can the hidden compliance state (refusal / partial / full_violation)
be recovered from judge scores alone, without labels?
"""
from itertools import permutations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.analysis.utils import COMPLIANCE_STATES, OUTPUTS_DIR, ensure_outputs_dir, load_transcript_rows, save_metrics

SCORE_COLUMNS = [f'judge_{s}' for s in COMPLIANCE_STATES]


def sq_dists(A, B):
    return np.maximum((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T, 0.0)


# ---------- PCA: S u_i = lambda_i u_i, z = U^T (x - mean) ----------

def pca(X):
    mean = X.mean(axis=0)
    S = (X - mean).T @ (X - mean) / len(X)
    eigvals, eigvecs = np.linalg.eigh(S)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = np.maximum(eigvals[order], 0.0), eigvecs[:, order]
    return mean, eigvecs, eigvals / eigvals.sum()


# ---------- K-means: minimise J = sum_n ||x_n - mu_{k(n)}||^2 ----------

def kmeans(X, k, n_init=10, max_iter=300, seed=42):
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        mu = [X[rng.integers(len(X))]]
        for _ in range(1, k):  # k-means++ seeding
            d2 = sq_dists(X, np.array(mu)).min(axis=1)
            mu.append(X[rng.choice(len(X), p=d2 / d2.sum())])
        mu = np.array(mu)
        for _ in range(max_iter):
            labels = sq_dists(X, mu).argmin(axis=1)  # E-step: nearest centre
            new_mu = np.array([X[labels == j].mean(axis=0) if np.any(labels == j) else mu[j] for j in range(k)])  # M-step
            done = np.allclose(new_mu, mu)
            mu = new_mu
            if done:
                break
        d2 = sq_dists(X, mu)
        labels = d2.argmin(axis=1)
        J = d2[np.arange(len(X)), labels].sum()
        if best is None or J < best[2]:
            best = (labels, mu, float(J))
    return best


# ---------- Gaussian mixture fitted with EM ----------

def log_gauss(X, mu, cov):
    L = np.linalg.cholesky(cov)
    z = np.linalg.solve(L, (X - mu).T)
    return -0.5 * (X.shape[1] * np.log(2 * np.pi) + 2 * np.log(np.diag(L)).sum() + (z ** 2).sum(axis=0))


def gmm_em(X, k, max_iter=500, tol=1e-6, reg=1e-6, seed=42):
    N, D = X.shape
    R = np.eye(k)[kmeans(X, k, seed=seed)[0]]  # initialise responsibilities from K-means
    lls = []
    for _ in range(max_iter):
        # M-step: N_k, pi_k, mu_k, Sigma_k from responsibilities
        Nk = R.sum(axis=0) + 1e-12
        pi, mu = Nk / N, (R.T @ X) / Nk[:, None]
        cov = np.array([(R[:, j, None] * (X - mu[j])).T @ (X - mu[j]) / Nk[j] + reg * np.eye(D) for j in range(k)])
        # E-step: gamma_nk = pi_k N(x_n|mu_k,Sigma_k) / sum_j pi_j N(x_n|mu_j,Sigma_j)
        log_joint = np.column_stack([np.log(pi[j]) + log_gauss(X, mu[j], cov[j]) for j in range(k)])
        m = log_joint.max(axis=1, keepdims=True)
        log_norm = m[:, 0] + np.log(np.exp(log_joint - m).sum(axis=1))
        R = np.exp(log_joint - log_norm[:, None])
        lls.append(float(log_norm.sum()))
        if len(lls) > 1 and abs(lls[-1] - lls[-2]) < tol:
            break
    n_params = (k - 1) + k * D + k * D * (D + 1) // 2
    bic = -2 * lls[-1] + n_params * np.log(N)
    return R.argmax(axis=1), lls, float(bic)


# ---------- Agglomerative clustering (Lance-Williams updates) ----------

def agglomerative(X, n_clusters=3, linkage='complete'):
    N = len(X)
    D = np.sqrt(sq_dists(X, X))
    np.fill_diagonal(D, np.inf)
    size, alive, members = np.ones(N), np.ones(N, dtype=bool), {i: [i] for i in range(N)}
    for _ in range(N - n_clusters):
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
        members[a] += members.pop(b)
    labels = np.empty(N, dtype=int)
    for k, idx in enumerate(members.values()):
        labels[idx] = k
    return labels


# ---------- evaluation ----------

def silhouette(X, labels):
    D = np.sqrt(sq_dists(X, X))
    s = np.zeros(len(X))
    for i in range(len(X)):
        own = labels == labels[i]
        if own.sum() > 1:
            a = D[i, own].sum() / (own.sum() - 1)
            b = min(D[i, labels == c].mean() for c in np.unique(labels) if c != labels[i])
            s[i] = (b - a) / max(a, b)
    return float(s.mean())


def adjusted_rand(y, labels):
    table = np.array([[np.sum((y == c) & (labels == k)) for k in np.unique(labels)] for c in np.unique(y)])
    c2 = lambda n: n * (n - 1) / 2
    sum_ij, sum_a, sum_b = c2(table).sum(), c2(table.sum(1)).sum(), c2(table.sum(0)).sum()
    expected = sum_a * sum_b / c2(len(y))
    return float((sum_ij - expected) / ((sum_a + sum_b) / 2 - expected))


def matched_accuracy(y, labels):
    ids = np.unique(labels)
    idx = np.searchsorted(ids, labels)
    return float(max(np.mean(np.array(p)[idx] == y) for p in permutations(np.unique(y), len(ids))))


def evaluate(name, X, labels, y, **extra):
    metrics = {'ari_vs_true_state': adjusted_rand(y, labels), 'accuracy_after_matching': matched_accuracy(y, labels), 'silhouette': silhouette(X, labels), **extra}
    save_metrics(name, metrics)
    return metrics


def conversation_features(df):
    g = df.sort_values('timestep_in_conversation').groupby('conversation_id')
    feats = g[SCORE_COLUMNS].mean()
    feats['max_full'] = g['judge_full_violation'].max()
    feats['n_turns'] = g.size()
    feats['final_full'] = g['judge_full_violation'].last()
    final_state = g['true_compliance_state'].last().map(COMPLIANCE_STATES.index)
    return feats.values.astype(float), final_state.values


def main():
    ensure_outputs_dir()
    df = load_transcript_rows()
    if 'true_compliance_state' not in df.columns:
        raise ValueError('Review 2 metrics require synthetic rows with true_compliance_state')
    df = df.dropna(subset=['true_compliance_state'])
    if df.empty:
        raise ValueError('Review 2 metrics require synthetic rows with true_compliance_state')
    y = df['true_compliance_state'].map(COMPLIANCE_STATES.index).values
    X = df[SCORE_COLUMNS].values
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    print('\n=== unsupervised (NumPy): PCA, K-means, GMM-EM, hierarchical ===')
    mean, U, ratio = pca(X)
    # the three probabilities sum to 1, so the last eigenvalue is ~0: keep the 2-D principal subspace
    Z = (X - mean) @ U[:, :2]
    save_metrics('pca_numpy', {'explained_variance_ratio': [round(float(r), 4) for r in ratio], 'components_kept': 2})

    ks = list(range(2, 7))
    km_runs = [kmeans(Z, k) for k in ks]
    gmm_runs = [gmm_em(Z, k) for k in ks]
    km_labels, _, _ = km_runs[ks.index(3)]
    gmm_labels, gmm_ll, _ = gmm_runs[ks.index(3)]
    evaluate('kmeans_numpy', Z, km_labels, y, k=3)
    evaluate('gmm_em_numpy', Z, gmm_labels, y, k=3, bic_by_k=dict(zip(ks, [round(r[2], 1) for r in gmm_runs])))

    C, final_state = conversation_features(df)
    C = (C - C.mean(axis=0)) / C.std(axis=0)
    hc_labels = agglomerative(C, 3, 'complete')
    evaluate('hierarchical_numpy', C, hc_labels, final_state, linkage='complete', level='conversation',
             ari_by_linkage={l: round(adjusted_rand(final_state, agglomerative(C, 3, l)), 3) for l in ('single', 'complete', 'average', 'ward')})

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, (title, labels) in zip(axes, [('true compliance state', y), ('K-means (k=3)', km_labels), ('GMM-EM (K=3)', gmm_labels)]):
        ax.scatter(Z[:, 0], Z[:, 1], c=labels, cmap='tab10', s=6)
        ax.set_title(title)
        ax.set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    fig.tight_layout()
    fig.savefig(f'{OUTPUTS_DIR}/unsupervised_clusters_pca.png', dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    axes[0].bar(range(1, len(ratio) + 1), ratio, color='#4c72b0')
    axes[0].set_title('PCA scree (explained variance ratio)')
    axes[1].plot(ks, [r[2] for r in km_runs], marker='o', label='distortion J')
    axes[1].set_title('K-means elbow')
    ax_s = axes[1].twinx()
    ax_s.plot(ks, [silhouette(Z, r[0]) for r in km_runs], marker='s', color='#dd8452')
    ax_s.set_ylabel('silhouette', color='#dd8452')
    axes[2].plot(gmm_ll, color='#55a868')
    axes[2].set_title('EM log-likelihood per iteration (K=3)')
    fig.tight_layout()
    fig.savefig(f'{OUTPUTS_DIR}/unsupervised_model_selection.png', dpi=120)
    plt.close(fig)
    print(f'unsupervised plots written to {OUTPUTS_DIR}/unsupervised_*.png')


if __name__ == '__main__':
    main()
