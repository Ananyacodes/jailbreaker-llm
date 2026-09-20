import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from src.analysis.utils import COMPLIANCE_STATES, OUTPUTS_DIR, ensure_outputs_dir, load_transcript_rows, save_metrics

SCORE_COLUMNS = [f'judge_{s}' for s in COMPLIANCE_STATES]


def evaluate(name: str, labels, X_scaled, true_state) -> dict:
    non_noise = labels != -1
    metrics = {
        'ari_vs_true_state': float(adjusted_rand_score(true_state, labels)),
        'silhouette': float(silhouette_score(X_scaled[non_noise], labels[non_noise])) if non_noise.sum() > 1 and len(set(labels[non_noise])) > 1 else float('nan'),
        'n_clusters_found': int(len(set(labels)) - (1 if -1 in labels else 0)),
    }
    save_metrics(f'clustering_{name}', metrics)
    return metrics


def plot_clusters(X_scaled, labels, true_state, name):
    ensure_outputs_dir()
    coords = PCA(n_components=2, random_state=42).fit_transform(X_scaled)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].scatter(coords[:, 0], coords[:, 1], c=labels, cmap='tab10', s=6)
    axes[0].set_title(f'{name}: discovered clusters (PCA of judge-score space)')
    true_codes = true_state.astype('category').cat.codes
    axes[1].scatter(coords[:, 0], coords[:, 1], c=true_codes, cmap='tab10', s=6)
    axes[1].set_title('true compliance state (for comparison)')
    fig.tight_layout()
    fig.savefig(f'{OUTPUTS_DIR}/clustering_{name}.png', dpi=120)
    plt.close(fig)


def main():
    df = load_transcript_rows()
    X_scaled = StandardScaler().fit_transform(df[SCORE_COLUMNS].values)
    true_state = df['true_compliance_state']
    print('\n=== clustering: unsupervised compliance-state discovery from judge scores alone ===')
    kmeans_labels = KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(X_scaled)
    evaluate('kmeans', kmeans_labels, X_scaled, true_state)
    plot_clusters(X_scaled, kmeans_labels, true_state, 'kmeans')
    dbscan_labels = DBSCAN(eps=0.25, min_samples=10).fit_predict(X_scaled)
    evaluate('dbscan', dbscan_labels, X_scaled, true_state)
    plot_clusters(X_scaled, dbscan_labels, true_state, 'dbscan')


if __name__ == '__main__':
    main()
