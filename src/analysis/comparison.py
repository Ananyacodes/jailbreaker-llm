import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.analysis.utils import OUTPUTS_DIR, ensure_outputs_dir


def load_all_metrics() -> dict:
    with open(f'{OUTPUTS_DIR}/metrics.json') as f:
        return json.load(f)


def plot_state_recovery_comparison(metrics: dict):
    supervised = {'Logistic Regression': metrics['classification_logistic_regression']['accuracy'], 'Random Forest (clf)': metrics['classification_random_forest']['accuracy'], 'SVM (RBF)': metrics['classification_svm_rbf']['accuracy']}
    unsupervised_ari = {'KMeans (scores)': metrics['clustering_kmeans']['ari_vs_true_state'], 'DBSCAN (scores)': metrics['clustering_dbscan']['ari_vs_true_state'], 'HMM (scores + order)': metrics['hmm']['ari_vs_true_state']}
    hmm_accuracy = metrics['hmm']['accuracy_after_state_matching']
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    axes[0].bar(list(supervised.keys()), list(supervised.values()), color='#4c72b0')
    axes[0].bar(['HMM (scores + order)'], [hmm_accuracy], color='#dd8452')
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel('accuracy vs true compliance state')
    axes[0].set_title('metadata-only classifiers vs. score+order-aware HMM')
    axes[0].tick_params(axis='x', rotation=20)
    axes[1].bar(list(unsupervised_ari.keys()), list(unsupervised_ari.values()), color=['#55a868', '#55a868', '#dd8452'])
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel('Adjusted Rand Index vs true compliance state')
    axes[1].set_title('unsupervised: snapshot clustering vs. order-aware HMM')
    axes[1].tick_params(axis='x', rotation=20)
    fig.suptitle('compliance-state recovery: how much does judge score + TURN ORDER help?')
    fig.tight_layout()
    ensure_outputs_dir()
    fig.savefig(f'{OUTPUTS_DIR}/comparison_state_recovery.png', dpi=120)
    plt.close(fig)


def plot_regression_comparison(metrics: dict):
    models = {'Linear Regression': metrics['regression_linear_regression'], 'Random Forest': metrics['regression_random_forest']}
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].bar(models.keys(), [m['rmse'] for m in models.values()], color='#4c72b0')
    axes[0].set_title('RMSE (lower is better)')
    axes[1].bar(models.keys(), [m['r2'] for m in models.values()], color='#55a868')
    axes[1].set_title('R² (higher is better)')
    fig.suptitle('regression: predicting eval-judge full_violation from live-judge signal')
    fig.tight_layout()
    fig.savefig(f'{OUTPUTS_DIR}/comparison_regression.png', dpi=120)
    plt.close(fig)


def print_summary_table(metrics: dict):
    print('\n' + '=' * 76)
    print('COMPARATIVE PERFORMANCE SUMMARY')
    print('=' * 76)
    print('\n-- regression (predict eval-judge full_violation from live-judge signal) --')
    for name in ['regression_linear_regression', 'regression_random_forest']:
        m = metrics[name]
        print(f"  {name:32s} RMSE={m['rmse']:.3f}  MAE={m['mae']:.3f}  R2={m['r2']:.3f}")
    print('\n-- compliance-state recovery (3 hidden states) --')
    for name in ['classification_logistic_regression', 'classification_random_forest', 'classification_svm_rbf']:
        m = metrics[name]
        print(f"  {name:36s} accuracy={m['accuracy']:.3f}  f1_macro={m['f1_macro']:.3f}")
    for name in ['clustering_kmeans', 'clustering_dbscan']:
        m = metrics[name]
        print(f"  {name:36s} ARI={m['ari_vs_true_state']:.3f}  silhouette={m['silhouette']:.3f}")
    m = metrics['hmm']
    print(f"  {'hmm (score + order)':36s} accuracy={m['accuracy_after_state_matching']:.3f}  ARI={m['ari_vs_true_state']:.3f}")
    print('=' * 76)


def main():
    metrics = load_all_metrics()
    print_summary_table(metrics)
    plot_state_recovery_comparison(metrics)
    plot_regression_comparison(metrics)
    print(f'\ncomparison plots written to {OUTPUTS_DIR}/comparison_*.png')


if __name__ == '__main__':
    main()
