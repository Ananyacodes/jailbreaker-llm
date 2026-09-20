import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.svm import SVC

from src.analysis.utils import COMPLIANCE_STATES, OUTPUTS_DIR, conversation_train_test_split, ensure_outputs_dir, load_transcript_rows, save_metrics

METADATA_FEATURES = ['timestep_in_conversation']


def build_features(df: pd.DataFrame):
    X = df[METADATA_FEATURES].copy()
    X = pd.concat([X, pd.get_dummies(df['technique'], prefix='technique')], axis=1)
    X = pd.concat([X, pd.get_dummies(df['target_model'], prefix='target')], axis=1)
    return X


def evaluate(name, model, X_train, y_train, X_test, y_test, save_confusion=False) -> dict:
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = {'accuracy': float(accuracy_score(y_test, preds)), 'f1_macro': float(f1_score(y_test, preds, average='macro'))}
    save_metrics(f'classification_{name}', metrics)
    if save_confusion:
        ensure_outputs_dir()
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ConfusionMatrixDisplay.from_predictions(y_test, preds, labels=COMPLIANCE_STATES, xticks_rotation=20, ax=ax)
        ax.set_title(f'metadata-only classification confusion matrix ({name})')
        fig.tight_layout()
        fig.savefig(f'{OUTPUTS_DIR}/classification_confusion_{name}.png', dpi=120)
        plt.close(fig)
    return metrics


def main():
    df = load_transcript_rows()
    train_df, test_df = conversation_train_test_split(df)
    X_train_raw, X_test_raw = (build_features(train_df), build_features(test_df))
    X_train, X_test = X_train_raw.align(X_test_raw, join='outer', axis=1, fill_value=0)
    y_train, y_test = (train_df['true_compliance_state'].values, test_df['true_compliance_state'].values)
    print('\n=== classification: predicting compliance state from turn METADATA ONLY (no judge score) ===')
    evaluate('logistic_regression', LogisticRegression(max_iter=1000), X_train, y_train, X_test, y_test)
    evaluate('random_forest', RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42), X_train, y_train, X_test, y_test, save_confusion=True)
    evaluate('svm_rbf', SVC(kernel='rbf', C=2.0), X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    main()
