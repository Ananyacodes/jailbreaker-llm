import numpy as np
from hmmlearn.hmm import GaussianHMM
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score, adjusted_rand_score, confusion_matrix

from src.analysis.utils import COMPLIANCE_STATES, load_transcript_rows, save_metrics

SCORE_COLUMNS = [f'judge_{s}' for s in COMPLIANCE_STATES]


def build_sequences(df):
    df = df.sort_values(['conversation_id', 'timestep_in_conversation'])
    lengths = df.groupby('conversation_id', sort=False).size().tolist()
    X = df[SCORE_COLUMNS].values
    true_codes = df['true_compliance_state'].astype('category').cat.codes.values
    return (X, lengths, true_codes)


def match_states_to_truth(true_codes: np.ndarray, decoded_states: np.ndarray, n_states: int) -> np.ndarray:
    cm = confusion_matrix(true_codes, decoded_states, labels=range(n_states))
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = dict(zip(col_ind, row_ind))
    return np.array([mapping[s] for s in decoded_states])


def main():
    df = load_transcript_rows()
    X, lengths, true_codes = build_sequences(df)
    print('\n=== HMM: decoding compliance state using judge scores + TURN ORDER ===')
    model = GaussianHMM(n_components=3, covariance_type='diag', n_iter=200, random_state=42)
    model.fit(X, lengths)
    decoded_states = model.predict(X, lengths)
    matched = match_states_to_truth(true_codes, decoded_states, n_states=3)
    metrics = {'accuracy_after_state_matching': float(accuracy_score(true_codes, matched)), 'ari_vs_true_state': float(adjusted_rand_score(true_codes, decoded_states)), 'log_likelihood': float(model.score(X, lengths))}
    save_metrics('hmm', metrics)
    print("\nlearned transition matrix (rows/cols in HMM's own, unmatched state order):")
    print(np.round(model.transmat_, 3))
    print(f'(true generating transition matrix used states {COMPLIANCE_STATES}: see data/generate_synthetic_campaign.py)')


if __name__ == '__main__':
    main()
