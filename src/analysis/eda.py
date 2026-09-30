import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from src.analysis.utils import COMPLIANCE_STATES, OUTPUTS_DIR, ensure_outputs_dir, load_transcript_rows

STATE_COLORS = {'refusal': '#8ecae6', 'partial_compliance': '#ffb703', 'full_violation': '#e63946'}


def summary_stats(df):
    print('\n=== dataset shape ===')
    print(df.shape, f"across {df['conversation_id'].nunique()} conversations")
    print('\n=== technique usage ===')
    print(df['technique'].value_counts())
    print('\n=== true compliance state distribution (turn-level) ===')
    if 'true_compliance_state' in df.columns:
        print(df['true_compliance_state'].value_counts())
    print('\n=== target model distribution ===')
    print(df['target_model'].value_counts())


def plot_score_distributions(df):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    for ax, state in zip(axes, COMPLIANCE_STATES):
        if 'true_compliance_state' in df.columns:
            for true_state in COMPLIANCE_STATES:
                sns.kdeplot(df.loc[df['true_compliance_state'] == true_state, f'judge_{state}'], ax=ax, label=true_state, color=STATE_COLORS[true_state])
            ax.legend(fontsize=6)
        else:
            sns.histplot(df[f'judge_{state}'], ax=ax, color=STATE_COLORS[state])
        ax.set_title(f'judge score: {state}')
    fig.tight_layout()
    fig.savefig(f'{OUTPUTS_DIR}/eda_score_distributions.png', dpi=120)
    plt.close(fig)


def plot_escalation_by_turn(df):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    by_turn_technique = df.groupby(['timestep_in_conversation', 'technique'])['judge_full_violation'].mean().reset_index()
    for technique in df['technique'].unique():
        sub = by_turn_technique[by_turn_technique['technique'] == technique]
        ax.plot(sub['timestep_in_conversation'], sub['judge_full_violation'], marker='o', label=technique, linewidth=1.2)
    ax.set_xlabel('turn index within conversation')
    ax.set_ylabel('mean judge full_violation score')
    ax.set_title('escalation curve: does full_violation probability rise with turns?')
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(f'{OUTPUTS_DIR}/eda_escalation_by_turn.png', dpi=120)
    plt.close(fig)


def plot_target_comparison(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    conv_outcome = df.sort_values('timestep_in_conversation').groupby('conversation_id').last()
    rate = conv_outcome.groupby('target_model')['judge_label'].apply(lambda s: (s == 'full_violation').mean())
    ax.bar(rate.index, rate.values, color='#dd8452')
    ax.set_ylabel('fraction of conversations ending in full_violation')
    ax.set_title('target model comparison: which is easiest to break?')
    ax.tick_params(axis='x', rotation=15)
    fig.tight_layout()
    fig.savefig(f'{OUTPUTS_DIR}/eda_target_comparison.png', dpi=120)
    plt.close(fig)


def main():
    ensure_outputs_dir()
    df = load_transcript_rows()
    summary_stats(df)
    plot_score_distributions(df)
    plot_escalation_by_turn(df)
    plot_target_comparison(df)
    print(f'\nEDA plots written to {OUTPUTS_DIR}/eda_*.png')


if __name__ == '__main__':
    main()
