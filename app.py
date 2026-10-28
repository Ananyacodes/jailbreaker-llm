"""Transparency UI for the Review 2 unsupervised pipeline.

Run from the repository root:   streamlit run app.py

Every stage (data, standardize, PCA, K-means, GMM-EM, hierarchical, scoring) is
shown with its intermediate numbers and can be replayed step by step. The
algorithms are the ones in src/analysis/unsupervised.py; src/analysis/trace.py
records their intermediate states (tests/test_trace.py proves the results are identical).
"""
import inspect
import json
import os
import time
import traceback
from contextlib import contextmanager

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.patches import Ellipse
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage as sp_linkage
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from src.analysis import external_data as ext
from src.analysis import scenarios
from src.analysis import trace
from src.analysis.unsupervised import SCORE_COLUMNS, adjusted_rand, conversation_features, silhouette
from src.analysis.utils import COMPLIANCE_STATES, load_transcript_rows

st.set_page_config(page_title='Unsupervised pipeline, step by step', layout='wide')


def _df_kwargs():
    """st.dataframe sizing differs between Streamlit versions; pick the one this install understands."""
    try:
        p = inspect.signature(st.dataframe).parameters
        if 'width' in p and 'stretch' in (str(p['width'].annotation) + str(p['width'].default) + str(getattr(p['width'].annotation, '__args__', ''))):
            return {'width': 'stretch'}
    except Exception:
        pass
    return {'use_container_width': True}


DF_KW = _df_kwargs()


@contextmanager
def guard(label):
    """A failure in one tab shows a readable error in that tab instead of breaking the whole page."""
    try:
        yield
    except Exception as e:  # noqa: BLE001
        st.error(f'The "{label}" tab hit an error: {e}')
        with st.expander('Technical details (paste this if you need help)'):
            st.code(traceback.format_exc())

PALETTE = ['#2F6FED', '#F2A20C', '#E5484D', '#16A34A', '#7C3AED', '#0EA5A4', '#DB2777', '#64748B', '#CA8A04', '#0369A1']
BASE_RGB = np.array([[int(c[i:i + 2], 16) / 255 for i in (1, 3, 5)] for c in PALETTE])
DATASETS = [
    'Campaign: baseline (simulated, your repo data)',
    'Campaign: noisy judge (simulated)',
    'Campaign: stealthy attacker (simulated)',
    'Campaign: persistent jailbreak (simulated)',
    'Anthropic red-team attempts (real, needs download)',
    'JailbreakBench attack artifacts (real, needs download)',
]
SIM_KEYS = {'Campaign: noisy judge (simulated)': 'noisy', 'Campaign: stealthy attacker (simulated)': 'stealthy', 'Campaign: persistent jailbreak (simulated)': 'persistent'}
LEVELS = ['Turn level (judge scores per turn)', 'Conversation level (one summary row per conversation)']


# ======================================================================
# data and computation (cached; each returns its own run time for the log)
# ======================================================================

def _sim_rows():
    try:
        return load_transcript_rows()
    except FileNotFoundError:
        from data.generate_synthetic_campaign import generate
        os.makedirs('data/transcripts', exist_ok=True)
        with open('data/transcripts/synthetic_campaign_pbl.jsonl', 'w') as f:
            for row in generate():
                f.write(json.dumps(row) + '\n')
        return load_transcript_rows()


@st.cache_data(show_spinner=False)
def get_data(key):
    """key = '<dataset>||<max rows>||<level>'. Raises ext.DatasetUnavailable (with instructions) for real datasets that are not downloaded."""
    t = time.time()
    name, max_rows, level = key.rsplit('||', 2)
    max_rows = int(max_rows)
    seq = None
    idx = None
    if name.startswith('Campaign'):
        df = (_sim_rows() if name.startswith('Campaign: baseline') else scenarios.simulate(SIM_KEYS[name])).dropna(subset=['true_compliance_state'])
        about = ('Baseline campaign from your repo (220 conversations, up to 6 turns, stops at the first success). ' if name.startswith('Campaign: baseline')
                 else scenarios.SCENARIOS[SIM_KEYS[name]]['about'] + ' ')
        about += 'Simulated: a hidden compliance state per turn drives the judge\'s three probabilities.'
        if level.startswith('Turn'):
            X = df[SCORE_COLUMNS].reset_index(drop=True)
            y = df['true_compliance_state'].map(COMPLIANCE_STATES.index).values
            seq = {'conv': df['conversation_id'].values, 't': df['timestep_in_conversation'].values}
            about = f'{len(df)} turns from {df["conversation_id"].nunique()} conversations; each row is the reward judge\'s three probabilities for one turn. ' + about
        else:
            C, final_state = conversation_features(df)
            X = pd.DataFrame(C, columns=['mean_refusal', 'mean_partial', 'mean_full', 'max_full', 'n_turns', 'final_full'])
            y = final_state
            about = (f'{len(X)} conversations; each row = mean of each score, max p(full), number of turns and last p(full). '
                     'The label is the conversation\'s final state. ') + about
        classes = COMPLIANCE_STATES
    else:
        X, y, classes, about = ext.load_redteam() if name.startswith('Anthropic') else ext.load_jbb()
        if len(X) > max_rows:  # hierarchical clustering and silhouette are O(N^2): use a seeded random sample
            idx = np.sort(np.random.default_rng(0).choice(len(X), max_rows, replace=False))
            about += f' (Random sample of {max_rows} of {len(X)} rows, seed 0, so the O(N^2) steps stay fast.)'
            X, y = X.iloc[idx].reset_index(drop=True), np.asarray(y)[idx]
    return {'X': X, 'y': np.asarray(y), 'classes': classes, 'about': about, 'seq': seq, 'idx': idx}, time.time() - t


@st.cache_data(show_spinner=False)
def prep(name, standardize):
    t = time.time()
    X = get_data(name)[0]['X'].values.astype(float)
    if standardize:
        Xs, mean, sd = trace.zscore(X)
    else:
        Xs, mean, sd = X - X.mean(axis=0), X.mean(axis=0), np.ones(X.shape[1])
    return {'Xs': Xs, 'mean': mean, 'sd': sd}, time.time() - t


@st.cache_data(show_spinner=False)
def run_pca(name, standardize):
    t = time.time()
    p = trace.pca_trace(prep(name, standardize)[0]['Xs'])
    return p, time.time() - t


def space(name, standardize, ncomp, use_pca):
    Xs = prep(name, standardize)[0]['Xs']
    if not use_pca:
        return Xs
    p = run_pca(name, standardize)[0]
    return (Xs - p['mean']) @ p['eigvecs'][:, :ncomp]


@st.cache_data(show_spinner=False)
def run_kmeans(name, standardize, ncomp, use_pca, k, seed):
    t = time.time()
    W = space(name, standardize, ncomp, use_pca)
    best, runs = trace.kmeans_trace(W, k, seed=seed)
    return {'best': best, 'Js': [r['J'] for r in runs]}, time.time() - t


@st.cache_data(show_spinner=False)
def run_k_sweep(name, standardize, ncomp, use_pca, seed):
    t = time.time()
    W = space(name, standardize, ncomp, use_pca)
    rows = []
    for k in range(2, 7):
        best, _ = trace.kmeans_trace(W, k, seed=seed)
        rows.append({'k': k, 'distortion J': best['J'], 'silhouette': silhouette(W, best['labels'])})
    return pd.DataFrame(rows), time.time() - t


@st.cache_data(show_spinner=False)
def run_gmm(name, standardize, ncomp, use_pca, k, seed):
    t = time.time()
    W = space(name, standardize, ncomp, use_pca)
    try:
        return {'out': trace.gmm_trace(W, k, seed=seed), 'error': None}, time.time() - t
    except np.linalg.LinAlgError as e:
        return {'out': None, 'error': f'covariance matrix is not positive definite ({e}); use fewer PCA components'}, time.time() - t


@st.cache_data(show_spinner=False)
def run_hier(name, standardize, ncomp, use_pca, link):
    t = time.time()
    W = space(name, standardize, ncomp, use_pca)
    merges, Z = trace.agglo_trace(W, link)
    return {'merges': merges, 'Z': Z}, time.time() - t


@st.cache_data(show_spinner=False)
def sim_turns(name):
    """The same rows (same order) that get_data turned into features at turn level, with their text columns."""
    df = _sim_rows() if name.startswith('Campaign: baseline') else scenarios.simulate(SIM_KEYS[name])
    return df.dropna(subset=['true_compliance_state']).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def redteam_raw():
    return ext.load_redteam_raw()


@st.cache_data(show_spinner=False)
def run_hmm(key, seed):
    t = time.time()
    d = get_data(key)[0]
    if d['seq'] is None:
        return None, 0.0
    out = trace.hmm_trace(d['X'].values.astype(float), d['seq']['conv'], d['seq']['t'], d['y'], seed=seed)
    return out, time.time() - t


# ======================================================================
# small plotting helpers
# ======================================================================

def fig_ax(w=6.2, h=4.4):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.spines[['top', 'right']].set_visible(False)
    return fig, ax


def show(fig):
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def color_by_labels(labels):
    return [PALETTE[int(l) % len(PALETTE)] for l in labels]


def pc_axes(ax, p2):
    ax.set_xlabel(p2[0])
    ax.set_ylabel(p2[1])


def mixed_colors(R):
    return np.clip(R @ BASE_RGB[: R.shape[1]], 0, 1)


def ellipse(ax, mean, cov2, color, nstd=(1, 2)):
    w, v = np.linalg.eigh(cov2)
    ang = np.degrees(np.arctan2(v[1, 1], v[0, 1]))
    for s, a in zip(nstd, (1.0, 0.45)):
        ax.add_patch(Ellipse(mean, 2 * s * np.sqrt(max(w[1], 1e-12)), 2 * s * np.sqrt(max(w[0], 1e-12)), angle=ang, fill=False, ec=color, lw=1.8, alpha=a))


# ======================================================================
# sidebar controls
# ======================================================================
st.title('Unsupervised pipeline, step by step')
st.caption('PCA, K-means, GMM-EM, hierarchical clustering and an HMM on red-team judge scores and real red-teaming datasets. Labels are never used to fit anything, only to score at the end.')

with st.sidebar:
    st.header('Run settings')
    name = st.selectbox('Dataset', DATASETS)
    is_ext = not name.startswith('Campaign')
    level = st.radio('Level', LEVELS, disabled=is_ext, help='Turn level keeps every turn (needed for the HMM). Conversation level summarises each conversation into one row.') if not is_ext else LEVELS[0]
    max_rows = st.number_input('Max rows (random sample, real datasets only)', 300, 3000, 1500, step=100, disabled=not is_ext, help='Hierarchical clustering and silhouette cost O(N^2), so large datasets are sampled.')
    dkey = f'{name}||{int(max_rows)}||{level}'
    try:
        data, _ = get_data(dkey)
    except ext.DatasetUnavailable as e:
        st.error(str(e))
        fetch = {'Anthropic': ext.fetch_redteam, 'JailbreakBench': ext.fetch_jbb}.get(name.split()[0])
        cmd = 'python -m src.analysis.external_data ' + ('redteam' if name.startswith('Anthropic') else 'jbb')
        st.code(cmd, language='bash')
        if fetch and st.button('Download now (needs internet)'):
            with st.spinner('Downloading...'):
                try:
                    fetch()
                    st.cache_data.clear()
                    st.rerun()
                except ext.DatasetUnavailable as e2:
                    st.error(str(e2))
        st.stop()
    n_classes = len(np.unique(data['y']))
    n_features = data['X'].shape[1]
    standardize = st.checkbox('Standardize features (z-score)', value=True, help='Turn this off to see how much scale differences between features change the clusters.')
    use_pca = st.checkbox('Cluster K-means and GMM in PCA space', value=n_features > 3 or (name.startswith('Campaign') and level.startswith('Turn')))
    max_c = max(2, min(n_features, 10))
    ncomp = st.slider('PCA components kept', 2, max_c, min(2, max_c), disabled=not use_pca) if max_c > 2 else 2
    k = st.number_input('Number of clusters k', 2, 8, n_classes, key=f'k_{name}_{level[:4]}')
    seed = st.number_input('Random seed', 0, 9999, 42, key='seed')
    link = st.selectbox('Hierarchical linkage', ['complete', 'single', 'average', 'ward'])
    hc_pca = st.checkbox('Hierarchical on PCA space too', value=False, help='The slides cluster conversations on the standardized features, so this is off by default.')
    show_labels = st.checkbox('Colour plots by true label (display only)', value=True)
    st.markdown('---')
    st.caption('Scoring uses the true labels only after clustering: matched accuracy, ARI, silhouette.')

# ======================================================================
# run the whole pipeline once, logging each step
# ======================================================================
LOG = []
def log(step, what, params, secs):
    LOG.append({'step': step, 'what': what, 'parameters': params, 'seconds': round(secs, 3)})

try:
    with st.spinner('Running pipeline...'):
        data, s = get_data(dkey); log(1, 'load data', f'{name}, {len(data["X"])} rows', s)
        X_raw = data['X'].values.astype(float); y = data['y']; classes = data['classes']
        pr, s = prep(dkey, standardize); log(2, 'standardize' if standardize else 'centre only', f'shape {X_raw.shape}', s)
        pc, s = run_pca(dkey, standardize); log(3, 'PCA (eigendecomposition of covariance)', f'{n_features} features', s)
        W = space(dkey, standardize, ncomp, use_pca)
        km, s = run_kmeans(dkey, standardize, ncomp, use_pca, int(k), int(seed)); log(4, 'K-means, 10 restarts, k-means++ seeding', f'k={int(k)}, seed={int(seed)}, space={W.shape[1]}-D', s)
        sweep, s = run_k_sweep(dkey, standardize, ncomp, use_pca, int(seed)); log(4, 'K-means sweep over k=2..6 (elbow, silhouette)', f'seed={int(seed)}', s)
        gm, s = run_gmm(dkey, standardize, ncomp, use_pca, int(k), int(seed)); log(5, 'GMM fitted by EM (init from K-means)', f'K={int(k)}, full covariance', s)
        Wh = space(dkey, standardize, ncomp, hc_pca and use_pca)
        hc, s = run_hier(dkey, standardize, ncomp, hc_pca and use_pca, link); log(6, f'agglomerative, {link} linkage (Lance-Williams)', f'{len(Wh)} points, {Wh.shape[1]}-D', s)
        seq = data['seq']
        hm = None
        if seq is not None:
            hm, s = run_hmm(dkey, int(seed)); log(7, 'HMM (hmmlearn GaussianHMM, 3 states, diag covariance)', f'{len(np.unique(seq["conv"]))} conversations, order-aware', s)
except Exception as e:  # noqa: BLE001
    st.error(f'The pipeline could not run with these settings: {e}')
    st.code(traceback.format_exc())
    st.stop()

Z2 = np.asarray(pc['eigvecs'])[:, :2]
P = (pr['Xs'] - pc['mean']) @ pc['eigvecs']  # all principal components
P2 = P[:, :2]
pc_names = [f'PC{i + 1}' for i in range(P.shape[1])]
feat_names = list(data['X'].columns)
disp_colors = color_by_labels(y) if show_labels else '#94A3B8'

tabs = st.tabs(['1 Data', '2 Standardize', '3 PCA', '4 K-means', '5 GMM-EM', '6 Hierarchical', '7 HMM (order-aware)', '8 Scoreboard and checks', '9 Run log', '10 Conversations'])

# ----------------------------------------------------------------------
with tabs[0], guard('Data'):
    st.subheader('1. The data')
    st.write(data['about'])
    c1, c2 = st.columns([1, 1])
    with c1:
        counts = pd.Series(y).map(lambda i: classes[i]).value_counts().rename('rows')
        st.write(f'**{X_raw.shape[0]} rows x {X_raw.shape[1]} features**, {n_classes} true classes (hidden from the algorithms).')
        st.dataframe(counts.to_frame().assign(share=lambda d: (d['rows'] / d['rows'].sum()).round(3)), **DF_KW)
        fig, ax = fig_ax(5, 2.6)
        ax.barh(counts.index[::-1], counts.values[::-1], color=PALETTE[: len(counts)][::-1])
        ax.set_xlabel('rows')
        show(fig)
    with c2:
        st.write('First 15 rows (features only, exactly what the algorithms receive):')
        st.dataframe(data['X'].head(15).round(4), **DF_KW)
        if show_labels:
            st.write('True label of those rows (used only for colouring and final scoring):')
            st.write(', '.join(classes[i] for i in y[:15]))
    with st.expander('Describe every feature'):
        st.dataframe(data['X'].describe().T.round(4), **DF_KW)

# ----------------------------------------------------------------------
with tabs[1], guard('Standardize'):
    st.subheader('2. Standardize')
    st.latex(r'z = \frac{x - \mu}{\sigma}')
    st.write('Each feature is rescaled to mean 0 and standard deviation 1, so a feature measured in hundreds does not drown out one measured in fractions. '
             + ('This run is standardized.' if standardize else '**This run is NOT standardized: features are only centred, so large-scale features dominate distances.**'))
    c1, c2 = st.columns(2)
    stat = pd.DataFrame({'mean before': X_raw.mean(0), 'std before': X_raw.std(0), 'mean after': pr['Xs'].mean(0), 'std after': pr['Xs'].std(0)}, index=feat_names).round(4)
    with c1:
        st.write('Per-feature statistics')
        st.dataframe(stat, **DF_KW)
    with c2:
        fig, ax = fig_ax(5.6, 3.4)
        idx = np.arange(len(feat_names))
        ax.bar(idx - 0.2, X_raw.std(0), 0.4, label='std before', color='#94A3B8')
        ax.bar(idx + 0.2, pr['Xs'].std(0), 0.4, label='std after', color='#0D9488')
        ax.set_yscale('log'); ax.set_xticks(idx); ax.set_xticklabels(feat_names, rotation=70, ha='right', fontsize=7); ax.legend()
        ax.set_title('Feature spread before vs after (log scale)', fontsize=10)
        show(fig)
    r1, r2 = st.columns(2)
    r1.write('Raw (first 8 rows)'); r1.dataframe(pd.DataFrame(X_raw[:8], columns=feat_names).round(4), **DF_KW)
    r2.write('After this step (first 8 rows)'); r2.dataframe(pd.DataFrame(pr['Xs'][:8], columns=feat_names).round(4), **DF_KW)

# ----------------------------------------------------------------------
with tabs[2], guard('PCA'):
    st.subheader('3. PCA')
    st.latex(r'S = \tfrac{1}{N}(X-\bar{x})^{T}(X-\bar{x}), \qquad S\,u_i = \lambda_i u_i, \qquad z = U^{T}(x-\bar{x})')
    st.write('Find the directions where the data varies most. Eigenvalues say how much variance each direction holds; eigenvectors are the directions.')
    ev = pd.DataFrame({'eigenvalue': pc['eigvals'], 'share of variance': pc['ratio'], 'cumulative': np.cumsum(pc['ratio'])}, index=pc_names).round(5)
    c1, c2 = st.columns([1, 1])
    with c1:
        st.write('Eigenvalues (sorted)')
        st.dataframe(ev.head(12), **DF_KW)
        fig, ax = fig_ax(5.4, 3.2)
        m = min(len(pc['ratio']), 15)
        ax.bar(range(1, m + 1), pc['ratio'][:m], color='#0D9488'); ax.plot(range(1, m + 1), np.cumsum(pc['ratio'])[:m], 'o-', color='#F2A20C')
        ax.set_xlabel('component'); ax.set_ylabel('share of variance'); ax.set_title('Scree plot (bars) and cumulative (line)', fontsize=10)
        show(fig)
        if use_pca:
            st.info(f'Clustering uses the first {ncomp} components, which keep {pc["ratio"][:ncomp].sum():.1%} of the variance.')
        if pc['ratio'][-1] < 1e-10 and n_features == 3:
            st.success('The last eigenvalue is 0: the three judge probabilities sum to 1, so only 2 dimensions carry information.')
    with c2:
        fig, ax = fig_ax(5.4, 4.4)
        ax.scatter(P2[:, 0], P2[:, 1], c=disp_colors, s=9, alpha=.75)
        pc_axes(ax, (f'PC1 ({pc["ratio"][0]:.1%})', f'PC2 ({pc["ratio"][1]:.1%})')); ax.set_title('Data projected onto PC1 and PC2', fontsize=10)
        show(fig)
    with st.expander('Covariance matrix S and loadings (what each component is made of)'):
        st.write('Covariance matrix S')
        st.dataframe(pd.DataFrame(pc['cov'], index=feat_names, columns=feat_names).round(3), **DF_KW)
        st.write('Loadings: weight of each original feature in each component (columns are the eigenvectors)')
        st.dataframe(pd.DataFrame(pc['eigvecs'][:, :min(5, P.shape[1])], index=feat_names, columns=pc_names[:5]).round(3), **DF_KW)
    with st.expander('First 10 projected points'):
        st.dataframe(pd.DataFrame(P[:10, :min(5, P.shape[1])], columns=pc_names[:5]).round(4), **DF_KW)

# ----------------------------------------------------------------------
with tabs[3], guard('K-means'):
    st.subheader('4. K-means')
    st.latex(r'J = \sum_n \lVert x_n - \mu_{k(n)} \rVert^2 \quad \text{(assign to nearest centre, then move each centre to the mean of its points)}')
    best = km['best']; hist = best['history']
    st.write(f'Run in a **{W.shape[1]}-D** space. 10 random restarts from k-means++ starts; the restart with the lowest J wins. '
             f'Winning run: #{best["run"] + 1}, converged in **{len(hist)} iterations**, final J = **{best["J"]:.3f}**.')
    t = st.slider('Replay iteration (0 = starting centres)', 0, len(hist), 0, key='km_t')
    c1, c2 = st.columns([1.15, 1])
    with c1:
        fig, ax = fig_ax(6.4, 4.8)
        if t == 0:
            ax.scatter(P2[:, 0], P2[:, 1], c='#CBD5E1', s=9)
            title = 'Start: k-means++ picks far-apart points as centres'
        else:
            h = hist[t - 1]
            ax.scatter(P2[:, 0], P2[:, 1], c=color_by_labels(h['labels']), s=9, alpha=.75)
            title = f'Iteration {t}: points join the nearest centre, then centres move (arrow)'
        # centres live in W-space; show them in the PC1/PC2 plane via nearest-member mean of the plotted points
        def to_plane(cs, labels):
            return np.array([P2[labels == j].mean(0) if np.any(labels == j) else P2[0] for j in range(len(cs))])
        if t == 0:
            lab0 = ((W[:, None, :] - best['init'][None]) ** 2).sum(-1).argmin(1)
            cplane = to_plane(best['init'], lab0)
            ax.scatter(cplane[:, 0], cplane[:, 1], s=220, marker='X', c='white', edgecolors='#0F172A', linewidths=1.8, zorder=5)
        else:
            lb = h['labels']
            before = to_plane(h['centers_before'], ((W[:, None, :] - h['centers_before'][None]) ** 2).sum(-1).argmin(1))
            after = to_plane(h['centers_after'], lb)
            for b_, a_ in zip(before, after):
                ax.annotate('', xy=a_, xytext=b_, arrowprops=dict(arrowstyle='->', color='#0F172A', lw=1.6))
            ax.scatter(before[:, 0], before[:, 1], s=90, facecolors='none', edgecolors='#0F172A', linewidths=1.6, zorder=5)
            ax.scatter(after[:, 0], after[:, 1], s=220, marker='X', c='white', edgecolors='#0F172A', linewidths=1.8, zorder=5)
        ax.set_title(title, fontsize=9.5); pc_axes(ax, ('PC1', 'PC2'))
        show(fig)
        st.caption('The picture always uses the PC1/PC2 plane; centre markers are drawn at the average position of their members there. Exact centre coordinates are in the table.')
    with c2:
        Js = [h['J'] for h in hist]
        fig, ax = fig_ax(5, 2.5)
        ax.plot(range(1, len(Js) + 1), Js, 'o-', color='#2F6FED')
        if t:
            ax.axvline(t, color='#E5484D', ls='--')
        ax.set_xlabel('iteration'); ax.set_ylabel('distortion J'); ax.set_title('J never goes up', fontsize=10)
        show(fig)
        if t:
            h = hist[t - 1]
            sizes = np.bincount(h['labels'], minlength=int(k))
            st.write('Centres after this iteration and cluster sizes')
            cdf = pd.DataFrame(h['centers_after'][:, :min(6, W.shape[1])], columns=[f'dim{i + 1}' for i in range(min(6, W.shape[1]))]).round(4)
            cdf.insert(0, 'size', sizes); cdf.index = [f'cluster {j}' for j in range(int(k))]
            st.dataframe(cdf, **DF_KW)
        else:
            st.write('Starting centres')
            st.dataframe(pd.DataFrame(best['init'][:, :min(6, W.shape[1])], columns=[f'dim{i + 1}' for i in range(min(6, W.shape[1]))], index=[f'centre {j}' for j in range(int(k))]).round(4), **DF_KW)
    st.markdown('**Why does a point join a cluster?** Pick a point and see its squared distance to every centre at this iteration.')
    pt = st.number_input('Point index', 0, len(W) - 1, 0, key='km_pt')
    cen_used = best['init'] if t == 0 else hist[t - 1]['centers_before']
    d2 = ((W[int(pt)] - cen_used) ** 2).sum(1)
    st.dataframe(pd.DataFrame({'squared distance to centre': d2.round(4).astype(str), 'chosen': ['<-- nearest' if i == d2.argmin() else '' for i in range(len(d2))]}, index=[f'centre {j}' for j in range(int(k))]).T, **DF_KW)
    c3, c4 = st.columns(2)
    with c3:
        st.write('All 10 restarts (lowest J wins)')
        st.dataframe(pd.DataFrame({'restart': range(1, 11), 'final J': np.round(km['Js'], 3).astype(str), 'winner': ['yes' if i == best['run'] else '' for i in range(10)]}).set_index('restart').T, **DF_KW)
    with c4:
        st.write('Choosing k: lower J is always better with more clusters, so look at the elbow and the silhouette peak')
        fig, ax = fig_ax(5, 2.6)
        ax.plot(sweep['k'], sweep['distortion J'], 'o-', color='#2F6FED'); ax.set_ylabel('J', color='#2F6FED'); ax.set_xlabel('k')
        ax2 = ax.twinx(); ax2.plot(sweep['k'], sweep['silhouette'], 's-', color='#F2A20C'); ax2.set_ylabel('silhouette', color='#F2A20C')
        show(fig)
        st.dataframe(sweep.round(3).set_index('k').T, **DF_KW)
    with st.expander('The code that runs'):
        st.code('labels = sq_dists(X, mu).argmin(axis=1)                       # assign\nnew_mu = np.array([X[labels == j].mean(axis=0) for j in range(k)])  # update\nif np.allclose(new_mu, mu): break                                   # converged', language='python')

# ----------------------------------------------------------------------
with tabs[4], guard('GMM-EM'):
    st.subheader('5. Gaussian mixture, fitted by EM')
    st.latex(r'p(x)=\sum_k \pi_k\,\mathcal{N}(x\mid\mu_k,\Sigma_k), \qquad \gamma_{nk}=\frac{\pi_k\,\mathcal{N}(x_n\mid\mu_k,\Sigma_k)}{\sum_j \pi_j\,\mathcal{N}(x_n\mid\mu_j,\Sigma_j)}')
    if gm['error']:
        st.error(gm['error'])
    else:
        g = gm['out']; gh = g['history']
        st.write(f'Initialised from the K-means labels. EM ran **{len(gh) - 1} iterations** (stops when the log-likelihood gain is below 1e-6). BIC = {g["bic"]:.1f}.')
        it = st.slider('Replay EM iteration (0 = initialisation from K-means)', 0, len(gh) - 1, 0, key='gm_t')
        h = gh[it]
        c1, c2 = st.columns([1.15, 1])
        with c1:
            fig, ax = fig_ax(6.4, 4.8)
            ax.scatter(P2[:, 0], P2[:, 1], c=mixed_colors(h['R']), s=10, alpha=.85)
            if use_pca:  # components are orthogonal PCs, so the PC1/PC2 marginal is the top-left 2x2 block
                for j in range(int(k)):
                    ellipse(ax, h['mu'][j][:2], h['cov'][j][:2, :2], '#0F172A')
                    ax.scatter(*h['mu'][j][:2], s=140, marker='X', c='white', edgecolors='#0F172A', linewidths=1.6, zorder=5)
            ax.set_title(f'Iteration {it}: colour blends where membership is shared' + ('' if use_pca else ' (ellipses shown only in PCA space)'), fontsize=9.5)
            pc_axes(ax, ('PC1', 'PC2'))
            show(fig)
            if use_pca:
                st.caption('Ellipses are the 1 and 2 standard-deviation contours of each component, taken from the learned covariance (PC1/PC2 marginal).')
        with c2:
            fig, ax = fig_ax(5, 2.6)
            ax.plot(range(1, len(g['lls']) + 1), g['lls'], color='#16A34A')
            if it:
                ax.axvline(it, color='#E5484D', ls='--')
            ax.set_xlabel('iteration'); ax.set_ylabel('log-likelihood'); ax.set_title('Never decreases', fontsize=10)
            show(fig)
            ptab = pd.DataFrame({'weight pi': h['pi'], 'effective points': (h['R'].sum(0))}, index=[f'component {j}' for j in range(int(k))]).round(4)
            st.write('Parameters used in this iteration (re-estimated from the responsibilities)')
            st.dataframe(ptab, **DF_KW)
            unsure = int((h['R'].max(1) < 0.9).sum())
            st.metric('Points the model is less than 90% sure about', unsure)
        st.markdown('**How is one point assigned?** The E-step arithmetic for a single point at the selected iteration:')
        gpt = st.number_input('Point index', 0, len(W) - 1, 0, key='gm_pt')
        log_pi, log_dens, log_joint, resp = trace.point_breakdown(W[int(gpt)], h['pi'], h['mu'], h['cov'])
        st.dataframe(pd.DataFrame({'log pi': log_pi, 'log N(x | mu, Sigma)': log_dens, 'log pi + log N': log_joint, 'responsibility': resp}, index=[f'component {j}' for j in range(int(k))]).round(4), **DF_KW)
        st.caption('Responsibility = exp(row) normalised to sum to 1. The cluster label is the component with the largest responsibility.')
        with st.expander('The code that runs'):
            st.code('Nk = R.sum(axis=0)\npi, mu = Nk / N, (R.T @ X) / Nk[:, None]                       # M-step\ncov = (R[:, j, None] * (X - mu[j])).T @ (X - mu[j]) / Nk[j] + reg * np.eye(D)\nlog_joint = np.log(pi[j]) + log_gauss(X, mu[j], cov[j])          # E-step\nR = np.exp(log_joint - log_norm[:, None])', language='python')

# ----------------------------------------------------------------------
with tabs[5], guard('Hierarchical'):
    st.subheader('6. Hierarchical (agglomerative) clustering')
    st.latex(r'd(\text{new}, c) \text{ from } d(a,c),\; d(b,c) \;\text{ via the Lance-Williams update for the chosen linkage}')
    merges, Zl = hc['merges'], hc['Z']
    N = len(Wh)
    st.write(f'{N} points in a **{Wh.shape[1]}-D** space, **{link}** linkage. Every point starts as its own cluster; the closest pair merges repeatedly until one cluster remains ({N - 1} merges). A cut then leaves k clusters.')
    c1, c2 = st.columns([1.2, 1])
    with c1:
        fig, ax = fig_ax(7, 4.2)
        cut_h = (Zl[N - int(k) - 1, 2] + Zl[N - int(k), 2]) / 2 if N - int(k) < len(Zl) else Zl[-1, 2]
        dendrogram(Zl, ax=ax, truncate_mode='lastp', p=min(30, N), no_labels=True, color_threshold=cut_h, above_threshold_color='#94A3B8')
        ax.axhline(cut_h, color='#E5484D', ls='--'); ax.set_ylabel('merge distance'); ax.set_title(f'Dendrogram (last {min(30, N)} merges shown), cut for k={int(k)}', fontsize=10)
        show(fig)
    with c2:
        st.write('Largest merge distances (the tall gaps are natural places to cut)')
        top = pd.DataFrame(merges[-8:], columns=['slot a', 'slot b', 'distance', 'new size'])
        top.index = range(N - 8, N)
        st.dataframe(top.round(4).iloc[::-1], **DF_KW)
    step = st.slider('Replay merge step (0 = every point alone)', 0, N - int(k), N - int(k), key='hc_t')
    lab = trace.labels_at(merges, N, N - step)
    c3, c4 = st.columns([1.2, 1])
    with c3:
        fig, ax = fig_ax(6.4, 4.6)
        nshow = N - step
        if nshow <= 25:
            ax.scatter(P2[:, 0], P2[:, 1], c=color_by_labels(lab), s=11, alpha=.8)
        else:
            ax.scatter(P2[:, 0], P2[:, 1], c='#CBD5E1', s=9)
        ax.set_title(f'After {step} merges: {nshow} clusters' + (' (grey until 25 or fewer remain)' if nshow > 25 else ''), fontsize=10); pc_axes(ax, ('PC1', 'PC2'))
        show(fig)
    with c4:
        if step:
            a_, b_, d_, sz = merges[step - 1]
            st.write(f'Merge #{step}: slot {a_} absorbed slot {b_} at distance **{d_:.4f}**, new cluster size **{sz}**.')
        else:
            st.write('No merges yet.')
        st.write('Recent merges up to this step')
        recent = pd.DataFrame(merges[max(0, step - 8):step], columns=['slot a', 'slot b', 'distance', 'new size'])
        recent.index = range(max(0, step - 8) + 1, step + 1)
        st.dataframe(recent.round(4), **DF_KW)
    st.caption('The ARI of every linkage is on the Scoreboard tab.')
    with st.expander('The code that runs'):
        st.code('a, b = np.unravel_index(np.argmin(D), D.shape)      # closest pair of clusters\nnew = np.maximum(dac, dbc)                                # complete linkage (farthest pair)\nD[a, o] = D[o, a] = new; D[b, :] = D[:, b] = np.inf       # merge b into a', language='python')

# ----------------------------------------------------------------------
with tabs[6], guard('HMM'):
    st.subheader('7. Hidden Markov model (uses the order of turns)')
    st.latex(r'P(z_t \mid z_{t-1}) = A_{z_{t-1} z_t}, \qquad x_t \mid z_t \sim \mathcal{N}(\mu_{z_t}, \mathrm{diag}\,\sigma^2_{z_t})')
    if hm is None:
        st.info('An HMM needs ordered turns inside conversations, with one set of scores per turn. Only the simulated campaign at turn level has that. '
                'The real datasets here have one row per conversation or per attack, so there is no sequence to model. This tab is therefore empty for this dataset.')
    else:
        st.write('The model assumes each conversation walks through hidden states turn by turn (a refusal tends to be followed by another refusal), and each state produces judge scores around its own average. '
                 'It learns the transition matrix A and the per-state score averages from the scores alone, then decodes the most likely state for every turn (Viterbi). '
                 '**This is the one model here that uses the hmmlearn library; the others are written from scratch in NumPy.**')
        cn = [classes[i] for i in range(len(classes))]
        c1, c2, c3 = st.columns(3)
        c1.metric('Accuracy after matching', f'{hm["accuracy"]:.4f}')
        c2.metric('ARI vs true state', f'{hm["ari"]:.4f}')
        c3.metric('Log-likelihood', f'{hm["log_likelihood"]:.1f}')
        a1, a2 = st.columns([1, 1])
        with a1:
            st.write('Learned transition matrix A (row = state now, column = state next turn), states named by the matching step')
            st.dataframe(pd.DataFrame(hm['transmat'], index=[f'from {c}' for c in cn], columns=[f'to {c}' for c in cn]).round(3), **DF_KW)
            stay = np.diag(hm['transmat'])
            st.write('Expected run length in each state (turns) = 1 / (1 - stay probability):')
            st.dataframe(pd.DataFrame({'stay probability': stay.round(3), 'expected turns': (1 / np.maximum(1 - stay, 1e-9)).round(2)}, index=cn).T, **DF_KW)
            st.caption('In scenarios that stop at the first success (baseline, noisy, stealthy) a high stay probability for full violation is partly a stopping-rule artefact: few turns follow a success. In the persistent-jailbreak scenario conversations continue, so the persistence is a real property of the data.')
        with a2:
            st.write('Learned score averages per state (emissions)')
            st.dataframe(pd.DataFrame(hm['means'], index=[f'state "{c}"' for c in cn], columns=[c.replace('judge_', 'mean ') for c in SCORE_COLUMNS]).round(3), **DF_KW)
            st.write('Start-of-conversation probabilities')
            st.dataframe(pd.DataFrame({'start probability': hm['startprob']}, index=cn).round(3).T, **DF_KW)
        st.markdown('**Watch it decode one conversation**')
        ids = list(hm['conv_ids'])
        cid = st.selectbox('Conversation', ids, index=0)
        sel = hm['conv_sorted'] == cid
        pred_cls = np.array([hm['match']['mapping'].get(int(v), -1) for v in hm['decoded_sorted'][sel]])
        true_cls = hm['y_sorted'][sel]
        Xc = hm['X_sorted'][sel]
        turns = np.arange(1, len(Xc) + 1)
        fig, ax = fig_ax(8, 3.2)
        for j, col in enumerate(SCORE_COLUMNS):
            ax.plot(turns, Xc[:, j], 'o-', color=PALETTE[j], label=col.replace('judge_', 'p('), lw=2)
        for tt, pc_ in zip(turns, pred_cls):
            ax.axvspan(tt - 0.5, tt + 0.5, ymin=0.0, ymax=0.06, color=PALETTE[int(pc_) % 10] if pc_ >= 0 else '#CBD5E1')
        ax.set_xticks(turns); ax.set_ylim(0, 1.02); ax.set_xlabel('turn'); ax.set_ylabel('judge score'); ax.legend(fontsize=8, ncol=3, loc='upper center')
        ax.set_title('Scores per turn; coloured strip at the bottom = decoded state', fontsize=10)
        show(fig)
        st.dataframe(pd.DataFrame({'turn': turns, **{c.replace('judge_', 'p_'): Xc[:, j].round(3) for j, c in enumerate(SCORE_COLUMNS)},
                                   'decoded state': [classes[v] if v >= 0 else '?' for v in pred_cls], 'true state': [classes[v] for v in true_cls],
                                   'correct': ['yes' if p == t_ else 'no' for p, t_ in zip(pred_cls, true_cls)]}).set_index('turn').astype(str), **DF_KW)
        with st.expander('The code that runs'):
            st.code('model = GaussianHMM(n_components=3, covariance_type="diag", n_iter=200, random_state=42)\nmodel.fit(X, lengths)               # EM over all conversations, lengths marks where each one ends\ndecoded = model.predict(X, lengths) # Viterbi: most likely state path per conversation\nrow_ind, col_ind = linear_sum_assignment(-cm)  # name the states by best match to the truth', language='python')

with tabs[7], guard('Scoreboard'):
    st.subheader('8. Scoreboard and cross-checks')
    st.write('Only now are the true labels used. **Matched accuracy**: name the clusters by the best one-to-one assignment to the true classes (Hungarian algorithm). **ARI**: pairs of points grouped together in both answers (1 perfect, 0 random). **Silhouette** needs no labels.')
    kk = int(k)
    methods, loglik = {}, {}
    methods['K-means (ours, NumPy)'] = km['best']['labels']
    methods['K-means (sklearn, reference)'] = KMeans(n_clusters=kk, n_init=10, random_state=int(seed)).fit_predict(W)
    if not gm['error']:
        methods['GMM-EM (ours, NumPy)'] = gm['out']['labels']
        try:
            sk_gmm = GaussianMixture(n_components=kk, covariance_type='full', init_params='kmeans', reg_covar=1e-6, random_state=int(seed)).fit(W)
            methods['GMM (sklearn, reference)'] = sk_gmm.predict(W)
            loglik['GMM (sklearn, reference)'] = float(sk_gmm.score(W) * len(W))
        except Exception:
            pass
        loglik['GMM-EM (ours, NumPy)'] = gm['out']['lls'][-1]
    if hm is not None:
        methods['HMM (hmmlearn, uses turn order)'] = hm['labels']
    methods[f'Hierarchical {link} (ours)'] = trace.labels_at(merges, N, kk)
    methods[f'Hierarchical {link} (scipy, reference)'] = fcluster(sp_linkage(Wh, link), kk, 'maxclust') - 1
    refs = {'K-means (ours, NumPy)': 'K-means (sklearn, reference)', 'GMM-EM (ours, NumPy)': 'GMM (sklearn, reference)', f'Hierarchical {link} (ours)': f'Hierarchical {link} (scipy, reference)'}
    table = []
    for m, labs in methods.items():
        Xm = Wh if m.startswith('Hier') else W
        sc = trace.score(Xm, np.asarray(labs), y)
        row = {'method': m, 'clusters': sc['n_clusters'], 'matched accuracy': round(sc['accuracy'], 4), 'ARI': round(sc['ari'], 4), 'silhouette': round(sc['silhouette'], 4)}
        if m in loglik:
            row['log-likelihood'] = round(loglik[m], 1)
        if m in refs and refs[m] in methods:
            row['agrees with reference (ARI)'] = round(adjusted_rand(np.asarray(methods[refs[m]]), np.asarray(labs)), 4)
        table.append(row)
    st.dataframe(pd.DataFrame(table).set_index('method'), **DF_KW)
    st.caption('"Agrees with reference" compares our from-scratch NumPy version with the library version on the same data: 1.0 means the same partition. '
               'K-means and hierarchical should match; GMM can settle in a different local optimum from a different start, so compare the log-likelihood (higher is the better fit) before reading a mismatch as a bug.')
    pick = st.selectbox('Show the label-matching step for', list(methods))
    ml = trace.match_labels(y, np.asarray(methods[pick]))
    c1, c2 = st.columns(2)
    with c1:
        st.write('Contingency table: rows = true class, columns = cluster id')
        st.dataframe(pd.DataFrame(ml['cm'], index=[classes[i] for i in ml['classes']], columns=[f'cluster {i}' for i in ml['ids']]), **DF_KW)
        st.write('Hungarian assignment: ' + ', '.join(f'cluster {c} -> {classes[t]}' for c, t in sorted(ml['mapping'].items())))
        st.write(f'Matched accuracy = {int((ml["pred"] == y).sum())} / {len(y)} = **{ml["accuracy"]:.4f}**')
    with c2:
        fig, ax = fig_ax(5.2, 4.2)
        cmx = np.array([[np.sum((y == c) & (ml['pred'] == p)) for p in ml['classes']] for c in ml['classes']])
        ax.imshow(cmx, cmap='Blues')
        for i in range(cmx.shape[0]):
            for j in range(cmx.shape[1]):
                ax.text(j, i, cmx[i, j], ha='center', va='center', color='#0F172A')
        ax.set_xticks(range(len(ml['classes']))); ax.set_xticklabels([classes[i] for i in ml['classes']], rotation=20); ax.set_yticks(range(len(ml['classes']))); ax.set_yticklabels([classes[i] for i in ml['classes']])
        ax.set_xlabel('predicted (after matching)'); ax.set_ylabel('true'); ax.set_title('Confusion matrix after matching', fontsize=10)
        show(fig)
    st.markdown('**Hierarchical: ARI for every linkage on this data** (selecting the best one by ARI uses the labels, so it is slightly optimistic)')
    lr = []
    for l_ in ('single', 'complete', 'average', 'ward'):
        mg = run_hier(dkey, standardize, ncomp, hc_pca and use_pca, l_)[0]['merges']
        lr.append({'linkage': l_, 'ARI': round(adjusted_rand(y, trace.labels_at(mg, N, kk)), 4)})
    st.dataframe(pd.DataFrame(lr).set_index('linkage').T, **DF_KW)
    st.markdown('**What the checks mean**')
    st.markdown('- Each from-scratch method is compared with the library version on the same data.\n- `pytest` proves the step-recording code here returns exactly the same answers as `src/analysis/unsupervised.py`.\n- Labels never enter the fitting, only this scoring step.')

# ----------------------------------------------------------------------
with tabs[8], guard('Run log'):
    st.subheader('9. Run log')
    st.write('Everything that was computed for the current settings, in order. A step that was already computed with the same settings is served from cache; its time is the original compute time.')
    st.dataframe(pd.DataFrame(LOG).set_index('step'), **DF_KW)
    st.json({'dataset': name, 'level': level if not is_ext else None, 'max_rows': int(max_rows) if is_ext else None, 'standardize': standardize, 'cluster_in_pca_space': use_pca, 'pca_components': ncomp if use_pca else None, 'k': int(k), 'seed': int(seed), 'linkage': link, 'hierarchical_in_pca_space': bool(hc_pca and use_pca)})


# ----------------------------------------------------------------------
def clip(t, n=900):
    t = str(t)
    return t if len(t) <= n else t[:n] + f' ... [{len(t) - n} more characters not shown]'


def score_bars(vals, labels):
    cols = st.columns(len(vals))
    for c, v, lab in zip(cols, vals, labels):
        c.progress(float(min(max(v, 0.0), 1.0)), text=f'{lab}: {v:.2f}')


with tabs[9], guard('Conversations'):
    st.subheader('10. The conversations themselves')
    km_labels = np.asarray(km['best']['labels'])
    km_map = trace.match_labels(y, km_labels)['mapping']
    km_name = lambda c: classes[km_map[int(c)]] if int(c) in km_map else 'unmatched'

    if name.startswith('Campaign'):
        st.info('**These campaigns are simulated, so the text is placeholder text, not real prompts and replies.** The generator writes a hidden compliance state per turn, then draws the judge scores from it. '
                'What you can see here is real for this project: the turn order, the attacker technique (baseline only), both judges\' scores, the true state, and what the unsupervised models decided for each turn. '
                'Real text appears if you run a real campaign (`src/eval/run_campaign.py`) and put its `.jsonl` in `data/transcripts/`.')
        if not level.startswith('Turn'):
            st.warning('Switch the sidebar Level to "Turn level" to see per-turn K-means and HMM decisions. Showing conversations without them.')
        df = sim_turns(name)
        final = df.sort_values('timestep_in_conversation').groupby('conversation_id')['true_compliance_state'].last()
        f1, f2 = st.columns(2)
        want = f1.selectbox('Show conversations that ended in', ['any'] + list(COMPLIANCE_STATES))
        ids = [c for c in final.index if want == 'any' or final[c] == want]
        cid = f2.selectbox(f'Conversation ({len(ids)} match)', ids)
        sub = df[df['conversation_id'] == cid].sort_values('timestep_in_conversation')
        turn_level = level.startswith('Turn') and len(df) == len(km_labels)
        dec = None
        if turn_level and hm is not None:
            dec = np.array([hm['match']['mapping'].get(int(v), -1) for v in hm['labels']])
        for i, r in sub.iterrows():
            t = int(r['timestep_in_conversation']) + 1
            st.markdown(f'**Turn {t}**' + (f" · technique `{r['technique']}`" if 'technique' in sub.columns and isinstance(r.get('technique'), str) else ''))
            with st.chat_message('user'):
                st.write(r['attacker_prompt'] if isinstance(r.get('attacker_prompt'), str) else '*(simulated attacker turn, no text stored)*')
            with st.chat_message('assistant'):
                st.write(r['target_response'] if isinstance(r.get('target_response'), str) else '*(simulated target reply, no text stored)*')
            score_bars([r[c] for c in SCORE_COLUMNS], ['refusal', 'partial', 'full violation'])
            if 'eval_refusal' in sub.columns and pd.notna(r.get('eval_refusal')):
                st.caption('Second judge (evaluation ensemble): ' + ', '.join(f"{n} {r['eval_' + n]:.2f}" for n in COMPLIANCE_STATES) + (' · flagged for human review' if r.get('eval_needs_human_review') else ''))
            verdicts = {'true state': r['true_compliance_state']}
            if turn_level:
                verdicts['K-means says'] = km_name(km_labels[i])
                if dec is not None:
                    verdicts['HMM says'] = classes[dec[i]] if dec[i] >= 0 else 'unmatched'
            st.caption(' · '.join(f'{k}: **{v}**' + (' ✓' if k != 'true state' and v == r['true_compliance_state'] else (' ✗' if k != 'true state' else '')) for k, v in verdicts.items()))
            st.divider()

    elif name.startswith('Anthropic'):
        st.info('Real red-team conversations written by crowdworkers against an Anthropic assistant. The **rating** is the human red-teamer\'s own success score (0 = failed, 4 = very successful). '
                'The clusters below were found WITHOUT the text, from the numeric features only, so reading the text is a way to sanity-check what a cluster contains.')
        show_text = st.checkbox('Show the conversation text. It is raw data from red-team attempts and can include offensive or harmful content.', value=False)
        try:
            raw = redteam_raw()
        except ext.DatasetUnavailable as e:
            st.error(str(e)); raw = None
        if raw is not None:
            idx = data.get('idx')
            sub = (raw.iloc[idx] if idx is not None else raw).reset_index(drop=True)
            if len(sub) != len(km_labels):
                st.error(f'Row count mismatch ({len(sub)} raw rows vs {len(km_labels)} feature rows). Delete data/external/red_team_attempts_raw.jsonl and re-download.')
            else:
                f1, f2 = st.columns(2)
                rating_pick = f1.selectbox('True rating', ['any'] + classes)
                cl_pick = f2.selectbox('K-means cluster', ['any'] + sorted(int(c) for c in np.unique(km_labels)))
                mask = np.ones(len(sub), dtype=bool)
                if rating_pick != 'any':
                    mask &= (y == classes.index(rating_pick))
                if cl_pick != 'any':
                    mask &= (km_labels == cl_pick)
                rows = np.flatnonzero(mask)[:300]
                st.caption(f'{int(mask.sum())} conversations match (listing the first {len(rows)}).')
                if len(rows):
                    pick = st.selectbox('Conversation', rows, format_func=lambda i: f'row {i} · rating {y[i]} · K-means cluster {km_labels[i]}')
                    st.write(f'Rating **{classes[y[pick]]}** · K-means cluster **{km_labels[pick]}** (named "{km_name(km_labels[pick])}" by the matching step)')
                    st.dataframe(data['X'].iloc[[pick]].T.rename(columns={pick: 'value'}).round(3).astype(str), **DF_KW)
                    if 'task_description' in sub.columns and show_text:
                        st.write('**Task the red-teamer was given:** ' + clip(sub.loc[pick, 'task_description'], 400))
                    if show_text:
                        for who, txt in ext.parse_transcript(sub.loc[pick, 'transcript']):
                            with st.chat_message('user' if who == 'Human' else 'assistant'):
                                st.write(clip(txt))
                    else:
                        n_turns = len(ext.parse_transcript(sub.loc[pick, 'transcript']))
                        st.write(f'{n_turns} messages in this conversation. Tick the box above to read them.')
    else:
        st.info('JailbreakBench rows are attack prompts and target replies built to elicit harmful output. This loader keeps only the numeric features (query count, token counts, lengths, refusal-phrase flag, attack method) and does not store the text, so there is nothing to display. '
                'Use the Data tab to see the features and the Scoreboard to see how well they separate jailbroken from not jailbroken.')
