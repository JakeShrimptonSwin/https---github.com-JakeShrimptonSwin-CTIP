import os
import numpy as np
import matplotlib.pyplot as plot
import seaborn as sns

from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

RANDOM_STATE = 42
OUT_DIR = "ml_outputs"

# silhouette_score compares every pair of points, so on 300k+ rows the full
# calculation would take hours. We score it on a random sample instead.
SIL_SAMPLE_SIZE = 5000

# Keep the PCA plot readable when the class contains many URLs.
PCA_SAMPLE_SIZE = 10000
URL_DISPLAY_LENGTH = 100

# how many of the most-different features to report/plot per cluster
TOP_N_FEATURES = 5


def _shorten_url(url, max_length=URL_DISPLAY_LENGTH):
    url = str(url)
    if len(url) <= max_length:
        return url
    return f'{url[:max_length - 3]}...'


def cluster(df, target_class=1):
    # NOTE: this uses one clustering model only - KMeans. Elbow method and
    # silhouette score below are not separate models, just two ways of
    # scoring different values of k so we can pick one before fitting KMeans.
    os.makedirs(OUT_DIR, exist_ok=True)

    numeric_df = df.select_dtypes(include=[np.number])
    feature_cols = [c for c in numeric_df.columns if c != 'label']
    print(f'Using {len(feature_cols)} numeric features for clustering (excluding label).')

# --------------------Filter to a single class--------------------
    # label is only used here to pick which rows to cluster - it is never
    # given to KMeans, so the grouping it finds is based purely on the
    # URL features, not on knowing the answer already.
    class_mask = (df['label'] == target_class).values
    X_cluster = df.loc[class_mask, feature_cols].fillna(0).reset_index(drop=True)
    urls_cluster = df.loc[class_mask, 'url'].reset_index(drop=True)

    print(f'Clustering within label={target_class}  (n={len(X_cluster)} rows)')

    # --- flag extreme outliers first, on the UNCAPPED values ---
    # A tiny number of pathological URLs (e.g. a phishing kit with a 25,000-char
    # payload crammed into the URL, vs ~50 chars typical) can be so extreme that
    # even RobustScaler can't stop them dominating KMeans - the model ends up
    # just splitting "the 2-3 freak rows" vs "everyone else", which LOOKS like a
    # great silhouette score but isn't real structure. We report these rows here
    # for transparency, then cap them below so they can't hijack the clustering.
    outlier_score_raw = np.linalg.norm(RobustScaler().fit_transform(X_cluster), axis=1)
    top_outliers = np.argsort(outlier_score_raw)[-3:][::-1]
    print('Most extreme URLs in this class (feature values are capped before clustering):')
    for idx in top_outliers:
        print(f'  {_shorten_url(urls_cluster.iloc[idx])}  (distance={outlier_score_raw[idx]:.1f})')

    # --- cap (winsorize) every feature at the 1st/99th percentile ---
    # This stops the handful of outliers above from single-handedly deciding
    # the cluster structure, while leaving the bulk of normal variation intact.
    lower = X_cluster.quantile(0.01)
    upper = X_cluster.quantile(0.99)
    X_cluster = X_cluster.clip(lower=lower, upper=upper, axis=1)

    X_scaled = RobustScaler().fit_transform(X_cluster)

# =====================================================================
# Step 1 - choose k (number of clusters)
# =====================================================================
    k_range = range(2, 9)
    inertias, sil_scores = [], []
    for k in k_range:
        km = MiniBatchKMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10, batch_size=2048)
        labels_k = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil = silhouette_score(
            X_scaled, labels_k,
            sample_size=min(SIL_SAMPLE_SIZE, len(X_scaled)),
            random_state=RANDOM_STATE,
        )
        sil_scores.append(sil)
        print(f'  k={k}: inertia={km.inertia_:.1f}, silhouette={sil:.4f}')

    fig, axes = plot.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(list(k_range), inertias, marker='o')
    axes[0].set_title('Elbow method (lower = tighter clusters)')
    axes[0].set_xlabel('k')
    axes[0].set_ylabel('Inertia')
    axes[1].plot(list(k_range), sil_scores, marker='o', color='orange')
    axes[1].set_title('Silhouette score (higher = better separated)')
    axes[1].set_xlabel('k')
    axes[1].set_ylabel('Silhouette')
    plot.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    plot.savefig(f'{OUT_DIR}/clustering_k_selection.png', dpi=150)
    plot.close()

    best_k = list(k_range)[int(np.argmax(sil_scores))]
    print(f'Chosen k={best_k} (highest silhouette score: {max(sil_scores):.3f})')

# =====================================================================
# Step 2 - fit the clustering model: KMeans, using the chosen k
# =====================================================================
    kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    result_df = X_cluster.copy()
    result_df['cluster'] = cluster_labels
    result_df['url'] = urls_cluster

# --------------------Visualise clusters--------------------
    # We have way more than 2 features, so we can't plot them directly.
    # PCA compresses all of them down into 2 "summary" numbers (PC1, PC2)
    # that keep as much of the original spread as possible, purely so we
    # can see on a 2D chart whether the clusters actually look separated.
    # PC1/PC2 don't correspond to any single real feature.
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_scaled)
    sample_size = min(PCA_SAMPLE_SIZE, len(coords))
    rng = np.random.default_rng(RANDOM_STATE)
    plot_indices = []
    unique_clusters = np.unique(cluster_labels)
    for cluster_id in unique_clusters:
        cluster_indices = np.flatnonzero(cluster_labels == cluster_id)
        cluster_sample_size = min(
            len(cluster_indices),
            max(1, round(sample_size * len(cluster_indices) / len(coords))),
        )
        plot_indices.extend(rng.choice(cluster_indices, size=cluster_sample_size, replace=False))

    fig, ax = plot.subplots(figsize=(9, 7))
    colors = sns.color_palette('colorblind', n_colors=len(unique_clusters))
    for color, cluster_id in zip(colors, unique_clusters):
        cluster_indices = np.flatnonzero(cluster_labels[plot_indices] == cluster_id)
        sampled_indices = np.asarray(plot_indices)[cluster_indices]
        ax.scatter(
            coords[sampled_indices, 0],
            coords[sampled_indices, 1],
            color=color,
            s=9,
            alpha=0.45,
            edgecolors='none',
            rasterized=True,
            label=f'Cluster {cluster_id} (n={np.count_nonzero(cluster_labels == cluster_id):,})',
        )
    ax.set_title(
        f'Clusters within label={target_class} (PCA, showing {len(plot_indices):,} '
        f'of {len(coords):,} URLs)'
    )
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax.legend(title='Cluster', markerscale=1.5)
    fig.tight_layout()
    fig.savefig(f'{OUT_DIR}/cluster_pca_scatter.png', dpi=180)
    plot.close()

# =====================================================================
# Investigate what each cluster actually represents.
# Composition (how many rows) alone doesn't explain a cluster, so for each
# one we compare its average feature values against the overall average
# for this class, pull out what differs the most, and back it with real
# example URLs.
# =====================================================================
    overall_mean = X_cluster.mean()
    overall_std = X_cluster.std(ddof=0).replace(0, np.nan)

    print(f'\nCluster profiles (within label={target_class}):')
    for c in sorted(result_df['cluster'].unique()):
        sub = result_df[result_df['cluster'] == c]
        cluster_mean = sub[feature_cols].mean()
        standardized_diff = ((cluster_mean - overall_mean) / overall_std).fillna(0)
        top_features = standardized_diff.abs().sort_values(ascending=False).head(TOP_N_FEATURES)

        print(f'\n--- Cluster {c} (n={len(sub)}, {len(sub)/len(result_df):.1%} of class) ---')
        for feat in top_features.index:
            print(
                f'  {feat}: {standardized_diff[feat]:+.2f} SD '
                f'(cluster={cluster_mean[feat]:.3f}, class={overall_mean[feat]:.3f})'
            )
        print('  Example URLs:')
        for u in sub['url'].sample(min(5, len(sub)), random_state=RANDOM_STATE):
            print(f'    {_shorten_url(u)}')

    print(f'\nClustering plots saved to ./{OUT_DIR}/')
    return result_df