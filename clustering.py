import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # file-only backend - same fix as classification.py
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

# how many of the most-different features to report/plot per cluster
TOP_N_FEATURES = 5


def cluster(df, target_class=1):
    # NOTE: this uses one clustering model only - KMeans. Elbow method and
    # silhouette score below are not separate models, just two ways of
    # scoring different values of k so we can pick one before fitting KMeans.
    os.makedirs(OUT_DIR, exist_ok=True)

    numeric_df = df.select_dtypes(include=[np.number])
    feature_cols = [c for c in numeric_df.columns if c != 'label']
    print(f'Using {len(feature_cols)} features for clustering:')
    print(feature_cols)

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
    print('Most extreme URLs in this class (excluded from clustering below, reported here):')
    for idx in top_outliers:
        print(f'  {urls_cluster.iloc[idx]}  (distance={outlier_score_raw[idx]:.1f})')

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
    plot.figure(figsize=(7, 6))
    sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=cluster_labels, palette='tab10', s=15)
    plot.title(f'Clusters within label={target_class} (PCA 2D view)')
    plot.xlabel('PC1')
    plot.ylabel('PC2')
    plot.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    plot.savefig(f'{OUT_DIR}/cluster_pca_scatter.png', dpi=150)
    plot.close()

# =====================================================================
# Investigate what each cluster actually represents.
# Composition (how many rows) alone doesn't explain a cluster, so for each
# one we compare its average feature values against the overall average
# for this class, pull out what differs the most, and back it with real
# example URLs. One profile chart is saved per cluster.
# =====================================================================
    overall_mean = X_cluster.mean()
    denom = overall_mean.replace(0, 1e-9).abs()

    print(f'\nCluster profiles (within label={target_class}):')
    for c in sorted(result_df['cluster'].unique()):
        sub = result_df[result_df['cluster'] == c]
        cluster_mean = sub[feature_cols].mean()
        rel_diff = (cluster_mean - overall_mean) / denom
        top_features = rel_diff.abs().sort_values(ascending=False).head(TOP_N_FEATURES)

        print(f'\n--- Cluster {c} (n={len(sub)}, {len(sub)/len(result_df):.1%} of class) ---')
        for feat in top_features.index:
            print(f'  {feat}: cluster={cluster_mean[feat]:.3f}  overall={overall_mean[feat]:.3f}  diff={rel_diff[feat]:+.1%}')
        print('  Example URLs:')
        for u in sub['url'].sample(min(5, len(sub)), random_state=RANDOM_STATE):
            print(f'    {u}')

        # Bar chart of the same top-differing features, for the report -
        # green = higher than the class average, red = lower.
        diffs = rel_diff[top_features.index].sort_values()
        colors = ['tab:red' if v < 0 else 'tab:green' for v in diffs]
        plot.figure(figsize=(7, 4))
        diffs.plot(kind='barh', color=colors)
        plot.axvline(0, color='black', linewidth=0.8)
        plot.title(f'Cluster {c} - how it differs from the class average')
        plot.xlabel('Relative difference vs. overall average')
        plot.tight_layout()
        os.makedirs(OUT_DIR, exist_ok=True)
        plot.savefig(f'{OUT_DIR}/cluster_{c}_profile.png', dpi=150)
        plot.close()

    print(f'\nPlots saved to ./{OUT_DIR}/')
    return result_df