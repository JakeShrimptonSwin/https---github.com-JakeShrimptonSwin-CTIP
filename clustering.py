import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

RANDOM_STATE = 42
DATA_PATH = "data/malicious_phish_updated.csv"
OUT_DIR = "ml_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_CLASS = 1

# =====================================================================
# 1. LOAD DATA
# =====================================================================
df = pd.read_csv(DATA_PATH)
print("Label distribution:")
print(df["label"].value_counts())

numeric_df = df.select_dtypes(include=[np.number])
feature_cols = [c for c in numeric_df.columns if c != "label"]
print(f"\nUsing {len(feature_cols)} numeric feature columns for clustering.")

X = df[feature_cols].copy().replace([np.inf, -np.inf], np.nan).fillna(0)

# =====================================================================
# 2. FILTER TO A SINGLE CLASS (labels not used for fitting — only to filter)
# =====================================================================
class_mask = (df["label"] == TARGET_CLASS).values
X_cluster = X.loc[class_mask].reset_index(drop=True)
urls_cluster = df.loc[class_mask, "url"].reset_index(drop=True)

print(f"\nClustering within label = {TARGET_CLASS}  (n={len(X_cluster)} rows)")

X_cluster_scaled = StandardScaler().fit_transform(X_cluster)

# =====================================================================
# 3. CHOOSE k VIA ELBOW + SILHOUETTE SCORE
# =====================================================================
SIL_SAMPLE_SIZE = 5000  # increase if you want a more precise score, at the cost of speed

k_range = range(2, 9)
inertias, sil_scores = [], []
for k in k_range:
    km = MiniBatchKMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10, batch_size=2048)
    labels_k = km.fit_predict(X_cluster_scaled)
    inertias.append(km.inertia_)
    sil = silhouette_score(
        X_cluster_scaled, labels_k,
        sample_size=min(SIL_SAMPLE_SIZE, len(X_cluster_scaled)),
        random_state=RANDOM_STATE,
    )
    sil_scores.append(sil)
    print(f"  k={k}: inertia={km.inertia_:.1f}, silhouette(sampled)={sil:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(list(k_range), inertias, marker="o")
axes[0].set_title("Elbow method")
axes[0].set_xlabel("k")
axes[0].set_ylabel("Inertia")
axes[1].plot(list(k_range), sil_scores, marker="o", color="orange")
axes[1].set_title("Silhouette score")
axes[1].set_xlabel("k")
axes[1].set_ylabel("Silhouette")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/clustering_k_selection.png", dpi=150)
plt.close()

best_k = list(k_range)[int(np.argmax(sil_scores))]
print(f"Selected k={best_k} by best silhouette score ({max(sil_scores):.3f})")

# =====================================================================
# 4. FIT FINAL KMEANS
# =====================================================================
kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
cluster_assignments = kmeans.fit_predict(X_cluster_scaled)

X_cluster_labeled = X_cluster.copy()
X_cluster_labeled["cluster"] = cluster_assignments
X_cluster_labeled["url"] = urls_cluster

# 2D PCA visualisation
pca = PCA(n_components=2, random_state=RANDOM_STATE)
coords = pca.fit_transform(X_cluster_scaled)
plt.figure(figsize=(7, 6))
sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=cluster_assignments, palette="tab10", s=15)
plt.title(f"KMeans clusters within label={TARGET_CLASS} (PCA projection)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/cluster_pca_scatter.png", dpi=150)
plt.close()

# =====================================================================
# 5. CLUSTER PROFILING — what makes each cluster different?
# =====================================================================
overall_mean = X_cluster[feature_cols].mean()
denom = overall_mean.replace(0, 1e-9).abs()

print("\n" + "=" * 60)
print(f"CLUSTER PROFILES (within label={TARGET_CLASS})")
print("=" * 60)

for c in sorted(X_cluster_labeled["cluster"].unique()):
    sub = X_cluster_labeled[X_cluster_labeled["cluster"] == c]
    cluster_mean = sub[feature_cols].mean()
    rel_diff = (cluster_mean - overall_mean) / denom
    top_features = rel_diff.abs().sort_values(ascending=False).head(3)

    print(f"\n--- Cluster {c}  (n={len(sub)}, {len(sub)/len(X_cluster_labeled):.1%} of class) ---")
    print("Top distinguishing features vs. within-class average:")
    for feat in top_features.index:
        print(
            f"  {feat:28s} cluster={cluster_mean[feat]:8.3f}  "
            f"overall={overall_mean[feat]:8.3f}  rel_diff={rel_diff[feat]:+.1%}"
        )
    print("Example URLs from this cluster:")
    for u in sub["url"].sample(min(5, len(sub)), random_state=RANDOM_STATE):
        print(f"    {u}")

X_cluster_labeled.to_csv(f"{OUT_DIR}/cluster_assignments_label{TARGET_CLASS}.csv", index=False)
print(f"\nDone. Plots + cluster CSV saved to ./{OUT_DIR}/")