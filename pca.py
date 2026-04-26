import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ── 1. Load & Preprocess ─────────────────────────────────────────────────────
df = pd.read_csv("heart.csv")

# Separate features and target
X = df.drop(columns=["target"])
y = df["target"]

# Median imputation for any missing values
X = X.fillna(X.median())

# ── 2. Standardize ───────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── 3. Fit PCA (all components first) ────────────────────────────────────────
pca_full = PCA()
pca_full.fit(X_scaled)

explained_var      = pca_full.explained_variance_ratio_
cumulative_var     = np.cumsum(explained_var)

print("=" * 60)
print("PCA — EXPLAINED VARIANCE PER COMPONENT")
print("=" * 60)
print(f"\n{'PC':<6} {'Explained Var %':<20} {'Cumulative Var %'}")
print("-" * 45)
for i, (ev, cv) in enumerate(zip(explained_var, cumulative_var), 1):
    print(f"PC{i:<4} {ev*100:<20.4f} {cv*100:.4f}")

# ── 4. Choose components that explain ≥ 95% variance ─────────────────────────
n_components = np.argmax(cumulative_var >= 0.95) + 1
print(f"\nComponents needed for 95% variance: {n_components}")

# ── 5. Final PCA Transform ───────────────────────────────────────────────────
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

print(f"\nOriginal shape : {X_scaled.shape}")
print(f"Reduced shape  : {X_pca.shape}")

# ── 6. Loadings (Feature Contributions) ──────────────────────────────────────
print("\n" + "=" * 60)
print("COMPONENT LOADINGS (feature weights per PC)")
print("=" * 60)

loadings = pd.DataFrame(
    pca.components_.T,
    index=X.columns,
    columns=[f"PC{i+1}" for i in range(n_components)]
).round(4)

print(loadings.to_string())

# ── 7. Top contributing feature per PC ───────────────────────────────────────
print("\n── Top Feature per Principal Component ──")
for col in loadings.columns:
    top_feat = loadings[col].abs().idxmax()
    print(f"  {col}: {top_feat}  (loading = {loadings.loc[top_feat, col]:.4f})")

# ── 8. PCA DataFrame preview ─────────────────────────────────────────────────
pca_df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(n_components)])
pca_df["target"] = y.values

print("\n── PCA-Transformed Data (first 5 rows) ──")
print(pca_df.head().round(4).to_string())