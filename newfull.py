import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, classification_report,confusion_matrix, mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.inspection import permutation_importance
from scipy.stats import chi2_contingency
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 0. LOAD & PREPARE
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv("heart.csv")
df = df.fillna(df.median(numeric_only=True))

X_raw = df.drop(columns=["target"])
y     = df["target"]

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# ─────────────────────────────────────────────────────────────────────────────
# 1. K-MEANS CLUSTERING  +  ELBOW  +  SILHOUETTE
# ─────────────────────────────────────────────────────────────────────────────
inertias, sil_scores = [], []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("K-Means: Elbow & Silhouette", fontsize=14, fontweight="bold")

axes[0].plot(K_range, inertias, "bo-")
axes[0].set_title("Elbow Method")
axes[0].set_xlabel("Number of Clusters (k)")
axes[0].set_ylabel("Inertia (WCSS)")
axes[0].grid(True)

axes[1].plot(K_range, sil_scores, "rs-")
axes[1].set_title("Silhouette Score")
axes[1].set_xlabel("Number of Clusters (k)")
axes[1].set_ylabel("Silhouette Score")
axes[1].grid(True)

plt.tight_layout()
plt.savefig("kmeans_elbow_silhouette.png", dpi=150)
plt.show()

best_k = K_range[np.argmax(sil_scores)]
print(f"Best k by Silhouette : {best_k}  (score = {max(sil_scores):.4f})")

km_best = KMeans(n_clusters=best_k, random_state=42, n_init=10)
cluster_labels = km_best.fit_predict(X_scaled)
df["cluster"] = cluster_labels

# PCA 2-D for cluster visualisation
pca2 = PCA(n_components=2)
X_pca2 = pca2.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca2[:, 0], X_pca2[:, 1],
                      c=cluster_labels, cmap="Set1", alpha=0.7, edgecolors="k", linewidths=0.3)
plt.colorbar(scatter, label="Cluster")
plt.title(f"K-Means Clusters (k={best_k}) — PCA 2-D Projection")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.tight_layout()
plt.savefig("kmeans_clusters.png", dpi=150)
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# 2. DATA VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

# 2a. Target distribution
plt.figure(figsize=(5, 4))
df["target"].value_counts().plot(kind="bar", color=["#2196F3", "#F44336"], edgecolor="k")
plt.title("Target Distribution (0=No Disease, 1=Disease)")
plt.xlabel("Target"); plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("target_distribution.png", dpi=150)
plt.show()

# 2b. Histograms for all features
df.drop(columns=["cluster"]).hist(figsize=(16, 12), bins=20, edgecolor="k", color="#42A5F5")
plt.suptitle("Feature Distributions", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("feature_histograms.png", dpi=150)
plt.show()

# 2c. Boxplots by target
num_cols = X_raw.select_dtypes(include=np.number).columns.tolist()
fig, axes = plt.subplots(3, 5, figsize=(20, 12))
axes = axes.flatten()
for i, col in enumerate(num_cols):
    df.boxplot(column=col, by="target", ax=axes[i], grid=False)
    axes[i].set_title(col, fontsize=9)
    axes[i].set_xlabel("Target")
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])
plt.suptitle("Feature Boxplots by Target", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("boxplots_by_target.png", dpi=150)
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# 3. CORRELATION
# ─────────────────────────────────────────────────────────────────────────────
corr_matrix = df.drop(columns=["cluster"]).corr()

plt.figure(figsize=(12, 9))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, square=True, cbar_kws={"shrink": 0.8})
plt.title("Correlation Heatmap", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=150)
plt.show()

print("\n── Top Correlations with Target ──")
print(corr_matrix["target"].drop("target").sort_values(key=abs, ascending=False).round(4))

# ─────────────────────────────────────────────────────────────────────────────
# 4. OUTLIER DETECTION  (IQR method)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("OUTLIER DETECTION (IQR Method)")
print("=" * 60)

outlier_report = {}
for col in num_cols:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    n_out = ((df[col] < lower) | (df[col] > upper)).sum()
    outlier_report[col] = n_out

outlier_df = pd.Series(outlier_report, name="Outlier Count").sort_values(ascending=False)
print(outlier_df.to_string())

# Boxplot of outliers
plt.figure(figsize=(14, 5))
df[num_cols].boxplot(grid=False, vert=True)
plt.xticks(rotation=45, ha="right")
plt.title("Boxplot — Outlier Overview (IQR)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outliers_boxplot.png", dpi=150)
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# 5. CHI-SQUARE TEST
# ─────────────────────────────────────────────────────────────────────────────
categorical_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
categorical_cols = [c for c in categorical_cols if c in df.columns]

print("\n" + "=" * 60)
print("CHI-SQUARE TEST  (feature vs target)")
print("=" * 60)
print(f"\n{'Feature':<12} {'Chi2':>10} {'p-value':>12} {'Significant?':>14}")
print("-" * 52)

for col in categorical_cols:
    ct = pd.crosstab(df[col], df["target"])
    chi2, p, dof, _ = chi2_contingency(ct)
    sig = "YES ✓" if p < 0.05 else "NO"
    print(f"{col:<12} {chi2:>10.4f} {p:>12.6f} {sig:>14}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. SMOTE  (handle class imbalance)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SMOTE — Synthetic Minority Oversampling")
print("=" * 60)

print(f"Before SMOTE — class distribution:\n{y.value_counts().to_string()}")

smote = SMOTE(random_state=42)
X_sm, y_sm = smote.fit_resample(X_scaled, y)

print(f"\nAfter SMOTE  — class distribution:\n{pd.Series(y_sm).value_counts().to_string()}")
print(f"\nShape before : {X_scaled.shape}")
print(f"Shape after  : {X_sm.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. CLASSIFICATION  (Random Forest on SMOTE data)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("CLASSIFICATION — Random Forest")
print("=" * 60)

X_tr, X_te, y_tr, y_te = train_test_split(X_sm, y_sm, test_size=0.2,
                                            random_state=42, stratify=y_sm)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_tr, y_tr)
y_pred_rf = rf.predict(X_te)

print("\nClassification Report:\n")
print(classification_report(y_te, y_pred_rf, target_names=["No Disease", "Disease"]))

# Confusion matrix
cm = confusion_matrix(y_te, y_pred_rf)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Disease", "Disease"],
            yticklabels=["No Disease", "Disease"])
plt.title("Confusion Matrix — Random Forest")
plt.ylabel("Actual"); plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()

# Feature importance
importances = pd.Series(rf.feature_importances_, index=X_raw.columns).sort_values(ascending=True)
plt.figure(figsize=(8, 6))
importances.plot(kind="barh", color="#42A5F5", edgecolor="k")
plt.title("Feature Importance — Random Forest")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# 8. REGRESSION  (Logistic + Linear)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("LOGISTIC REGRESSION")
print("=" * 60)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_tr, y_tr)
y_pred_lr = lr.predict(X_te)

print("\nClassification Report:\n")
print(classification_report(y_te, y_pred_lr, target_names=["No Disease", "Disease"]))

print("\n" + "=" * 60)
print("LINEAR REGRESSION  (target as continuous)")
print("=" * 60)

X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
lin_reg = LinearRegression()
lin_reg.fit(X_tr_r, y_tr_r)
y_pred_lin = lin_reg.predict(X_te_r)

mse  = mean_squared_error(y_te_r, y_pred_lin)
rmse = np.sqrt(mse)
r2   = r2_score(y_te_r, y_pred_lin)

print(f"\nMSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# Actual vs Predicted
plt.figure(figsize=(7, 5))
plt.scatter(y_te_r, y_pred_lin, alpha=0.6, edgecolors="k", linewidths=0.3, color="#EF5350")
plt.plot([y_te_r.min(), y_te_r.max()], [y_te_r.min(), y_te_r.max()], "k--", lw=1.5)
plt.title("Linear Regression — Actual vs Predicted")
plt.xlabel("Actual"); plt.ylabel("Predicted")
plt.tight_layout()
plt.savefig("linear_regression_actual_vs_pred.png", dpi=150)
plt.show()

print("\n✓ All analyses complete.")