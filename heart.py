# ==========================================
# PCA + DATA PREPROCESSING (HEART DATASET)
# ==========================================

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ==========================================
# 2. LOAD DATASET
# ==========================================
# Download dataset from Kaggle and place CSV in same folder
df = pd.read_csv("heart.csv")

print("Dataset Shape:", df.shape)


# ==========================================
# PART 2(a): PRINT FIRST 20 ROWS
# ==========================================
print("\nFirst 20 rows:\n")
print(df.head(20))


# ==========================================
# PART 2(b): HANDLE MISSING VALUES
# ==========================================
# Check missing values
print("\nMissing Values:\n", df.isnull().sum())

# Replace missing values with mean (for numerical columns)
for col in df.columns:
    if df[col].isnull().sum() > 0:
        mean_value = df[col].mean()
        df[col].fillna(mean_value, inplace=True)
        print(f"Filled missing values in {col} with mean: {mean_value}")


# ==========================================
# PART 2(c): CORRELATION HEATMAP
# ==========================================
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


# ==========================================
# PART 1(b): SCALE THE DATASET
# ==========================================
# Separate features and target
X = df.drop('target', axis=1)  # change 'target' if dataset uses different name
y = df['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nData scaled successfully!")


# ==========================================
# PART 1(c): APPLY PCA
# ==========================================
pca = PCA()
X_pca = pca.fit_transform(X_scaled)


# ==========================================
# PART 1(d): EXPLAINED VARIANCE
# ==========================================
explained_variance = pca.explained_variance_ratio_

print("\nExplained Variance Ratio:\n", explained_variance)


# Plot explained variance
plt.figure(figsize=(8,5))
plt.plot(np.cumsum(explained_variance), marker='o')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA - Explained Variance")
plt.grid()
plt.show()


# ==========================================
# SELECT NUMBER OF COMPONENTS (e.g., 95%)
# ==========================================
n_components = np.argmax(np.cumsum(explained_variance) >= 0.95) + 1
print(f"\nNumber of components to retain 95% variance: {n_components}")


# Apply PCA with selected components
pca_final = PCA(n_components=n_components)
X_reduced = pca_final.fit_transform(X_scaled)

print("\nReduced Dataset Shape:", X_reduced.shape)


# ==========================================
# FINAL DISPLAY RESULTS
# ==========================================
print("\n===== FINAL OUTPUT =====")
print("Original Shape:", X.shape)
print("Reduced Shape after PCA:", X_reduced.shape)
print("Top 5 PCA Transformed Rows:\n", X_reduced[:5])