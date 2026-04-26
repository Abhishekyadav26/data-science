# ==========================================
# COMPLETE MACHINE LEARNING LAB EXPERIMENT
# ==========================================

# 1. IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression

from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score

from scipy.stats import chi2_contingency

from imblearn.over_sampling import SMOTE


# ==========================================
# 2. LOAD DATASET
# ==========================================
df = pd.read_csv("heart.csv")   # change filename if needed

print("Dataset Shape:", df.shape)
print("\nFirst 20 rows:\n", df.head(20))


# ==========================================
# 3. DATA PREPROCESSING
# ==========================================

# Missing values
print("\nMissing Values:\n", df.isnull().sum())

# Fill missing values with mean
for col in df.columns:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mean(), inplace=True)

print("\nMissing values handled.")


# ==========================================
# 4. CORRELATION
# ==========================================
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


# ==========================================
# 5. OUTLIER DETECTION (BOXPLOT)
# ==========================================
plt.figure(figsize=(12,6))
sns.boxplot(data=df)
plt.xticks(rotation=90)
plt.title("Outlier Detection")
plt.show()


# ==========================================
# 6. FEATURE SCALING
# ==========================================
X = df.drop('target', axis=1)   # change if needed
y = df['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ==========================================
# 7. PCA
# ==========================================
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

explained_variance = pca.explained_variance_ratio_

plt.plot(np.cumsum(explained_variance), marker='o')
plt.title("PCA Explained Variance")
plt.xlabel("Components")
plt.ylabel("Cumulative Variance")
plt.grid()
plt.show()

n_components = np.argmax(np.cumsum(explained_variance) >= 0.95) + 1
print("Number of components (95% variance):", n_components)

pca_final = PCA(n_components=n_components)
X_reduced = pca_final.fit_transform(X_scaled)


# ==========================================
# 8. K-MEANS CLUSTERING
# ==========================================
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Elbow Method
plt.plot(range(1,11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Clusters")
plt.ylabel("WCSS")
plt.show()

# Silhouette Score
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    print(f"Clusters={i}, Silhouette Score={score:.4f}")


# ==========================================
# 9. DATA VISUALIZATION
# ==========================================
sns.histplot(df['age'], kde=True)
plt.title("Age Distribution")
plt.show()

sns.scatterplot(x=df.iloc[:,0], y=df.iloc[:,1], hue=y)
plt.title("Sample Scatter Plot")
plt.show()


# ==========================================
# 10. CLASSIFICATION (Decision Tree)
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("\nClassification Results:")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# ==========================================
# 11. REGRESSION (Linear Regression)
# ==========================================
reg = LinearRegression()
reg.fit(X_train, y_train)

y_pred_reg = reg.predict(X_test)

print("\nRegression Results:")
print("MSE:", mean_squared_error(y_test, y_pred_reg))
print("R2 Score:", r2_score(y_test, y_pred_reg))


# ==========================================
# 12. CHI-SQUARE TEST
# ==========================================
# Example using two categorical columns (adjust if needed)
contingency_table = pd.crosstab(df['sex'], df['target'])

chi2, p, dof, expected = chi2_contingency(contingency_table)

print("\nChi-Square Test:")
print("Chi2:", chi2)
print("p-value:", p)


# ==========================================
# 13. SMOTE (LAST EXPERIMENT)
# ==========================================
smote = SMOTE(random_state=42)
X_sm, y_sm = smote.fit_resample(X, y)

print("\nBefore SMOTE:\n", y.value_counts())
print("\nAfter SMOTE:\n", pd.Series(y_sm).value_counts())

# Train again after SMOTE
X_train_sm, X_test_sm, y_train_sm, y_test_sm = train_test_split(X_sm, y_sm, test_size=0.2, random_state=42)

clf_sm = DecisionTreeClassifier()
clf_sm.fit(X_train_sm, y_train_sm)

y_pred_sm = clf_sm.predict(X_test_sm)

print("\nAfter SMOTE Classification Report:")
print(confusion_matrix(y_test_sm, y_pred_sm))
print(classification_report(y_test_sm, y_pred_sm))


# ==========================================
# END OF EXPERIMENT
# ==========================================