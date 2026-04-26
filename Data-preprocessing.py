import pandas as pd
import numpy as np

# ── 1. Load Dataset ──────────────────────────────────────────────────────────
df = pd.read_csv("heart.csv")

# ── 2. Dataset Description ───────────────────────────────────────────────────
print("=" * 60)
print("DATASET DESCRIPTION")
print("=" * 60)

print(f"\nShape : {df.shape[0]} rows × {df.shape[1]} columns")

print("\n── Column Names & Data Types ──")
print(df.dtypes.to_string())

print("\n── First 5 Rows ──")
print(df.head().to_string())

print("\n── Statistical Summary (Numerical) ──")
print(df.describe().round(2).to_string())

print("\n── Statistical Summary (Categorical) ──")
cat_cols = df.select_dtypes(include=["object", "category"]).columns
if len(cat_cols):
    print(df[cat_cols].describe().to_string())
else:
    print("No categorical columns found.")

# ── 3. Missing Value Analysis ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("MISSING VALUE ANALYSIS")
print("=" * 60)

missing_count  = df.isnull().sum()
missing_pct    = (missing_count / len(df) * 100).round(2)
missing_report = pd.DataFrame({
    "Missing Count"  : missing_count,
    "Missing %"      : missing_pct,
    "Data Type"      : df.dtypes
}).sort_values("Missing Count", ascending=False)

print("\n── Per-Column Missing Values ──")
print(missing_report.to_string())

total_missing = missing_count.sum()
print(f"\nTotal missing cells : {total_missing}")
print(f"Overall missing %   : {(total_missing / df.size * 100):.2f}%")
print(f"Complete rows       : {df.dropna().shape[0]} / {len(df)}")

# ── 4. Handle Missing Values ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("HANDLING MISSING VALUES")
print("=" * 60)

df_clean = df.copy()

num_cols = df_clean.select_dtypes(include=[np.number]).columns
cat_cols = df_clean.select_dtypes(include=["object", "category"]).columns

# Numerical  → median imputation
for col in num_cols:
    if df_clean[col].isnull().sum() > 0:
        median_val = df_clean[col].median()
        df_clean[col].fillna(median_val, inplace=True)
        print(f"  [NUM] '{col}' → filled with median ({median_val:.4f})")

# Categorical → mode imputation
for col in cat_cols:
    if df_clean[col].isnull().sum() > 0:
        mode_val = df_clean[col].mode()[0]
        df_clean[col].fillna(mode_val, inplace=True)
        print(f"  [CAT] '{col}' → filled with mode ('{mode_val}')")

if total_missing == 0:
    print("  No missing values found — dataset is complete.")

# ── 5. Post-cleaning Verification ────────────────────────────────────────────
print("\n── Missing Values After Cleaning ──")
print(df_clean.isnull().sum().to_string())
print(f"\nFinal dataset shape : {df_clean.shape}")

# ── 6. Target Distribution ───────────────────────────────────────────────────
print("\n── Target Column Distribution ──")
if "target" in df_clean.columns:
    dist = df_clean["target"].value_counts()
    pct  = df_clean["target"].value_counts(normalize=True).mul(100).round(2)
    print(pd.DataFrame({"Count": dist, "%": pct}).to_string())
