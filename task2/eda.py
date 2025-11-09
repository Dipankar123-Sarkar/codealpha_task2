# ============================================
# TASK 2: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================

# 📦 Importing Libraries
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, f_oneway, zscore

# ============================================
# STEP 1: LOAD DATASET
# ============================================

def load_dataset(file_path: Path | str | None = None) -> pd.DataFrame:
    """Load dataset from the given path or the same folder as this script.

    Returns a pandas DataFrame. Exits the program with a helpful message if file not found.
    """
    if file_path is None:
        # CSV expected in same folder as this script
        file_path = Path(__file__).parent / "car_sell.csv"
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"ERROR: dataset not found at: {file_path}")
        sys.exit(1)
    return pd.read_csv(file_path)


def safe_print_head(df: pd.DataFrame, n: int = 5) -> None:
    print("🔹 First {} rows of the dataset:".format(n))
    print(df.head(n), "\n")

# ============================================
# STEP 2: BASIC INFORMATION
# ============================================

def basic_info(df: pd.DataFrame) -> None:
    print("🔹 Basic Information about Dataset:")
    # df.info() prints directly
    df.info()
    print()

    print("🔹 Shape of Dataset:", df.shape, "\n")

    print("🔹 Data Types:")
    print(df.dtypes, "\n")

# ============================================
# STEP 3: SUMMARY STATISTICS
# ============================================

def summary_statistics(df: pd.DataFrame) -> None:
    print("🔹 Summary Statistics:")
    print(df.describe(include='all'), "\n")

# ============================================
# STEP 4: CHECK MISSING & DUPLICATE VALUES
# ============================================

def missing_and_duplicates(df: pd.DataFrame) -> None:
    print("🔹 Missing Values in Each Column:")
    print(df.isnull().sum(), "\n")

    print("🔹 Number of Duplicate Rows:", df.duplicated().sum(), "\n")

# ============================================
# STEP 5: UNIVARIATE ANALYSIS
# ============================================

# Histograms for numeric variables
def univariate_analysis(df: pd.DataFrame) -> None:
    # Histograms for numeric variables
    print("📊 Plotting histograms for numeric variables...")
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) > 0:
        df[numeric_cols].hist(bins=20, figsize=(12, 10), color='skyblue', edgecolor='black')
        plt.suptitle("Distribution of Numerical Features")
        plt.tight_layout()
        plt.show()
    else:
        print("No numeric columns available to plot histograms.")

# Countplots for categorical columns (if any)
    # Countplots for categorical columns (if any)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        plt.figure(figsize=(8, 4))
        # If too many categories, show top 20
        counts = df[col].value_counts(dropna=False)
        if counts.shape[0] > 30:
            top = counts.nlargest(20).index
            sns.countplot(x=col, data=df[df[col].isin(top)], order=top, palette="pastel")
            plt.title(f"Count Plot of top 20 {col}")
        else:
            sns.countplot(x=col, data=df, palette="pastel")
            plt.title(f"Count Plot of {col}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# ============================================
# STEP 6: BIVARIATE ANALYSIS
# ============================================

# Correlation heatmap for numeric variables
def bivariate_analysis(df: pd.DataFrame) -> None:
    # Correlation heatmap for numeric variables
    plt.figure(figsize=(8,6))
    corr = df.corr(numeric_only=True)
    if corr.shape[0] > 1:
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.show()
    else:
        print("Not enough numeric columns to compute correlation heatmap.")

# Scatter plot between two numeric features (example)
    # Scatter plot between two numeric features (example)
    if 'Sales' in df.columns and 'Discount' in df.columns:
        sns.scatterplot(x='Discount', y='Sales', data=df)
        plt.title("Discount vs Sales")
        plt.tight_layout()
        plt.show()

# Box plot (categorical vs numeric example)
    # Box plot (categorical vs numeric example)
    if 'Region' in df.columns and 'Profit' in df.columns:
        sns.boxplot(x='Region', y='Profit', data=df, palette="Set2")
        plt.title("Profit Distribution Across Regions")
        plt.tight_layout()
        plt.show()

# ============================================
# STEP 7: HYPOTHESIS TESTING
# ============================================

# Example 1: Correlation Test (Pearson)
def hypothesis_testing(df: pd.DataFrame) -> None:
    # Example 1: Correlation Test (Pearson)
    if 'Discount' in df.columns and 'Sales' in df.columns:
        x = df['Discount'].dropna()
        y = df['Sales'].dropna()
        # align indices
        common_idx = x.index.intersection(y.index)
        x = x.loc[common_idx]
        y = y.loc[common_idx]
        if len(x) >= 2 and x.nunique() > 1 and y.nunique() > 1:
            corr, pval = pearsonr(x, y)
            print("🔹 Correlation between Discount and Sales:", round(corr, 3))
            print("P-value:", round(pval, 5))
            if pval < 0.05:
                print("✅ Significant correlation exists.\n")
            else:
                print("❌ No significant correlation.\n")
        else:
            print("Not enough valid/variable data for Pearson correlation between Discount and Sales.")

# Example 2: ANOVA Test (Profit difference by Region)
    # Example 2: ANOVA Test (Profit difference by Region)
    if 'Region' in df.columns and 'Profit' in df.columns:
        groups = []
        for _, g in df.groupby('Region'):
            vals = g['Profit'].dropna()
            if len(vals) >= 2:
                groups.append(vals)
        if len(groups) > 1:
            try:
                f_stat, p_val = f_oneway(*groups)
                print("🔹 ANOVA Test for Profit by Region:")
                print("F-Statistic:", round(f_stat, 3), "| P-value:", round(p_val, 5))
                if p_val < 0.05:
                    print("✅ Significant difference in Profit across Regions.\n")
                else:
                    print("❌ No significant difference found.\n")
            except Exception as e:
                print("ANOVA could not be performed:", e)
        else:
            print("Not enough groups with sufficient data to perform ANOVA on Profit by Region.")

# ============================================
# STEP 8: DETECT OUTLIERS
# ============================================

# Outlier detection using IQR
def detect_outliers(df: pd.DataFrame) -> None:
    # Outlier detection using IQR
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        if IQR == 0 or np.isnan(IQR):
            print(f"🔹 {col}: IQR is zero or NaN — skipping outlier detection for this column")
            continue
        outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))]
        print(f"🔹 {col}: {outliers.shape[0]} outliers detected")

# Boxplots for numeric features
    # Boxplots for numeric features
    for col in numeric_cols:
        plt.figure(figsize=(6, 3))
        sns.boxplot(x=df[col], color='lightblue')
        plt.title(f"Boxplot of {col}")
        plt.tight_layout()
        plt.show()

# ============================================
# STEP 9: DETECT POTENTIAL DATA ISSUES
# ============================================

# 1. Missing values
def detect_data_issues(df: pd.DataFrame) -> None:
    # 1. Missing values
    missing_data = df.isnull().sum()
    missing_cols = missing_data[missing_data > 0]
    if not missing_cols.empty:
        print("⚠️ Columns with missing values:")
        print(missing_cols)
    else:
        print("✅ No missing values detected.")

# 2. Duplicates
    # 2. Duplicates
    if df.duplicated().sum() > 0:
        print("⚠️ Duplicates detected:", df.duplicated().sum())
    else:
        print("✅ No duplicate records found.")

# 3. Inconsistent data types
    # 3. Inconsistent data types
    print("\n🔹 Data types summary:")
    print(df.dtypes)

# 4. Extreme outliers (using z-score)
    # 4. Extreme outliers (using z-score)
    num_df = df.select_dtypes(include=np.number)
    if not num_df.empty:
        # Fill a bit for the purpose of z-score calculation so we can compute across columns
        filled = num_df.fillna(num_df.mean())
        z_scores = np.abs(zscore(filled))
        outliers_z = (z_scores > 3).sum().sum()
        print(f"🔹 Extreme outliers (z > 3): {outliers_z}")
    else:
        print("No numeric columns to compute z-scores for extreme outliers.")

# ============================================
# STEP 10: SUMMARY OF FINDINGS
# ============================================

def summary_findings(df: pd.DataFrame) -> None:
    missing_cols = df.isnull().sum()[df.isnull().sum() > 0]
    print("\n📋 SUMMARY OF EDA FINDINGS")
    print("""
1. Dataset contains {} rows and {} columns.
2. Numeric & categorical features identified.
3. Missing values detected in {} columns.
4. {} duplicate records found.
5. Correlation and ANOVA tests conducted to identify significant relationships.
6. Outliers detected in several numeric columns.
7. Further steps: handle missing data, remove duplicates, treat outliers, and normalize features.
""".format(df.shape[0], df.shape[1], missing_cols.shape[0], df.duplicated().sum()))


def run_eda(file_path: Path | str | None = None) -> None:
    df = load_dataset(file_path)
    safe_print_head(df)
    basic_info(df)
    summary_statistics(df)
    missing_and_duplicates(df)
    univariate_analysis(df)
    bivariate_analysis(df)
    hypothesis_testing(df)
    detect_outliers(df)
    detect_data_issues(df)
    summary_findings(df)


if __name__ == "__main__":
    # Allow optional custom path via CLI argument
    csv_arg = None
    if len(sys.argv) > 1:
        csv_arg = sys.argv[1]
    run_eda(csv_arg)
