import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(
    page_title="Student Marks EDA",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Student Academic Performance - EDA Dashboard")

st.write(
    """
    This dashboard shows Exploratory Data Analysis (EDA) 
    for the **Student Marks Dataset (marks.csv)**.
    """
)

# -------------------------------
# Load data
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("marks.csv")
    return df

df = load_data()

# -------------------------------
# Show raw data
# -------------------------------
st.subheader("🔍 Raw Data Preview")
st.dataframe(df.head())

# -------------------------------
# Basic info
# -------------------------------
st.subheader("📌 Dataset Information")

col1, col2 = st.columns(2)

with col1:
    st.write("**Shape (rows, columns):**", df.shape)
    st.write("**Columns:**")
    st.write(list(df.columns))

with col2:
    st.write("**Data Types:**")
    st.write(df.dtypes)

st.subheader("📊 Summary Statistics (Numeric Columns)")
st.dataframe(df.describe())

# -------------------------------
# Missing values
# -------------------------------
st.subheader("❓ Missing Values")

missing = df.isnull().sum()
st.write("**Missing values per column:**")
st.write(missing)

fig, ax = plt.subplots(figsize=(8, 3))
sns.heatmap(df.isnull(), cbar=False, ax=ax)
ax.set_title("Missing Values Heatmap")
st.pyplot(fig)

# -------------------------------
# Column type separation
# -------------------------------
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

st.write("**Numeric Columns:**", numeric_cols)
st.write("**Categorical Columns:**", categorical_cols)

# -------------------------------
# Univariate Analysis (Numeric)
# -------------------------------
st.subheader("📈 Univariate Analysis (Numeric Columns)")

if len(numeric_cols) > 0:
    selected_num_col = st.selectbox("Select a numeric column", numeric_cols)

    if selected_num_col:
        c1, c2 = st.columns(2)

        with c1:
            fig, ax = plt.subplots()
            sns.histplot(df[selected_num_col], kde=True, ax=ax)
            ax.set_title(f"Distribution of {selected_num_col}")
            st.pyplot(fig)

        with c2:
            fig, ax = plt.subplots()
            sns.boxplot(x=df[selected_num_col], ax=ax)
            ax.set_title(f"Boxplot of {selected_num_col}")
            st.pyplot(fig)
else:
    st.write("No numeric columns found.")

# -------------------------------
# Univariate Analysis (Categorical)
# -------------------------------
st.subheader("📊 Univariate Analysis (Categorical Columns)")

if len(categorical_cols) > 0:
    selected_cat_col = st.selectbox("Select a categorical column", categorical_cols)

    if selected_cat_col:
        fig, ax = plt.subplots()
        sns.countplot(x=df[selected_cat_col], ax=ax)
        ax.set_title(f"Count Plot of {selected_cat_col}")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        st.pyplot(fig)
else:
    st.write("No categorical columns found.")

# -------------------------------
# Correlation Heatmap
# -------------------------------
st.subheader("🔗 Correlation Between Numeric Features")

if len(numeric_cols) > 1:
    fig, ax = plt.subplots(figsize=(10, 6))
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)
else:
    st.write("Not enough numeric columns for correlation heatmap.")

# -------------------------------
# Bivariate Analysis
# -------------------------------
st.subheader("📉 Bivariate Analysis (Feature vs Target)")

if len(numeric_cols) > 1:
    # Default target: Final Exam Marks if present
    if "Final Exam Marks (out of 100)" in numeric_cols:
        default_target_index = numeric_cols.index("Final Exam Marks (out of 100)")
    else:
        default_target_index = 0

    target_col = st.selectbox("Select target (Y-axis)", numeric_cols, index=default_target_index)

    feature_options = [col for col in numeric_cols if col != target_col]
    feature_col = st.selectbox("Select feature (X-axis)", feature_options)

    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=feature_col, y=target_col, ax=ax)
    ax.set_title(f"{feature_col} vs {target_col}")
    st.pyplot(fig)
else:
    st.write("Need at least 2 numeric columns for bivariate analysis.")

# -------------------------------
# Outlier Detection (IQR)
# -------------------------------
st.subheader("⚠ Outlier Detection (IQR Method)")

if len(numeric_cols) > 0:
    outlier_col = st.selectbox("Select column for outlier check", numeric_cols)

    Q1 = df[outlier_col].quantile(0.25)
    Q3 = df[outlier_col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[outlier_col] < lower) | (df[outlier_col] > upper)][outlier_col]

    st.write(f"Number of outliers in **{outlier_col}**: {len(outliers)}")

    fig, ax = plt.subplots()
    sns.boxplot(x=df[outlier_col], ax=ax)
    ax.axvline(lower, linestyle='--', label='Lower bound')
    ax.axvline(upper, linestyle='--', label='Upper bound')
    ax.set_title(f"Outliers in {outlier_col}")
    ax.legend()
    st.pyplot(fig)
else:
    st.write("No numeric columns available for outlier detection.")

st.success("✅ EDA Dashboard Loaded Successfully!")

