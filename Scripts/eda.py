import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure 'output/' directory exists
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the cleaned dataset
df = pd.read_csv("cleaned_defects.csv")

# Set Seaborn style
sns.set_style("whitegrid")

# 🔹 1. Plot Most Frequent Defects in Manufacturing
plt.figure(figsize=(10, 5))
sns.countplot(y=df["defect_type"], order=df["defect_type"].value_counts().index, hue=df["defect_type"], palette="viridis", legend=False)
plt.title("Most Frequent Defects in Manufacturing")
plt.xlabel("Count")
plt.ylabel("Defect Type")
plt.savefig(os.path.join(output_dir, "defect_distribution.png"), dpi=300, bbox_inches="tight")  # Save plot
plt.show()

# 🔹 2. Plot Daily Defects Trend (if 'date' column exists)
if 'date' in df.columns and 'defect_count' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
    daily_defects = df.groupby('date')['defect_count'].sum()

    plt.figure(figsize=(12, 5))
    plt.plot(daily_defects.index, daily_defects.values, marker="o", linestyle="-", color="red")
    plt.title("Daily Defects Trend")
    plt.xlabel("Date")
    plt.ylabel("Total Defects")
    plt.grid()
    plt.savefig(os.path.join(output_dir, "daily_defects_trend.png"), dpi=300, bbox_inches="tight")  # Save plot
    plt.show()

# 🔹 3. Correlation Heatmap (only for numeric columns)
numeric_cols = df.select_dtypes(include=["number"])

if not numeric_cols.empty:  # Check if numeric columns exist
    plt.figure(figsize=(8, 6))
    sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("📉 Correlation Between Variables")
    plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"), dpi=300, bbox_inches="tight")  # Save plot
    plt.show()
else:
    print("No numeric columns available for correlation heatmap.")
