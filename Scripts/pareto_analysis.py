import os
import pandas as pd
import matplotlib.pyplot as plt

# Ensure 'output/' directory exists
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load dataset
df = pd.read_csv("cleaned_defects.csv")
print("📌 Available columns:", df.columns.tolist())  # Debugging step

# Ensure 'defect_type' exists
if "defect_type" not in df.columns:
    raise KeyError("❌ 'defect_type' column missing! Required for analysis.")

# Count occurrences of each defect type
defect_counts = df["defect_type"].value_counts()

# Cumulative percentage
cumulative_percentage = defect_counts.cumsum() / defect_counts.sum() * 100

# 🔹 Plot Pareto Chart
fig, ax1 = plt.subplots(figsize=(12, 6))

# Bar plot (Defect frequencies)
ax1.bar(defect_counts.index, defect_counts, color="blue", label="Defect Count")
ax1.set_ylabel("Defect Count", fontsize=12)
ax1.set_xlabel("Defect Type", fontsize=12)
ax1.set_xticklabels(defect_counts.index, rotation=45, ha="right")
ax1.grid(axis="y", linestyle="dashed", alpha=0.5)

# Line plot (Cumulative percentage)
ax2 = ax1.twinx()
ax2.plot(defect_counts.index, cumulative_percentage, color="red", marker="o", linestyle="dashed", label="Cumulative %")
ax2.set_ylabel("Cumulative Percentage", fontsize=12)
ax2.axhline(80, color="gray", linestyle="dashed", alpha=0.7, label="80% Threshold")

# 🔹 Legends
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")

plt.title("Pareto Analysis of Defects (80/20 Rule)", fontsize=14)

# Save plot
output_path = os.path.join(output_dir, "pareto_analysis.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"✅ Pareto chart saved at: {output_path}")

# Show plot
plt.show()
