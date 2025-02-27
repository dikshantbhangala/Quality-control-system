import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load cleaned dataset
df = pd.read_csv("cleaned_defects.csv")

# If dataset has a 'date' column
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
    daily_defects = df.groupby('date')['defect_count'].sum()

    # Calculate mean and standard deviation
    mean = daily_defects.mean()
    std_dev = daily_defects.std()

    # Upper & Lower Control Limits (±3 sigma)
    ucl = mean + (3 * std_dev)
    lcl = mean - (3 * std_dev)

    # Plot Control Chart
    plt.figure(figsize=(12,5))
    plt.plot(daily_defects.index, daily_defects.values, marker="o", linestyle="-", label="Defect Count")
    plt.axhline(mean, color="green", linestyle="dashed", label="Mean")
    plt.axhline(ucl, color="red", linestyle="dashed", label="UCL (Upper Control Limit)")
    plt.axhline(lcl, color="blue", linestyle="dashed", label="LCL (Lower Control Limit)")
    
    plt.title("📈 Control Chart for Defect Monitoring")
    plt.xlabel("Date")
    plt.ylabel("Total Defects")
    plt.legend()
    plt.grid()
    plt.show()
