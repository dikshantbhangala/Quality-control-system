import pandas as pd

df = pd.read_csv("defects_data.csv")
if 'Date' in df.columns:  
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Select only numeric columns for filling missing values
numeric_cols = df.select_dtypes(include=['number']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
df.to_csv("cleaned_defects.csv",index=False)
print("✅ Data Cleaning Complete! Cleaned data saved as 'cleaned_defects.csv'")