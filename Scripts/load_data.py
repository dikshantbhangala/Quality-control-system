import pandas as pd 
df = pd.read_csv("defects_data.csv")
print("Dataset Preview:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nMissing Values in Dataset:")
print(df.isnull().sum())