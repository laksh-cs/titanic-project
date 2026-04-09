import pandas as pd

# Load dataset
df = pd.read_csv('data/train.csv')

# Show basic data
print(df.head())

# Show structure
print(df.info())

# Show statistics
print(df.describe())