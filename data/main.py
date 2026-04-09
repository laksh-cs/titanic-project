import pandas as pd

# Load dataset
df = pd.read_csv('data/train.csv')

# Show data
print(df.head())
print(df.info())