import pandas as pd

# Load dataset
df = pd.read_csv('data/train.csv')

# Show basic data
print(df.head())

# Show structure
print(df.info())

# Show statistics
print(df.describe())

# ========================
# Data Preprocessing (Improved)
# ========================

# Use MEAN instead of median (improvement)
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

df.drop('Cabin', axis=1, inplace=True)

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)