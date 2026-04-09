import pandas as pd

# ========================
# Data Gathering
# ========================
df = pd.read_csv('data/train.csv')

print(df.head())
print(df.info())
print(df.describe())

# ========================
# Data Preprocessing
# ========================
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

df.drop('Cabin', axis=1, inplace=True)

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

print(df.info())