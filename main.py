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

# ========================
# EDA (Exploratory Data Analysis)
# ========================

print("\nSurvival Count:")
print(df['Survived'].value_counts())

print("\nSurvival by Gender:")
print(df.groupby('Sex')['Survived'].mean())

print("\nAverage Age:")
print(df['Age'].mean())

# ========================
# Feature Engineering
# ========================

# Create new feature: Family Size
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# Drop unnecessary columns
df.drop(['Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)

# Check data
print("\nAfter Feature Engineering:")
print(df.head())