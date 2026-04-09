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

# ========================
# Model Training
# ========================

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Define features and target
X = df.drop('Survived', axis=1)
y = df['Survived']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ========================
# Model Evaluation
# ========================

from sklearn.metrics import accuracy_score, confusion_matrix

y_pred = model.predict(X_test)

print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))