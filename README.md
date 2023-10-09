# PRODIGY_DS_02
This repository contains a data cleaning and exploratory data analysis (EDA) of the Titanic dataset from Kaggle. The EDA explores the relationships between variables and identifies patterns and trends in the data. The results of the EDA can be used to inform machine learning models and other downstream tasks.


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Titanic dataset
df = pd.read_csv(r'C:datasetrain.csv',  delimiter=',')t

# Display the first few rows of the dataset
print(df.head())

# Step 2: Explore the dataset
print(df.head())
print(df.isnull().sum())
print(df.info())

# Step 3: Data Cleaning
# Check if 'Cabin' column exists before attempting to drop it
if 'Cabin' in df.columns:
    df.drop('Cabin', axis=1, inplace=True)

# Handle missing values in 'Age'
df['Age'].fillna(df['Age'].median(), inplace=True)

# Step 4: Exploratory Data Analysis (EDA)
plt.figure(figsize=(14, 8))

plt.subplot(2, 2, 1)
sns.countplot(x='Survived', data=df)
plt.title('Survival Distribution')

plt.subplot(2, 2, 2)
sns.countplot(x='Survived', hue='Sex', data=df)
plt.title('Survival Distribution by Gender')

plt.subplot(2, 2, 3)
sns.countplot(x='Survived', hue='Pclass', data=df)
plt.title('Survival Distribution by Class')

plt.subplot(2, 2, 4)
sns.histplot(df['Age'], kde=True)
plt.title('Age Distribution')

plt.tight_layout()
plt.show()
