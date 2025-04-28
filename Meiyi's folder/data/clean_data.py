import numpy as np
import pandas as pd

# Load dataset
df = pd.read_csv('/Users/meiyi/Desktop/documents/GT/Spring_2025/Project/archive/Food Ingredients and Recipe Dataset with Image Name Mapping.csv')

# Verify columns
print("Available columns:", df.columns)

# Select ingredient column
ingredient_col = 'Cleaned_Ingredients' if 'Cleaned_Ingredients' in df.columns else 'Ingredients'
df[ingredient_col] = df[ingredient_col].fillna("")

# Data Analysis
print("Dataset Info:")
df.info()
print("\nMissing Values:")
print(df.isnull().sum())
# Drop rows with any missing values
df = df.dropna()
print(df.isnull().sum())
print("\nTop 10 most common ingredients:")
ingredient_counts = df[ingredient_col].str.split(', ').explode().value_counts().head(10)
print(ingredient_counts)