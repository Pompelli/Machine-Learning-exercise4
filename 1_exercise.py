# Import necessary libraries
import pandas as pd
from palmerpenguins import load_penguins
import seaborn as sns
import matplotlib.pyplot as plt

# (a) Load the data
penguins = load_penguins()

# Display the first few rows
print("First rows of the data:")
print(penguins.head())

# (b) Check and clean missing values
print("\nNumber of missing values per column:")
print(penguins.isnull().sum())

# Remove rows with missing values
penguins_cleaned = penguins.dropna()

print("\nData after cleaning (no missing values):")
print(penguins_cleaned.isnull().sum())

# (c) Display available features
print("\nAvailable features:")
print(penguins_cleaned.columns.tolist())

# (d) Visualization: Scatterplot - Bill Length vs Bill Depth
plt.figure(figsize=(8, 6))
sns.scatterplot(data=penguins_cleaned, x='bill_length_mm', y='bill_depth_mm', hue='species')
plt.title("Bill Length vs. Bill Depth, colored by species")
plt.xlabel("Bill Length (mm)")
plt.ylabel("Bill Depth (mm)")
plt.legend(title="Species")
plt.show()

# Pairplot for all features
sns.pairplot(penguins_cleaned, hue='species', diag_kind='kde')
plt.suptitle('Feature combinations by species', y=1.02)
plt.show()

# (e) Best features for classification
print("\nRecommended features for classification: 'flipper_length_mm' and 'bill_depth_mm'")
