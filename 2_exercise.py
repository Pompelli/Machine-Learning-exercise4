import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
from palmerpenguins import load_penguins

# (a) Load and filter the dataset
penguins = load_penguins()

# Filter the dataset to include only relevant columns: species, bill_length_mm, and bill_depth_mm.
penguins_filtered = penguins.dropna(subset=['species', 'bill_length_mm', 'bill_depth_mm'])

# Keep only two species: Chinstrap and Gentoo for binary classification.
penguins_filtered = penguins_filtered[penguins_filtered['species'].isin(['Chinstrap', 'Gentoo'])]

# Select the features for training: bill length and bill depth.
X = penguins_filtered[['bill_length_mm', 'bill_depth_mm']]  # Features
y = penguins_filtered['species']  # Target variable (species)

# Map the target variable (species) to numerical values: 0 for Chinstrap, 1 for Gentoo.
y = y.map({'Chinstrap': 0, 'Gentoo': 1})

# (b) Hard margin classification: Remove outliers if necessary
# Compute the Z-scores for each feature to identify potential outliers.
X_zscore = X.apply(zscore)

# Set a threshold for outlier removal (absolute Z-score > 3 is considered an outlier).
outliers = (np.abs(X_zscore) > 3).any(axis=1)

# Remove the outliers from the dataset for a clean separation of classes.
X_filtered = X[~outliers]
y_filtered = y[~outliers]

# Visualize the data after outlier removal.
plt.figure(figsize=(8, 6))
sns.scatterplot(data=X_filtered, x='bill_length_mm', y='bill_depth_mm', hue=y_filtered, palette='coolwarm')
plt.title("Data after Removing Outliers")
plt.xlabel('Bill Length (mm)')
plt.ylabel('Bill Depth (mm)')
plt.show()

# Train a hard margin SVM using LinearSVC. Set C to a very high value (1e10) to enforce a hard margin.
svm_hard = LinearSVC(C=1e10, random_state=69)
svm_hard.fit(X_filtered, y_filtered)

# Create a grid for visualizing the decision boundary and margin.
xx, yy = np.meshgrid(
    np.linspace(X_filtered['bill_length_mm'].min() - 1, X_filtered['bill_length_mm'].max() + 1, 100),
    np.linspace(X_filtered['bill_depth_mm'].min() - 1, X_filtered['bill_depth_mm'].max() + 1, 100)
)

# Predict the class for each point in the grid.
Z = svm_hard.predict(np.vstack([xx.ravel(), yy.ravel()]).T)

# Plot the decision boundary and margin.
plt.figure(figsize=(8, 6))

# Fill the decision regions.
plt.contourf(xx, yy, Z.reshape(xx.shape), alpha=0.75, cmap=plt.cm.coolwarm)

# Plot the training data points.
plt.scatter(X_filtered['bill_length_mm'], X_filtered['bill_depth_mm'], c=y_filtered, cmap=plt.cm.coolwarm, edgecolors='k', marker='o', s=100, label="Training Data")

# Draw margin lines and the decision boundary.
decision_function = svm_hard.decision_function(np.vstack([xx.ravel(), yy.ravel()]).T)
plt.contour(xx, yy, decision_function.reshape(xx.shape), levels=[-1, 0, 1], linewidths=2, colors='k', linestyles=['--', '-', '--'])

plt.title("Hard Margin SVM - Decision Boundary and Margin")
plt.xlabel('Bill Length (mm)')
plt.ylabel('Bill Depth (mm)')
plt.show()

# (c) Soft Margin SVM: Train with and without StandardScaler
scaler = StandardScaler()

# Scale the features using StandardScaler to normalize the data.
X_scaled = scaler.fit_transform(X_filtered)

# Split the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_filtered, test_size=0.2, random_state=69)

# Train a soft margin SVM (default LinearSVC with moderate C).
svm_soft = LinearSVC(random_state=69)
svm_soft.fit(X_train, y_train)

# Create a grid for visualizing the decision boundary and margin with scaled features.
xx, yy = np.meshgrid(
    np.linspace(X_train[:, 0].min() - 1, X_train[:, 0].max() + 1, 100),
    np.linspace(X_train[:, 1].min() - 1, X_train[:, 1].max() + 1, 100)
)

# Predict the class for each point in the grid.
Z = svm_soft.predict(np.vstack([xx.ravel(), yy.ravel()]).T)

# Plot the decision boundary and margin for scaled features.
plt.figure(figsize=(8, 6))

# Fill the decision regions.
plt.contourf(xx, yy, Z.reshape(xx.shape), alpha=0.75, cmap=plt.cm.coolwarm)

# Plot the scaled training data points.
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.coolwarm, edgecolors='k', marker='o', s=100, label="Training Data")

# Draw margin lines and the decision boundary.
decision_function = svm_soft.decision_function(np.vstack([xx.ravel(), yy.ravel()]).T)
plt.contour(xx, yy, decision_function.reshape(xx.shape), levels=[-1, 0, 1], linewidths=2, colors='k', linestyles=['--', '-', '--'])

plt.title("Soft Margin SVM - Decision Boundary and Margin with StandardScaler")
plt.xlabel('Bill Length (scaled)')
plt.ylabel('Bill Depth (scaled)')
plt.show()

# Train a soft margin SVM without scaling.
svm_soft_no_scaling = LinearSVC(random_state=69)
svm_soft_no_scaling.fit(X_train, y_train)

# Predict the class for each point in the grid without scaling.
Z_no_scaling = svm_soft_no_scaling.predict(np.vstack([xx.ravel(), yy.ravel()]).T)

# Plot the decision boundary and margin without scaling.
plt.figure(figsize=(8, 6))

# Fill the decision regions.
plt.contourf(xx, yy, Z_no_scaling.reshape(xx.shape), alpha=0.75, cmap=plt.cm.coolwarm)

# Plot the training data points without scaling.
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.coolwarm, edgecolors='k', marker='o', s=100, label="Training Data")

# Draw margin lines and the decision boundary.
decision_function_no_scaling = svm_soft_no_scaling.decision_function(np.vstack([xx.ravel(), yy.ravel()]).T)
plt.contour(xx, yy, decision_function_no_scaling.reshape(xx.shape), levels=[-1, 0, 1], linewidths=2, colors='k', linestyles=['--', '-', '--'])

plt.title("Soft Margin SVM - Decision Boundary and Margin without StandardScaler")
plt.xlabel('Bill Length (mm)')
plt.ylabel('Bill Depth (mm)')
plt.show()
