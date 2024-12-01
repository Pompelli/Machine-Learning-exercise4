import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from palmerpenguins import load_penguins

penguins = load_penguins()

penguins_filtered = penguins.dropna(subset=['species', 'bill_length_mm', 'bill_depth_mm'])
penguins_filtered = penguins_filtered[penguins_filtered['species'].isin(['Adelie', 'Chinstrap', 'Gentoo'])]

X = penguins_filtered[['bill_length_mm', 'bill_depth_mm']]
y = penguins_filtered['species']


y = y.map({'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2})

# (b) Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=187)

# (c) Train the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=69)
clf.fit(X_train, y_train)

# (d) Visualize the Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(clf, filled=True, feature_names=['Bill Length', 'Bill Depth'], class_names=['Adelie', 'Chinstrap', 'Gentoo'], rounded=True, fontsize=10)
plt.title("Decision Tree Visualization")
plt.show()

# (e) Visualize the decision boundaries

x_min, x_max = X_train['bill_length_mm'].min() - 1, X_train['bill_length_mm'].max() + 1
y_min, y_max = X_train['bill_depth_mm'].min() - 1, X_train['bill_depth_mm'].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

# Predict the class for each point in the grid
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.75, cmap=plt.cm.coolwarm)

# Plot the training points
plt.scatter(X_train['bill_length_mm'], X_train['bill_depth_mm'], c=y_train, cmap=plt.cm.coolwarm, edgecolors='k', marker='o', s=100, label="Training Data")
plt.title("Decision Tree - Decision Boundary")
plt.xlabel('Bill Length (mm)')
plt.ylabel('Bill Depth (mm)')
plt.show()

'''
(f) Rotating the Features:

We swap the two features (bill_length_mm and bill_depth_mm) 
and retrain the decision tree model. We also visualize the decision
 tree and boundaries again after this rotation to see how it affects the decision process.

'''

# (f) Rotate the features and re-train the model
X_rotated = X[['bill_depth_mm', 'bill_length_mm']]  # Swap the columns

# Train a new decision tree with rotated features
X_train_rot, X_test_rot, y_train_rot, y_test_rot = train_test_split(X_rotated, y, test_size=0.3, random_state=42)
clf_rot = DecisionTreeClassifier(random_state=42)
clf_rot.fit(X_train_rot, y_train_rot)

# Visualize the decision tree for the rotated features
plt.figure(figsize=(12, 8))
plot_tree(clf_rot, filled=True, feature_names=['Bill Depth', 'Bill Length'], class_names=['Adelie', 'Chinstrap', 'Gentoo'], rounded=True, fontsize=10)
plt.title("Rotated Decision Tree Visualization")
plt.show()

# Visualize the decision boundaries with rotated features
xx, yy = np.meshgrid(np.linspace(X_train_rot['bill_depth_mm'].min() - 1, X_train_rot['bill_depth_mm'].max() + 1, 100),
                     np.linspace(X_train_rot['bill_length_mm'].min() - 1, X_train_rot['bill_length_mm'].max() + 1, 100))

Z_rot = clf_rot.predict(np.c_[xx.ravel(), yy.ravel()])
Z_rot = Z_rot.reshape(xx.shape)

# Plot the decision boundary with rotated features
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z_rot, alpha=0.75, cmap=plt.cm.coolwarm)
plt.scatter(X_train_rot['bill_depth_mm'], X_train_rot['bill_length_mm'], c=y_train_rot, cmap=plt.cm.coolwarm, edgecolors='k', marker='o', s=100, label="Training Data")
plt.title("Decision Tree - Rotated Features Decision Boundary")
plt.xlabel('Bill Depth (mm)')
plt.ylabel('Bill Length (mm)')
plt.show()


