import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from palmerpenguins import load_penguins


penguins = load_penguins()

penguins_filtered = penguins.dropna(subset=['species', 'bill_length_mm', 'bill_depth_mm'])
penguins_filtered = penguins_filtered[penguins_filtered['species'].isin(['Chinstrap', 'Gentoo'])]

X = penguins_filtered[['bill_length_mm', 'bill_depth_mm']]
y = penguins_filtered['species'].map({'Chinstrap': 0, 'Gentoo': 1})  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)

# (a) Create a pipeline for polynomial SVC with degree 3

degree = 3  
poly_svc_pipeline = make_pipeline(
    StandardScaler(),  # Scale the features to have zero mean and unit variance
    PolynomialFeatures(degree),  # Transform the features to polynomial features of the given degree
    SVC(kernel='poly', degree=degree, random_state=69)  # Train SVC with a polynomial kernel
)

# Train model
poly_svc_pipeline.fit(X_train, y_train)

# Evaluate the model on the test data
accuracy = poly_svc_pipeline.score(X_test, y_test)
print(f"Polynomial SVC accuracy: {accuracy:.2f}")

# (b) Define a function to visualize the decision boundary
def plot_decision_boundary(X, y, model, degree, ax=None):
    # Create a mesh grid for plotting the decision boundary
    h = 0.02  # Step size in the mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Predict the class labels for each point in the grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)  # Reshape the result to match the grid
    
    if ax is None:
        ax = plt.gca()  # If no axis is provided, use the current axis
    ax.contourf(xx, yy, Z, alpha=0.75, cmap=plt.cm.coolwarm)  # Plot the decision boundary
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k', marker='o', s=100)  # Plot the data points
    ax.set_title(f"Polynomial SVC (Degree={degree})")
    ax.set_xlabel('Bill Length (scaled)')
    ax.set_ylabel('Bill Depth (scaled)')

# (b) Visualize the decision boundary for different degrees of polynomial
fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # Create subplots for different degrees

# Train and plot for polynomial degrees 2, 3, and 5
for i, degree in enumerate([2, 3, 5]):
    model = make_pipeline(StandardScaler(), PolynomialFeatures(degree), SVC(kernel='poly', degree=degree, random_state=42))
    model.fit(X_train, y_train)
    plot_decision_boundary(X_train.to_numpy(), y_train.to_numpy(), model, degree, axes[i])  # Plot the decision boundary for each degree

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()  # Display the plot

# (c) Use the kernel trick with a polynomial kernel for SVC

# Create an SVC model with a polynomial kernel of degree 3
svm_poly_kernel = SVC(kernel='poly', degree=3, random_state=69)
svm_poly_kernel.fit(X_train, y_train)

# Evaluate the polynomial kernel model on the test data
accuracy_kernel = svm_poly_kernel.score(X_test, y_test)
print(f"Polynomial kernel SVC accuracy (using kernel trick): {accuracy_kernel:.2f}")
#IDK WHY I GET 1 (100% here)!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# (d) Try using the Gaussian RBF kernel and perform GridSearchCV to find the best parameters
param_grid = {
    'C': [0.1, 1, 10],  # Regularization parameter
    'gamma': ['scale', 'auto', 0.1, 1, 10]  # Kernel coefficient
}

# Create an SVC model with an RBF kernel
svm_rbf = SVC(kernel='rbf', random_state=42)

# Perform GridSearchCV with cross-validation to find the best parameters for the RBF kernel
grid_search = GridSearchCV(svm_rbf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Print the best parameters found by GridSearchCV
print(f"Best parameters for RBF kernel: {grid_search.best_params_}")

# Evaluate the best RBF model on the test data
best_model_rbf = grid_search.best_estimator_
accuracy_rbf = best_model_rbf.score(X_test, y_test)
print(f"RBF kernel SVC accuracy: {accuracy_rbf:.2f}")
