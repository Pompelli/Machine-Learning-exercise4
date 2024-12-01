import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from palmerpenguins import load_penguins

# Load the Palmer Penguins dataset
penguins = load_penguins()

# Remove rows with missing values
penguins = penguins.dropna()

# Select two features and two species
species = ['Adelie', 'Chinstrap']
data = penguins[penguins['species'].isin(species)][['bill_length_mm', 'bill_depth_mm', 'species']]

# Encode the species as numeric values (Adelie = 0, Chinstrap = 1)
data['species'] = data['species'].map({'Adelie': 0, 'Chinstrap': 1})

# Split the data into features and target
X = data[['bill_length_mm', 'bill_depth_mm']]
y = data['species']

# Function to visualize the decision boundaries
def plot_decision_boundary(X, y, model, ax, title="Decision Boundary"):
    # Set the step size for the mesh grid used to visualize the decision boundary
    h = 0.02  # Step size in the mesh (the smaller the value, the finer the resolution)
    
    # Determine the minimum and maximum values for both axes (X and Y) and add some margin
    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
    
    # Create the mesh grid that covers all combinations of X and Y within the defined range
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),   # X values
                         np.arange(y_min, y_max, h))   # Y values
    
    # Make predictions for every point in the mesh grid (for all possible combinations of x and y)
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])  # .ravel() flattens the 2D array into a 1D array
    Z = Z.reshape(xx.shape)  # Reshape Z back to the shape of the mesh grid
    
    # Visualize the decision boundaries as colored regions (according to the model's classifications)
    ax.contourf(xx, yy, Z, alpha=0.8)  # Fill contours (based on the predicted class labels)
    
    # Scatter plot of the actual data points, colored according to their true labels (y)
    scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, edgecolors='k', marker='o', cmap=plt.cm.coolwarm)
    
    # Set the title for the plot
    ax.set_title(title)
    
    return scatter


# (a) Polynomial SVC with pipeline
# Create a model with StandardScaler and PolynomialFeatures
degree = 3
model_poly = make_pipeline(StandardScaler(), 
                           PolynomialFeatures(degree), 
                           SVC(kernel='linear', C=1))

# Train the model
model_poly.fit(X, y)

# Make predictions on the training dataset
y_pred_poly = model_poly.predict(X)

# Calculate the accuracy
accuracy_poly = accuracy_score(y, y_pred_poly)
print(f'Accuracy with Polynomial SVC (Degree {degree}): {accuracy_poly * 100:.2f}%')

# (b) Visualize the decision boundaries for different polynomial degrees
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, degree in enumerate([1, 2, 3]):
    model = make_pipeline(StandardScaler(), 
                          PolynomialFeatures(degree),
                          SVC(kernel='linear', C=1))
    model.fit(X, y)
    scatter = plot_decision_boundary(X, y, model, axes[i], title=f"Polynomial Degree {degree}")

fig.tight_layout()
plt.show()

# (c) Using a Polynomial Kernel (Kernel Trick)
model_poly_kernel = make_pipeline(StandardScaler(), 
                                   SVC(kernel='poly', degree=3, C=1))

model_poly_kernel.fit(X, y)

# Visualize the decision boundary with the Polynomial Kernel
fig, ax = plt.subplots(figsize=(8, 6))
plot_decision_boundary(X, y, model_poly_kernel, ax, title="Polynomial Kernel")
plt.show()

# (d) Using an RBF Kernel (Gaussian Kernel) and optimizing the parameters
param_grid = {
    'svc__C': [0.1, 1, 10],
    'svc__gamma': ['scale', 'auto', 0.1, 1]
}

model_rbf = make_pipeline(StandardScaler(), 
                          SVC(kernel='rbf'))

grid_search = GridSearchCV(model_rbf, param_grid, cv=5)
grid_search.fit(X, y)

# Best parameters from GridSearchCV
best_params = grid_search.best_params_
print(f"Best parameters for RBF Kernel: {best_params}")

# Use the best model for prediction and visualization
best_model = grid_search.best_estimator_

# Visualize the decision boundary with the best RBF Kernel
fig, ax = plt.subplots(figsize=(8, 6))
plot_decision_boundary(X, y, best_model, ax, title="RBF Kernel with Optimized Parameters")
plt.show()

# Check accuracy on the training data
y_pred_rbf = best_model.predict(X)
accuracy_rbf = accuracy_score(y, y_pred_rbf)
print(f'Accuracy with RBF Kernel: {accuracy_rbf * 100:.2f}%')
