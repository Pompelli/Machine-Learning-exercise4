import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

'''max_depth 
controls how deep the tree can grow, thereby regulating the complexity of the model's structure. 
A higher max_depth allows the tree to capture more intricate patterns in the data,
potentially leading to overfitting if the depth is too large.

min_samples_leaf 
influences how many data points are required to form a leaf. 
It helps determine how detailed the tree's predictions can be and whether it tends to learn unnecessary finer structures. 
A higher value for min_samples_leaf forces the tree to generalize better by having more samples per leaf,
which can help prevent overfitting.'''


np.random.seed(42)
# Generate synthetic data: Non-linear relationship (sin data)

X = np.sort(5 * np.random.rand(80, 1), axis=0)  # 80 data points, sorted
y = np.sin(X).ravel() 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit SVR with Polynomial kernel for different degrees
degrees = [2, 3, 4]  # Polynomial degrees to test
plt.figure(figsize=(12, 8))

for degree in degrees:
    svr = SVR(kernel='poly', degree=degree, C=100, epsilon=0.1)
    svr.fit(X_train, y_train)
    
    # Generate predictions for both train and test sets
    y_pred = svr.predict(X_test)
    
    # Plot the polynomial fit
    X_grid = np.linspace(0, 5, 1000).reshape(-1, 1)
    plt.plot(X_grid, svr.predict(X_grid), label=f"Poly degree {degree}")
    
    print(f"Degree {degree} - MSE: {mean_squared_error(y_test, y_pred):.3f}")

plt.scatter(X_train, y_train, color='red', label='Training data')
plt.scatter(X_test, y_test, color='blue', label='Test data')
plt.title("SVM Polynomial Fit for Different Degrees")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

# Fit Decision Tree for Regression with different depths and min_samples_leaf
depths = [3, 5, 7]  # Different depths of the tree
min_samples_leaf_values = [1, 5, 10]  # Different min_samples_leaf values
plt.figure(figsize=(12, 8))

for depth in depths:
    for min_samples_leaf in min_samples_leaf_values:
        tree = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=min_samples_leaf)
        tree.fit(X_train, y_train)
        
        # Predict using the decision tree model
        y_pred = tree.predict(X_test)
        
        # Plot the decision tree fit
        X_grid = np.linspace(0, 5, 1000).reshape(-1, 1)
        plt.plot(X_grid, tree.predict(X_grid), label=f"Depth {depth}, min_samples_leaf {min_samples_leaf}")
        
        print(f"Depth {depth}, min_samples_leaf {min_samples_leaf} - MSE: {mean_squared_error(y_test, y_pred):.3f}")

plt.scatter(X_train, y_train, color='red', label='Training data')
plt.scatter(X_test, y_test, color='blue', label='Test data')
plt.title("Decision Tree Regression for Different Depths and min_samples_leaf")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
