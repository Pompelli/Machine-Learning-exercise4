import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from palmerpenguins import load_penguins

# (a) Load the dataset

penguins = load_penguins()



# only use  'species', 'bill_length_mm', 'bill_depth_mm', and 'flipper_length_mm'
penguins_filtered = penguins.dropna(subset=['species', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm'])

# Filter to only include two species: 'Chinstrap' and 'Gentoo'
penguins_filtered = penguins_filtered[penguins_filtered['species'].isin(['Chinstrap', 'Gentoo'])]

############################################################################################################

# (b) Feature selection
# Select the features we want to use to train the model: bill length and bill depth
X = penguins_filtered[['bill_length_mm', 'bill_depth_mm']]  # Features
y = penguins_filtered['species']  

# Convert target variable (species) to (0, 1)
# necessary cause the SVM model requires numerical val
y = y.map({'Chinstrap': 0, 'Gentoo': 1})

############################################################################################################

# (c) Split training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)

############################################################################################################

# (d) Scale  data

# scale data cause SVMs sensitive to the scale of input
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) 
X_test_scaled = scaler.transform(X_test)  

############################################################################################################

# (e) Train SVM 

# linear SVM  model to classify 
svm_soft = LinearSVC(random_state=69)
svm_soft.fit(X_train_scaled, y_train) 

############################################################################################################

#(f) Create a grid for visualizing 
xx, yy = np.meshgrid(
    np.linspace(X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1, 100),
    np.linspace(X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1, 100)
)


Z = svm_soft.predict(np.vstack([xx.ravel(), yy.ravel()]).T)


plt.figure(figsize=(8, 6))

# Plot the decision boundary
plt.contourf(xx, yy, Z.reshape(xx.shape), alpha=0.75, cmap=plt.cm.coolwarm)

# Plot the training data points on top
plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap=plt.cm.coolwarm, edgecolors='k', marker='o', s=100, label="Training Data")

plt.title("Soft Margin SVM - Decision Boundary")
plt.xlabel('Bill Length (scaled)')
plt.ylabel('Bill Depth (scaled)')

plt.show()
