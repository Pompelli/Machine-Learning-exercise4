import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
from palmerpenguins import load_penguins

# (a) Lade und filtere den Datensatz
penguins = load_penguins()

# Nur 'species', 'bill_length_mm', 'bill_depth_mm' und 'flipper_length_mm' verwenden
penguins_filtered = penguins.dropna(subset=['species', 'bill_length_mm', 'bill_depth_mm'])

# Filtere nur zwei Arten: 'Chinstrap' und 'Gentoo'
penguins_filtered = penguins_filtered[penguins_filtered['species'].isin(['Chinstrap', 'Gentoo'])]

# Wähle die Merkmale für das Training: Schnabellänge und Schnabeltiefe
X = penguins_filtered[['bill_length_mm', 'bill_depth_mm']]  # Merkmale
y = penguins_filtered['species']  # Zielvariable

# Konvertiere die Zielvariable (Arten) in numerische Werte (0, 1)
y = y.map({'Chinstrap': 0, 'Gentoo': 1})

# (b) Harte Margin-Klassifikation: Entferne Ausreißer (falls notwendig)
# Berechne Z-Scores für die Merkmale
X_zscore = X.apply(zscore)

# Setze einen Schwellenwert für die Ausreißerentfernung (normalerweise 2 oder 3)
outliers = (np.abs(X_zscore) > 3).any(axis=1)

# Entferne Ausreißer aus dem Datensatz
X_filtered = X[~outliers]
y_filtered = y[~outliers]

# Visualisiere die Daten nach der Ausreißerentfernung
plt.figure(figsize=(8, 6))
sns.scatterplot(data=X_filtered, x='bill_length_mm', y='bill_depth_mm', hue=y_filtered, palette='coolwarm')
plt.title("Daten nach Entfernen von Ausreißern")
plt.xlabel('Schnabellänge (mm)')
plt.ylabel('Schnabeltiefe (mm)')
plt.show()

# (b) Harte Margin SVM: LinearSVC mit C auf einen sehr hohen Wert gesetzt (harte Margin)
svm_hard = LinearSVC(C=1e10, random_state=69)
svm_hard.fit(X_filtered, y_filtered)

# Visualisiere die Entscheidungsgrenze (harte Margin) und die Margen (Street)
xx, yy = np.meshgrid(
    np.linspace(X_filtered['bill_length_mm'].min() - 1, X_filtered['bill_length_mm'].max() + 1, 100),
    np.linspace(X_filtered['bill_depth_mm'].min() - 1, X_filtered['bill_depth_mm'].max() + 1, 100)
)

Z = svm_hard.predict(np.vstack([xx.ravel(), yy.ravel()]).T)

plt.figure(figsize=(8, 6))

# Zeichne die Entscheidungsgrenze
plt.contourf(xx, yy, Z.reshape(xx.shape), alpha=0.75, cmap=plt.cm.coolwarm)

# Zeichne die Trainingsdatenpunkte
plt.scatter(X_filtered['bill_length_mm'], X_filtered['bill_depth_mm'], c=y_filtered, cmap=plt.cm.coolwarm, edgecolors='k', marker='o', s=100, label="Trainingsdaten")

# Zeichne die Margen (die "Street" zwischen den Klassen)
decision_function = svm_hard.decision_function(np.vstack([xx.ravel(), yy.ravel()]).T)
plt.contour(xx, yy, decision_function.reshape(xx.shape), levels=[-1, 0, 1], linewidths=2, colors='k', linestyles=['--', '-', '--'])

plt.title("Harte Margin SVM - Entscheidungsgrenze und Margin (Street)")
plt.xlabel('Schnabellänge (mm)')
plt.ylabel('Schnabeltiefe (mm)')
plt.show()

# (c) Weiche Margin SVM: Trainiere mit StandardScaler und ohne StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_filtered)

# Splitte die Daten
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_filtered, test_size=0.2, random_state=69)

# Trainiere die SVM mit weicher Margin
svm_soft = LinearSVC(random_state=69)
svm_soft.fit(X_train, y_train)

# Visualisiere die Entscheidungsgrenze der weichen Margin SVM und die Margen
xx, yy = np.meshgrid(
    np.linspace(X_train[:, 0].min() - 1, X_train[:, 0].max() + 1, 100),
    np.linspace(X_train[:, 1].min() - 1, X_train[:, 1].max() + 1, 100)
)

Z = svm_soft.predict(np.vstack([xx.ravel(), yy.ravel()]).T)

plt.figure(figsize=(8, 6))

# Zeichne die Entscheidungsgrenze für die weiche Margin SVM
plt.contourf(xx, yy, Z.reshape(xx.shape), alpha=0.75, cmap=plt.cm.coolwarm)

# Zeichne die Trainingsdatenpunkte
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.coolwarm, edgecolors='k', marker='o', s=100, label="Trainingsdaten")

# Zeichne die Margen (die "Street" zwischen den Klassen)
decision_function = svm_soft.decision_function(np.vstack([xx.ravel(), yy.ravel()]).T)
plt.contour(xx, yy, decision_function.reshape(xx.shape), levels=[-1, 0, 1], linewidths=2, colors='k', linestyles=['--', '-', '--'])

plt.title("Weiche Margin SVM - Entscheidungsgrenze und Margin (Street) mit StandardScaler")
plt.xlabel('Schnabellänge (skaliert)')
plt.ylabel('Schnabeltiefe (skaliert)')
plt.show()

# (c) Ohne StandardScaler: Trainiere die weiche Margin SVM ohne Skalierung
svm_soft_no_scaling = LinearSVC(random_state=69)
svm_soft_no_scaling.fit(X_train, y_train)

# Visualisiere die Entscheidungsgrenze ohne Skalierung
Z_no_scaling = svm_soft_no_scaling.predict(np.vstack([xx.ravel(), yy.ravel()]).T)

plt.figure(figsize=(8, 6))

# Zeichne die Entscheidungsgrenze ohne Skalierung
plt.contourf(xx, yy, Z_no_scaling.reshape(xx.shape), alpha=0.75, cmap=plt.cm.coolwarm)

# Zeichne die Trainingsdatenpunkte
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.coolwarm, edgecolors='k', marker='o', s=100, label="Trainingsdaten")

# Zeichne die Margen (die "Street" zwischen den Klassen)
decision_function_no_scaling = svm_soft_no_scaling.decision_function(np.vstack([xx.ravel(), yy.ravel()]).T)
plt.contour(xx, yy, decision_function_no_scaling.reshape(xx.shape), levels=[-1, 0, 1], linewidths=2, colors='k', linestyles=['--', '-', '--'])

plt.title("Weiche Margin SVM - Entscheidungsgrenze und Margin (Street) ohne StandardScaler")
plt.xlabel('Schnabellänge (mm)')
plt.ylabel('Schnabeltiefe (mm)')
plt.show()
