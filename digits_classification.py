# Digits Classification using K-Nearest Neighbors (KNN)

# Import necessary libraries
from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns # type: ignore
import matplotlib.pyplot as plt

# Load the digits dataset
digits = datasets.load_digits()

# Create a DataFrame with feature names
data = pd.DataFrame(digits.data, columns=digits.feature_names)

# Adding the target column
data['target'] = digits.target

# Display the first few rows of the dataset
print(data.head())

# Splitting the dataset into features (X) and target (y)
X = data.drop('target', axis=1).values
y = data['target'].values

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the K-Nearest Neighbors model (with k=3)
model = KNeighborsClassifier(n_neighbors=3)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model using confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix using heatmap
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Display classification report
print(classification_report(y_test, y_pred))
