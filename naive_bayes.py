import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load the diabetes dataset
diabetes = load_diabetes()
df = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)

# Create a binary outcome
df['Outcome'] = (diabetes.target >= 140).astype(int)

# Display the first few rows of the DataFrame
print(df.head())

# Define features and target
X = df.drop('Outcome', axis=1)  # Features
y = df['Outcome']                # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Naive Bayes classifier
model = GaussianNB()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test set
y_preds = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_preds)
print("Accuracy: ", accuracy * 100, "%")

# Perform cross-validation
crv_scores = cross_val_score(model, X, y, cv=5)
print("Cross Validation:", crv_scores * 100, "%")
