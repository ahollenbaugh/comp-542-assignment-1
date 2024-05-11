import pandas as pd
import numpy as np

# Read in the dataset:
dataset = pd.read_csv("C:\\Users\\aghol\\OneDrive\\Desktop\\COMP 542\\Assignment1\\drug200.csv", sep=',')

# Convert categorical data into numerical data:
from sklearn.preprocessing import LabelEncoder
# Identify which features contain categorical data:
categorical_cols = [col for col, dtype in dataset.dtypes.items() if dtype == 'object']
# Assign an integer to each category:
le = LabelEncoder()
for col in categorical_cols:
    dataset[col] = le.fit_transform(dataset[col])
print(dataset)

# Prepare decision tree classifier:
features = ["Age", "Sex", "BP", "Cholesterol"]
target = "Drug"
# Split data into training and testing sets:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset[features], dataset[target], test_size=0.2, random_state=42)

def evaluate_decision_tree(criterion, max_depth):
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)
    # Train the model:
    clf.fit(X_train, y_train)
    # Make predictions on the test set:
    y_pred = clf.predict(X_test)
    # Calculate accuracy:
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_test, y_pred)
    # print(f"Criterion: {criterion}, Max Depth: {max_depth}, Accuracy: {accuracy:.2f}")
    return accuracy

# Evaluate Decision Tree with different criteria (Gini, entropy, log loss)
criteria = ["gini", "entropy", "log_loss"]
accuracies = {}
for criterion in criteria:
    accuracy = evaluate_decision_tree(criterion, None)  # Default max_depth
    accuracies[criterion + "_default"] = accuracy

# Evaluate Decision Tree with different max_depth values
max_depths = [None, 3, 10]  # Include default, shallow, and deeper tree
for max_depth in max_depths:
    for criterion in criteria:
        accuracy = evaluate_decision_tree(criterion, max_depth)
        accuracies[f"{criterion}_depth_{max_depth}"] = accuracy

# Print and export accuracy results (modify as needed for PDF generation)
print("\nSummary of Accuracies:")
for key, accuracy in accuracies.items():
    print(f"{key}: {accuracy:.2f}")
 