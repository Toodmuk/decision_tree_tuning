#----- Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
# Load the Dataset
# Load dataset
file_path = "launches_report.xlsx"
data = pd.read_excel(file_path, sheet_name="Space Launches")
# Inspect the data
print(data.head())


#----- Data Preprocessing
# Select relevant features and target variable
data = data[["Provider", "Rocket", "Mission", "Launch Pad", "Status"]]

# Drop rows with missing values
data.dropna(inplace=True)

# Convert "Status" to a binary target (e.g., "Launch Successful" vs others)
data["Target"] = data["Status"].apply(lambda x: 1 if "Successful" in x else 0)

# Drop the original "Status" column
data.drop("Status", axis=1, inplace=True)

# Encode categorical features using one-hot encoding
data_encoded = pd.get_dummies(data, columns=["Provider", "Rocket", "Mission", "Launch Pad"], drop_first=True)

# Separate features and target
X = data_encoded.drop("Target", axis=1)
y = data_encoded["Target"]


#----- SPLIT and TRAIN the DATA (Decision Tree)
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Decision Tree Classifier
classifier = DecisionTreeClassifier(random_state=42)
classifier.fit(X_train, y_train)

# Evaluate the model
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Initial Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))


#----- Hyperparameter Tuning
# Define parameter grid
param_grid = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

#----- Perform Grid Search
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best parameters and accuracy
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validated Accuracy:", grid_search.best_score_)

# Retrain model with best parameters
best_classifier = grid_search.best_estimator_
best_classifier.fit(X_train, y_train)


#----- Evaluate the Tuned Model
# Predict with the tuned model
y_pred_tuned = best_classifier.predict(X_test)

# Evaluate performance
accuracy_tuned = accuracy_score(y_test, y_pred_tuned)
print(f"Tuned Model Accuracy: {accuracy_tuned:.2f}")
print("\nClassification Report (Tuned):\n", classification_report(y_test, y_pred_tuned))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_tuned)
print("\nConfusion Matrix:\n", conf_matrix)

# Visualize Confusion Matrix
import seaborn as sns
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
            xticklabels=["Not Successful", "Successful"], 
            yticklabels=["Not Successful", "Successful"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# Visualize the Decision Tree
# Graphical Visualization of the Decision Tree
plt.figure(figsize=(15, 10))
plot_tree(best_classifier, feature_names=X.columns, 
          class_names=["Not Successful", "Successful"], filled=True)
plt.title("Decision Tree Visualization")
plt.show()

# Export Decision Tree as Text Rules
# Display decision tree rules
rules = export_text(best_classifier, feature_names=list(X.columns))
print("\nDecision Tree Rules:\n")
print(rules)

# Save the rules to a file
with open("decision_tree_rules.txt", "w") as f:
    f.write(rules)


# Analyze Feature Importance
# Feature Importance
feature_importances = pd.DataFrame({
    "Feature": X.columns,
    "Importance": best_classifier.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importances:\n")
print(feature_importances)

# Plot feature importances
feature_importances = pd.DataFrame({
    "Feature": X.columns,
    "Importance": best_classifier.feature_importances_
}).sort_values(by="Importance", ascending=False)

# Plot feature importances
plt.figure(figsize=(12, 8))
plt.barh(feature_importances["Feature"], 
         feature_importances["Importance"], color="skyblue")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance Visualization")
plt.gca().invert_yaxis()  # To have the most important features on top
plt.tight_layout()  # Ensures that labels and title fit into the plot
plt.show()

# **Visualizing the Decision Tree:**
# Limit the depth of the tree for better readability
plt.figure(figsize=(15, 10))
plot_tree(best_classifier, 
          feature_names=X.columns, 
          class_names=["Not Successful", "Successful"], 
          filled=True, 
          max_depth=3)  # Limiting the depth for clarity
plt.title("Decision Tree Visualization (Limited Depth)")
plt.show()

# Alternatively, print the tree rules in the terminal:
rules = export_text(best_classifier, feature_names=list(X.columns))
print("\nDecision Tree Rules:\n")
print(rules)

# **Feature Importance Visualization:**
# Improve feature importance visualization with adjusted layout
feature_importances = pd.DataFrame({
    "Feature": X.columns,
    "Importance": best_classifier.feature_importances_
}).sort_values(by="Importance", ascending=False)

# Plot feature importances
plt.figure(figsize=(12, 8))
plt.barh(feature_importances["Feature"], feature_importances["Importance"], color="skyblue")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance Visualization")
plt.gca().invert_yaxis()  # To have the most important features on top
plt.tight_layout()  # Ensures that labels and title fit into the plot
plt.show()
