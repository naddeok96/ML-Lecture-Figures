from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_text

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Create a DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)

# Fit the model
clf.fit(X, y)

# Print the tree structure
tree_rules = export_text(clf, feature_names=iris['feature_names'])
print("Multivariate Decision Tree:")
print(tree_rules)

# Create a DecisionTreeClassifier
clf_univariate = DecisionTreeClassifier(random_state=0)

# Fit the model using only the 'petal length' feature
clf_univariate.fit(X[:, 2].reshape(-1, 1), y)

# Print the tree structure
tree_rules_univariate = export_text(clf_univariate, feature_names=['petal length'])
print("Univariate Decision Tree:")
print(tree_rules_univariate)
