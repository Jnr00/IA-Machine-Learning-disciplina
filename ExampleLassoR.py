from sklearn.linear_model import Lasso
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# Load Boston housing dataset
boston = load_boston()
X = boston.data
y = boston.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create a Lasso regression object with alpha=1.0
lasso = Lasso(alpha=1.0)

# Train the model
lasso.fit(X_train, y_train)

# Make predictions on the test set
y_pred = lasso.predict(X_test)
