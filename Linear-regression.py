import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Sample dataset
X = np.array([1, 2, 3, 4, 5]).reshape((-1, 1))
y = np.array([2, 4, 5, 4, 5])

# Create and fit the model
model = LinearRegression()
model.fit(X, y)

# Print the coefficients
print("Coefficient of determination (R^2): %.2f" % model.score(X, y))
print("Intercept: %.2f" % model.intercept_)
print("Slope: %.2f" % model.coef_)

# Predict values
y_pred = model.predict(X)

# Plot the data and the regression line
plt.scatter(X, y, label="Data points")
plt.plot(X, y_pred, label="Regression line", color="red")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
