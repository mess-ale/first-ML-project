import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the Boston housing dataset
boston = load_boston()
X = boston.data
y = boston.target
data = pd.DataFrame(X, columns=boston.feature_names)
data["SalePrice"] = y

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop("SalePrice", axis=1), data["SalePrice"], test_size=0.2, random_state=42)

# Train the linear regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Make predictions
predictions = lr.predict(X_test)

# Calculate the root mean squared error
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(rmse)
