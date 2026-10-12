# import numpy as np 
# import pandas as pd 
# import os
# from pandas import read_csv
# column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
# data = read_csv('housing.csv', header=None, delimiter=r"\s+", names=column_names)
# print(data.head(5))

# import numpy as np 
# import pandas as pd 
# import os
# from pandas import read_csv
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score
# import matplotlib.pyplot as plt
# import seaborn as sns

# # 1. Load the dataset using your local file method
# column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
# data = read_csv('housing.csv', header=None, delimiter=r"\s+", names=column_names)

# print("--- First 5 rows of data ---")
# print(data.head(5))

# # 2. Define Features (X) and Target (y)
# X = data.drop('MEDV', axis=1)
# y = data['MEDV']

# print("\n--- First 5 house prices (y) ---")
# print(y.head())

# # 3. Split the data into Training and Testing sets (80% train, 20% test)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 4. Initialize and train the Linear Regression model
# model = LinearRegression()
# model.fit(X_train, y_train)

# # 5. Print Intercept (beta_0) and Coefficients
# print(f"\n--- Model Parameters ---")
# print(f"Intercept (beta_0): {model.intercept_:.4f}")

# coefficients_df = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
# print("\nCoefficients (beta_1 to beta_13):")
# print(coefficients_df)

# # 6. Make predictions and evaluate performance
# y_pred = model.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)
# r2 = r2_score(y_test, y_pred)

# print(f"\n--- Model Performance ---")
# print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
# print(f"R-squared (R2) Score: {r2:.4f}")

# # 7. Generate Scatter Graph (RM vs MEDV with trendline)
# plt.figure(figsize=(10, 6))
# sns.scatterplot(x='RM', y='MEDV', data=data, alpha=0.7, color='teal')
# sns.regplot(x='RM', y='MEDV', data=data, scatter=False, color='darkorange')

# plt.title('Relationship Between Room Count and House Price', fontsize=14, fontweight='bold')
# plt.xlabel('Average Number of Rooms (RM)', fontsize=12)
# plt.ylabel('Median House Price in $1,000s (MEDV)', fontsize=12)
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.show()





import numpy as np 
import pandas as pd 
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the dataset
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data = read_csv('housing.csv', header=None, delimiter=r"\s+", names=column_names)

# 2. Use ONLY ONE feature (RM - Average number of rooms)
X = data[['RM']]
y = data['MEDV']

# Print the value of y (first 5 rows for clarity)
print("--- Target Values (y / MEDV) ---")
print(y.head())
print("\n" + "="*40 + "\n")

# 3. Split the data into Training and Testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Get model parameters
beta_0 = model.intercept_
beta_1 = model.coef_[0]

# Print the Linear Regression Equation
print("--- Linear Regression Equation ---")
print(f"MEDV = {beta_0:.4f} + ({beta_1:.4f} * RM)")
print("\n" + "="*40 + "\n")

# 5. Make predictions
y_pred = model.predict(X_test)

# 6. Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# 7. Print output in your exact requested format
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R² Score: {r2}")

# 8. Generate Scatter Graph with Regression Line
plt.figure(figsize=(10, 6))
sns.scatterplot(x='RM', y='MEDV', data=data, alpha=0.7, color='teal', label='Actual Data')
plt.plot(X_test, y_pred, color='darkorange', linewidth=2, label='Regression Line')

plt.title('Simple Linear Regression: Room Count vs House Price', fontsize=14, fontweight='bold')
plt.xlabel('Average Number of Rooms (RM)', fontsize=12)
plt.ylabel('Median House Price in $1,000s (MEDV)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()