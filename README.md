# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the dataset into a DataFrame and explore its contents to understand the data structure. 
2.Separate the dataset into independent (X) and dependent (Y) variables, and split them into training and testing sets.
3.Create a linear regression model and fit it using the training data.
4.Predict the results for the testing set and plot the training and testing sets with fitted lines.
5.Calculate error metrics (MSE, MAE, RMSE) to evaluate the model’s performance.
## Program:

Program to implement the simple linear regression model for predicting the marks scored.
# Developed by: Dixun Devotta S
# RegisterNumber:  212224060073

# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored
```
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
```
# 1) Load dataset
```
df = pd.read_csv(r"C:\\Users\\admin\\OneDrive\\Desktop\\ML\\exp_2_dataset_student_scores.csv")   # CSV should have two columns, e.g. "Hours","Scores"
print("First 5 rows:\n", df.head(), "\n")
print("Last 5 rows:\n", df.tail(), "\n")
```
# 2) Prepare input (X) and output (Y)
# Assume CSV columns: Hours (feature) and Scores (target)
```
X = df.iloc[:, :-1].values   # all rows, all columns except last -> shape (n_samples, 1)
Y = df.iloc[:, -1].values    # all rows, last column -> shape (n_samples,)
print("X (features):", X.flatten())
print("Y (targets):", Y)
```

# 3) Split data into training and testing sets
```
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=0)
print("\nTraining samples:", len(X_train), " Testing samples:", len(X_test))
```
# 4) Create and train the model
```
regressor = LinearRegression()
regressor.fit(X_train, Y_train)   # fit on training data
```
# 5) Predict on the test set
```
Y_pred = regressor.predict(X_test)
print("\nPredicted values:", np.round(Y_pred, 2))
print("Actual values   :", Y_test)
```
# 6) Plot training results
```
plt.figure(figsize=(6,4))
plt.scatter(X_train, Y_train, color="orange", label="Training data")
plt.plot(X_train, regressor.predict(X_train), color="red", label="Fitted line")
plt.title("Hours vs Scores (Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.legend()
plt.grid(True)
plt.show()
```
# 7) Plot testing results (use X_test sorted for a nicer line)
```
order = np.argsort(X_test.flatten())
X_test_sorted = X_test.flatten()[order]
Y_test_sorted = Y_test[order]
Y_pred_sorted = Y_pred[order]
plt.figure(figsize=(6,4))
plt.scatter(X_test, Y_test, color="blue", label="Test data")
plt.plot(X_test_sorted, Y_pred_sorted, color="green", label="Predictions")
plt.title("Hours vs Scores (Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.legend()
plt.grid(True)
plt.show()
```
# 8) Evaluation metrics
```
mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
print("\nMean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
```
# 9) Example: predict for new students
```
new_hours = np.array([[2.5], [8.0]])   # shape must be (n_samples, 1)
pred_new = regressor.predict(new_hours)
print("\nPredictions for new hours", new_hours.flatten(), "=>", np.round(pred_new,2))
```
## Output:

# Load dataset
<img width="707" height="387" alt="image" src="https://github.com/user-attachments/assets/cee17a36-ff39-4168-b43f-e26b5095dba9" />

# Prepare input (X) and output (Y)
<img width="897" height="141" alt="image" src="https://github.com/user-attachments/assets/8eb18320-a603-4f5b-9882-938a2e21c8c8" />

# Split data into training and testing sets
<img width="690" height="76" alt="image" src="https://github.com/user-attachments/assets/3a1343a1-32db-49c7-8b4e-7e872df0e1d4" />

# Create and train the model
<img width="562" height="145" alt="image" src="https://github.com/user-attachments/assets/95b82071-0882-4f5f-a2fa-7a5dfbead0bc" />

# Predict on the test set
<img width="781" height="112" alt="image" src="https://github.com/user-attachments/assets/a60bd1f4-541d-42f5-b491-5d1c026c4a68" />

# Plot training results
<img width="766" height="537" alt="image" src="https://github.com/user-attachments/assets/076a5919-694a-4b9b-8598-a1ab36a0a618" />

# Plot testing results (use X_test sorted for a nicer line)
<img width="826" height="522" alt="image" src="https://github.com/user-attachments/assets/6bddadd1-7b13-4c22-b25d-afb102c14a98" />

# Evaluation metrics
<img width="690" height="131" alt="image" src="https://github.com/user-attachments/assets/dbddeb88-f0fc-46c2-9fdf-b8293be84c81" />

# Example: predict for new students
<img width="638" height="81" alt="image" src="https://github.com/user-attachments/assets/fce3045b-e41d-46df-8cf4-5ae893eb886d" />

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
