import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Load the dataset (Pima Indians Diabetes dataset)
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 
           'DiabetesPedigreeFunction', 'Age', 'Outcome']
data = pd.read_csv(url, names=columns)

# Prepare the input and output
X = data.iloc[:, :-1].values  # Features (all except 'Outcome')
Y = data.iloc[:, -1].values   # Target ('Outcome')

# Split the dataset into training and testing sets (80-20 split)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, Y_train)

# Model predictions
Y_pred = lr_model.predict(X_test)

# Evaluation Metrics
print(f'R-squared: {metrics.r2_score(Y_test, Y_pred):.3f}')
print(f'Mean Absolute Error (MAE): {metrics.mean_absolute_error(Y_test, Y_pred):.3f}')
print(f'Mean Squared Error (MSE): {metrics.mean_squared_error(Y_test, Y_pred):.3f}')
print(f'Root Mean Squared Error (RMSE): {np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)):.3f}')

# Optional: Plotting the predictions vs actual
plt.scatter(Y_test, Y_pred, color='blue')
plt.plot([0, 1], [0, 1], color='red')  # Ideal line where predictions = actual values
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.title('Linear Regression: Actual vs Predicted')
plt.show()
