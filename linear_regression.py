import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ----- Load Data -----
data_frame = pd.read_csv("student_record.csv")


# Features and target
X = data_frame[["study_hours", "attendance", "previous_score"]]
y = data_frame["current_score"]

# ----- Train Linear Regression Model -----
model = LinearRegression()

model.fit(X, y)

# ----- Display model parameters -----
print("Model coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.2f}")
print(f"Intercept: {model.intercept_:.2f}\n")

# ----- User Prediction -----
print("Enter new student details to predict current score:")
study_hours = float(input("Study hours: "))
attendance = float(input("Attendance %: "))
previous_score = float(input("Previous score: "))

user_df = pd.DataFrame({
    "study_hours": [study_hours],
    "attendance": [attendance],
    "previous_score": [previous_score]
})

predicted_score = model.predict(user_df)[0]
predicted_score = np.clip(predicted_score, 0, 100)
print(f"\nPredicted current score: {predicted_score:.2f}\n")

# ----- Model Evaluation -----
y_pred_train = model.predict(X)
mse = mean_squared_error(y, y_pred_train)
r2 = r2_score(y, y_pred_train)

print(f"Training MSE: {mse:.2f}")
print(f"Training R2 Score: {r2:.2f}\n")


# ----- Visualization with slope line -----
plt.figure(figsize=(8,5))

# Scatter actual data
plt.scatter(data_frame["study_hours"], y, color="blue", label="Actual Scores")

# Scatter predicted point
plt.scatter([study_hours], [predicted_score], color="red", s=100, label="Predicted Score")

# ----- Regression line for study_hours -----
# Create range of study_hours values
study_hours_range = np.linspace(data_frame["study_hours"].min(), data_frame["study_hours"].max(), 100)

# Keep other features at their mean
attendance_mean = data_frame["attendance"].mean()
previous_score_mean = data_frame["previous_score"].mean()

X_line = pd.DataFrame({
    "study_hours": study_hours_range,
    "attendance": [attendance_mean]*100,
    "previous_score": [previous_score_mean]*100
})

y_line = model.predict(X_line)

plt.plot(study_hours_range, y_line, color="green", linewidth=2, label="Slope Line")

plt.xlabel("Study Hours")
plt.ylabel("Current Score")
plt.title("Study Hours vs Current Score with Slope")
plt.legend()
plt.grid(True)
plt.show()
