import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ----- Load dataset -----
df = pd.read_csv("American_Housing_Data.csv")

# ----- Convert to numeric -----
for col in ["Beds", "Baths", "Living Space",
            "Zip Code Population", "Zip Code Density",
            "Median Household Income", "Latitude", "Longitude"]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# ----- Drop rows with missing values -----
df = df.dropna()

# ----- Define X and y (using only 3 features) -----
X = df[["Living Space", "Baths", "Median Household Income"]]  # Selected features
y = df["Price"]

# ----- Train-test split -----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----- Train model -----
model = LinearRegression()
model.fit(X_train, y_train)

# ----- Evaluate -----
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print("---- Training Performance ----")
print("Training R² Score:", r2_score(y_train, y_train_pred))
print("Training MSE:", mean_squared_error(y_train, y_train_pred))

print("\n---- Testing Performance ----")
print("Testing R² Score:", r2_score(y_test, y_test_pred))
print("Testing MSE:", mean_squared_error(y_test, y_test_pred))

# ----- Display model parameters -----
print("\nModel coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.2f}")
print(f"Intercept: {model.intercept_:.2f}")

# ----- Visualization: Living Space vs Price -----
plt.figure(figsize=(8,5))
plt.scatter(df["Living Space"], y, color="blue", label="Actual Prices", alpha=0.6)

# Regression line
living_range = np.linspace(df["Living Space"].min(), df["Living Space"].max(), 100)
bath_mean = df["Baths"].mean()
income_mean = df["Median Household Income"].mean()

X_line = pd.DataFrame({
    "Living Space": living_range,
    "Baths": [bath_mean]*100,
    "Median Household Income": [income_mean]*100
})

y_line = model.predict(X_line)
plt.plot(living_range, y_line, color="red", linewidth=2, label="Regression Line")

plt.xlabel("Living Space (sq ft)")
plt.ylabel("Price")
plt.title("Living Space vs House Price")
plt.legend()
plt.grid(True)
plt.show()

# ----- Optional Visualization: Baths vs Price -----
plt.figure(figsize=(8,5))
plt.scatter(df["Baths"], y, color="green", alpha=0.6)
bath_range = np.linspace(df["Baths"].min(), df["Baths"].max(), 100)
X_line_bath = pd.DataFrame({
    "Living Space": [df["Living Space"].mean()]*100,
    "Baths": bath_range,
    "Median Household Income": [income_mean]*100
})
y_line_bath = model.predict(X_line_bath)
plt.plot(bath_range, y_line_bath, color="red", linewidth=2, label="Regression Line")
plt.xlabel("Baths")
plt.ylabel("Price")
plt.title("Baths vs House Price")
plt.legend()
plt.grid(True)
plt.show()

# ----- Optional Visualization: Median Household Income vs Price -----
plt.figure(figsize=(8,5))
plt.scatter(df["Median Household Income"], y, color="purple", alpha=0.6)
income_range = np.linspace(df["Median Household Income"].min(), df["Median Household Income"].max(), 100)
X_line_income = pd.DataFrame({
    "Living Space": [df["Living Space"].mean()]*100,
    "Baths": [bath_mean]*100,
    "Median Household Income": income_range
})
y_line_income = model.predict(X_line_income)
plt.plot(income_range, y_line_income, color="red", linewidth=2, label="Regression Line")
plt.xlabel("Median Household Income")
plt.ylabel("Price")
plt.title("Median Household Income vs House Price")
plt.legend()
plt.grid(True)
plt.show()
