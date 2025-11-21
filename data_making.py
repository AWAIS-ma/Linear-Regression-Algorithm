import pandas as pd
import numpy as np

# Random seed for reproducibility
np.random.seed(42)

# Number of students
n_students = 200

# Generate data
data_frame = pd.DataFrame({
    "student_id": range(1, n_students + 1),
    "study_hours": np.random.randint(1, 10, size=n_students),
    "attendance": np.random.randint(50, 101, size=n_students),  # 50% to 100%
    "previous_score": np.random.randint(40, 91, size=n_students)  # 40 to 90
})

# Current score depends on study_hours, attendance, previous_score + some noise
data_frame["current_score"] = (
    0.4 * data_frame["study_hours"] * 10 + 
    0.3 * data_frame["attendance"] +
    0.3 * data_frame["previous_score"] +
    np.random.normal(0, 5, n_students)  # random noise
).round(0).astype(int)

# Save to CSV
data_frame.to_csv("student_data.csv", index=False)
print("Student dataset saved as 'student_data.csv'")
