import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
data = pd.read_csv(
    os.path.join(os.path.dirname(__file__), "data","student_data.csv")
)
# Input features
X = data[["study_hours", "attendance", "previous_score", "assignment_score"]]

# Target
y = data["final_score"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Model trained successfully!")
print("Mean Absolute Error:", mae)
print("R2 Score:", r2)

# Example prediction
new_student = [[7, 90, 80, 85]]

predicted_score = model.predict(new_student)

print("Predicted Final Score:", predicted_score[0])