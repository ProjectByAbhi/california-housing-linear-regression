import joblib
import numpy as np

# Load model
model = joblib.load("linear_model.pkl")

# Example input (you can change values)
sample_input = np.array([[8.3, 41, 6.9, 1.02, 322, 2.5, 37.88, -122.23]])

prediction = model.predict(sample_input)

print("Predicted House Value:", prediction[0])