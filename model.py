import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
data = pd.read_csv("house_data.csv")

# Input features
X = data[["SquareFeet", "Bedrooms", "Bathrooms"]]

# Output
y = data["Price"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("Model trained successfully!")