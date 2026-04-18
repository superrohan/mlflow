import pickle
import pandas as pd
from sklearn.metrics import accuracy_score

# Load model
with open('model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Load dataset
dataset = pd.read_csv('gender_classification_v7.csv')

X_test = dataset.drop('gender', axis=1)
y_test = dataset['gender']

# Predict
y_pred = loaded_model.predict(X_test)

# FIX: map predictions
mapping = {0: 'Female', 1: 'Male'}
y_pred = [mapping[i] for i in y_pred]

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")