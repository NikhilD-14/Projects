import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from collections import Counter

# Load dataset
data = pd.read_csv(r"C:\Users\DELL\OneDrive\Documents\OneDrive\Desktop\streamlit\diabetes_prediction_dataset.csv")

# Example: Encoding the 'Gender' column
label_encoder = LabelEncoder()
data['gender'] = label_encoder.fit_transform(data['gender'])  # Male -> 1, Female -> 0
data['smoking_history'] = label_encoder.fit_transform(data['smoking_history'])

# Features and target
X = data.drop(columns="diabetes")  # Assuming 'Outcome' is the target column
y = data["diabetes"]



# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model with Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8)
model.fit(X_train_scaled, y_train)

# Evaluate model
y_pred = model.predict(X_test_scaled)
predictions_count = Counter(y_pred)
majority_class = predictions_count.most_common(1)[0][0]

# Map the majority class to 'Diabetes: Yes' or 'Diabetes: No'
prediction_label = "Diabetes: Yes" if majority_class == 1 else "Diabetes: No"

# Print the majority prediction
print(f"Majority Prediction: {prediction_label}")
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

