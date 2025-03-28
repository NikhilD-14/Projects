import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Load dataset
df = pd.read_csv("C:\\Users\DELL\Downloads\python\diet_recommendation_corrected_calories.csv")

# Encode categorical variables
label_enc = LabelEncoder()
df['Gender'] = label_enc.fit_transform(df['Gender'])
df['Diet Goal'] = label_enc.fit_transform(df['Diet Goal'])
df['Exercise Type'] = label_enc.fit_transform(df['Exercise Type'])

# Features and targets
X = df[['Weight (kg)', 'Height (cm)', 'Age', 'Gender', 'Diet Goal']]
y_weight = df['Ideal Weight (kg)']
y_calories = df['Calories']
y_exercise = df['Exercise Type']

# Train-test split
X_train, X_test, y_weight_train, y_weight_test = train_test_split(X, y_weight, test_size=0.2, random_state=42)
X_train, X_test, y_calories_train, y_calories_test = train_test_split(X, y_calories, test_size=0.2, random_state=42)
X_train, X_test, y_exercise_train, y_exercise_test = train_test_split(X, y_exercise, test_size=0.2, random_state=42)

# Train models
weight_model = RandomForestRegressor()
calorie_model = RandomForestRegressor()
exercise_model = RandomForestClassifier()

weight_model.fit(X_train, y_weight_train)
calorie_model.fit(X_train, y_calories_train)
exercise_model.fit(X_train, y_exercise_train)

def get_diet_recommendation(weight, height, age, gender, diet_goal, diet_type):
    gender_encoded = label_enc.transform([gender.lower().capitalize()])[0]
    diet_goal_encoded = label_enc.transform([diet_goal.lower().capitalize()])[0]
    
    input_data = [[weight, height, age, gender_encoded, diet_goal_encoded]]
    ideal_weight = weight_model.predict(input_data)[0]
    calorie_intake = calorie_model.predict(input_data)[0]
    exercise = exercise_model.predict(input_data)[0]
    
    exercise_name = df[df['Exercise Type'] == exercise]['Exercise Type'].values[0]
    
    veg_foods = {
        "eat": ["Fruits", "Vegetables", "Nuts", "Legumes", "Whole Grains"],
        "avoid": ["Processed Foods", "Sugary Drinks", "Refined Grains"]
    }
    
    non_veg_foods = {
        "eat": ["Lean Meat", "Fish", "Eggs", "Dairy", "Legumes"],
        "avoid": ["Fried Meat", "Processed Meats", "High-fat Dairy"]
    }
    
    diet_recommendation = {
        "Ideal Weight (kg)": round(ideal_weight, 2),
        "Calories to Consume": round(calorie_intake, 2),
        "Recommended Exercise": exercise_name,
        "Foods to Eat": veg_foods['eat'] if diet_type.lower() == "veg" else non_veg_foods['eat'],
        "Foods to Avoid": veg_foods['avoid'] if diet_type.lower() == "veg" else non_veg_foods['avoid']
    }
    
    return diet_recommendation

# Example Usage
user_input = {
    "weight": 70,
    "height": 175,
    "age": 25,
    "gender": "male",
    "diet_goal": "weight loss",
    "diet_type": "veg"
}

recommendation = get_diet_recommendation(**user_input)
print(recommendation)
