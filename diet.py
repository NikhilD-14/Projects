import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("C:\\Users\DELL\Downloads\python\diet_recommendation_corrected_calories.csv")

# Preprocessing
scaler = StandardScaler()
df[['Weight (kg)', 'Height (cm)', 'Age']] = scaler.fit_transform(df[['Weight (kg)', 'Height (cm)', 'Age']])

# Train models
X = df[['Weight (kg)', 'Height (cm)', 'Age']]
weight_model = RandomForestRegressor().fit(X, df['Ideal Weight (kg)'])
calorie_model = RandomForestRegressor().fit(X, df['Calories'])

def get_diet_recommendation(weight, height, age, diet_type):
    input_data = scaler.transform([[weight, height, age]])
    ideal_weight = round(weight_model.predict(input_data)[0], 2)
    calorie_intake = round(calorie_model.predict(input_data)[0], 2)
    
    diet_options = {
        "veg": ("✅ Fruits, Vegetables, Nuts, Whole Grains", "❌ Junk Food, Processed Sugar, Refined Grains"),
        "non-veg": ("✅ Lean Meat, Fish, Eggs, Dairy", "❌ Fried Meat, Processed Meats, High-fat Dairy")
    }
    foods_to_eat, foods_to_avoid = diet_options[diet_type]
    
    weight_diff = round(abs(weight - ideal_weight), 2)
    if weight < ideal_weight - 2:
        weight_goal = f"📈 Gain {weight_diff} kg with a high-calorie, protein-rich diet."
    elif weight > ideal_weight + 2:
        weight_goal = f"📉 Lose {weight_diff} kg with a calorie deficit and balanced diet."
    else:
        weight_goal = "✅ Maintain your current weight with a balanced diet and exercise."
    
    return f"<b>Ideal Weight:</b> {ideal_weight} kg\n\n **Calories:** {calorie_intake} kcal\n\n**🥗 Foods to Eat:** {foods_to_eat}\n\n**🚫 Foods to Avoid:** {foods_to_avoid}\n\n{weight_goal}"

# Streamlit UI
st.set_page_config(page_title="Diet Recommendation", page_icon="🥗", layout="wide")

theme_style = """
    <style>
        .stApp { background-color: black ; }
        .stButton > button { background-color: green; color: white; }
        .stNumberInput label, .stSelectbox label { color: lightgreen; font-weight: bold; }
    </style>
"""
st.markdown(theme_style, unsafe_allow_html=True)

st.markdown("<h1 style='color:lightgreen;'>🌱 Diet Recommendation</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='color:pink;'> Get a personalized diet and fitness plan tailored to your needs! 🎯</h2> ",unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    weight = st.number_input("⚖️ Weight (kg)", 30, 200, 70)
    height = st.number_input("📏 Height (cm)", 100, 220, 175)
with col2:
    age = st.number_input("🎂 Age", 10, 100, 25)
    diet_type = st.selectbox("🥩 Diet Type", ["veg", "non-veg"])

st.markdown("---")

if st.button("🔍 Get My Diet Plan"):
    recommendation = get_diet_recommendation(weight, height, age, diet_type)
    st.markdown(f"<div style='color:pink;'>{recommendation}</div>", unsafe_allow_html=True)
