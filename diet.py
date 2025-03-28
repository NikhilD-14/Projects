import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv("C:\Users\DELL\Downloads\cleaned_diet_recommendations_dataset.csv")

encoder=LabelEncoder()
column=['Gender','Physical_Activity_Level','Disease_Type','Dietary_Restrictions','Diet_Recommendation']
for col in column:
    df[col]=encoder.fit_tranform(df[col])

scale=StandardScaler()
df_scaled=scale.fit_transform(df[['Age','Weight_kg','Height_cm','BMI','']])

knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
knn.fit(df_scaled) 

def recommend_diet(user_input):
    input_scaled = scale.transform([user_input])
    distances, indices = knn.kneighbors(input_scaled)
    return df.iloc[indices[0]]

# Streamlit UI
st.title("Diet Recommendation System")

# User Inputs
age = st.slider("Age", 10, 80, 25)
weight = st.number_input("Weight (kg)", 30, 150, 70)
height = st.number_input("Height (cm)", 100, 220, 170)
preference = st.selectbox("Diet Preference", ["Vegetarian", "Non-Vegetarian"])
goal = st.selectbox("Goal", ["Weight Loss", "Muscle Gain", "Maintain Weight"])

# Convert inputs into feature array
user_features = [weight, height, age]  # Placeholder, adjust based on dataset

if st.button("Get Diet Plan"):
    recommendations = recommend_diet(user_features)
    st.write("### Recommended Diet Plan:")
    st.dataframe(recommendations)
