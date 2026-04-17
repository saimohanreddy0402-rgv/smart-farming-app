import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

# Load data
data = pd.read_csv("crop_data.csv")

# Encode categorical values
le_soil = LabelEncoder()
le_crop = LabelEncoder()

data['soil'] = le_soil.fit_transform(data['soil'])
data['crop'] = le_crop.fit_transform(data['crop'])

# Features & target
X = data[['soil', 'rainfall', 'temperature', 'crop']]
y = data['yield']

# Train model
model = LinearRegression()
model.fit(X, y)

# App UI
st.title("Smart AI Farming Assistant")

st.write("Enter farm details:")

soil = st.selectbox("Soil Type", ["loamy", "sandy", "clay"])
crop = st.selectbox("Crop Type", ["rice", "wheat", "maize"])
rainfall = st.slider("Rainfall (mm)", 100, 1000, 500)
temperature = st.slider("Temperature (°C)", 15, 40, 25)

if st.button("Predict Yield"):
    soil_val = le_soil.transform([soil])[0]
    crop_val = le_crop.transform([crop])[0]

    prediction = model.predict([[soil_val, rainfall, temperature, crop_val]])

    st.success(f" Predicted Yield: {prediction[0]:.2f} tons/hectare")

    # Recommendations
    st.subheader(" Recommendations")

    if rainfall < 500:
        st.write("Increase irrigation")
    else:
        st.write("Irrigation level is good")

    if soil == "sandy":
        st.write("Use organic fertilizers")
    elif soil == "clay":
        st.write("⚠ Improve drainage system")
    else:
        st.write("Soil condition is balanced")