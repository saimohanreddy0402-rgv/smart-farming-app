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

# Irrigation
if rainfall < 400:
    st.warning(" Increase irrigation frequency to improve yield")
elif rainfall > 800:
    st.warning(" Reduce irrigation and improve drainage to avoid waterlogging")
else:
    st.info(" Maintain current irrigation schedule")

# Soil-based recommendation
if soil == "sandy":
    st.warning(" Add organic compost to improve water retention")
elif soil == "clay":
    st.warning(" Improve soil drainage and avoid overwatering")
elif soil == "loamy":
    st.success(" Use balanced fertilizers (NPK) for better yield")

# Temperature
if temperature > 35:
    st.warning("🌡 Use shade nets or irrigation to reduce heat stress")
elif temperature < 20:
    st.warning("❄ Crop growth may slow - consider protective measures")
