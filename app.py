import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Explainable AQI Estimation",
    page_icon="🌍",
    layout="wide"
)

regression_model = joblib.load("aqi_regression_model.pkl")
classification_model = joblib.load("aqi_classification_model.pkl")

st.title("🌍 Explainable AQI Estimation")
st.write(
    "Estimate Air Quality Index using pollutant-level AQI values "
    "with ensemble machine learning."
)

st.divider()

st.subheader("Enter Pollutant AQI Values")

col1, col2 = st.columns(2)

with col1:
    co = st.number_input(
        "CO AQI",
        min_value=0.0,
        value=20.0
    )

    ozone = st.number_input(
        "Ozone AQI",
        min_value=0.0,
        value=50.0
    )

with col2:
    no2 = st.number_input(
        "NO2 AQI",
        min_value=0.0,
        value=30.0
    )

    pm25 = st.number_input(
        "PM2.5 AQI",
        min_value=0.0,
        value=100.0
    )

input_data = pd.DataFrame({
    "CO AQI Value": [co],
    "Ozone AQI Value": [ozone],
    "NO2 AQI Value": [no2],
    "PM2.5 AQI Value": [pm25]
})

st.divider()

if st.button("Predict AQI", type="primary"):

    predicted_aqi = regression_model.predict(input_data)[0]

    predicted_category = classification_model.predict(
        input_data
    )[0]

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Predicted AQI",
            f"{predicted_aqi:.2f}"
        )

    with col2:
        st.metric(
            "AQI Category",
            predicted_category
        )

    st.subheader("Input Values")

    st.dataframe(
        input_data,
        use_container_width=True
    )

    st.info(
        "The prediction represents the model's estimated AQI "
        "based on the entered pollutant AQI values."
    )
