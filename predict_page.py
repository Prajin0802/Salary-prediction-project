import streamlit as st
import joblib
import numpy as np


def load_model():
    with open("saved_steps.pkl", "rb") as file:
        data = joblib.load(file)
    return data


data = load_model()

regressor = data["model"]
le_country = data["le_country"]
le_education = data["le_EdLevel"]


def show_predict_page():
    st.title("Software Developer Salary Prediction")

    st.write("""### We need some information to predict the salary""")

    countries = (
        "United States",
        "India",
        "United Kingdom",
        "Germany",
        "Canada",
        "Brazil",
        "France",
        "Spain",
        "Australia",
        "Netherlands",
        "Poland",
        "Italy",
        "Russian Federation",
        "Sweden",
    )

    education = (
        "Less than a Bachelors",
        "Bachelor’s degree",
        "Master’s degree",
        "Post grad",
    )

    country = st.selectbox("Country", countries)
    education = st.selectbox("Education Level", education)

    experience = st.slider("Years of Experience", 0, 50, 3)

    ok = st.button("Calculate Salary")
    if ok:
        x = np.array([[country, education, experience]])
        x[:, 0] = le_country.fit_transform(X[:, 0])
        x[:, 1] = le_education.fit_transform(X[:, 1])

        salary = regressor.predict(X)
        st.subheader(f"The estimated salary is ${salary[0]:.2f}")
