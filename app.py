import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import numpy as np
import tensorflow as tf
import streamlit as st

TF_ENABLE_ONEDNN_OPTS = 0

st.title(":pill: Individual medical costs billed by health insurance :pill:")

st.header("Data Information")

st.markdown("""
---
**Age**: age of primary beneficiary

**Sex**: insurance contractor gender, female, male

**Bmi**: Body mass index, providing an understanding of body, weights that are relatively high or low relative to height,
objective index of body weight (kg / m ^ 2) using the ratio of height to weight, ideally 18.5 to 24.9

**Children**: Number of children covered by health insurance / Number of dependents

**Smoker**: Smoking

**Region**: the beneficiary's residential area in the US, northeast, southeast, southwest, northwest.
""")

st.markdown("""---""")

age = st.number_input(label="Age", min_value=0)

sex = st.selectbox("Sex", ["Male", "Female"])

bmi = st.number_input(label="BMI")

children = st.number_input(label="Children", min_value=0)

smoker = st.selectbox("Smoker", ["No", "Yes"])

region = st.selectbox("Region", ["Northeast", "Southeast", "Southwest", "Northwest"])

def calculate(age, sex, bmi, children, smoker, region):
    data = {"age":[age],
        "sex":[sex],
        "bmi":[bmi],
        "children":[children],
        "smoker":[smoker],
        "region":[region]}
    
    print(type(bmi))

    for key in data.keys():
        if type(data[key][0]) == str:
            data[key][0] = data[key][0].lower()

    df = pd.DataFrame(data)

    insurance = pd.read_csv("insurance.csv")
    X = insurance.drop("charges", axis=1)
            
    ct = sklearn.compose.make_column_transformer(
        (sklearn.preprocessing.MinMaxScaler(), ["age", "bmi", "children"]),
        (sklearn.preprocessing.OneHotEncoder(handle_unknown="ignore"), ["sex", "smoker", "region"])
    )

    ct.fit(insurance)

    normalized_df = ct.transform(df)
    
    model = tf.keras.models.load_model("model.keras")

    result = model.predict(normalized_df)

    return int(result[0])

if st.button(label="Calculate"):
    result = calculate(age, sex, bmi, children, smoker, region)
    st.markdown(f"Price of individual medical costs billed by health insurance is: **{result}$**")


