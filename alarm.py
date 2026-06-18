import streamlit as st
import pandas as pd
import joblib

import sys
print(sys.executable)

def preprocess(df, training=True):

    df = df.copy()

    df["Raised At"] = pd.to_datetime(df["Raised At"])

    import re

    def convert_duration(duration):
        hours = minutes = seconds = 0

        h = re.search(r'(\d+)h', str(duration))
        m = re.search(r'(\d+)m', str(duration))
        s = re.search(r'(\d+)s', str(duration))

        if h:
            hours = int(h.group(1))
        if m:
            minutes = int(m.group(1))
        if s:
            seconds = int(s.group(1))

        return hours*3600 + minutes*60 + seconds

    if training:
        df["Duration_sec"] = df["Duration"].apply(convert_duration)

    df["Start_Hours"] = df["Raised At"].dt.hour

    df["Start_Minutes"] = df["Raised At"].dt.minute

    df["Start_Second"] = df["Raised At"].dt.second

    df["Start_Day"] = df["Raised At"].dt.day_name()

    df["Start_Month"] = df["Raised At"].dt.month

    df["is_fault"] = (
        df["Alarm Name"]
        .str.contains(
            "FAULT",
            case=False,
            na=False
        )
        .astype(int)
    )

    df["is_stop"] = (
        df["Alarm Name"]
        .str.contains(
            "STOP",
            case=False,
            na=False
        )
        .astype(int)
    )

    df["is_safety"] = (
        df["Alarm Name"]
        .str.contains(
            "SAFETY|HUMAN|LIGHT CURTAIN|DOOR",
            case=False,
            na=False
        )
        .astype(int)
    )

    df["is_manual"] = (
        df["Alarm Name"]
        .str.contains(
            "MANUAL",
            case=False,
            na=False
        )
        .astype(int)
    )

    df["communication_fault"] = (
        df["Alarm Name"]
        .str.contains(
            "NET|CC-LINK|MELSEC",
            case=False,
            na=False
        )
        .astype(int)
    )

    df["camera_alarm"] = (
        df["Alarm Name"]
        .str.contains(
            "CAMERA|SCAN",
            case=False,
            na=False
        )
        .astype(int)
    )

    def severity(x):

        x = str(x).upper()

        if "FAULT" in x or "STOP" in x:
            return "HIGH"

        elif "OPEN" in x:
            return "MEDIUM"

        else:
            return "LOW"

    df["severity"] = (
        df["Alarm Name"]
        .apply(severity)
    )

    df["alarm_length"] = (
        df["Alarm Name"]
        .str.len()
    )

    # this is the freq veriable use in rere_alarm columns formation
    freq = (
        df["Alarm Name"]
        .value_counts()
    )

    df["rare_alarm"] = (
        df["Alarm Name"]
        .map(freq)
        .lt(5)
        .astype(int)
    )

    # remove COMNE part from the Equipment Name
    df["Equipment Name"] = df["Equipment Name"].str.replace(" PANEL", "", regex=False)

    df['Alarm_name'] = (
        df['Alarm Name']
        .str.split(' ', n=1)
        .str[1]
    )

    df["Alarm_name"] = df["Alarm_name"].str.replace(" ", "", regex=False)

    obj_col = df.select_dtypes(include="object").columns

    df[obj_col] = df[obj_col].apply(lambda x: x.str.lower())

    df = df.drop(columns=['Raised At', 'Alarm Name', 'Severity', 'Start_Month'])

    df = df.dropna()

    return df

# model load
model = joblib.load("pipeline.pkl")

st.title("🚨Alarm Duration Prediction System")

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:

    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)

    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file, engine="openpyxl", header=5)

    df.columns = df.columns.str.strip()

    st.write(df.columns.tolist())

    st.write("Original Data")

    st.dataframe(df.head())

    # Predict button
    if st.button("Predict"):

        # preprocessing
        new_data = preprocess(df, training=False)

        # prediction
        pred = model.predict(new_data)

        # add prediction column
        new_data["Prediction"] = pred

        st.success("Prediction completed successfully!")

        st.write("Prediction Result")
        st.dataframe(new_data)

        csv = new_data.to_csv(index=False)

        st.download_button(
            "Download Prediction CSV",
            csv,
            file_name="prediction.csv",
            mime="text/csv"
        )