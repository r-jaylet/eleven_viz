import pandas as pd
import streamlit as st


def show():
    st.title("GPS Data")
    st.write("### Overview")
    st.write(
        "This page displays GPS tracking data. Below, you can explore the first 10 records."
    )

    # Load GPS CSV file
    try:
        df = pd.read_csv("data/CFC GPS Data.csv")
        st.dataframe(df.head(10), use_container_width=True)
    except FileNotFoundError:
        st.error(
            "GPS data file not found. Please ensure data exists in the directory."
        )
