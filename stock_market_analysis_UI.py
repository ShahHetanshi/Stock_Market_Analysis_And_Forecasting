import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# App Title
st.title("Stock Market Analysis and Forecasting")

# Sidebar for user inputs
st.sidebar.header("User Input")

# Upload dataset
uploaded_file = st.sidebar.file_uploader("Upload Historical Stock Data (CSV)", type="csv")

if uploaded_file is not None:
    # Load data
    data = pd.read_csv(uploaded_file, index_col='Date', parse_dates=['Date'])
    st.subheader("Uploaded Data Preview")
    st.dataframe(data.head())

    # Select stock or column for analysis
    columns = data.columns.tolist()
    selected_column = st.sidebar.selectbox("Select a column for analysis", columns)

    # Visualize selected column
    st.subheader(f"Visualization of {selected_column}")
    plt.figure(figsize=(10, 5))
    plt.plot(data[selected_column], label=selected_column)
    plt.legend()
    plt.grid()
    st.pyplot(plt)

    # Moving Average
    st.sidebar.subheader("Moving Average")
    window_size = st.sidebar.slider("Select window size", 1, 50, 5)
    data[f"MA_{window_size}"] = data[selected_column].rolling(window=window_size).mean()

    st.subheader(f"Moving Average ({window_size} periods)")
    plt.figure(figsize=(10, 5))
    plt.plot(data[selected_column], label=selected_column)
    plt.plot(data[f"MA_{window_size}"], label=f"MA_{window_size}")
    plt.legend()
    plt.grid()
    st.pyplot(plt)

    # Forecasting Section
    st.sidebar.subheader("Forecasting")
    forecasting_model = st.sidebar.selectbox("Select Forecasting Model", ["Linear Regression", "ARIMA", "LSTM"])

    st.subheader("Forecasting")
    st.write(f"Forecasting model selected: {forecasting_model}")

    if forecasting_model == "Linear Regression":
        st.write("Running Linear Regression model...")

        # Prepare data for Linear Regression
        data = data.dropna(subset=[selected_column])  # Drop NaN values
        X = np.arange(len(data)).reshape(-1, 1)  # Time as feature
        y = data[selected_column].values

        # Train-test split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Fit the model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Plot results
        plt.figure(figsize=(10, 5))
        plt.plot(data.index[:split_idx], y_train, label="Training Data")
        plt.plot(data.index[split_idx:], y_test, label="Actual Data")
        plt.plot(data.index[split_idx:], y_pred, label="Predicted Data")
        plt.legend()
        plt.grid()
        st.pyplot(plt)

    elif forecasting_model == "ARIMA":
        st.write("ARIMA model is not implemented yet.")
    elif forecasting_model == "LSTM":
        st.write("LSTM model is not implemented yet.")

else:
    st.write("Please upload a CSV file to proceed.")