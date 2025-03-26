from flask import Flask

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)

@app.route("/")
def home():
    return "<p>Home Page</p>"


@app.route("/gen")
def gen():
    # Set random seed for reproducibility
    np.random.seed(42)

    # Number of samples
    data_size = 500

    # Generate random data for key risk factors
    age = np.random.randint(20, 80, data_size)
    bmi = np.random.uniform(18.5, 40, data_size)
    fasting_glucose = np.random.uniform(80, 180, data_size)
    hba1c = np.random.uniform(4.5, 9.5, data_size)
    blood_pressure_systolic = np.random.randint(110, 180, data_size)
    blood_pressure_diastolic = np.random.randint(70, 110, data_size)
    physical_activity = np.random.randint(0, 10, data_size)
    diabetes_history = np.random.randint(0, 2, data_size)
    smoker = np.random.randint(0, 2, data_size)

    # Generate outcome based on glucose and HbA1c levels (simple rule-based approach)
    def classify_diabetes(glucose, hba1c):
        if glucose > 125 or hba1c > 6.5:
            return 1  # Diabetic
        return 0  # Non-Diabetic

    outcome = [classify_diabetes(fg, hba) for fg, hba in zip(fasting_glucose, hba1c)]

    # Create DataFrame
    df = pd.DataFrame({
        'Age': age,
        'BMI': bmi,
        'Fasting Glucose': fasting_glucose,
        'HbA1c': hba1c,
        'Blood Pressure Systolic': blood_pressure_systolic,
        'Blood Pressure Diastolic': blood_pressure_diastolic,
        'Physical Activity': physical_activity,
        'Diabetes History': diabetes_history,
        'Smoker': smoker,
        'Outcome': outcome
    })

    # Save dataset
    df.to_csv('diabetes_dummy_data.csv', index=False)

    # Display first few rows
    print(df.head())

    return "<p>Gen Page</p>"

@app.route("/eda")
def eda():
    # Load the dataset
    df = pd.read_csv('diabetes_dummy_data.csv')

    # Basic info and summary statistics
    print("---INFO---")
    print(df.info())
    print("---Describe---")
    print(df.describe())

    # Check for missing values
    print("\nMissing values in each column:\n", df.isnull().sum())

    # Class distribution
    plt.figure(figsize=(6,4))
    print("---SNS Graph---")
    sns.countplot(x='Outcome', data=df, palette='coolwarm')
    plt.title('Diabetes Outcome Distribution')
    plt.show()

    # Histograms for numerical features
    print("---Histogram Graph---")
    df.hist(figsize=(12, 8), bins=20, edgecolor='black')
    plt.tight_layout()
    plt.show()

    # Correlation heatmap
    print("---Heatmap Graph---")
    plt.figure(figsize=(10,6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Feature Correlation Heatmap')
    plt.show()
    return "<p>EDA performed</p>"



