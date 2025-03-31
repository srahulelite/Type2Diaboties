from flask import Flask, request, render_template

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import pickle

app = Flask(__name__)

# Load model and scaler if available
try:
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
except FileNotFoundError:
    model, scaler = None, None

@app.route("/")
def home():
    return render_template("index.html")


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

@app.route("/train")
def train():
    global model, scaler
    
    # Load the dataset
    df = pd.read_csv('diabetes_dummy_data.csv')

    # Selecting features and target variable
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']

    # Splitting into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Standardizing the numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model Training - Logistic Regression
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    # Save model and scaler
    pickle.dump(model, open("model.pkl", "wb"))
    pickle.dump(scaler, open("scaler.pkl", "wb"))

    # Predictions
    y_pred = model.predict(X_test_scaled)

    # Model Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    return f"Model trained! Accuracy: {accuracy:.2f} <br><pre>{report}</pre>"

@app.route("/predict", methods=["POST"]) 
def predict():
    if not model or not scaler:
        return "Model not trained yet! Please train first."
    
    # Get form data
    form_data = [float(request.form[key]) for key in request.form.keys()]
    
    # Preprocess input
    input_scaled = scaler.transform([form_data])
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1] * 100
    
    result = "Diabetes Risk High" if prediction == 1 else "Low Risk"
    return render_template("index.html", prediction=result, probability=probability)




