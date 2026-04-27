import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("heart.csv")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient details below:")

age = st.number_input("Age", 1, 120)
sex = st.selectbox("Sex (0=Female, 1=Male)", [0, 1])
cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure")
chol = st.number_input("Cholesterol")
fbs = st.selectbox("Fasting Blood Sugar >120 (0/1)", [0, 1])
restecg = st.selectbox("Rest ECG (0-2)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate")
exang = st.selectbox("Exercise Induced Angina (0/1)", [0, 1])
oldpeak = st.number_input("Oldpeak")
slope = st.selectbox("Slope (0-2)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
thal = st.selectbox("Thal (0-3)", [0, 1, 2, 3])

if st.button("Predict"):

    input_data = np.array([[age, sex, cp, trestbps, chol,
                            fbs, restecg, thalach,
                            exang, oldpeak, slope, ca, thal]])

    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")

    st.write("Prediction Probability:")
    st.write(probability)

st.subheader("📊 Model Performance")

st.write("Model Accuracy:", round(accuracy * 100, 2), "%")

cm = confusion_matrix(y_test, y_pred)

fig1, ax1 = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', ax=ax1)
ax1.set_title("Confusion Matrix")
ax1.set_xlabel("Predicted")
ax1.set_ylabel("Actual")
st.pyplot(fig1)

st.subheader("📈 Chest Pain vs Heart Disease")

fig2, ax2 = plt.subplots()
sns.countplot(x='cp', hue='target', data=df, ax=ax2)
ax2.set_title("Chest Pain Type vs Heart Disease")
st.pyplot(fig2)