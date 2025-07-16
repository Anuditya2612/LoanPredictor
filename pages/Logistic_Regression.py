import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

st.title("📈 Performance of Logistic Regressor")

# Load data and model
df = pd.read_csv("data/Bank_Personal_Loan_Modelling.csv")
model = joblib.load("model/logistic_model.pkl")
scaler = joblib.load("model/scaler.pkl")

X = df.drop(columns=['Personal Loan', 'ID', 'ZIP Code'])
y = df['Personal Loan']
X_scaled = scaler.transform(X)

# Predict
y_pred = model.predict(X_scaled)
y_proba = model.predict_proba(X_scaled)[:, 1]

st.markdown("""
## About LOGISTIC REGRESSION
- This is a simple linear classification technique under the supervised learning.
- It estimates the probability of target feature using sigmoid function
""")

st.write("### Accuracy of the model is:\n")
st.text(accuracy_score(y,y_pred))

st.write("### Classification Report")
st.text(classification_report(y, y_pred))

st.write("### Confusion Matrix")
cm = confusion_matrix(y, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
st.pyplot(plt)
plt.clf()

st.write("### ROC (Receiver Operating Characteristic) AUC Score")
st.write(roc_auc_score(y, y_proba))

st.markdown("""
## Summary of the model used -
- Here we make use of Logistic Regression to classify whether the customer is likely to accept or reject his/her loan offer.
- And the model achieves an accuracy of 95.24%
- Alongside we take a look at the classification report with the confusion matrix, and finally the ROC score.         
""")