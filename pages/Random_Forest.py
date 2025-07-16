import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree

st.title("🌲 Random Forest Evaluation")

df = pd.read_csv("data/Bank_Personal_Loan_Modelling.csv")
model = joblib.load("model/random_forest.pkl")
scaler = joblib.load("model/scaler.pkl")

X = df.drop(columns=['Personal Loan', 'ID', 'ZIP Code'])
y = df['Personal Loan']
X_scaled = scaler.transform(X)

# Predict
y_pred = model.predict(X_scaled)
y_proba = model.predict_proba(X_scaled)[:, 1]

st.markdown("""
## About RANDOM FOREST CLASSIFIER
-  A classification technique under the UN-Supervised learning.
- It estimates the probability of target feature by minimizing the 'gini' impurity
- We have the choice to set the no. of neighbors (estimators) to be considered and also the depth of the tree(s) formed
""")

st.write("### Classification Report")
st.text(classification_report(y, y_pred))

st.write("### Confusion Matrix")
cm = confusion_matrix(y, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
st.pyplot(plt)
plt.clf()

st.write("### ROC AUC Score")
st.write(roc_auc_score(y, y_proba))

st.write("### F1 Score")
st.write(f1_score(y,y_pred))

# Feature Importances
st.write("### Feature Importances")
importances = model.feature_importances_
features = X.columns
sns.barplot(x=importances, y=features)
plt.title("Random Forest Feature Importance")
plt.tight_layout()
st.pyplot(plt)
plt.clf()

#Visualizing a tree from the forest
st.subheader("🌲 Let's visualize one tree (50th) from the forest")

plt.figure(figsize=(20,10))
plot_tree(model.estimators_[49],feature_names=X.columns,class_names=['Rejected','Accepted'],
          filled=True, rounded=True, max_depth=5)

st.pyplot(plt)
plt.clf()