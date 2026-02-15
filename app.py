import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report

st.title("Heart Disease Classification App")

model_name = st.selectbox("Select Model", 
["LogisticRegression", "DecisionTreeClassifier", "KNeighborsClassifier", 
 "GaussianNB", "RandomForestClassifier", "XGBClassifier"])

file = st.file_uploader("Upload CSV File", type=["csv"])

if file is not None:

    data = pd.read_csv(file)
    X = data.drop("target", axis=1)
    y = data["target"]

    model = joblib.load("model/" + model_name + ".pkl")

    if model_name in ["LogisticRegression", "KNeighborsClassifier"]:
        scaler = joblib.load("model/scaler.pkl")
        X = scaler.transform(X)

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:,1]

    st.write("Accuracy:", accuracy_score(y, y_pred))
    st.write("AUC:", roc_auc_score(y, y_prob))
    st.write("Precision:", precision_score(y, y_pred))
    st.write("Recall:", recall_score(y, y_pred))
    st.write("F1 Score:", f1_score(y, y_pred))
    st.write("MCC:", matthews_corrcoef(y, y_pred))

    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d")
    st.pyplot(fig)

    st.text(classification_report(y, y_pred))
