
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
from sklearn.cluster import KMeans
import plotly.express as px
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="AI Packaging Dashboard", layout="wide")
st.title("🚀 Biodegradable Packaging AI Decision Engine")

uploaded_file = st.file_uploader("Upload Dataset CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    df_encoded = df.copy()
    encoders = {}
    for col in df_encoded.columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        encoders[col] = le

    X = df_encoded.drop("switch_likelihood", axis=1)
    y = df_encoded["switch_likelihood"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)

    st.subheader("Classification Metrics")
    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.write("Precision:", precision_score(y_test, y_pred, average='weighted'))
    st.write("Recall:", recall_score(y_test, y_pred, average='weighted'))
    st.write("F1 Score:", f1_score(y_test, y_pred, average='weighted'))

    # ROC
    try:
        fpr, tpr, _ = roc_curve(y_test, y_prob[:,1], pos_label=1)
        fig = px.area(x=fpr, y=tpr, title="ROC Curve")
        st.plotly_chart(fig)
    except:
        st.write("ROC not available for multi-class")

    # Confusion Matrix
    st.write("Confusion Matrix")
    st.write(confusion_matrix(y_test, y_pred))

    # Feature importance
    importances = clf.feature_importances_
    feat_df = pd.DataFrame({"Feature": X.columns, "Importance": importances}).sort_values(by="Importance", ascending=False)
    st.bar_chart(feat_df.set_index("Feature"))

    # Clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    df["Cluster"] = kmeans.fit_predict(X)
    st.subheader("Customer Segments")
    st.plotly_chart(px.histogram(df, x="Cluster"))

    # Association rules
    df_bool = df_encoded.applymap(lambda x: 1 if x > 0 else 0)
    freq_items = apriori(df_bool, min_support=0.1, use_colnames=True)
    rules = association_rules(freq_items, metric="confidence", min_threshold=0.5)
    st.subheader("Association Rules")
    st.dataframe(rules[["antecedents","consequents","support","confidence","lift"]])

    # Strategy
    st.subheader("Strategy Recommendations")
    st.write("High probability + high spend → contracts")
    st.write("High probability + low spend → discounts")
    st.write("Low probability + high spend → educate + samples")
    st.write("Low probability + low spend → low priority")

    # Prediction
    st.subheader("Predict New Customer")
    input_data = {}
    for col in X.columns:
        input_data[col] = st.selectbox(col, encoders[col].classes_)

    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])
        for col in input_df.columns:
            input_df[col] = encoders[col].transform(input_df[col])
        pred = clf.predict(input_df)[0]
        prob = clf.predict_proba(input_df).max()
        st.success(f"Prediction: {pred} (Confidence: {prob:.2f})")
