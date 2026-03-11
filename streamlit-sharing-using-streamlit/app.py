import streamlit as st
import joblib
import numpy as np
import pandas as pd
import json

# Load trained models
dt_classifier = joblib.load("./model/decisiontree_classifier_baseline.pkl")
naive_bayes = joblib.load("./model/naive_Bayes_classifier_optimum.pkl")
svm_classifier = joblib.load("./model/support_vector_classifier_optimum.pkl")
random_forest = joblib.load("./model/random_forest_classifier_optimum.pkl")

# Load encoders and scalers
label_encoders_4 = joblib.load("./model/label_encoders_4.pkl")
label_encoders_5 = joblib.load("./model/label_encoders_5.pkl")
scaler_4 = joblib.load("./model/scaler_4.pkl")
scaler_5 = joblib.load("./model/scaler_5.pkl")

# Load association rules
with open("./rules/association_rules_groceries.json", "r") as f:
    association_rules = json.load(f)

# Online shoppers features
ONLINE_SHOPPERS_FEATURES = [
    "Administrative", "Administrative_Duration", "Informational",
    "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues", "SpecialDay",
    "Month", "OperatingSystems", "Browser", "Region", "TrafficType",
    "VisitorType", "Weekend"
]

CATEGORICAL_COLS = ['Month', 'VisitorType', 'Weekend']

# Streamlit page config
st.set_page_config(
    page_title="ML Model Predictions",
    page_icon="🤖",
    layout="wide"
)

st.title("ML Model Predictions")

# Sidebar model selector
model_choice = st.sidebar.selectbox(
    "Select Model",
    [
        "Decision Tree Classifier",
        "Naive Bayes Classifier",
        "SVM Classifier",
        "Random Forest Classifier",
        "Product Recommender"
    ]
)

# Decision Tree Classifier
if model_choice == "Decision Tree Classifier":
    st.header("Customer Churn Prediction")
    st.write("Predict churn using a Decision Tree Classifier.")

    with st.form("dt_form"):
        monthly_fee = st.number_input("Monthly Fee", min_value=0.0, step=1.0)
        customer_age = st.number_input("Customer Age", min_value=0, step=1)
        support_calls = st.number_input("Support Calls", min_value=0, step=1)
        submitted = st.form_submit_button("Predict")

    if submitted:
        X = np.array([[monthly_fee, customer_age, support_calls]])
        prediction = dt_classifier.predict(X)
        st.success(f"### Predicted Class: {int(prediction[0])}")

# Naive Bayes Classifier
elif model_choice == "Naive Bayes Classifier":
    st.header("Online Shoppers Purchase Prediction (Naive Bayes)")
    st.write("Predict purchasing intention using Naive Bayes.")

    with st.form("nb_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            administrative = st.number_input("Administrative", value=0)
            administrative_duration = st.number_input("Administrative Duration", value=0.0)
            informational = st.number_input("Informational", value=0)
            informational_duration = st.number_input("Informational Duration", value=0.0)
            product_related = st.number_input("Product Related", value=0)
            product_related_duration = st.number_input("Product Related Duration", value=0.0)

        with col2:
            bounce_rates = st.number_input("Bounce Rates", value=0.0, format="%.4f")
            exit_rates = st.number_input("Exit Rates", value=0.0, format="%.4f")
            page_values = st.number_input("Page Values", value=0.0)
            special_day = st.number_input("Special Day", value=0.0, min_value=0.0, max_value=1.0)
            month = st.selectbox("Month", ["Jan", "Feb", "Mar", "Apr", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
            operating_systems = st.number_input("Operating Systems", value=1, min_value=1)

        with col3:
            browser = st.number_input("Browser", value=1, min_value=1)
            region = st.number_input("Region", value=1, min_value=1)
            traffic_type = st.number_input("Traffic Type", value=1, min_value=1)
            visitor_type = st.selectbox("Visitor Type", ["Returning_Visitor", "New_Visitor", "Other"])
            weekend = st.selectbox("Weekend", ["TRUE", "FALSE"])

        submitted = st.form_submit_button("Predict")

    if submitted:
        data = {
            "Administrative": administrative,
            "Administrative_Duration": administrative_duration,
            "Informational": informational,
            "Informational_Duration": informational_duration,
            "ProductRelated": product_related,
            "ProductRelated_Duration": product_related_duration,
            "BounceRates": bounce_rates,
            "ExitRates": exit_rates,
            "PageValues": page_values,
            "SpecialDay": special_day,
            "Month": month,
            "OperatingSystems": operating_systems,
            "Browser": browser,
            "Region": region,
            "TrafficType": traffic_type,
            "VisitorType": visitor_type,
            "Weekend": weekend
        }

        new_data = pd.DataFrame([data])
        for col in CATEGORICAL_COLS:
            new_data[col] = label_encoders_4[col].transform(new_data[col])
        new_data = new_data[ONLINE_SHOPPERS_FEATURES]

        prediction = naive_bayes.predict(new_data)[0]
        st.success(f"### Prediction: {'Purchase' if prediction == 1 else 'No Purchase'}")

# SVM Classifier
elif model_choice == "SVM Classifier":
    st.header("Online Shoppers Purchase Prediction (SVM)")
    st.write("Predict purchasing intention using Support Vector Machine.")

    with st.form("svm_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            administrative = st.number_input("Administrative", value=0)
            administrative_duration = st.number_input("Administrative Duration", value=0.0)
            informational = st.number_input("Informational", value=0)
            informational_duration = st.number_input("Informational Duration", value=0.0)
            product_related = st.number_input("Product Related", value=0)
            product_related_duration = st.number_input("Product Related Duration", value=0.0)

        with col2:
            bounce_rates = st.number_input("Bounce Rates", value=0.0, format="%.4f")
            exit_rates = st.number_input("Exit Rates", value=0.0, format="%.4f")
            page_values = st.number_input("Page Values", value=0.0)
            special_day = st.number_input("Special Day", value=0.0, min_value=0.0, max_value=1.0)
            month = st.selectbox("Month", ["Jan", "Feb", "Mar", "Apr", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
            operating_systems = st.number_input("Operating Systems", value=1, min_value=1)

        with col3:
            browser = st.number_input("Browser", value=1, min_value=1)
            region = st.number_input("Region", value=1, min_value=1)
            traffic_type = st.number_input("Traffic Type", value=1, min_value=1)
            visitor_type = st.selectbox("Visitor Type", ["Returning_Visitor", "New_Visitor", "Other"])
            weekend = st.selectbox("Weekend", ["TRUE", "FALSE"])

        submitted = st.form_submit_button("Predict")

    if submitted:
        data = {
            "Administrative": administrative,
            "Administrative_Duration": administrative_duration,
            "Informational": informational,
            "Informational_Duration": informational_duration,
            "ProductRelated": product_related,
            "ProductRelated_Duration": product_related_duration,
            "BounceRates": bounce_rates,
            "ExitRates": exit_rates,
            "PageValues": page_values,
            "SpecialDay": special_day,
            "Month": month,
            "OperatingSystems": operating_systems,
            "Browser": browser,
            "Region": region,
            "TrafficType": traffic_type,
            "VisitorType": visitor_type,
            "Weekend": weekend
        }

        new_data = pd.DataFrame([data])
        for col in CATEGORICAL_COLS:
            new_data[col] = label_encoders_4[col].transform(new_data[col])
        new_data = new_data[ONLINE_SHOPPERS_FEATURES]
        scaled_data = scaler_4.transform(new_data)

        prediction = svm_classifier.predict(scaled_data)[0]
        st.success(f"### Prediction: {'Purchase' if prediction == 1 else 'No Purchase'}")

# Random Forest Classifier
elif model_choice == "Random Forest Classifier":
    st.header("Online Shoppers Purchase Prediction (Random Forest)")
    st.write("Predict purchasing intention using Random Forest.")

    with st.form("rf_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            administrative = st.number_input("Administrative", value=0)
            administrative_duration = st.number_input("Administrative Duration", value=0.0)
            informational = st.number_input("Informational", value=0)
            informational_duration = st.number_input("Informational Duration", value=0.0)
            product_related = st.number_input("Product Related", value=0)
            product_related_duration = st.number_input("Product Related Duration", value=0.0)

        with col2:
            bounce_rates = st.number_input("Bounce Rates", value=0.0, format="%.4f")
            exit_rates = st.number_input("Exit Rates", value=0.0, format="%.4f")
            page_values = st.number_input("Page Values", value=0.0)
            special_day = st.number_input("Special Day", value=0.0, min_value=0.0, max_value=1.0)
            month = st.selectbox("Month", ["Jan", "Feb", "Mar", "Apr", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
            operating_systems = st.number_input("Operating Systems", value=1, min_value=1)

        with col3:
            browser = st.number_input("Browser", value=1, min_value=1)
            region = st.number_input("Region", value=1, min_value=1)
            traffic_type = st.number_input("Traffic Type", value=1, min_value=1)
            visitor_type = st.selectbox("Visitor Type", ["Returning_Visitor", "New_Visitor", "Other"])
            weekend = st.selectbox("Weekend", ["TRUE", "FALSE"])

        submitted = st.form_submit_button("Predict")

    if submitted:
        data = {
            "Administrative": administrative,
            "Administrative_Duration": administrative_duration,
            "Informational": informational,
            "Informational_Duration": informational_duration,
            "ProductRelated": product_related,
            "ProductRelated_Duration": product_related_duration,
            "BounceRates": bounce_rates,
            "ExitRates": exit_rates,
            "PageValues": page_values,
            "SpecialDay": special_day,
            "Month": month,
            "OperatingSystems": operating_systems,
            "Browser": browser,
            "Region": region,
            "TrafficType": traffic_type,
            "VisitorType": visitor_type,
            "Weekend": weekend
        }

        new_data = pd.DataFrame([data])
        for col in CATEGORICAL_COLS:
            new_data[col] = label_encoders_5[col].transform(new_data[col])
        new_data = new_data[ONLINE_SHOPPERS_FEATURES]
        scaled_data = scaler_5.transform(new_data)

        prediction = random_forest.predict(scaled_data)[0]
        st.success(f"### Prediction: {'Purchase' if prediction == 1 else 'No Purchase'}")

# Product Recommender
elif model_choice == "Product Recommender":
    st.header("Product Recommendations")
    st.write("Get product recommendations based on association rules from grocery data.")

    # Sample products for reference
    st.info("Sample products: whole milk, yogurt, rolls/buns, tropical fruit, pip fruit, other vegetables, citrus fruit, soda, bottled water, root vegetables")

    with st.form("recommender_form"):
        products_input = st.text_area(
            "Enter products (comma-separated)",
            placeholder="e.g., whole milk, yogurt, rolls/buns"
        )
        submitted = st.form_submit_button("Get Recommendations")

    if submitted and products_input:
        input_products = set([p.strip().lower() for p in products_input.split(",") if p.strip()])

        if input_products:
            recommendations = []
            for rule in association_rules:
                antecedents = set(rule['antecedents'])
                if antecedents.issubset(input_products):
                    for consequent in rule['consequents']:
                        if consequent not in input_products:
                            recommendations.append({
                                'product': consequent,
                                'confidence': rule['confidence'],
                                'lift': rule['lift'],
                                'support': rule['support'],
                                'based_on': list(antecedents)
                            })

            # Remove duplicates and sort by lift
            seen = set()
            unique_recs = []
            for rec in sorted(recommendations, key=lambda x: x['lift'], reverse=True):
                if rec['product'] not in seen:
                    seen.add(rec['product'])
                    unique_recs.append(rec)

            if unique_recs:
                st.success(f"### Found {len(unique_recs)} recommendation(s)")
                for rec in unique_recs[:10]:
                    st.write(f"**{rec['product']}** (Lift: {rec['lift']:.4f}, Confidence: {rec['confidence']:.4f})")
                    st.caption(f"Based on: {', '.join(rec['based_on'])}")
            else:
                st.warning("No recommendations found for these products.")
        else:
            st.error("Please enter at least one product.")
