import gradio as gr
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
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
VISITOR_TYPES = ["Returning_Visitor", "New_Visitor", "Other"]
WEEKENDS = ["TRUE", "FALSE"]


# Decision Tree Classifier
def predict_dt(monthly_fee, customer_age, support_calls):
    X = np.array([[monthly_fee, customer_age, support_calls]])
    prediction = dt_classifier.predict(X)
    return int(prediction[0])


# Online Shopper prediction helper
def build_online_shopper_data(administrative, administrative_duration, informational,
                               informational_duration, product_related, product_related_duration,
                               bounce_rates, exit_rates, page_values, special_day,
                               month, operating_systems, browser, region, traffic_type,
                               visitor_type, weekend):
    return {
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


# Naive Bayes Classifier
def predict_naive_bayes(*args):
    data = build_online_shopper_data(*args)
    new_data = pd.DataFrame([data])
    for col in CATEGORICAL_COLS:
        new_data[col] = label_encoders_4[col].transform(new_data[col])
    new_data = new_data[ONLINE_SHOPPERS_FEATURES]
    prediction = naive_bayes.predict(new_data)[0]
    return "Purchase" if prediction == 1 else "No Purchase"


# SVM Classifier
def predict_svm(*args):
    data = build_online_shopper_data(*args)
    new_data = pd.DataFrame([data])
    for col in CATEGORICAL_COLS:
        new_data[col] = label_encoders_4[col].transform(new_data[col])
    new_data = new_data[ONLINE_SHOPPERS_FEATURES]
    scaled_data = scaler_4.transform(new_data)
    prediction = svm_classifier.predict(scaled_data)[0]
    return "Purchase" if prediction == 1 else "No Purchase"


# Random Forest Classifier
def predict_rf(*args):
    data = build_online_shopper_data(*args)
    new_data = pd.DataFrame([data])
    for col in CATEGORICAL_COLS:
        new_data[col] = label_encoders_5[col].transform(new_data[col])
    new_data = new_data[ONLINE_SHOPPERS_FEATURES]
    scaled_data = scaler_5.transform(new_data)
    prediction = random_forest.predict(scaled_data)[0]
    return "Purchase" if prediction == 1 else "No Purchase"


# Product Recommender
def recommend_products(products_text):
    if not products_text.strip():
        return "Please enter at least one product."

    input_products = set([p.strip().lower() for p in products_text.split(",") if p.strip()])

    if not input_products:
        return "Please enter valid product names."

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
                        'based_on': list(antecedents)
                    })

    # Remove duplicates and sort by lift
    seen = set()
    unique_recs = []
    for rec in sorted(recommendations, key=lambda x: x['lift'], reverse=True):
        if rec['product'] not in seen:
            seen.add(rec['product'])
            unique_recs.append(rec)

    if not unique_recs:
        return "No recommendations found for these products."

    result = f"Found {len(unique_recs)} recommendation(s):\n\n"
    for rec in unique_recs[:10]:
        result += f"- {rec['product']} (Lift: {rec['lift']:.4f}, Confidence: {rec['confidence']:.4f})\n"
        result += f"  Based on: {', '.join(rec['based_on'])}\n\n"

    return result


# Online shopper inputs
def get_online_shopper_inputs():
    return [
        gr.Number(label="Administrative", value=0),
        gr.Number(label="Administrative Duration", value=0.0),
        gr.Number(label="Informational", value=0),
        gr.Number(label="Informational Duration", value=0.0),
        gr.Number(label="Product Related", value=0),
        gr.Number(label="Product Related Duration", value=0.0),
        gr.Number(label="Bounce Rates", value=0.0),
        gr.Number(label="Exit Rates", value=0.0),
        gr.Number(label="Page Values", value=0.0),
        gr.Number(label="Special Day", value=0.0),
        gr.Dropdown(label="Month", choices=MONTHS, value="Jan"),
        gr.Number(label="Operating Systems", value=1),
        gr.Number(label="Browser", value=1),
        gr.Number(label="Region", value=1),
        gr.Number(label="Traffic Type", value=1),
        gr.Dropdown(label="Visitor Type", choices=VISITOR_TYPES, value="Returning_Visitor"),
        gr.Dropdown(label="Weekend", choices=WEEKENDS, value="FALSE")
    ]


# Build Gradio interface with tabs
with gr.Blocks(title="ML Model Predictions") as demo:
    gr.Markdown("# ML Model Predictions")

    with gr.Tabs():
        # Decision Tree Tab
        with gr.TabItem("Decision Tree"):
            gr.Markdown("## Customer Churn Prediction")
            gr.Markdown("Predict churn using a Decision Tree Classifier.")
            with gr.Row():
                with gr.Column():
                    dt_monthly = gr.Number(label="Monthly Fee", value=0)
                    dt_age = gr.Number(label="Customer Age", value=0)
                    dt_calls = gr.Number(label="Support Calls", value=0)
                    dt_btn = gr.Button("Predict")
                with gr.Column():
                    dt_output = gr.Number(label="Predicted Class")
            dt_btn.click(predict_dt, inputs=[dt_monthly, dt_age, dt_calls], outputs=dt_output)

        # Naive Bayes Tab
        with gr.TabItem("Naive Bayes"):
            gr.Markdown("## Online Shoppers Purchase Prediction (Naive Bayes)")
            with gr.Row():
                with gr.Column():
                    nb_inputs = get_online_shopper_inputs()
                    nb_btn = gr.Button("Predict")
                with gr.Column():
                    nb_output = gr.Textbox(label="Prediction")
            nb_btn.click(predict_naive_bayes, inputs=nb_inputs, outputs=nb_output)

        # SVM Tab
        with gr.TabItem("SVM"):
            gr.Markdown("## Online Shoppers Purchase Prediction (SVM)")
            with gr.Row():
                with gr.Column():
                    svm_inputs = get_online_shopper_inputs()
                    svm_btn = gr.Button("Predict")
                with gr.Column():
                    svm_output = gr.Textbox(label="Prediction")
            svm_btn.click(predict_svm, inputs=svm_inputs, outputs=svm_output)

        # Random Forest Tab
        with gr.TabItem("Random Forest"):
            gr.Markdown("## Online Shoppers Purchase Prediction (Random Forest)")
            with gr.Row():
                with gr.Column():
                    rf_inputs = get_online_shopper_inputs()
                    rf_btn = gr.Button("Predict")
                with gr.Column():
                    rf_output = gr.Textbox(label="Prediction")
            rf_btn.click(predict_rf, inputs=rf_inputs, outputs=rf_output)

        # Product Recommender Tab
        with gr.TabItem("Product Recommender"):
            gr.Markdown("## Product Recommendations")
            gr.Markdown("Get product recommendations based on association rules from grocery data.")
            gr.Markdown("*Sample products: whole milk, yogurt, rolls/buns, tropical fruit, pip fruit, other vegetables, citrus fruit, soda, bottled water, root vegetables*")
            with gr.Row():
                with gr.Column():
                    rec_input = gr.Textbox(
                        label="Enter products (comma-separated)",
                        placeholder="e.g., whole milk, yogurt, rolls/buns"
                    )
                    rec_btn = gr.Button("Get Recommendations")
                with gr.Column():
                    rec_output = gr.Textbox(label="Recommendations", lines=10)
            rec_btn.click(recommend_products, inputs=rec_input, outputs=rec_output)

if __name__ == "__main__":
    demo.launch()
