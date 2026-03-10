import json
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)

CORS(
    app,
    supports_credentials=False,
    resources={
        r"/api/*": {
            "origins": [
                "https://127.0.0.1",
                "https://localhost",
                "https://127.0.0.1:443",
                "https://localhost:443",
                "http://127.0.0.1",
                "http://localhost",
                "http://127.0.0.1:5000",
                "http://localhost:5000",
                "http://127.0.0.1:5500",
                "http://localhost:5500",
            ]
        }
    },
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
RULES_PATH = BASE_DIR / "rules" / "association_rules_groceries.json"

ONLINE_SHOPPERS_FEATURES = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]

ONLINE_SHOPPERS_NUMERIC_FEATURES = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
]

KNN_NUMERIC_FIELDS = [
    "Customer_rating",
    "Cost_of_the_Product",
    "Prior_purchases",
    "Discount_offered",
    "Weight_in_gms",
]

KNN_OPTIONAL_FEATURE_LIST_KEY = "features"


def _read_json_payload():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return None, _error_response(
            "Request body must be valid JSON object.", status_code=400
        )
    return data, None


def _error_response(message, status_code=400, details=None):
    payload = {"error": message}
    if details is not None:
        payload["details"] = details
    return jsonify(payload), status_code


def _load_association_rules(rules_path):
    if not rules_path.exists():
        return []

    with rules_path.open("r", encoding="utf-8") as f:
        parsed = json.load(f)

    raw_rules = parsed.get("rules", parsed) if isinstance(parsed, dict) else parsed
    normalized_rules = []
    for rule in raw_rules:
        antecedents = [
            str(item).strip().lower()
            for item in rule.get("antecedents", [])
            if str(item).strip()
        ]
        consequents = [
            str(item).strip().lower()
            for item in rule.get("consequents", [])
            if str(item).strip()
        ]
        if not antecedents or not consequents:
            continue
        normalized_rules.append(
            {
                "antecedents": antecedents,
                "consequents": consequents,
                "support": float(rule.get("support", 0.0)),
                "confidence": float(rule.get("confidence", 0.0)),
                "lift": float(rule.get("lift", 0.0)),
            }
        )
    return normalized_rules


def _encode_online_shopper_categories(frame, label_encoders):
    for column in ["Month", "VisitorType", "Weekend"]:
        encoder = label_encoders[column]
        raw_value = frame.at[0, column]
        if column == "Weekend" and isinstance(raw_value, str):
            lowered = raw_value.strip().lower()
            if lowered in {"true", "1", "yes"}:
                raw_value = True
            elif lowered in {"false", "0", "no"}:
                raw_value = False
        try:
            encoded_value = encoder.transform([raw_value])[0]
        except ValueError:
            accepted = [str(v) for v in encoder.classes_.tolist()]
            raise ValueError(
                f"Invalid value for '{column}'. Accepted values: {accepted}"
            ) from None
        frame[column] = frame[column].astype(object)
        frame.at[0, column] = encoded_value


def _build_online_shopper_frame(data, label_encoders):
    missing = [key for key in ONLINE_SHOPPERS_FEATURES if key not in data]
    if missing:
        raise ValueError(
            f"Missing required fields: {missing}. "
            f"Expected fields: {ONLINE_SHOPPERS_FEATURES}"
        )

    frame = pd.DataFrame([ {key: data[key] for key in ONLINE_SHOPPERS_FEATURES} ], dtype=object)
    for column in ONLINE_SHOPPERS_NUMERIC_FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="raise")

    _encode_online_shopper_categories(frame, label_encoders)
    frame = frame[ONLINE_SHOPPERS_FEATURES].astype(float)
    return frame


def _build_knn_feature_vector(data):
    if KNN_OPTIONAL_FEATURE_LIST_KEY in data:
        feature_values = data.get(KNN_OPTIONAL_FEATURE_LIST_KEY)
        if not isinstance(feature_values, list) or len(feature_values) != 8:
            raise ValueError("'features' must be a list with exactly 8 numeric values.")
        try:
            return [[float(value) for value in feature_values]]
        except (TypeError, ValueError):
            raise ValueError("'features' must contain only numeric values.") from None

    missing = [
        key for key in KNN_NUMERIC_FIELDS + ["Shipping_Mode"] if key not in data
    ]
    if missing:
        raise ValueError(
            f"Missing required fields: {missing}. "
            f"Required fields: {KNN_NUMERIC_FIELDS + ['Shipping_Mode']}"
        )

    try:
        numeric_values = [float(data[field]) for field in KNN_NUMERIC_FIELDS]
    except (TypeError, ValueError):
        raise ValueError(
            f"Numeric fields must be valid numbers: {KNN_NUMERIC_FIELDS}"
        ) from None

    shipping_mode = str(data["Shipping_Mode"]).strip()
    shipping_df = pd.DataFrame([{"Shipping Mode": shipping_mode}])
    try:
        encoded_shipping = onehot_encoder_3.transform(shipping_df)
    except ValueError:
        accepted = [str(v) for v in onehot_encoder_3.categories_[0].tolist()]
        raise ValueError(
            f"Invalid Shipping_Mode '{shipping_mode}'. Accepted values: {accepted}"
        ) from None

    if hasattr(encoded_shipping, "toarray"):
        encoded_row = encoded_shipping.toarray()[0].tolist()
    else:
        encoded_row = encoded_shipping[0].tolist()
    return [numeric_values + encoded_row]


def _load_models():
    return {
        "decisiontree_classifier_baseline": joblib.load(
            str(MODEL_DIR / "decisiontree_classifier_baseline.pkl")
        ),
        "decisiontree_regressor_optimum": joblib.load(
            str(MODEL_DIR / "decisiontree_regressor_optimum.pkl")
        ),
        "naive_bayes_classifier_optimum": joblib.load(
            str(MODEL_DIR / "naive_Bayes_classifier_optimum.pkl")
        ),
        "knn_classifier_optimum": joblib.load(str(MODEL_DIR / "knn_classifier_optimum.pkl")),
        "support_vector_classifier_optimum": joblib.load(
            str(MODEL_DIR / "support_vector_classifier_optimum.pkl")
        ),
        "random_forest_classifier_optimum": joblib.load(
            str(MODEL_DIR / "random_forest_classifier_optimum.pkl")
        ),
        "label_encoders_1b": joblib.load(str(MODEL_DIR / "label_encoders_1b.pkl")),
        "label_encoders_2": joblib.load(str(MODEL_DIR / "label_encoders_2.pkl")),
        "label_encoders_4": joblib.load(str(MODEL_DIR / "label_encoders_4.pkl")),
        "label_encoders_5": joblib.load(str(MODEL_DIR / "label_encoders_5.pkl")),
        "onehot_encoder_3": joblib.load(str(MODEL_DIR / "onehot_encoder_3.pkl")),
        "scaler_4": joblib.load(str(MODEL_DIR / "scaler_4.pkl")),
        "scaler_5": joblib.load(str(MODEL_DIR / "scaler_5.pkl")),
    }


MODELS = _load_models()

decisiontree_classifier_baseline = MODELS["decisiontree_classifier_baseline"]
decisiontree_regressor_optimum = MODELS["decisiontree_regressor_optimum"]
naive_bayes_classifier_optimum = MODELS["naive_bayes_classifier_optimum"]
knn_classifier_optimum = MODELS["knn_classifier_optimum"]
support_vector_classifier_optimum = MODELS["support_vector_classifier_optimum"]
random_forest_classifier_optimum = MODELS["random_forest_classifier_optimum"]

label_encoders_1b = MODELS["label_encoders_1b"]
label_encoders_2 = MODELS["label_encoders_2"]
label_encoders_4 = MODELS["label_encoders_4"]
label_encoders_5 = MODELS["label_encoders_5"]
onehot_encoder_3 = MODELS["onehot_encoder_3"]
scaler_4 = MODELS["scaler_4"]
scaler_5 = MODELS["scaler_5"]

ASSOCIATION_RULES = _load_association_rules(RULES_PATH)


@app.route("/api/v1/models/decision-tree-classifier/predictions", methods=["POST"])
def predict_decision_tree_classifier():
    data, error = _read_json_payload()
    if error:
        return error

    required_fields = ["monthly_fee", "customer_age", "support_calls"]
    missing = [field for field in required_fields if field not in data]
    if missing:
        return _error_response(
            f"Missing required fields: {missing}. Required fields: {required_fields}",
            status_code=400,
        )

    try:
        new_data = pd.DataFrame(
            [
                {
                    "monthly_fee": float(data["monthly_fee"]),
                    "customer_age": float(data["customer_age"]),
                    "support_calls": float(data["support_calls"]),
                }
            ]
        )
    except (TypeError, ValueError):
        return _error_response(
            "monthly_fee, customer_age and support_calls must be numeric values.",
            status_code=400,
        )

    prediction = decisiontree_classifier_baseline.predict(new_data)[0]
    return jsonify({"Predicted Class = ": int(prediction)})


@app.route("/api/v1/models/decision-tree-regressor/predictions", methods=["POST"])
def predict_decision_tree_regressor():
    data, error = _read_json_payload()
    if error:
        return error

    required_fields = [
        "PaymentDate",
        "CustomerType",
        "BranchSubCounty",
        "ProductCategoryName",
        "QuantityOrdered",
    ]
    missing = [field for field in required_fields if field not in data]
    if missing:
        return _error_response(
            f"Missing required fields: {missing}. Required fields: {required_fields}",
            status_code=400,
        )

    try:
        new_data = pd.DataFrame([data])
        new_data["PaymentDate"] = pd.to_datetime(
            new_data["PaymentDate"], errors="raise"
        )
        new_data["QuantityOrdered"] = pd.to_numeric(
            new_data["QuantityOrdered"], errors="raise"
        )

        datetime_columns = new_data.select_dtypes(include=["datetime64"]).columns
        categorical_cols = new_data.select_dtypes(
            exclude=["int64", "float64", "datetime64[ns]"]
        ).columns

        for col in categorical_cols:
            if col in new_data and col in label_encoders_1b:
                new_data[col] = label_encoders_1b[col].transform(new_data[col])

        new_data["PaymentDate_year"] = new_data["PaymentDate"].dt.year
        new_data["PaymentDate_month"] = new_data["PaymentDate"].dt.month
        new_data["PaymentDate_day"] = new_data["PaymentDate"].dt.day
        new_data["PaymentDate_dayofweek"] = new_data["PaymentDate"].dt.dayofweek
        new_data = new_data.drop(columns=datetime_columns)

        expected_features = [
            "CustomerType",
            "BranchSubCounty",
            "ProductCategoryName",
            "QuantityOrdered",
            "PaymentDate_year",
            "PaymentDate_month",
            "PaymentDate_day",
            "PaymentDate_dayofweek",
        ]
        new_data = new_data[expected_features]
    except KeyError as exc:
        return _error_response(
            f"Missing expected column: {str(exc)}",
            status_code=400,
        )
    except (TypeError, ValueError) as exc:
        return _error_response(str(exc), status_code=400)

    prediction = decisiontree_regressor_optimum.predict(new_data)[0]
    return jsonify({"Predicted Percentage Profit per Unit = ": float(prediction)})


@app.route("/api/v1/models/naive-bayes-classifier/predictions", methods=["POST"])
def predict_naive_bayes_classifier():
    data, error = _read_json_payload()
    if error:
        return error

    try:
        new_data = _build_online_shopper_frame(data, label_encoders_2)
    except (TypeError, ValueError) as exc:
        return _error_response(str(exc), status_code=400)

    prediction = naive_bayes_classifier_optimum.predict(new_data)[0]
    return jsonify({"Predicted Class = ": int(prediction)})


@app.route("/api/v1/models/knn-classifier/predictions", methods=["POST"])
def predict_knn_classifier():
    data, error = _read_json_payload()
    if error:
        return error

    try:
        feature_vector = _build_knn_feature_vector(data)
    except (TypeError, ValueError) as exc:
        return _error_response(str(exc), status_code=400)

    prediction = knn_classifier_optimum.predict(feature_vector)[0]
    return jsonify({"Predicted Class = ": int(prediction)})


@app.route(
    "/api/v1/models/support-vector-classifier/predictions", methods=["POST"]
)
def predict_support_vector_classifier():
    data, error = _read_json_payload()
    if error:
        return error

    try:
        new_data = _build_online_shopper_frame(data, label_encoders_4)
        scaled_data = scaler_4.transform(new_data)
    except (TypeError, ValueError) as exc:
        return _error_response(str(exc), status_code=400)

    prediction = support_vector_classifier_optimum.predict(scaled_data)[0]
    return jsonify({"Predicted Class = ": int(prediction)})


@app.route("/api/v1/models/random-forest-classifier/predictions", methods=["POST"])
def predict_random_forest_classifier():
    data, error = _read_json_payload()
    if error:
        return error

    try:
        new_data = _build_online_shopper_frame(data, label_encoders_5)
        scaled_data = scaler_5.transform(new_data)
    except (TypeError, ValueError) as exc:
        return _error_response(str(exc), status_code=400)

    prediction = random_forest_classifier_optimum.predict(scaled_data)[0]
    return jsonify({"Predicted Class = ": int(prediction)})


@app.route("/api/v1/recommenders/association-rules/recommendations", methods=["POST"])
def recommend_from_association_rules():
    data, error = _read_json_payload()
    if error:
        return error

    if not ASSOCIATION_RULES:
        return _error_response(
            (
                "Association rules are not available. Ensure rules file exists at "
                f"'{RULES_PATH}'."
            ),
            status_code=503,
        )

    items = data.get("items")
    if not isinstance(items, list) or not items:
        return _error_response(
            "'items' must be a non-empty list of product names.",
            status_code=400,
        )

    top_n = data.get("top_n", 5)
    try:
        top_n = int(top_n)
        if top_n < 1:
            raise ValueError
    except (TypeError, ValueError):
        return _error_response("'top_n' must be a positive integer.", status_code=400)

    basket = {str(item).strip().lower() for item in items if str(item).strip()}
    if not basket:
        return _error_response(
            "After normalization, 'items' is empty. Provide valid product names.",
            status_code=400,
        )

    scored_recommendations = {}
    for rule in ASSOCIATION_RULES:
        antecedents = set(rule["antecedents"])
        consequents = set(rule["consequents"])
        if not antecedents.issubset(basket):
            continue

        for product in consequents - basket:
            score = float(rule["confidence"]) * float(rule["lift"])
            existing = scored_recommendations.get(product)
            if existing is None or score > existing["score"]:
                scored_recommendations[product] = {
                    "product": product,
                    "score": score,
                    "support": float(rule["support"]),
                    "confidence": float(rule["confidence"]),
                    "lift": float(rule["lift"]),
                    "matched_antecedents": sorted(antecedents),
                }

    ranked = sorted(
        scored_recommendations.values(),
        key=lambda item: (item["score"], item["confidence"], item["lift"]),
        reverse=True,
    )[:top_n]

    for entry in ranked:
        entry["score"] = round(entry["score"], 6)
        entry["support"] = round(entry["support"], 6)
        entry["confidence"] = round(entry["confidence"], 6)
        entry["lift"] = round(entry["lift"], 6)

    return jsonify(
        {
            "input_items": sorted(basket),
            "recommendations": ranked,
            "rules_loaded": len(ASSOCIATION_RULES),
        }
    )


if __name__ == "__main__":
    app.run(debug=True)

