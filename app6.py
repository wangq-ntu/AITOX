import os
import uuid
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, session
from werkzeug.utils import secure_filename
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, matthews_corrcoef, classification_report
)
import logging

# —————— Logging Configuration ——————
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# —————— Configuration ——————
UPLOAD_FOLDER = "Uploads"
ALLOWED_EXTENSIONS = {"xlsx", "csv"}

# Paths to your serialized objects
MODEL_MAP = {
    "Linear SVM": "models/Linear_SVM_model.pkl",
    "RBF SVM":    "models/RBF_SVM_model.pkl",
    "MLP":        "models/mlp_model.pkl",
}
SCALER_PATH = "models/scaler.pkl"
LE_PATH     = "models/le.pkl"
FEATURE_DEF = "models/X_test_a.csv"  # Reference CSV for feature columns

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# —————— Flask setup ——————
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = os.urandom(24)  # For session management

# —————— Load reference feature-list ——————
logger.info("Loading reference feature columns…")
try:
    feat_df = pd.read_csv(FEATURE_DEF)
    FEATURE_COLS = feat_df.columns.tolist()
    logger.info(f"Will use feature columns: {FEATURE_COLS}")
except Exception as e:
    logger.error(f"Error loading feature definition: {str(e)}")
    raise

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET"])
def index():
    """Render upload / model-select form for both inputs."""
    return render_template("index.html", models=MODEL_MAP.keys())

@app.route("/initial_predict", methods=["POST"])
def initial_predict():
    # ——— File upload check for input data ———
    if "file" not in request.files:
        return "No input data file uploaded", 400
    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        return "Please upload a .xlsx or .csv file for input data", 400

    # ——— Save & read input data file ———
    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(path)
    try:
        if filename.endswith(".xlsx"):
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path)
    except Exception as e:
        logger.error(f"Error reading input data file: {str(e)}")
        return "Error reading input data file", 400

    # ——— Check required columns for input data ———
    required = {"Drugs"}
    if not required.issubset(df.columns):
        return f"Input data file must include columns: {required}", 400

    # ——— Prepare features ———
    X_raw = df.drop(columns=list(required))

    # ——— Align to FEATURE_COLS ———
    missing = set(FEATURE_COLS) - set(X_raw.columns)
    if missing:
        return f"Uploaded data is missing these feature columns: {missing}", 400
    X_aligned = X_raw[FEATURE_COLS]

    # ——— Load scaler and label encoder ———
    try:
        scaler = joblib.load(SCALER_PATH)
        le = joblib.load(LE_PATH)
    except Exception as e:
        logger.error(f"Error loading scaler or label encoder: {str(e)}")
        return "Error loading preprocessing objects", 500

    # ——— Scale features ———
    try:
        X_scaled = scaler.transform(X_aligned)
    except Exception as e:
        logger.error(f"Error in feature scaling: {str(e)}")
        return "Error processing features", 400

    # ——— Load the chosen model ———
    choice = request.form.get("model_choice")
    model_path = MODEL_MAP.get(choice)
    if model_path is None or not os.path.exists(model_path):
        return f"Model '{choice}' not found", 400
    try:
        model = joblib.load(model_path)
    except Exception as e:
        logger.error(f"Error loading model {choice}: {str(e)}")
        return f"Error loading model '{choice}'", 500

    # ——— Run prediction (Output 1) ———
    try:
        y_pred = model.predict(X_scaled)
        decoded = le.inverse_transform(y_pred)
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return "Error during prediction", 500

    # ——— Store data for final prediction ———
    session["predictions"] = decoded.tolist()
    session["X_data"] = df[["Drugs", "Concentration"]].to_json()  # Store as JSON to preserve structure
    session["model_choice"] = choice

    # ——— Render predictions (Output 1) ———
    pred_df = pd.DataFrame({
        'Drug': df["Drugs"],
        'Concentration': df["Concentration"],
        'Predicted Toxicity': decoded
    })
    pred_table_html = pred_df.to_html(classes="table table-striped", index=True, float_format="%.4f")

    return render_template(
        "predictions.html",
        model_name=choice,
        pred_table=pred_table_html
    )

@app.route("/final_predict", methods=["POST"])
def final_predict():
    # ——— Retrieve data from session ———
    predictions = session.get("predictions")
    X_data_json = session.get("X_data")
    model_choice = session.get("model_choice")
    if not all([predictions, X_data_json, model_choice]):
        return "Session data missing. Please start over.", 400

    # ——— File upload check for true values ———
    if "true_file" not in request.files:
        return "No true values file uploaded", 400
    true_file = request.files["true_file"]
    if true_file.filename == "" or not allowed_file(true_file.filename):
        return "Please upload a .xlsx or .csv file for true values", 400

    # ——— Save & read true values file ———
    true_filename = secure_filename(true_file.filename)
    true_unique_name = f"{uuid.uuid4().hex}_{true_filename}"
    true_path = os.path.join(app.config["UPLOAD_FOLDER"], true_unique_name)
    true_file.save(true_path)
    try:
        if true_filename.endswith(".xlsx"):
            true_df = pd.read_excel(true_path)
        else:
            true_df = pd.read_csv(true_path)
    except Exception as e:
        logger.error(f"Error reading true values file: {str(e)}")
        return "Error reading true values file", 400

    # ——— Check required column for true values ———
    if "Toxicity" not in true_df.columns:
        return "True values file must include 'Toxicity' column", 400

    # ——— Load label encoder ———
    try:
        le = joblib.load(LE_PATH)
        y_true = le.transform(true_df["Toxicity"])
        y_pred = le.transform(predictions)  # Convert predictions back to numeric
    except Exception as e:
        logger.error(f"Error processing true labels: {str(e)}")
        return "Error processing true labels", 500

    # ——— Compute metrics (Output 2) ———
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="binary")
    rec = recall_score(y_true, y_pred, average="binary")
    f1 = f1_score(y_true, y_pred, average="binary")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    mcc = matthews_corrcoef(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred).tolist()
    class_report = classification_report(y_true, y_pred, output_dict=False)

    # ——— Log metrics ———
    logger.info(f"--- {model_choice} Metrics ---")
    logger.info(f"Accuracy    {acc:.4f}")
    logger.info(f"Precision   {prec:.4f}")
    logger.info(f"Recall      {rec:.4f}")
    logger.info(f"Specificity {spec:.4f}")
    logger.info(f"F1-Score    {f1:.4f}")
    logger.info(f"MCC         {mcc:.4f}")
    logger.info(f"Confusion Matrix:\n{conf_matrix}")
    logger.info(f"Classification Report:\n{class_report}")

    # ——— Reconstruct X_data from JSON ———
    try:
        X_data = pd.read_json(X_data_json)
        logger.info(f"X_data after deserialization:\n{X_data.head().to_string()}")
    except Exception as e:
        logger.error(f"Error reconstructing X_data: {str(e)}")
        return "Error processing stored data", 500

    # ——— Build comparison table (Output 3) ———
    comp_df = pd.DataFrame({
        'Drug': X_data["Drugs"],
        'Concentration': X_data["Concentration"],
        'True Toxicity': true_df["Toxicity"],
        'Predicted Label': predictions
    })
    comp_table_html = comp_df.to_html(classes="table table-striped", index=True, float_format="%.4f")

    # ——— Metrics for rendering ———
    metrics = {
        "Accuracy": f"{acc:.4f}",
        "Precision": f"{prec:.4f}",
        "Recall": f"{rec:.4f}",
        "Specificity": f"{spec:.4f}",
        "F1 Score": f"{f1:.4f}",
        "MCC": f"{mcc:.4f}"
    }

    return render_template(
        "results.html",
        model_name=model_choice,
        n_samples=len(true_df),
        metrics=metrics,
        comp_table=comp_table_html,
        confusion_matrix=conf_matrix,
        classification_report=class_report
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)