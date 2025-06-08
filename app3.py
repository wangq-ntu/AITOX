import os
import uuid
import numpy as np
import numpy.random._pickle  # Register MT19937, PCG64, etc.
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import pandas as pd
import joblib
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
ALLOWED_EXTENSIONS = {"csv"}

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
    """Render upload / model-select form."""
    return render_template("index.html", models=MODEL_MAP.keys())

@app.route("/predict", methods=["POST"])
def predict():
    # ——— File upload check ———
    if "file" not in request.files:
        return "No file uploaded", 400
    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        return "Please upload a .csv file", 400

    # ——— Save & read CSV ———
    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(path)
    try:
        df = pd.read_csv(path)
    except Exception as e:
        logger.error(f"Error reading CSV: {str(e)}")
        return "Error reading CSV file", 400

    # ——— Check required columns ———
    required = {"Toxicity", "Drugs", "index"}
    if not required.issubset(df.columns):
        return f"CSV must include columns: {required}", 400

    # ——— Prepare features & labels ———
    X_raw = df.drop(columns=list(required))
    y_true_df = df["Toxicity"]

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

    # ——— Scale features and transform labels ———
    try:
        X_scaled = scaler.transform(X_aligned)
        y_true = le.transform(y_true_df)
    except Exception as e:
        logger.error(f"Error in feature scaling or label transformation: {str(e)}")
        return "Error processing features or labels", 400

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

    # ——— Run prediction ———
    try:
        y_pred = model.predict(X_scaled)
        decoded = le.inverse_transform(y_pred)
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return "Error during prediction", 500

    # ——— Compute metrics ———
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="binary")
    rec = recall_score(y_true, y_pred, average="binary")
    f1 = f1_score(y_true, y_pred, average="binary")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    mcc = matthews_corrcoef(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred).tolist()  # Convert to list for rendering
    class_report = classification_report(y_true, y_pred, output_dict=False)  # String format for display

    # ——— Log metrics ———
    logger.info(f"--- {choice} Metrics ---")
    logger.info(f"Accuracy    {acc:.4f}")
    logger.info(f"Precision   {prec:.4f}")
    logger.info(f"Recall      {rec:.4f}")
    logger.info(f"Specificity {spec:.4f}")
    logger.info(f"F1-Score    {f1:.4f}")
    logger.info(f"MCC         {mcc:.4f}")
    logger.info(f"Confusion Matrix:\n{conf_matrix}")
    logger.info(f"Classification Report:\n{class_report}")

    # ——— Build comparison table ———
    comp_df = pd.DataFrame({
        'Drug':df['Drugs'],
        'Concentration':df['Concentration'],
        "True Toxicity": y_true_df,
        "Predicted Label": decoded
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
        model_name=choice,
        n_samples=len(df),
        metrics=metrics,
        comp_table=comp_table_html,
        confusion_matrix=conf_matrix,
        classification_report=class_report
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)