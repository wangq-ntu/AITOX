import os
import uuid
import numpy as np
import numpy.random._pickle   # register MT19937, PCG64, etc.
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import pandas as pd

import numpy as np


from numpy.ma.core import MAError

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans
from sklearn import datasets, decomposition
from sklearn.manifold import TSNE
from sklearn.feature_selection import RFE, VarianceThreshold
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingClassifier, ExtraTreesClassifier, BaggingClassifier, AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score




import pandas as pd
import joblib               


from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report,
    confusion_matrix, matthews_corrcoef
)

# —————— Configuration ——————
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"csv"}

# Paths to your serialized objects
MODEL_PATHS = {
    "Linear SVM": "models/Linear_SVM_model.pkl",
    "RBF SVM":    "models/RBF_SVM_model.pkl",
    "MLP":        "models/mlp_model.pkl",
}
SCALER_PATH = "models/scaler.pkl"
LE_PATH     = "models/le.pkl"
FEATURE_DEF = "models/X_test_a.csv"  # your reference CSV of feature columns

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# —————— Flask setup ——————
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# —————— Load models & preprocessing ——————
print("Loading models and preprocessing objects…")
Linear_svm_model = joblib.load(MODEL_PATHS["Linear SVM"])
RBF_svm_model    = joblib.load(MODEL_PATHS["RBF SVM"])
mlp_model        = joblib.load(MODEL_PATHS["MLP"])
scaler           = joblib.load(SCALER_PATH)
le               = joblib.load(LE_PATH)

# Load the reference feature‐list
feat_df      = pd.read_csv(FEATURE_DEF)
FEATURE_COLS = feat_df.columns.tolist()
print(f"Will use feature columns: {FEATURE_COLS}")
print("…done.")


def allowed_file(fname):
    return "." in fname and fname.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", models=MODEL_PATHS.keys())


@app.route("/predict", methods=["POST"])
def predict():
    # ——— File upload check ———
    if "file" not in request.files:
        return "No file uploaded", 400
    f = request.files["file"]
    if f.filename == "" or not allowed_file(f.filename):
        return "Please upload a .csv file", 400

    # ——— Save & read CSV ———
    filename = secure_filename(f.filename)
    uniq = f"{uuid.uuid4().hex}_{filename}"
    path = os.path.join(app.config["UPLOAD_FOLDER"], uniq)
    f.save(path)
    df = pd.read_csv(path)

    # ——— Check required columns ———
    required = {"Toxicity","Drugs","index"}
    if not required.issubset(df.columns):
        return f"CSV must include columns: {required}", 400

    # ——— Prepare features & labels ———
    X_raw     = df.drop(columns=list(required))
    y_true_df = df["Toxicity"]
    y_true    = le.transform(y_true_df)

    # ——— Align to FEATURE_COLS ———
    missing = set(FEATURE_COLS) - set(X_raw.columns)
    if missing:
        return f"Uploaded data is missing these feature columns: {missing}", 400
    X_aligned = X_raw[FEATURE_COLS]

    # ——— Scale ———
    X_scaled = scaler.transform(X_aligned)

    # ——— Select & apply model ———
    choice = request.form.get("model_choice")
    if choice not in MODEL_PATHS:
        return f"Unknown model: {choice}", 400
    model = {
        "Linear SVM": Linear_svm_model,
        "RBF SVM":    RBF_svm_model,
        "MLP":        mlp_model
    }[choice]

    y_pred = model.predict(X_scaled)
    decoded = le.inverse_transform(y_pred)

    # ——— Compute metrics ———
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="binary")
    rec  = recall_score(y_true, y_pred, average="binary")
    f1   = f1_score(y_true, y_pred, average="binary")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn / (tn + fp) if (tn+fp)>0 else 0.0
    mcc  = matthews_corrcoef(y_true, y_pred)

    # ——— Log to console ———
    print(f"--- {choice} Metrics ---")
    print(f"Accuracy    {acc:.4f}")
    print(f"Precision   {prec:.4f}")
    print(f"Recall      {rec:.4f}")
    print(f"Specificity {spec:.4f}")
    print(f"F1‑Score    {f1:.4f}")
    print(f"MCC         {mcc:.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred))

    # ——— Build comparison & result tables ———
    comp_df = pd.DataFrame({
        "True Toxicity":    y_true_df,
        "Predicted Label":  decoded
    })

    # HTML for metrics & table
    metrics = {
        "Accuracy":    f"{acc:.4f}",
        "Precision":   f"{prec:.4f}",
        "Recall":      f"{rec:.4f}",
        "Specificity": f"{spec:.4f}",
        "F1 Score":    f"{f1:.4f}",
        "MCC":         f"{mcc:.4f}"
    }
    comp_table_html = comp_df.to_html(classes="table table-bordered", index=True)

    return render_template(
        "results.html",
        model_name=choice,
        n_samples=len(df),
        metrics=metrics,
        comp_table=comp_table_html
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
