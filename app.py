import os
import uuid
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import pandas as pd
import joblib

# where to store uploaded CSVs
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# mapping form choice → model file
MODEL_MAP = {
    "Linear SVM": "models/Linear_svm_model.pkl",
    "RBF SVM":    "models/RBF_svm_model.pkl",
    "MLP":        "models/mlp_model.pkl",
}

ALLOWED_EXTENSIONS = {"csv"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )

@app.route("/", methods=["GET"])
def index():
    """Render upload / model‐select form."""
    return render_template("index.html", models=MODEL_MAP.keys())


@app.route("/predict", methods=["POST"])
def predict():
    # 1) check file
    if "file" not in request.files:
        return "No file part", 400
    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        return "No selected CSV or wrong extension", 400

    # 2) save it
    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(path)

    # 3) read data
    df = pd.read_csv(path)

    # 4) load the chosen model
    choice = request.form.get("model_choice")
    model_path = MODEL_MAP.get(choice)
    if model_path is None or not os.path.exists(model_path):
        return f"Model '{choice}' not found.", 400
    model = joblib.load(model_path)

    # 5) run prediction
    #    assumes your models support predict_proba
    #    and that the CSV contains exactly the features they expect
    scores = model.predict_proba(df)[:, 1]  # probability for “class 1”
    preds  = model.predict(df)

    # 6) assemble results
    df_result = df.copy()
    df_result["score"]      = scores
    df_result["prediction"] = preds

    # 7) render the results table
    #    to_html(autoescape=True) so Flask will show it safely
    results_html = df_result.to_html(classes="table table-striped", index=False, float_format="%.4f")

    return render_template("results.html",
                           table=results_html,
                           model_name=choice,
                           n_samples=len(df_result))


if __name__ == "__main__":
    # debug=True for hot‐reload during development; switch off in prod
    app.run(host="0.0.0.0", port=5000, debug=True)
