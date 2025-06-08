import os
import uuid
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, session, redirect, url_for, flash
from werkzeug.utils import secure_filename
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, matthews_corrcoef, classification_report
)
import logging
import re

# —————— Logging Configuration ——————
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# —————— Configuration ——————
UPLOAD_FOLDER = "Uploads"
ALLOWED_EXTENSIONS = {"xls", "csv"}

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Paths to your serialized objects
MODEL_MAP = {
    "MLP": "models/mlp_model_1604.pkl",  # Matches Python script
}
SCALER_PATH = "models/scaler.pkl"
LE_PATH = "models/le.pkl"
FEATURE_DEF = "models/X_test_a.csv"  # Ensure this matches /content/drive/MyDrive/sfa/27Mar25/X_test_a.csv

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this!

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load scaler and label encoder once
scaler = joblib.load(SCALER_PATH)
le = joblib.load(LE_PATH)

# Load feature definition columns
X_test_a = pd.read_csv(FEATURE_DEF)
logger.info(f"X_test_a columns: {list(X_test_a.columns)}")

# Load all models upfront
models = {}
for key, path in MODEL_MAP.items():
    try:
        models[key] = joblib.load(path)
        logger.info(f"Loaded model '{key}' from {path}")
    except Exception as e:
        logger.error(f"Failed to load model {key} from {path}: {e}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_gene_file(filepath):
    required_columns = ['log2FoldChange', 'gene_name']
    try:
        df = pd.read_csv(filepath, sep='\t')
    except Exception as e:
        logger.error(f"Error reading file {filepath}: {e}")
        return None, f"Failed to read the uploaded file: {e}"

    if not all(col in df.columns for col in required_columns):
        return None, f"File must contain columns: {required_columns}"

    gene_df = df[required_columns].copy()
    logger.info(f"Input file columns: {list(gene_df.columns)}")
    
    # Use original filename without UUID and derive exact prefix
    original_filename = os.path.basename(filepath).split('_', 1)[-1]  # Remove UUID prefix
    # Robustly extract subfolder_name to match notebook's directory (e.g., HEPARGH100vsHEPARGCO)
    subfolder_name = re.sub(r'_deg\.xls$', '', original_filename)  # Remove '_deg.xls'
    subfolder_name = re.sub(r'_deg$', '', subfolder_name)  # Remove '_deg' if present
    logger.info(f"Derived subfolder_name: {subfolder_name}")

    concentration = 0
    if '100' in original_filename:
        concentration = 65.1940
    elif '125' in original_filename:
        concentration = 81.4925
    elif '50' in original_filename:
        concentration = 32.5970
    gene_df['Concentration'] = concentration

    tokens_to_isolate = ["RG", "G2"]
    prefix = None
    for token in tokens_to_isolate:
        if token in subfolder_name:
            match = re.search(r'H(\d+)', subfolder_name)
            if match:
                number = match.group(1)
                prefix = f"{token}_H_{number}"
                break
    if not prefix:
        prefix = subfolder_name
    logger.info(f"Using prefix: {prefix}")

    old_cols = list(gene_df.columns)
    gene_df.columns = [f'{col}' if col != 'Concentration' else col for col in old_cols]
    logger.info(f"Renamed columns: {list(gene_df.columns)}")

    return gene_df, None

def prepare_features(gene_df):
    gene_df_indexed = gene_df.set_index('gene_name')

# Now transpose
    gene_df2 = gene_df_indexed.T
    #logger.info(f"gene_df2 columns: {list(gene_df2.columns)}")
    #gene_df2.columns = gene_df2.iloc[0]
    #gene_df2 = gene_df2[1:]
    concentration_value = gene_df['Concentration'].iloc[0]
    gene_df2.insert(0, "Concentration", concentration_value)
    gene_df3 = gene_df2[:-1]
    #gene_df3 = gene_df2.drop("Concentration", axis=0)
    logger.info(f"gene_df3 columns: {list(gene_df3.columns)}")

    Xv_test = gene_df3.copy()  
    Xv_test = Xv_test.loc[:, ~Xv_test.columns.duplicated()]
    
    # Create a DataFrame for missing columns with zeros
    missing_cols = [col for col in X_test_a.columns if col not in Xv_test.columns]
    logger.info(f"Missing columns: {missing_cols}")
    if missing_cols:
        # Create a DataFrame with zeros for missing columns
        missing_data = pd.DataFrame(0, index=Xv_test.index, columns=missing_cols)
        # Concatenate Xv_test with missing columns
        Xv_test = pd.concat([Xv_test, missing_data], axis=1)
    
    # Create a de-fragmented copy
    Xv_test = Xv_test.copy()
    # Align columns exactly as model expects
    try:
        Xv_test_a = Xv_test[X_test_a.columns]
    except Exception as e:
        logger.error(f"Error aligning features: {e}")
        return None, f"Error aligning columns: {e}"

    # Check for columns with all zero values
    zero_columns = [col for col in Xv_test_a.columns if (Xv_test_a[col] == 0).all()]
    logger.info(f"Columns with all zeros in Xv_test_a: {zero_columns}")
    logger.info(f"Number of zero columns: {len(zero_columns)}")
    logger.info(f"Final feature columns: {list(Xv_test_a.columns)}")

    return Xv_test_a, None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check file upload
        if 'file' not in request.files:
            flash("No file part in the request")
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash("No file selected")
            return redirect(request.url)
        if not allowed_file(file.filename):
            flash(f"File extension not allowed. Allowed: {ALLOWED_EXTENSIONS}")
            return redirect(request.url)

        # Save uploaded file securely
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        logger.info(f"File saved to {file_path}")

        gene_df, error = process_gene_file(file_path)
        if error:
            flash(error)
            return redirect(request.url)

        Xv_test_a, error = prepare_features(gene_df)
        if error:
            flash(error)
            return redirect(request.url)

        Xv_test_scaled = scaler.transform(Xv_test_a)
        logger.info(f"Scaled features shape: {Xv_test_scaled.shape}")
        logger.info(f"Scaled features sample: {Xv_test_scaled.flatten()[:10]}")  # Log first 10 values

        # Model choice - currently default to MLP
        model_choice = request.form.get('model_choice', 'MLP')
        if model_choice not in models:
            flash(f"Model '{model_choice}' not found, using MLP")
            model_choice = 'MLP'
        model = models[model_choice]

        y_pred = model.predict(Xv_test_scaled)
        logger.info(f"Raw predictions: {y_pred}")
        decoded = le.inverse_transform(y_pred)
        logger.info(f"Decoded predictions: {decoded}")
        # Capitalize predictions for consistency
        decoded = np.array([item.capitalize() for item in decoded])
        logger.info(f"Capitalized predictions: {decoded}")

        results_df = pd.DataFrame({'Prediction': decoded})

        return render_template('predictions.html',
                               model_name=model_choice,
                               pred_table=results_df.to_html(classes='table table-striped table-bordered', index=False))

    models_list = list(models.keys())
    return render_template('index.html', models=models_list)

if __name__ == '__main__':
    app.run(debug=True)