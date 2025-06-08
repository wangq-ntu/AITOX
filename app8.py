from flask import Flask, render_template, request, redirect, url_for, flash
import os
import pandas as pd
import re
import joblib

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management

# Load models and scaler on startup
mlp_model = joblib.load('models/mlp_model.pkl')
Linear_svm_model = joblib.load('models/Linear_SVM_model.pkl')
RBF_svm_model = joblib.load('models/RBF_SVM_model.pkl')
scaler = joblib.load('models/scaler.pkl')
le = joblib.load('models/le.pkl')

X_test_a = pd.read_csv('model/X_test_a.csv')  # feature columns reference

def process_gene_file(uploaded_file):
    required_columns = ['log2FoldChange', 'gene_name']
    try:
        df = pd.read_csv(uploaded_file, sep='\t')
    except Exception as e:
        return None, f"Failed to read the uploaded file: {e}"

    if not all(col in df.columns for col in required_columns):
        return None, f"File must contain columns: {required_columns}"

    gene_df = df[required_columns].copy()
    subfolder_name = os.path.splitext(uploaded_file.filename)[0]

    concentration = 0
    if '100' in uploaded_file.filename:
        concentration = 65.1940
    elif '125' in uploaded_file.filename:
        concentration = 81.4925
    elif '50' in uploaded_file.filename:
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

    old_cols = list(gene_df.columns)
    gene_df.columns = [f'{prefix}_{col}' if col != 'Concentration' else col for col in old_cols]

    return gene_df, None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part in request.')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected.')
            return redirect(request.url)

        gene_df, error = process_gene_file(file)
        if error:
            flash(error)
            return redirect(request.url)

        gene_df2 = gene_df.T
        gene_df2.columns = gene_df2.iloc[0]
        gene_df2 = gene_df2[1:]
        concentration_value = gene_df['Concentration'].iloc[0]
        gene_df2.insert(0, "Concentration", concentration_value)
        gene_df3 = gene_df2.drop("Concentration", axis=0)

        Xv_test = gene_df3.copy()

        new_columns = ['AL353704.1', 'ASS1P2', 'MT-TQ', 'AC090579.1']
        for col in new_columns:
            if col not in Xv_test.columns:
                Xv_test[col] = 0

        try:
            Xv_test_a = Xv_test[X_test_a.columns]
        except Exception as e:
            flash(f"Error aligning columns: {e}")
            return redirect(request.url)

        Xv_test_scaled = scaler.transform(Xv_test_a)

        # Here, only MLP model is used; extend to other models if needed
        yv_pred = mlp_model.predict(Xv_test_scaled)
        decoded = le.inverse_transform(yv_pred)

        results_df = pd.DataFrame({'Prediction': decoded})

        return render_template('predictions.html',
                               model_name='MLP Classifier',
                               pred_table=results_df.to_html(classes='table table-striped table-bordered', index=False))

    models = ['MLP Classifier', 'Linear SVM', 'RBF SVM']
    return render_template('index.html', models=models)


if __name__ == '__main__':
    app.run(debug=True)
