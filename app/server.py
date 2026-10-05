from flask import Flask, request, render_template
import pandas as pd
import joblib
import os
import sys
import csv
from datetime import datetime

app = Flask(__name__)
MODEL_PATH = "model/supervised_model.pkl"
LOG_FILE = "logs/predictions.csv"

# Ensure logs folder exists
os.makedirs("logs", exist_ok=True)

# Load model
model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    print(f"Warning: Model file not found at {MODEL_PATH}. Please train and save the model first.")

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        if model is None:
            prediction = "Model not loaded. Please train the model."
        else:
            try:
                amount = float(request.form['amount'])
                hour = int(request.form['hour'])
                dow = int(request.form['day_of_week'])

                input_data = pd.DataFrame([[amount, hour, dow]], columns=['amount', 'hour', 'day_of_week'])
                result = model.predict(input_data)[0]
                prediction = 'Suspicious' if result == 1 else 'Not Suspicious'

                # Log to CSV
                with open(LOG_FILE, "a", newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([datetime.now(), amount, hour, dow, result])
            except Exception as e:
                prediction = f"Error: {e}"
    return render_template('index.html', prediction=prediction)

@app.route('/dashboard')
def dashboard():
    try:
        if not os.path.exists(LOG_FILE):
            return render_template("dashboard.html", total=0, suspicious=0, data=[])

        df = pd.read_csv(LOG_FILE, names=['timestamp', 'amount', 'hour', 'day', 'prediction'])
        suspicious = (df['prediction'] == 1).sum()
        total = len(df)
        recent = df.tail(10).to_dict(orient='records')
        return render_template("dashboard.html", total=total, suspicious=suspicious, data=recent)
    except Exception as e:
        return f"Dashboard error: {e}"

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    except SystemExit:
        sys.stderr.write("Server failed to start due to environment restrictions.\n")
        sys.exit(1)
