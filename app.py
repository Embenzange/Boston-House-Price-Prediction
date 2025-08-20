import os
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

with open('house_price_predictor_mini_project.pkl', 'rb') as f:
    model=pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Extract form data
            features = [
                float(request.form['Unnamed: 0']),
                float(request.form['crim']),
                float(request.form['zn']),
                float(request.form['indus']),
                float(request.form['chas']),
                float(request.form['nox']),
                float(request.form['rm']),
                float(request.form['age']),
                float(request.form['dis']),
                float(request.form['rad']),
                float(request.form['tax']),
                float(request.form['ptratio']),
                float(request.form['black']),
                float(request.form['lstat']),
            ]
            
            # Convert to numpy array and reshape for prediction
            features_array = np.array(features).reshape(1, -1)
            
            # Make prediction
            prediction = model.predict(features_array)
            output = round(prediction[0], 2)
            
            return render_template('index.html', prediction_text=f'Predicted House Price: ${output}')
        except Exception as e:
            return render_template('index.html', prediction_text=f'Error: {str(e)}')
    return render_template('index.html')    

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render sets PORT dynamically
    app.run(host="0.0.0.0", port=port, debug=False)