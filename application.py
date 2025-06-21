import pickle
import numpy as np
from flask import Flask, request, render_template

# Flask app setup
application = Flask(__name__)
app = application

# Load model and scaler
ridge_model = pickle.load(open('Ridge_Lasso_Elastic/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('Ridge_Lasso_Elastic/scaler.pkl', 'rb'))

# Home page
@app.route('/')
def index():
    return render_template('index.html')

# Prediction page
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        try:
            # Extracting and converting input data from the form
            data = [
                float(request.form['Temperature']),
                float(request.form['RH']),
                float(request.form['Ws']),
                float(request.form['Rain']),
                float(request.form['FFMC']),
                float(request.form['DMC']),
                float(request.form['DC']),
                float(request.form['ISI']),
                float(request.form['BUI']),
                int(request.form['Classes']),
                int(request.form['Region'])
            ]

            # Scale the data and make prediction
            scaled_data = standard_scaler.transform([data])
            prediction = ridge_model.predict(scaled_data)

            return render_template('home.html', result=round(prediction[0], 2))

        except Exception as e:
            return f"An error occurred: {str(e)}"

    else:
        return render_template('home.html')

# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
