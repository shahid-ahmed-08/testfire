import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify, render_template
application=Flask(__name__)
app=application

ridge_model=pickle.load(open('Ridge_Lasso_Elastic/ridge.pkl','rb'))
standard_scaler=pickle.load(open('Ridge_Lasso_Elastic/scaler.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        pass
    else:
        return render_template('home.html')

if __name__=="__main__":
    app.run(host="0.0.0.0")