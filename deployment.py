import numpy as np 
import pandas as pd 

import sklearn
from sklearn import ensemble

import pickle

import flask
from flask import Flask, request, jsonify, render_template

print('Good to Go!')

# App Name
app = Flask(__name__, template_folder='templates', static_folder='static')


#Load the saved model
def load_model():
    return pickle.load(open('rfmodel.pkl', 'rb'))


# Home Page
@app.route('/')
def index():
    return render_template('index.html')



@app.route('/submission', methods=['POST'])
def submission():
    features = ['Gender', 'Married', 'Dependents',
    		  'Education', 'Self_Employed', 'ApplicantIncome',
    		  'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
    		  'Credit_History', 'Property_Area']

    values = []
    for i in features:
    	try:
    		values.append(int(request.form[i]))
    	except:
    		values.append(int(0))

    model = load_model()

    prediction = model.predict([values])
    print(prediction)
    if prediction[0] == 0:
    	output = 'Rejected'
    elif prediction[0] == 1:
    	output = 'Approved'

    return render_template('result.html', output=output)

if __name__ == "__main__":
	app.run()



