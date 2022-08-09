import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
model2 = pickle.load(open('model2.pkl', 'rb'))

@app.route('/')
def home():
    return "App is Working"



@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    print(data.values())
    prediction = model.predict(pd.DataFrame([np.array(list(data.values()))]))
    prediction_prob = model.predict_proba(pd.DataFrame([np.array(list(data.values()))]))

    isDefault = int(prediction[0])
    prob = float(prediction_prob[0][0])

   

    ret ={
        'is_deafult': isDefault,
        'safe_factor': prob,
        'safe_amount': calculate_amount(data) if isDefault==1 else data['FINANCE_AMOUNT'],
        'arreas_rentals': get_arreas_rentals(data) if isDefault==1 else -1
    }
 
    return jsonify(ret)

def calculate_amount(data):
    finance_amout=[data['FINANCE_AMOUNT']]
    probs =[1]
    i=0

    while probs[-1]>=0.5:
        curr_amount = finance_amout[-1]-5000
        if(curr_amount< finance_amout[0]/2):
            return curr_amount
        data['FINANCE_AMOUNT'] = curr_amount
        finance_amout.append(curr_amount)
        p = model.predict_proba(pd.DataFrame([np.array(list(data.values()))]))
        probs.append(p[0][1])
        print("Amount: ", curr_amount, "prob: ",p[0][1])
        i+=1
    return finance_amout[-1]


def get_arreas_rentals(data):
    arreas_rentals = int(model2.predict(pd.DataFrame([np.array(list(data.values()))]))*data['NO_OF_RENTAL'])
    return arreas_rentals

 

if __name__ == "__main__":
    app.run(debug=True)