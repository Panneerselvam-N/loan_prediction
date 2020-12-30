#import numpy as np
import pandas as pd 
import joblib
import json
import numpy as np
from flask import Flask,render_template,url_for,request,redirect

# Load column names from json file
json_loc= open('features_ass3.json',)

features=json.load(json_loc)
columns=np.array(features["data_columns"])

#load model
model=joblib.load("model_ass3.pkl")
x=0
def prediction(dip,income,coincome,amount,history,area,graduate,self_emplyed):
    
    area_ind= np.where(columns==area)[0][0]
    graduate_ind=np.where(columns==graduate)[0][0]
    self_emplyed_ind = np.where(columns==self_emplyed)[0][0]

    x=np.zeros((12))
    
    x[0]=dip
    x[1]=income
    x[2]=coincome
    x[3]=amount
    x[4]=history

    if area_ind >=0:
        x[area_ind]=1
    if graduate_ind >= 0:
        x[graduate_ind]=1
    if self_emplyed_ind >=0:
        x[self_emplyed_ind]=1
    

    ans=model.predict([x])[0]
    return ans

#prediction(0,6000,2250.0,265.0,0.0,'property_area_semiurban','education_graduate','self_employed_no')

app=Flask(__name__)

@app.route('/',methods=['GET','POST'])
def upload():
    if request.method =='POST':
        dip = request.form['dependents']
        income = request.form['applicantincome']
        coincome = request.form['coapplicantincome']
        amount = request.form['loanamount']
        history = request.form['credit_history']
        area= request.form['area']
        graduate = request.form['education']
        self_employed = request.form['self_emp']

        result = prediction(dip,income,coincome,amount,history,area,graduate,self_employed)
        if result== 1:
            return render_template('index.html',result="Approved")
        else:
            return render_template('index.html',result="Rejected")

    

    return render_template('index.html')

if __name__== "__main__":
    app.run(debug=True)
