import pandas as pd 
import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
train=pd.read_csv("alldata.csv")
train.drop(columns=["classLabel"],inplace=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    global data , model
    features = [str(x) for x in request.form.values()]
    x=features[0]
    x=x.split(",")
    l = np.array(x)
    np.reshape(l, (1, 21))
    p=pd.DataFrame([l],columns=['variable1','variable2','variable3','variable4','variable5','variable6','variable7','variable8','variable9','variable10','variable11','variable12','variable13','variable14','variable15','variable16','variable17','variable18','variable19','variable20','variable21'])
    p.drop(columns="variable20",inplace=True)
    p.rename(columns={'variable21':'variable20'}, inplace=True)
    print(p.columns)
    print(train.columns)
    data = pd.concat([train, p], axis=0)
    data=data.reset_index(drop=True)
    data["variable2"]=pd.to_numeric(data["variable2"])
    data["variable3"]=pd.to_numeric(data["variable3"])
    data["variable4"]=pd.to_numeric(data["variable4"])
    data["variable5"]=pd.to_numeric(data["variable5"])
    data["variable10"]=pd.to_numeric(data["variable10"])
    data["variable11"]=pd.to_numeric(data["variable11"])
    data["variable14"]=pd.to_numeric(data["variable14"])
    data["variable17"]=pd.to_numeric(data["variable17"])
    data["variable18"]=pd.to_numeric(data["variable18"])
    data["variable19"]=pd.to_numeric(data["variable19"])
    data["variable20"]=pd.to_numeric(data["variable20"])
    data=pd.get_dummies(data)
    print(data.columns)
    n=data.iloc[-1,:]
    n=n.values.reshape(1,47)
    m=model.predict(n)
    if m==1:
        output="yes"
    else:
        output="no"

    return render_template('index.html', prediction_text='predicted class is {}'.format(output))



if __name__ == "__main__":
    app.run(debug=True)