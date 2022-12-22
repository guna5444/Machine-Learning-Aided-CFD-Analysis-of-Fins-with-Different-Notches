from flask import Flask,render_template,request,redirect
from flask_cors import CORS,cross_origin
import pickle
import pandas as pd
import numpy as np

app=Flask(__name__)
cors=CORS(app)
model=pickle.load(open('LinearRegressionModel.pkl','rb'))
fin=pd.read_csv('fin.csv')

@app.route('/',methods=['GET','POST'])
def index():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
@cross_origin()
def predict():

    h=int(request.form.get('h'))

    p=int(request.form.get('p'))
    k=int(request.form.get('k'))
    a=int(request.form.get('a'))

    prediction=model.predict(pd.DataFrame(columns=['h', 'p', 'k', 'a'],
                              data=np.array([h,p,k,a]).reshape(1, 4)))
    print(prediction)

    return str(np.round(prediction[0],2))



if __name__=='__main__':
    app.run()