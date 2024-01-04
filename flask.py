#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask,render_template, request 
import io
from io import StringIO
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

def feature_engineering(df):
    df.columns = ['AverageRainingDays', 'clonesize', 'AverageOfLowerTRange',
                  'AverageOfUpperTRange', 'honeybee', 'osmia', 'bumbles', 'andrena']

def scaler(df):
    sc = StandardScaler()
    x = df[['AverageRainingDays', 'clonesize', 'AverageOfLowerTRange',
            'AverageOfUpperTRange', 'honeybee', 'osmia', 'bumbles', 'andrena']]
    x = sc.fit_transform(x)
    return x

@app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    f = request.files['data_file']
    if not f:
        return render_template('index.html', prediction_text="No file selected")
    
    # Read the CSV file from the uploaded file
    df = pd.read_csv(f)

    # Assuming feature engineering is necessary
    feature_engineering(df)

    # Scale the features
    x = scaler(df)

    # Load the pre-trained model
    loaded_model = pickle.load(open("linear_model.pkl", 'rb'))

    # Make predictions
    result = loaded_model.predict(x)

    return render_template('index.html', prediction_text="Predicted crop yield is/are: {}".format(result))

if __name__ == "__main__":
    app.run()


# In[ ]:




