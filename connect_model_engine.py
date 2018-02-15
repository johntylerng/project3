#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 10:52:27 2018

@author: cheeloongng
"""

import requests
import pickle
import pandas as pd
from sklearn.externals import joblib

API_HOST = 'http://13.59.233.239:5001'

TEST_API = '/test_endpoint'
PREDICT_API = '/predict'
TRAIN_API = '/train_endpoint'
TRAIN_MODEL_NO_FILE_API = '/train_endpoint_without_file'
TRAINING_FILE_PATH = 'data/USvideos.csv'
TRAINING_FILE2_PATH = 'data/US_category_id.json'
UPDATE_MODEL_API = '/update_model'


df= pd.read_csv(TRAINING_FILE_PATH)

TRAINING_DATA = df.to_dict('records')

TEST_DATA=df.sample(10).to_dict('records')

def test_endpoint():
    print('<client>Test connection')
    r = requests.get(API_HOST + TEST_API)
    
    if r.status_code == 200:
        print('connection successful')
        print("request's text>", r.text)
    else:
        print('Problem',r.status_code)

def train_model_without_file():
    print("Trying train endpoint...")
    #for 10b
    r = requests.get(API_HOST + TRAIN_MODEL_NO_FILE_API)
    
    if r.status_code == 200:
        print(r.text)
    else:
        print("Status code indicates a problem:", r.status_code)

def train_model():
    print("Trying train endpoint...sending training data")
    
    r = requests.post(API_HOST + TRAIN_API,json=TRAINING_DATA)
    
    if r.status_code == 200:
        print(r.text)
    else:
        print("Status code indicates a problem:", r.status_code)
        
def predict():
    print("predicting")
    
    r= requests.post(API_HOST+PREDICT_API ,json=TEST_DATA)
    
    if r.status_code == 200:
        print(r.text)
    else:
        print("Status code indicates a problem:", r.status_code)
#        
#def update_model_with_new_data():
#    print("Sending new data to train model...")
#    
#    r = requests.post(API_HOST + UPDATE_MODEL_API,json=NEW_TRAINING_DATA)
#    
#    if r.status_code == 200:
#        print(r.text)
#    else:
#        print("Status code indicates a problem:", r.status_code)



def main():
    test_endpoint()
    train_model_without_file()
#    train_model()
#    predict()
    

# Entry point for application (i.e. program starts here)
if __name__ == '__main__':
    main()