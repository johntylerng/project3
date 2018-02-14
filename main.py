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

#TRAINING_DATA = df.to_dict('records')
#
#TEST_DATA=[{'citations': 99.799999999999997,
#  'country_cat': 28.0,
#  'income': 97.799999999999997,
#  'international outlook': 64.0,
#  'num_students': 2243.0,
#  'research': 97.599999999999994,
#  'revised_world_rank': 1.0,
#  'teaching': 95.599999999999994,
#  'tuition_fee': 47577.0},
# {'citations': 98.799999999999997,
#  'country_cat': 27.0,
#  'income': 73.099999999999994,
#  'international outlook': 94.400000000000006,
#  'num_students': 19919.0,
#  'research': 98.900000000000006,
#  'revised_world_rank': 2.0,
#  'teaching': 86.5,
#  'tuition_fee': 41806.5},
# {'citations': 99.900000000000006,
#  'country_cat': 28.0,
#  'income': 63.299999999999997,
#  'international outlook': 76.299999999999997,
#  'num_students': 15596.0,
#  'research': 96.200000000000003,
#  'revised_world_rank': 3.0,
#  'teaching': 92.5,
#  'tuition_fee': 47940.0},
# {'citations': 97.0,
#  'country_cat': 27.0,
#  'income': 55.0,
#  'international outlook': 91.5,
#  'num_students': 18812.0,
#  'research': 96.700000000000003,
#  'revised_world_rank': 4.0,
#  'teaching': 88.200000000000003,
#  'tuition_fee': 45120.0}]

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

#def train_model():
#    print("Trying train endpoint...")
#    
#    r = requests.post(API_HOST + TRAIN_API,json=TRAINING_DATA)
#    
#    if r.status_code == 200:
#        print(r.text)
#    else:
#        print("Status code indicates a problem:", r.status_code)
#        
#def predict():
#    print("predicting")
#    
#    r= requests.post(API_HOST+PREDICT_API ,json=TEST_DATA)
#    
#    if r.status_code == 200:
#        print(r.text)
#    else:
#        print("Status code indicates a problem:", r.status_code)
##        
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
    #train_model_without_file()
    #train_model()
    #predict()
    

# Entry point for application (i.e. program starts here)
if __name__ == '__main__':
    main()