#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 10:08:29 2018

@author: cheeloongng
"""
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier

MODEL_DIRECTORY = 'model'
MODEL_FILE_NAME = '%s/model.pkl' %(MODEL_DIRECTORY)
DATA_FILE_PATH = 'data/'
MODEL_COLUMNS_FILE_NAME = '%s/model_columns.pkl' % MODEL_DIRECTORY

uk_mean_tf=0
us_mean_tf=0
uk_mean_ms=0
us_mean_ms=0


def train(raw_data):
    print("Training data sample:\n", raw_data.head(2))
#    raw_data['revised_world_rank'] = raw_data.apply(extract_hyphen,axis=1)
#    for feature in ['income','total_score','num_students',\
#                    'international_students','student_staff_ratio']:
#        raw_data[feature].fillna(raw_data[feature].mean(), inplace=True )
#    for feature in ['female_male_ratio','tuition_fee','median_salary']:
#        raw_data[feature].fillna(0, inplace=True )
#        
#    filtered_data_uk = raw_data[raw_data.country == 'United Kingdom']
#    uk_mean_tf= filtered_data_uk['tuition_fee'].mean()
#    filtered_data_us = raw_data[raw_data.country == 'United States of America']
#    us_mean_tf= filtered_data_us['tuition_fee'].mean()
#    raw_data['tuition_fee'] = raw_data.apply(assign_missing_values_tuition,axis=1)
#    
#    filtered_data2_uk = raw_data[raw_data.country == 'United Kingdom']
#    uk_mean_ms= filtered_data2_uk['median_salary'].mean()
#    filtered_data2_us = raw_data[raw_data.country == 'United States of America']
#    us_mean_ms= filtered_data2_us['median_salary'].mean()
#    raw_data['median_salary'] = raw_data.apply(assign_missing_values_salary,axis=1)
#    
##    print('us_mean_ms>',us_mean_ms)
##    print('uk_mean_ms>',uk_mean_ms)
##    print('us_mean_tf>',us_mean_tf)
##    print('uk_mean_tf>',uk_mean_tf)
#    
##    print(raw_data.head(2))
#    
##    weight_bins = [12213.999, 30495.769, 38787.259,197400.0]
##    group_names = ['low', 'good', 'excellent']
##    raw_data['salary_bins'] = pd.cut(raw_data['median_salary'],weight_bins,\
##                                  labels=group_names)
#    
#    raw_data['salary_bins'] = raw_data.apply(assign_salary_band,axis=1)
#    raw_data.country = raw_data.country.astype('category')
#    raw_data['country_cat'] = raw_data['country'].cat.codes
#    
#    print('empty>',raw_data.isnull().any())
    
    
#    X = raw_data.drop(['world_rank','university_name', 'country', 'total_score',\
#                  'student_staff_ratio','international_students','female_male_ratio',\
#                  'year','median_salary','salary_bins'], axis=1)
    
#    features_name = X.columns
#    print('features in X:',X.columns)
#    X = np.array(X)
#    y = raw_data.salary_bins
#    y = np.array(y)
#    
#    
#    print(X[:3,:])
#    print(y[:20])

    X,y,features_name = transform(raw_data)
    
    np.random.seed(42)
    shuffle_index = np.random.permutation(raw_data.shape[0])
    X, y= X[shuffle_index],y[shuffle_index]
    X_train_validate, X_test,y_train_validate, y_test= train_test_split(X,y,test_size=0.20,\
                                                                  random_state=0)

    X_train, X_validate, y_train, y_validate = train_test_split(X_train_validate,y_train_validate,\
                                                            test_size=0.25, random_state=0)
    
    #model = DecisionTreeClassifier(min_samples_leaf=5)
    model = SGDClassifier()
    model.fit(X_train,y_train)
    start = time.time() 
    model.fit(X_train, y_train)
    model_columns = list(features_name)
    
    print('Trained in %.1f seconds' % (time.time() - start))
    print('Model  validation score: %s' % model.score(X_validate, y_validate))
    
    return model_columns, model

def predict(df,model):

    #print(df)
    predictions = model.predict(df).tolist()
    #predictions = [int(prediction) for prediction in predictions]

    return {'predictions': predictions}
    
#
#def update(df, model):
#    print('enter update')
#    X,y,features_name = transform(raw_data)
#    
#    np.random.seed(42)
#    model.partialfit(X,y)
#    
#    return model


def extract_hyphen(row):
    if row['world_rank'].isdigit():
        return int(float(row['world_rank']))
    else:
        value_list = row['world_rank'].replace('-', ' ').split(' ')
        value_list = list(map(int, value_list))
        return int(np.mean(value_list))
            

def assign_missing_values_tuition(row):
    global uk_mean_tf, us_mean_tf
    if row['tuition_fee']>0:
        return row['tuition_fee']
    if (row['country'] == 'United Kingdom') and (row['tuition_fee']==0):
        return uk_mean_tf
    elif (row['country'] == 'United States of America') and (row['tuition_fee']==0):
        return us_mean_tf
    elif (row['country'] == 'Canada') and (row['tuition_fee']==0):
        return us_mean_tf
    else:
        return uk_mean_tf
    
def assign_missing_values_salary(row):
    global uk_mean_ms, us_mean_ms
    if row['median_salary']>0:
        return row['median_salary']
    if (row['country'] == 'United Kingdom') and (row['median_salary']==0):
        return uk_mean_ms
    elif (row['country'] == 'United States of America') and (row['median_salary']==0):
        return us_mean_ms
    elif (row['country'] == 'Canada') and (row['median_salary']==0):
        return us_mean_ms
    else:
        return uk_mean_ms
    
def assign_salary_band(row):
    if row['median_salary'] < 30495.769:
        return 'low'
    elif row['median_salary'] < 38787.259:
        return 'good'
    else:
        return 'excellent'
    

def transform(raw_data):
    raw_data['revised_world_rank'] = raw_data.apply(extract_hyphen,axis=1)
    for feature in ['income','total_score','num_students',\
                    'international_students','student_staff_ratio']:
        raw_data[feature].fillna(raw_data[feature].mean(), inplace=True )
    for feature in ['female_male_ratio','tuition_fee','median_salary']:
        raw_data[feature].fillna(0, inplace=True )
        
    filtered_data_uk = raw_data[raw_data.country == 'United Kingdom']
    uk_mean_tf= filtered_data_uk['tuition_fee'].mean()
    filtered_data_us = raw_data[raw_data.country == 'United States of America']
    us_mean_tf= filtered_data_us['tuition_fee'].mean()
    raw_data['tuition_fee'] = raw_data.apply(assign_missing_values_tuition,axis=1)
    
    filtered_data2_uk = raw_data[raw_data.country == 'United Kingdom']
    uk_mean_ms= filtered_data2_uk['median_salary'].mean()
    filtered_data2_us = raw_data[raw_data.country == 'United States of America']
    us_mean_ms= filtered_data2_us['median_salary'].mean()
    raw_data['median_salary'] = raw_data.apply(assign_missing_values_salary,axis=1)
    
#    print('us_mean_ms>',us_mean_ms)
#    print('uk_mean_ms>',uk_mean_ms)
#    print('us_mean_tf>',us_mean_tf)
#    print('uk_mean_tf>',uk_mean_tf)
    
#    print(raw_data.head(2))
    
#    weight_bins = [12213.999, 30495.769, 38787.259,197400.0]
#    group_names = ['low', 'good', 'excellent']
#    raw_data['salary_bins'] = pd.cut(raw_data['median_salary'],weight_bins,\
#                                  labels=group_names)
    
    raw_data['salary_bins'] = raw_data.apply(assign_salary_band,axis=1)
    raw_data.country = raw_data.country.astype('category')
    raw_data['country_cat'] = raw_data['country'].cat.codes
    
    X = raw_data.drop(['world_rank','university_name', 'country', 'total_score',\
                  'student_staff_ratio','international_students','female_male_ratio',\
                  'year','median_salary','salary_bins'], axis=1)
    features_name = X.columns
    #print('features in X:',X.columns)
    X = np.array(X)
    y = raw_data.salary_bins
    y = np.array(y)
    
    
    print(X[:3,:])
    print(y[:20])
    
    return X, y, features_name

