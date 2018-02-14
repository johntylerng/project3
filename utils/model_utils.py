#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 10:08:29 2018

@author: cheeloongng
"""
import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier



MODEL_DIRECTORY = 'model'
MODEL_FILE_NAME = '%s/model.pkl' %(MODEL_DIRECTORY)
DATA_FILE_PATH = 'data/'
MODEL_COLUMNS_FILE_NAME = '%s/model_columns.pkl' % MODEL_DIRECTORY

total_vote_average=0
average_rating =0



def train(data):
    print("Training data sample:\n", data.head(2))
    
    data['trending_date']= pd.to_datetime(data.trending_date,format='%y.%d.%m')
    data.publish_time = pd.to_datetime(data.publish_time, \
                                       format='%Y-%m-%dT%H:%M:%S.%fZ')
    data.rename(columns={'publish_time':'published_time'}, inplace=True)
    data.insert(4,'published_date',data.published_time.dt.date)
    data.published_time = data.published_time.dt.time
    
    id_to_category={}
    category_info = pd.read_json(DATA_FILE_PATH +'US_category_id.json')
    for category in category_info['items']:
        id_to_category[pd.to_numeric(category['id'])]=category['snippet']['title']
    data.insert(4,'category',data['category_id'].map(id_to_category))
    

    data['total_vote'] = data['likes']+data['dislikes']
    data['rating']=data['likes'] - data['dislikes']

    total_vote_average = data['total_vote'].mean()
    average_rating = data['rating'].mean()
    data['weighted_rating'] = data.apply(compute_weighted_rating,axis=1)
    
    data['video_bins'] = pd.qcut(data['weighted_rating'],
                                 q=3,
                                 labels=["below","good","excellent"])
    
    data['tags'] = data['tags'].apply(remove_punctuation)
    data['title']= data['title'].apply(remove_punctuation)
    print('title',data.title.sample(2))



    count_vectorizer = CountVectorizer()

    cv = count_vectorizer.fit_transform(data['tags'])
    new_df = pd.DataFrame(cv.toarray(), columns=count_vectorizer.get_feature_names())
    data = pd.concat([data,new_df], axis=1)

    cv = count_vectorizer.fit_transform(data['title'])
    title_df = pd.DataFrame(cv.toarray(), columns=count_vectorizer.get_feature_names())
    data = pd.concat([data,title_df], axis=1)
    
    X = data.drop(['video_id','title','channel_title','trending_date', 'category', 'published_date',\
                  'published_time','thumbnail_link','comments_disabled',\
                  'ratings_disabled','video_error_or_removed','description',\
               'total_vote','rating','weighted_rating','video_bins','tags'],axis=1)

    features_name = X.columns
    print('features in X:',X.columns)
    X = np.array(X)
    y = data['video_bins']
    y = np.array(y)
    np.random.seed(42)
    
    #split data into 60%, 20%, 20%
    X_train_validate, X_test,y_train_validate,\
    y_test= train_test_split(X,y,test_size=0.20,random_state=0)

    X_train, X_validate, y_train, y_validate = \
    train_test_split(X_train_validate,y_train_validate,\
                     test_size=0.25, random_state=0)

    
    


    X,y,features_name = transform(raw_data)
    
    np.random.seed(42)
    shuffle_index = np.random.permutation(raw_data.shape[0])
    X, y= X[shuffle_index],y[shuffle_index]
    X_train_validate, X_test,y_train_validate, y_test= train_test_split(X,y,test_size=0.20,\
                                                                  random_state=0)

    X_train, X_validate, y_train, y_validate = train_test_split(X_train_validate,y_train_validate,\
                                                            test_size=0.25, random_state=0)
    
    #model = DecisionTreeClassifier(min_samples_leaf=5)
    model = RandomForestClassifier(max_features='auto', \
                                   min_samples_leaf=20, n_estimators=10)
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

def compute_weighted_rating(row):
    num = (total_vote_average*average_rating)+ (row['total_vote']*row['rating'])
    return num /(total_vote_average+(row['total_vote']))

def remove_punctuation(row):
    return re.sub('[(,|\"&)$@#]'," ",row)
