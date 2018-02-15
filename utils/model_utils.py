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
from scipy import sparse
from sklearn.externals import joblib




MODEL_DIRECTORY = 'model'
MODEL_FILE_NAME = '%s/model.pkl' %(MODEL_DIRECTORY)
DATA_FILE_PATH = 'data/'
MODEL_COLUMNS_FILE_NAME = '%s/model_columns.pkl' % MODEL_DIRECTORY

total_vote_average=0
average_rating =0




def train(data):
#    global total_vote_average, average_rating
##    print("Training data sample:\n", data.head(2))
#    
#    data['trending_date']= pd.to_datetime(data.trending_date,format='%y.%d.%m')
#    data.publish_time = pd.to_datetime(data.publish_time, \
#                                       format='%Y-%m-%dT%H:%M:%S.%fZ')
#    data.rename(columns={'publish_time':'published_time'}, inplace=True)
#    data.insert(4,'published_date',data.published_time.dt.date)
#    data.published_time = data.published_time.dt.time
#    
#    id_to_category={}
#    category_info = pd.read_json(DATA_FILE_PATH +'US_category_id.json')
#    for category in category_info['items']:
#        id_to_category[pd.to_numeric(category['id'])]=category['snippet']['title']
#    data.insert(4,'category',data['category_id'].map(id_to_category))
#    
#
#    data['total_vote'] = data['likes']+data['dislikes']
#    data['rating']=data['likes'] - data['dislikes']
#
#    total_vote_average = data['total_vote'].mean()
#    average_rating = data['rating'].mean()
##    print('total vote average',total_vote_average)
##    print('average rating', average_rating)
#    
#    data['weighted_rating'] = data.apply(compute_weighted_rating,axis=1)
#    
##    print((data[['weighted_rating']].head(2)))
#    data['video_bins'] = data.apply(assign_category_band,axis=1)
#    
#    data['tags'] = data['tags'].apply(remove_punctuation)
#    data['title']= data['title'].apply(remove_punctuation)
##    print('title',data.title.sample(2))
#    
#    
#    X = data.drop(['video_id','title','channel_title','trending_date', 'category', 'published_date',\
#                  'published_time','thumbnail_link','comments_disabled',\
#                  'ratings_disabled','video_error_or_removed','description',\
#               'total_vote','rating','weighted_rating','video_bins','tags'],axis=1)
#
#
#
#    count_vectorizer = CountVectorizer(stop_words='english')
#    
#
#    word_count_tag=count_vectorizer.fit_transform(data['tags'])
##    new_df = pd.DataFrame(cv.toarray(), columns=count_vectorizer.get_feature_names())
##    data = pd.concat([data,new_df], axis=1)
#    
#    
##    num_feats = count_vectorizer.get_feature_names()
##    print(num_feats[:10])
#    
#    Z = np.array(X)
#    Z = sparse.hstack((word_count_tag, sparse.csr_matrix(X)))
#    
#    word_count_title=count_vectorizer.fit_transform(data['title'])
#    Z= sparse.hstack((word_count_title,Z))
#    
#    
##    data = pd.concat([data,new_df], axis=1)
##    cv = count_vectorizer.fit_transform(data['title'])
##    title_df = pd.DataFrame(cv.toarray(), columns=count_vectorizer.get_feature_names())
##    data = pd.concat([data,title_df], axis=1)
#    
##    X = data.drop(['video_id','title','channel_title','trending_date', 'category', 'published_date',\
##                  'published_time','thumbnail_link','comments_disabled',\
##                  'ratings_disabled','video_error_or_removed','description',\
##               'total_vote','rating','weighted_rating','video_bins','tags'],axis=1)
#    features_name = list(X.columns)
##    print('X feature name',features_name)
#    for i in count_vectorizer.get_feature_names():
#        features_name.append(i.encode('utf-8'))
#    
##    X= np.array(Z)
#    
##    features_name.append(count_vectorizer.get_feature_names)
##    print('features in X:',X.columns)
##    X = np.array(X)
#    y = data['video_bins']
#    y = np.array(y)
    
    Z, y, features_name = cleaning_data(data)
    np.random.seed(42)
#    
#    split data into 60%, 20%, 20%
    X_train_validate, X_test,y_train_validate,\
    y_test= train_test_split(Z,y,test_size=0.20,random_state=0)
#
    X_train, X_validate, y_train, y_validate = \
    train_test_split(X_train_validate,y_train_validate,\
                     test_size=0.25, random_state=0)
#
#    
#    #model = DecisionTreeClassifier(min_samples_leaf=5)
    model = RandomForestClassifier(max_features='auto', \
                                   min_samples_leaf=20, n_estimators=10)
    start = time.time() 
    model.fit(X_train, y_train)
    model_columns = list(features_name)
     
    print('Trained in %.1f seconds' % (time.time() - start))
    print('Model  validation score: %s' % model.score(X_validate, y_validate))
    
    
    return model_columns, model

def predict(data,model):

    model_columns = joblib.load(MODEL_COLUMNS_FILE_NAME)
    print(model_columns)
    
#    Z, y, test_data_features_name = cleaning_data(data)
#    Z= pd.DataFrame(Z.toarray(),columns=test_data_features_name)
#    print(Z.shape)
#    feature_not_in_list = list(set(model_columns) - set(test_data_features_name))
    print(len(feature_not_in_list))
#    Z = Z.drop(feature_not_in_list,axis=1) 
    
    
    predictions = None
#    predictions = model.predict(Z).tolist()
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

def cleaning_data(data):
    global total_vote_average, average_rating
#    print("Training data sample:\n", data.head(2))
    
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
#    print('total vote average',total_vote_average)
#    print('average rating', average_rating)
    
    data['weighted_rating'] = data.apply(compute_weighted_rating,axis=1)
    
#    print((data[['weighted_rating']].head(2)))
    data['video_bins'] = data.apply(assign_category_band,axis=1)
    
    data['tags'] = data['tags'].apply(remove_punctuation)
    data['title']= data['title'].apply(remove_punctuation)
#    print('title',data.title.sample(2))
    
    
    X = data.drop(['video_id','title','channel_title','trending_date', 'category', 'published_date',\
                  'published_time','thumbnail_link','comments_disabled',\
                  'ratings_disabled','video_error_or_removed','description',\
               'total_vote','rating','weighted_rating','video_bins','tags'],axis=1)



    count_vectorizer = CountVectorizer(stop_words='english')
    

    word_count_tag=count_vectorizer.fit_transform(data['tags'])
#    new_df = pd.DataFrame(cv.toarray(), columns=count_vectorizer.get_feature_names())
#    data = pd.concat([data,new_df], axis=1)
    
    
#    num_feats = count_vectorizer.get_feature_names()
#    print(num_feats[:10])
    
    Z = np.array(X)
    Z = sparse.hstack((word_count_tag, sparse.csr_matrix(X)))
    
    word_count_title=count_vectorizer.fit_transform(data['title'])
    Z= sparse.hstack((word_count_title,Z))
    print('Z shape', Z.shape)
    
#    data = pd.concat([data,new_df], axis=1)
#    cv = count_vectorizer.fit_transform(data['title'])
#    title_df = pd.DataFrame(cv.toarray(), columns=count_vectorizer.get_feature_names())
#    data = pd.concat([data,title_df], axis=1)
    
#    X = data.drop(['video_id','title','channel_title','trending_date', 'category', 'published_date',\
#                  'published_time','thumbnail_link','comments_disabled',\
#                  'ratings_disabled','video_error_or_removed','description',\
#               'total_vote','rating','weighted_rating','video_bins','tags'],axis=1)
    features_name = list(X.columns)
#    print('X feature name',features_name)
    for i in count_vectorizer.get_feature_names():
        features_name.append(i.encode('utf-8'))
    
#    X= np.array(Z)
    
#    features_name.append(count_vectorizer.get_feature_names)
#    print('features in X:',X.columns)
#    X = np.array(X)
    y = data['video_bins']
    y = np.array(y)
    return Z, y, features_name

    
def assign_category_band(row):
#    print(row['weighted_rating'])
    if row['weighted_rating'] <= 36673:
        return 'below'
    elif row['weighted_rating'] < 40520:
        return 'good'
    else:
        return 'excellent'
    



def compute_weighted_rating(row):
    global total_vote_average, average_rating
    num = (total_vote_average*average_rating)+ (row['total_vote']*row['rating'])
    return num /(total_vote_average+row['total_vote'])


def remove_punctuation(row):
    return re.sub('[(,|\"&)$@#]'," ",row)
