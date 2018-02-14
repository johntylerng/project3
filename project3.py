import numpy as np
import pandas as pd
#from pandas.plotting import scatter_matrix
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
#from sklearn.svm import SVC
#from sklearn.metrics import classification_report,confusion_matrix,\
#precision_score, recall_score, precision_recall_curve, average_precision_score
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
#from sklearn import tree
#from sklearn.ensemble import RandomForestClassifier
#import seaborn as sns
#from matplotlib.pyplot import pie, axis, show,title
import pickle
from sklearn.externals import joblib

def test_endpoint():
    print("Trying test endpoint...")
    url = API_HOST + TEST_API
    print("URL is", url)

    # Try to access the URL. The response will be stored in 'r'.
    r = requests.get(url)

    # The response status code tells us whether or not we were
    # successful in accessing the URL. Generally, HTTP status codes
    # starting with 2 are good and ones starting with 4 or 5 are bad.
    # HTTP status codes:
    # https://en.wikipedia.org/wiki/List_of_HTTP_status_codes
    if r.status_code == 200:
        print("Success!")
        print(r.text)
    else:
        print("Status code indicates a problem:", r.status_code)




raw_data = pd.read_csv('university_data_2016.csv')

def extract_hyphen(row):
    if row['world_rank'].isdigit():
        return int(float(row['world_rank']))
    else:
        value_list = row['world_rank'].replace('-', ' ').split(' ')
        value_list = list(map(int, value_list))
        return int(np.mean(value_list))
            
raw_data['revised_world_rank'] = raw_data.apply(extract_hyphen,axis=1)

for feature in ['income','total_score','num_students','international_students',\
                'student_staff_ratio']:
    raw_data[feature].fillna(raw_data[feature].mean(), inplace=True )


for feature in ['female_male_ratio','tuition_fee','median_salary']:
    raw_data[feature].fillna(0, inplace=True )

filtered_data_uk = raw_data[raw_data.country == 'United Kingdom']
uk_mean= filtered_data_uk['tuition_fee'].mean()
filtered_data_us = raw_data[raw_data.country == 'United States of America']
us_mean= filtered_data_us['tuition_fee'].mean()

def assign_missing_values_tuition(row):
    if row['tuition_fee']>0:
        return row['tuition_fee']
    if (row['country'] == 'United Kingdom') and (row['tuition_fee']==0):
        return uk_mean
    elif (row['country'] == 'United States of America') and \
    (row['tuition_fee']==0):
        return us_mean
    elif (row['country'] == 'Canada') and (row['tuition_fee']==0):
        return us_mean
    else:
        return uk_mean
    
raw_data['tuition_fee'] = raw_data.apply(assign_missing_values_tuition,axis=1)

filtered_data_uk = raw_data[raw_data.country == 'United Kingdom']
uk_mean= filtered_data_uk['median_salary'].mean()
filtered_data_us = raw_data[raw_data.country == 'United States of America']
us_mean= filtered_data_us['median_salary'].mean()

def assign_missing_values_salary(row):
    if row['median_salary']>0:
        return row['median_salary']
    if (row['country'] == 'United Kingdom') and (row['median_salary']==0):
        return uk_mean
    elif (row['country'] == 'United States of America') and (row['median_salary']==0):
        return us_mean
    elif (row['country'] == 'Canada') and (row['median_salary']==0):
        return us_mean
    else:
        return uk_mean
    
raw_data['median_salary'] = raw_data.apply(assign_missing_values_salary,axis=1)

raw_data['salary_bins'] = pd.qcut(raw_data['median_salary'],
                                 q=3,
                                 labels=["low","good","excellent"])

raw_data.country = raw_data.country.astype('category')
raw_data['country_cat'] = raw_data['country'].cat.codes

X = raw_data.drop(['world_rank','university_name', 'country', 'total_score',\
                  'student_staff_ratio','international_students',\
                  'female_male_ratio','year','median_salary','salary_bins'], axis=1)

features_name = X.columns
print('features in X:',X.columns)
X = np.array(X)
y = raw_data.salary_bins
y = np.array(y)
np.random.seed(42)
shuffle_index = np.random.permutation(raw_data.shape[0])
X, y= X[shuffle_index],y[shuffle_index]

#split data into 60%, 20%, 20%
X_train_validate, X_test,y_train_validate, y_test= train_test_split(X,y,test_size=0.20,\
                                                                  random_state=0)

X_train, X_validate, y_train, y_validate = train_test_split(X_train_validate,y_train_validate,\
                                                            test_size=0.25, random_state=0)

decision_tree_classifier = DecisionTreeClassifier(min_samples_leaf=5)
decision_tree_classifier.fit(X_train,y_train)

# save the model to disk
filename = 'finalized_classifier_model.sav'
joblib.dump(decision_tree_classifier, 'model_file2.pkl')
#pickle.dump(X_test, open('test_data', 'wb'))

df = pd.concat([X_train, y_train], axis=1)
df.to_csv('out_project.csv')












