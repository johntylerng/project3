import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as rf

# inputs
include = ['Age', 'Sex', 'Embarked', 'Survived']
dependent_variable = include[-1]

MODEL_DIRECTORY = 'model'
MODEL_FILE_NAME = '%s/model.pkl' % MODEL_DIRECTORY
MODEL_COLUMNS_FILE_NAME = '%s/model_columns.pkl' % MODEL_DIRECTORY

def train(df):
    x=read_csv('out_project2_x.csv')
    y= read_csv('out_proect2_y.csv')
    
    
    
    df_ = df[include]
    print("Training data sample:\n", df_.head())
    categoricals = []  # going to one-hot encode categorical variables

    for col, col_type in df_.dtypes.items():
        if col_type == 'O':
            categoricals.append(col)
        else:
            df_[col].fillna(0, inplace=True)  # fill NA's with 0 for ints/floats, too generic

    # get_dummies effectively creates one-hot encoded variables
    df_ohe = pd.get_dummies(df_, columns=categoricals, dummy_na=True)

    x = df_ohe[df_ohe.columns.difference([dependent_variable])]
    y = df_ohe[dependent_variable]

    # capture a list of columns that will be used for prediction
    model_columns = list(x.columns)
    print("Model columns are", model_columns)

    model = rf()
    start = time.time()
    model.fit(x, y)
    print('Trained in %.1f seconds' % (time.time() - start))
    print('Model training score: %s' % model.score(x, y))

    return model_columns, model


def predict(input_df, model, columns):
    print("Input data frame is...\n")
    print("-----------")
    print(input_df)
    print("-----------")
    query = pd.get_dummies(input_df)

    # Give all missing values a value of 0
    query = query.reindex(columns=columns, fill_value=0)

    predictions = model.predict(query).tolist()
    predictions = [int(prediction) for prediction in predictions]

    return {'predictions': predictions}




