import sys
import os
import shutil
import traceback
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.externals import joblib

from utils import model_utils2

app = Flask(__name__)

#TRAINING_FILE_PATH = 'data/titanic.csv'
TRAINING_FILE_PATH='data/USvideos.csv'

# These will be populated at training time
model_columns = None
model = None


@app.route('/test_endpoint', methods=['GET'])
def test_function():
    print("I made my own endpoint!")
    return "I'm hungry"


@app.route('/predict', methods=['POST'])
def predict():
    if model:
        try:
            input_df = pd.DataFrame(request.json)
            predictions = model_utils.predict(input_df, model, model_columns)
            return jsonify(predictions)
        except Exception as e:
            return jsonify({'error': str(e), 'trace': traceback.format_exc()})
    else:
        print('You need to train a model before you can make predictions.')
        return 'error: no model'


@app.route('/train', methods=['GET'])
def train():
    #df = pd.read_csv(TRAINING_FILE_PATH)
    df_x = pd.read_csv()

    # We use a global here so we can modify the model and list of column names
    global model_columns, model
    model_columns, model = model_utils.train(df)
    joblib.dump(model_columns, model_utils.MODEL_COLUMNS_FILE_NAME)
    joblib.dump(model, model_utils.MODEL_FILE_NAME)

    return 'Success'


@app.route('/train_new', methods=['POST'])
def train_new():
    df = pd.DataFrame(request.json)

    # We use a global here so we can modify the model and list of column names
    global model_columns, model
    model_columns, model = model_utils.train(df)
    joblib.dump(model_columns, model_utils.MODEL_COLUMNS_FILE_NAME)
    joblib.dump(model, model_utils.MODEL_FILE_NAME)

    return 'Success'


@app.route('/wipe', methods=['GET'])
def wipe():
    try:
        shutil.rmtree('model')
        os.makedirs(model_utils.MODEL_DIRECTORY)
        return 'Model wiped'
    except Exception as e:
        print(str(e))
        return 'Could not remove and recreate the model directory'


if __name__ == '__main__':
    try:
        port = int(sys.argv[1])
    except Exception as e:
        port = 5000

    try:
        model = joblib.load(model_utils.MODEL_FILE_NAME)
        print('model loaded')
        model_columns = joblib.load(model_utils.MODEL_COLUMNS_FILE_NAME)
        print('model columns loaded')
    except Exception as e:
        print('No model here')
        print('Train first')
        print(str(e))
        model = None

    app.run(host='0.0.0.0', port=port, debug=True)
