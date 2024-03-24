from flask import Flask, request, render_template, send_file
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from flask import render_template
import matplotlib
from flask import render_template_string
from xgboost import plot_tree
import matplotlib.pyplot as plt

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import sklearn
import os
from io import BytesIO
import base64
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from flask import render_template_string


app = Flask(__name__)

df = pd.DataFrame()
X_train, X_test, y_train, y_test = None, None, None, None
xgb_model = XGBClassifier()
xgb_accuracy, svm_accuracy, accuracy, fs_accuracy, ann_accuracy = None, None, None, None, None
precision, xgb_precision, svm_precision, fs_precision, ann_precision = None, None, None, None, None
recall,xgb_recall, svm_recall,fs_recall,ann_recall= None, None, None, None, None
f1,xgb_f1,svm_f1,fs_f1,ann_f1= None, None, None, None, None


label_encoders = {}
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    global df
    if 'file' not in request.files:
        return render_template('index.html', message='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', message='No selected file')

    if file:
        try:
            df = pd.read_csv(file)
            table_html = df.head(10).to_html()
            return render_template('index.html', table=table_html)

        except Exception as e:
            return render_template('index.html', message=f'Error: {str(e)}')

label_encoders = {}
@app.route('/preprocess')
def preprocess():
    global df, label_encoders
    if df.empty:
        return render_template('index.html', message='DataFrame is empty. Upload a file first.')

    for column in df.columns:
        if df[column].dtype == 'object':
            label_encoders[column] = sklearn.preprocessing.LabelEncoder()
            df[column] = label_encoders[column].fit_transform(df[column])

    table_html = df.head(1000).to_html()
    return render_template('index.html', table=table_html, message='Preprocessing completed.')

@app.route('/split')
def split():
    global df, X_train, X_test, y_train, y_test
    if df is not None and not df.empty:
        try:
            X = df[['Soil_color', 'pH', 'Rainfall', 'Temperature', 'Crop']]
            Y = df['Fertilizer']
            

            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=21)
            message = f'Split completed successfully. Shapes: X_train={X_train.shape}, X_test={X_test.shape}, y_train={y_train.shape}, y_test={y_test.shape}'
            return render_template('index.html', message=message)

        except Exception as e:
            return render_template('index.html', message=f'Error: {str(e)}')

    else:
        return render_template('index.html', message='Error: Data not loaded or empty. Please click "Show" first.')


@app.route('/random_forest')
def random_forest():
    global X_train, X_test, y_train, y_test, accuracy, precision,recall,f1

    if X_train is None or X_test is None or y_train is None or y_test is None:
        return render_template('error.html', message='Data not split. Please click "Split" first.')

    try:
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        y_pred = rf_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        app.config['accuracy'] = accuracy

        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        print(f'Random Forest Metrics:')
        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1-Score: {f1:.4f}')

        return render_template('random_forest_result.html', accuracy=accuracy, precision=precision, recall=recall, f1=f1)

    except Exception as e:
        return render_template('error.html', message=f'Error: {str(e)}')


@app.route('/xgboost')
def xgboost():
    global X_train, X_test, y_train, y_test, xgb_accuracy,xgb_precision,xgb_recall,xgb_f1

    if X_train is None or X_test is None or y_train is None or y_test is None:
        return render_template('error.html', message='Data not split. Please click "Split" first.')
    try:
        
        xgb_model = XGBClassifier()
        xgb_model.fit(X_train, y_train)    
        y_pred = xgb_model.predict(X_test)

        
        xgb_accuracy = accuracy_score(y_test, y_pred)
        app.config['xgb_accuracy'] = xgb_accuracy

        xgb_precision = precision_score(y_test, y_pred, average='weighted')
        xgb_recall = recall_score(y_test, y_pred, average='weighted')
        xgb_f1 = f1_score(y_test, y_pred, average='weighted')

        print(f'XGBoost Metrics:')
        print(f'Accuracy: {xgb_accuracy:.4f}')
        print(f'Precision: {xgb_precision:.4f}')
        print(f'Recall: {xgb_recall:.4f}')
        print(f'F1-Score: {xgb_f1:.4f}')

        return render_template('xgboost_result.html', xgb_accuracy=xgb_accuracy, xgb_precision=xgb_precision, xgb_recall=xgb_recall, xgb_f1=xgb_f1)

    except Exception as e:
        return render_template('error.html', message=f'Error: {str(e)}')


@app.route('/svm')
def svm():
    global X_train, X_test, y_train, y_test,svm_accuracy,svm_precision,svm_recall,svm_f1

    if X_train is None or X_test is None or y_train is None or y_test is None:
        return render_template('error.html', message='Data not split. Please click "Split" first.')

    try:
        clf = SVC(kernel='linear')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        svm_accuracy = accuracy_score(y_test, y_pred)
        
        svm_precision = precision_score(y_test, y_pred, average='weighted')
        svm_recall = recall_score(y_test, y_pred, average='weighted')
        svm_f1 = f1_score(y_test, y_pred, average='weighted')

        print(f'SVM Metrics:')
        print(f'Accuracy: {svm_accuracy:.4f}')
        print(f'Precision: {svm_precision:.4f}')
        
        print(f'Recall: {svm_recall:.4f}')
        print(f'F1-Score: {svm_f1:.4f}')
        return render_template('svm_result.html', svm_accuracy=svm_accuracy, svm_precision=svm_precision, svm_recall=svm_recall, svm_f1=svm_f1)

    except Exception as e:
        return render_template('error.html', message=f'Error: {str(e)}')

@app.route('/ann')
def ann():
    global X_train, X_test, y_train, y_test,ann_accuracy,ann_precision,ann_recall,ann_f1

    if X_train is None or X_test is None or y_train is None or y_test is None:
        return render_template('error.html', message='Data not split. Please click "Split" first.')


    try:
        ann_model = MLPClassifier(
            hidden_layer_sizes=(100,),
            max_iter=10,  
            solver='adam',
            activation='relu',
            batch_size=80,
            random_state=42
        )

        ann_model.fit(X_train, y_train)

        y_pred = ann_model.predict(X_test)

        ann_accuracy = accuracy_score(y_test, y_pred)
        app.config['ann_accuracy'] = ann_accuracy
        ann_precision = precision_score(y_test, y_pred, average='weighted')
        ann_recall = recall_score(y_test, y_pred, average='weighted')
        ann_f1 = f1_score(y_test, y_pred, average='weighted')

        print(f'ANN Metrics:')
        print(f'Accuracy: {ann_accuracy:.4f}')
        print(f'Precision: {ann_precision:.4f}')
        print(f'Recall: {ann_recall:.4f}')
        print(f'F1-Score: {ann_f1:.4f}')


        return render_template('ann_result.html', ann_accuracy=ann_accuracy, ann_precision=ann_precision, ann_recall=ann_recall, ann_f1=ann_f1)

    except Exception as e:
        return render_template('error.html', message=f'Error: {str(e)}')

@app.route('/make_prediction')
def make_prediction():
    return render_template('make_prediction.html')

@app.route('/make_prediction_result', methods=['POST'])
def make_prediction_result():
    global xgb_model, X_train, y_train, label_encoders

    if request.method == 'POST':
        Soil_color = float(request.form['Soil_color'])
        pH = float(request.form['pH'])
        Rainfall = float(request.form['Rainfall'])
        Temperature = float(request.form['Temperature'])
        Crop = float(request.form['Crop'])

        input_values = [[Soil_color, pH, Rainfall, Temperature, Crop]]
        print("Input values:", input_values)

        if xgb_model is None:
            xgb_model = XGBClassifier()

        try:
            input_df = pd.DataFrame(input_values, columns=X_train.columns)
            xgb_model.predict(input_df)
        except (AttributeError, Exception, sklearn.exceptions.NotFittedError) as e:
            xgb_model.fit(X_train, y_train)

        prediction = xgb_model.predict(input_df)

        prediction_fertilizer_name = label_encoders['Fertilizer'].inverse_transform(prediction)

        print("Prediction:", prediction_fertilizer_name)

        return render_template('prediction_result.html', prediction=prediction_fertilizer_name[0])

    return render_template('make_prediction.html')



if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5010, debug=True)