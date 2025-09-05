#import flask module
from flask import Flask, render_template, request, json, jsonify
from flask_restful import Api
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import csv
# import cors
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
api = Api(app)


#Test route
@app.route('/')
def hello_world():
    return 'Hello World'


@app.route("/predict-dynamic", methods=['GET', 'POST'])
def predict_test():
    if request.method == "POST":
        with open('dynamic.pkl', 'rb') as f:
            model = pickle.load(f) #load the pckle file

            # try:
            right = request.get_json()["temp"]

            row = right
            X = pd.DataFrame([row])
            predict_class = model.predict(X)[0] #predict the result
            prob = model.predict_proba(X)[0]
            max_prob = prob[np.argmax(prob)]
        # except:
        # pass
        # print(predict_class)
        data = {'predict': predict_class, 'probability': max_prob} #send the class and probability to the front end
        return jsonify(data)
    return render_template("index.html")

#route
@app.route("/predict-static-sign", methods=['GET', 'POST'])
def predict_static_sign():
    if request.method == "POST":
        try:
            with open('staticsign.pkl', 'rb') as f:
                model = pickle.load(f)

                right = request.get_json()["temp"]
                
                # Validate input data
                if not right or len(right) != 63:  # 21 landmarks * 3 coordinates
                    return jsonify({
                        'predict': 'none',
                        'probability': 0.0,
                        'error': 'Invalid landmark data'
                    }), 400

                row = right
                X = pd.DataFrame([row])
                
                # Make prediction
                predict_class = model.predict(X)[0]
                prob = model.predict_proba(X)[0]
                max_prob = prob[np.argmax(prob)]
                
                # Add confidence threshold
                confidence_threshold = 0.6
                if max_prob < confidence_threshold:
                    return jsonify({
                        'predict': 'none',
                        'probability': max_prob,
                        'message': 'Low confidence prediction'
                    })
                
                data = {
                    'predict': predict_class, 
                    'probability': max_prob,
                    'all_probabilities': prob.tolist()  # Return all class probabilities for debugging
                }
                return jsonify(data)
                
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return jsonify({
                'predict': 'none',
                'probability': 0.0,
                'error': 'Prediction failed'
            }), 500
            
    return render_template("index.html")


# Train Model Output
@app.route("/train-model", methods=['GET', 'POST'])
def train():
    if request.method == "POST":
        file_name = request.get_json()["file_name"]
        model_name = request.get_json()["model_name"]

        df = pd.read_csv(file_name)
        y = df['class']
        x = df.drop('class', axis=1)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1234)

        pipelines = {
            'lr': make_pipeline(StandardScaler(), LogisticRegression()),
            'rc': make_pipeline(StandardScaler(), RidgeClassifier()),
            'rf': make_pipeline(StandardScaler(), RandomForestClassifier()),
            'gb': make_pipeline(StandardScaler(), GradientBoostingClassifier()),
            'svm': make_pipeline(StandardScaler(), SVC()),
        }
        fit_models = {}
        # for algo, pipeline in pipelines.items():
        model = pipelines['gb'].fit(x_train, y_train)
        fit_models['gb'] = model
        model.predict(x_test)

        for algo, model in fit_models.items():
            yhat = model.predict(x_test)
            print(algo, accuracy_score(y_test, yhat))

        with open(model_name, 'wb') as f:
            pickle.dump(fit_models['gb'], f)


        return {"message":"Trainded Sign Language Model Successfully"}

# Create CSV data file
@app.route("/create-csv", methods=['POST'])
def create_csv():
    if request.method == "POST":
        landmarks = ['class_name']
        for val in range(1, 22):
            landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val)]

        with open(request.get_json()["filename"], mode='w', newline='') as f:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(landmarks)

    return {"message": "Successfully Created CSV file"}

@app.route("/save-csv-data", methods=['POST'])
def save_csv_data():
    if request.method == "POST":
        hand_points = request.get_json()["temp"]
        class_name = request.get_json()["className"]
        row = hand_points
        row.insert(0, class_name)
        with open(request.get_json()["filename"], mode='a', newline='') as f:
          csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
          csv_writer.writerow(row)

    return {"message": "Success Inserted Data"}


#Main function
if __name__ == '__main__':
    app.run()
