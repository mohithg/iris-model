from flask import Flask, request, jsonify
from sklearn.externals import joblib
import dmm_predict

app = Flask(__name__)


@app.route("/")
def hello():
    return "Hello World!"

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            data = request.get_json()
            predict_data = [[data.get('sepal.length'),data.get('sepal.width'),data.get('petal.length'),data.get('petal.width')]]

            ml_algo = joblib.load("./randomforest_iris.pkl")
        except ValueError:
            return jsonify("Please enter a number")
        prediction = ml_algo.predict(predict_data).tolist()[0]
        prediction_column = 'variety'
        # dmm_predict.send_to_dmm(data, prediction, prediction_column)
        return jsonify({'variety': prediction})

if __name__ == '__main__':
    app.run(debug=True)