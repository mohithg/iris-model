from flask import Flask, request, jsonify
from sklearn.externals import joblib

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
        return jsonify(ml_algo.predict(predict_data).tolist())

if __name__ == '__main__':
    app.run(debug=True)