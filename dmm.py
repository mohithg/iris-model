import requests

API_HOST="http://localhost:8080/api"

REGISTER_MODEL = f"{API_HOST}/register_model"
REGISTER_PREDICTION = f"{API_HOST}/register_prediction"

def register_model():
    data = {
        "algorithm": "feature_extraction",
        "feature_columns": [
            "feature1",
            "feature2"
        ],
        "hyper_params": [
            {
                "kernel": "gaussain"
            },
            {
                "no_of_trees": 12
            }
        ],
        "label_columns": [
            "label1",
            "label2"
        ],
        "metadata": [
            {
                "name": "PL Feature Collection"
            },
            {
                "RAM": "16 GB"
            },
            {
                "done_by": "some one"
            }
        ],
        "metrics": [
            {
                "mcc": "0.938",
                "auc": "0.999",
                "precision": "0.914",
                "recall": "0.913",
                "f1": "0.914",
                "accuracy": "91%",
                "rules": "3 rules applied"
            }
        ],
        "model_path": "project_id.experiment_id.model_id.version_number",
        "model_type": "classification",
        "prediction_columns": [
            "variety"
        ],
        "predictions_data": {
            "column_names": [
                "column1",
                "column2"
            ],
            "num_rows": 20,
            "row_id_column_name": "row_id"
        },
        "training_data": {
            "column_names": [
                "column1",
                "column2"
            ],
            "num_rows": 20,
            "row_id_column_name": "row_id",
            "url": "swagger_server/iris.csv"
        }
    }
    # sending post request and saving response as response object
    r = requests.post(url=REGISTER_MODEL, json=data)
    print(r)

def register_prediction(modelId):
    data = {
      "accuracies": {
        "accuracy1": 0.3,
        "accuracy2": 0.4
      },
      "accuracy_metrics": [""],
      "actual_data": {
        "column_names": [""],
        "num_rows": 0,
        "row_id_column_name": "row_id_column_name",
        "url": "url"
      },
      "feature_columns": ["variety"],
      "label_columns": ["sepal.length", "sepal.width", "petal.length", "petal.width"],
      "model": "model",
      "prediction_data": [{
        "petal.length": "0.98",
        "petal.width": "0.98",
        "sepal.length": "0.98",
        "sepal.width": "0.98"
      }]
    }
    data['model_id'] = modelId
    # sending post request and saving response as response object
    r = requests.post(url=REGISTER_PREDICTION, json=data)
    print(r)

register_model()

register_prediction('5b2762617e6d7631cd2b2d0c')
