import requests

API_ENDPOINT = "http://localhost:8080/api/register_prediction"

def send_to_dmm(data, prediction, prediction_column):
    prediction_data = data
    prediction_data[prediction_column] = prediction[0]
    data_to_send = {
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
        "model_id": "5b2762617e6d7631cd2b2d0c",
        "prediction_data": [prediction_data]
    }
    # sending post request and saving response as response object
    print(data_to_send)
    r = requests.post(url=API_ENDPOINT, json=data_to_send)

    print(r)