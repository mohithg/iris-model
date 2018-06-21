from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
import pandas as pd
import numpy as np

# Set the feature names
feature_names = ['sepal.length', 'sepal.width', 'petal.length', 'petal.width']

# Create a data frame from the iris dataset
df = pd.read_csv('iris.csv',header=0)

x = df[feature_names]
y = df['variety']

# Split as 80% training and 20% Testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=12345)

# Writing Train and Test to CSV files
x_train.to_csv("x_train.csv",  index = False)
x_test.to_csv("x_test.csv",  index = False)
y_train.to_csv("y_train.csv",  index = False)
y_test.to_csv("y_test.csv",  index = False)

# Initialize Random Forest Algorithm
forest = RandomForestClassifier()

# Sample List of possible parameters
param_grid = {
    'n_estimators': [5, 10, 20, 30],
    'random_state': [0],
    'n_jobs': [1],
    'min_samples_split': [3, 5, 10],
    'max_depth': [3, 5, 10, 15, 20]
}

# Run GridSearch Based on the above Hyper Parameters & Perform 5 fold Cross Validation
forestGrid = GridSearchCV(forest, param_grid,cv=5)
fgFit = forestGrid.fit(x_train, y_train)

# set the best params to fit random forest classifier
forest.set_params(**fgFit.best_params_)
forest.fit(x_train, y_train)

# Train Prediction / Evaluation
train_predictions = forest.predict(x_train)

print("Train Performance of the model")
print("Accuracy:")
print(accuracy_score(y_train, train_predictions))
print("Confusion Matrix:")
print(confusion_matrix(y_train, train_predictions))
print("Classification Report:")
print(classification_report(y_train, train_predictions))

test_predictions = forest.predict(x_test)

print("Test Performance of the model")
print("Accuracy:")
print(accuracy_score(y_test, test_predictions))
print("Confusion Matrix:")
print(confusion_matrix(y_test, test_predictions))
print("Classification Report:")
print(classification_report(y_test, test_predictions))

# Sample Prediction using a Dataframe slice
print("Sample Prediction - 1 ( Using Dataframe )")
print("Input:")
print(x_test.iloc[[1]])
print("Output:")
print(forest.predict(x_test.iloc[[1]]))


# Sample Prediction as a Numpy Array
# Convert JSON to numpy array and use it for prediction
# IMPORTANT : Keep the order of input the same
print("Sample Prediction - 1 ( Using Numpy Array )")
print("Input:")
t = np.array([6.4,3.2,5.3,2.3])
t = t.reshape(1,-1)
print(t)
print("Output:")
print(forest.predict(t))

# save the model as pickle# save
joblib.dump(forest, 'randomforest_iris.pkl', compress=True, protocol=2)