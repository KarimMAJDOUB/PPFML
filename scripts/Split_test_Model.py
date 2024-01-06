# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 15:28:48 2023

@author: Larkem Oussama 
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

df = pd.read_csv('Final_Data.csv')
data = df.drop(['ON_STREAM_HRS'], axis=1)

def get_features_labels(data, target_column):
    target_columns = ['BORE_OIL_VOL', 'BORE_GAS_VOL', 'BORE_WAT_VOL']

    X = data.drop(target_columns, axis=1)
    Y = data[target_column]

    return X, Y


def split_train_test(X, Y, test_size=0.3):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=42)

    return X_train, X_test, y_train, y_test


def Train_and_evaluate(model_name, X_train, X_test, y_train, y_test, index):
    models = {
        'Random Forest': RandomForestRegressor(),
        'Decision Tree': DecisionTreeRegressor(),
        'Gradient Boosting': GradientBoostingRegressor(),
        'KNeighbors': KNeighborsRegressor(),
        'SVR': SVR()
    }

    if model_name not in models:
        return f"Model '{model_name}' not found."

    model = models[model_name]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    score = model.score(X_test, y_test)

    plt.figure()
    plt.scatter(X_test[index], y_test, label=index)
    plt.scatter(X_test[index], y_pred, label=model_name)
    plt.title(f"Actual vs. Predicted Values (Model: {model_name})")
    plt.legend()
    plt.show()

    print(f"{model_name} Model Evaluation")
    print(f"MSE: {mse}")
    print(f"Score : {score}")
    print("---")

    print(f"\nHere are the hyperparameters of the {model_name} model:")
    print("---------------------------------------------------------------")
    for key, value in model.get_params().items():
        print(f"{key}: {value}")
    print("---------------------------------------------------------------")

    return {
        score: 'Relax, it will get better'
    }
