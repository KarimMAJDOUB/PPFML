# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 15:10:23 2024

@author: LARKEM OUSSAMA
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import numpy as np

df = pd.read_csv('Final_Data.csv')
data = df.drop(['ON_STREAM_HRS'], axis=1)

def get_features_labels(data, target_column):
    target_columns = ['BORE_OIL_VOL', 'BORE_GAS_VOL', 'BORE_WAT_VOL']
    X = data.drop(target_columns, axis=1)
    Y = data[target_column]
    return X, Y

def split_train_test(X, Y, test_size=0.3):
    return train_test_split(X, Y, test_size=test_size, random_state=42)

def build_model(input_shape, output_shape):
    input_data = Input(shape=input_shape, name='Input')
    dense1 = Dense(256, activation=tf.nn.relu)(input_data)
    dense2 = Dense(256, activation=tf.nn.relu)(dense1)
    output = Dense(output_shape, name='output')(dense2)

    model = Model(input_data, output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae', 'mse', 'mape'])


    model.summary()

    tf.keras.utils.plot_model(
        model, to_file='model.png', show_shapes=True, show_dtype=True,
        show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96
    )

    return model

def train_model(model, train_features, train_labels, test_features, test_labels, epochs=500, batch_size=128):
    history = model.fit(
        train_features,
        train_labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(test_features, test_labels)
    )
    return history

def plot_loss(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.grid('both')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

def evaluate_model(model, test_features, test_labels):
    return model.evaluate(test_features, test_labels)

def make_predictions(model, test_features):
    return model.predict(test_features)

def plot_actual_vs_predicted(test_labels, predictions, column_index):
    test_labels = test_labels.values.reshape(-1)

    plt.figure(figsize=(8, 6))
    plt.scatter(test_labels, predictions, label='DL')
    plt.scatter(test_labels, test_labels, label='Actual')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Actual vs Predicted - Column {column_index}')
    plt.legend()
    plt.grid()
    plt.show()

def calculate_additional_metrics(train_labels, test_labels, model, train_features, test_features):
    metrics = {}

    train_predictions = model.predict(train_features).reshape(-1)  
    test_predictions = model.predict(test_features).reshape(-1)    

    metrics['mae-train'] = mean_absolute_error(train_labels, train_predictions)
    metrics['mse-train'] = mean_squared_error(train_labels, train_predictions)
    metrics['r2-train'] = r2_score(train_labels, train_predictions)
    metrics['pearson-train'] = pearsonr(train_labels, train_predictions)[0]

    metrics['mae-test'] = mean_absolute_error(test_labels, test_predictions)
    metrics['mse-test'] = mean_squared_error(test_labels, test_predictions)
    metrics['r2-test'] = r2_score(test_labels, test_predictions)
    metrics['pearson-test'] = pearsonr(test_labels, test_predictions)[0]

    return metrics

def print_metrics(metrics):
    for key, value in metrics.items():
        if 'train' in key:
            print(f'{key} - Train: {value:.7f}')
        else:
            print(f'{key} - Test: {value:.7f}')