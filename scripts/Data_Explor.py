# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 19:59:03 2023

@author: Oussama Larkem
@email : ousslarkem@gmail.com
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pygwalker as pyg

def load_data(file_path):
    file_extension = file_path.split('.')[-1]  # Extract the file extension
    if file_extension.lower() == 'csv':
        data = pd.read_csv(file_path)
    elif file_extension.lower() in ['xls', 'xlsx']:
        data = pd.read_excel(file_path)
    # Add more conditions for other formats if needed (e.g., JSON, SQL, etc.)
    else:
        print("Unsupported file format. Please provide a valid file.")
        data = None
    return data

def missing_values(data):
    missing_values = data.isnull().sum()
    percentage_missing = (missing_values / len(data)) * 100
    high_perc = percentage_missing[percentage_missing > 50]
    a = len(high_perc)
    print("We have", a, "Columns with More Than 50% Missing Values:", high_perc)
    
    plt.figure(figsize=(10, 6))
    percentage_missing.plot(kind='bar', color='skyblue')
    plt.title('Percentage of Missing Values')
    plt.xlabel('Columns')
    plt.ylabel('Missing (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return percentage_missing

def drop_repeated_columns(data):
    repeated_columns = [column for column in data.columns if data[column].nunique() == 1]
    update_data = data.drop(repeated_columns + ['BORE_WI_VOL'], axis=1)
    print("--- 1 ---\nRepeated Columns:\n", repeated_columns)
    print("--- 2 ---\nwe have\n", len(repeated_columns), "columns with repeated content")
    print('--- 3 ---\nThe shape of our  updated data is :\n ', update_data.shape)
    print('--- 4 ---\nThe columns of our updated data are :\n ' , update_data.columns)
    return update_data

def export_to_csv(data):
    file_name = input("Enter the desired file name: ")
    data.to_csv(file_name + ".csv")
    
def plot_correlation(data):
    corr = data.corr()
    sns.heatmap(corr, annot=True, cmap="Oranges")
    plt.show()



