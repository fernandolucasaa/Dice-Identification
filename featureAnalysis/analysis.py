# -*- coding: utf-8 -*-
# analysis.py --- Analyse features

# Copyright (c) 2011-2016  Fabio Morooka <fabio.morooka@gmail.com> and Fernando Amaral <fernando.lucasaa@gmail.com>

# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# code:

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import string

# separate the data set loaded (npy file) in features and targets

def separate_array(data):    
    image = []
    labels = []
    for line in data:
        image.append(np.array(line[:-1]))
        labels.append(line[-1])
    
    return image, labels

 # Load the npy files with the features computed and targets and separate

X_train_class, Y_train_class = separate_array(np.load('./MINGAU/train_classes.npy'))
X_validation_class, Y_validation_class = separate_array(np.load('./MINGAU/validation_classes.npy'))

X_test_class, Y_test_class = separate_array(np.load('./MINGAU/test_classes.npy'))
X_test_photos_class, Y_test_photos_class = separate_array(np.load('./MINGAU/test_photos_classes.npy'))

# Create a dataframe with the features and targets

def join_data(X_data, Y_data):
    new_array = []
    if len(X_data) == len(Y_data):
        for i in range(len(X_data)):
            line = np.asarray(X_data[i]).reshape(-1).tolist()
            line.append(Y_data[i])
            
            new_array.append(line)

    return pd.DataFrame(new_array)

# For a specific class, calculate the average for all features and 
# return a list with the values
#
# Parameters:
# - df_class: dataset's dataframe (features and targets)
# - number: class whose features will be computed
#
# Return:
# - average: list with all mean values  

def average_features(df_class, number):
    
    # create a dataframe with features and target
    df_class_selected = df_class.loc[df_class[12] == number]
    
    # average for each column (features)
    average = (df_class_selected.mean(axis=0)).tolist()
    
    # remove the last column (target)
    average = average[:len(average)-1]
    
    return average

# For all classes in a data set, calculate the average for all features
#
# Parameters:
# - X_class, Y_class
#
# Return:
# - avg_features_classes: list will all mean values (2 dimensions) 

def average_features_classes(X_class, Y_class):
    
    # join the data set
    df_class = join_data(X_class, Y_class)
    
    avg_features_classes = []
    
    for i in range(1,7):
        avg_features_classes.append(average_features(df_class, i))
    
    return avg_features_classes

def plotBarGraph(x, y, title, ylabel, file_name, dataset_name):    
    
    plt.figure(figsize=(6,6))
    plt.grid()
    plt.bar(x, y)
    plt.title(title)
    plt.xlabel('Classes')
    plt.ylabel(ylabel)
    plt.savefig('./graphs_features' + '/' + dataset_name + '/' + file_name + '_' + dataset_name)
    
    return 0
def plot_average_features(X_class, Y_class, dataset_name):

    avg = average_features_classes(X_class, Y_class)
    avg_t = ((np.asarray(avg)).T).tolist()

    digits = [i for i in range(1,7)]
    keys = ['var', 'std', 'mean_grad_M', 'std_grad_M', 'mean_grad_D', 'std_grad_D', 'mean_PC_X', 'std_PC_X', 'active_PC_X', 'mean_PC_Y', 'std_PC_Y', 'active_PC_Y']

    for i in range(12):
        plotBarGraph(digits, avg_t[i], keys[i], keys[i], keys[i], dataset_name)
    
    return

 plot_average_features(X_train_class, Y_train_class, "train")

 plot_average_features(X_validation_class, Y_validation_class, "validation")

 plot_average_features(X_test_class, Y_test_class, "test")

 plot_average_features(X_test_photos_class, Y_test_photos_class, "test_photos")

 def plot_graph(avg_values, title, dataset_name):
    
    keys = ['var', 'std', 'mean_grad_M', 'std_grad_M', 'mean_grad_D', 'std_grad_D', 'mean_PC_X', 'std_PC_X', 'active_PC_X', 'mean_PC_Y', 'std_PC_Y', 'active_PC_Y']
    
    plt.figure(figsize=(24,10))
    plt.grid()
    plt.plot(keys,avg_values)
    plt.xlabel("Features")
    plt.ylabel("Values")
    plt.title(title)
    plt.legend("123456")
    plt.savefig('./graphs_features' + '/' + dataset_name)
    plt.show()
    
    return

 avg_train = average_features_classes(X_train_class, Y_train_class)
avg_train_t = (np.asarray(avg_train).T).tolist()

plot_graph(avg_train_t, "Train", "train")

avg_validation = average_features_classes(X_validation_class, Y_validation_class)
avg_validation_t = (np.asarray(avg_validation).T).tolist()

plot_graph(avg_validation_t, "Validation", "validation")

avg_test = average_features_classes(X_test_class, Y_test_class)
avg_test_t = (np.asarray(avg_test).T).tolist()

plot_graph(avg_test_t, "Test", "test")

avg_test_photos = average_features_classes(X_test_photos_class, Y_test_photos_class)
avg_test_photos_t = (np.asarray(avg_test_photos).T).tolist()

plot_graph(avg_test_photos_t, "Test photos", "test_photos")

