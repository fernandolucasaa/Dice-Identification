# -*- coding: utf-8 -*-
# generateDatabase.py --- Reading binary MNIST files algorithm

# Copyright (c) 2019-2020  Fabio Morooka <fabio.morooka@gmail.com> and Fernando Amaral <fernando.lucasaa@gmail.com>

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

from collections import Counter
from keras.datasets import mnist
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# This function verify if a file exists and if so it will remove it.
def remove_file(filename):
    if os.path.isfile(filename):
        print("Removing " + filename)
        os.remove(filename)
    else:
        print(filename + "file does not exist, creating one!")

#Extract the numbers from database
def selectDiceNumbers(img_list_all, labels_list_all):
    img_final = []
    labels_final = []
    diceNumbers = [1, 2, 3, 4, 5, 6]

    if len(img_list_all) == len(labels_list_all):
        initialSize = range(len(img_list_all))
    else:
        print("Size of lists must be the same!")

    for i in initialSize:
        if labels_list_all[i] in diceNumbers:
            img_final.append(img_list_all[i])
            labels_final.append(labels_list_all[i])
    
    return img_final, labels_final

# Load dataset from keras and then make a filter to use only dice numbers
def load_datasets():
    print("\nLoading Training dataset and Testing dataset from MNIST...")
    (img_train_all, labels_train_all),(img_test_all, labels_test_all) = mnist.load_data()
    
    img_train, labels_train = selectDiceNumbers(img_train_all, labels_train_all)
    img_test, labels_test = selectDiceNumbers(img_test_all, labels_test_all)

    print("Finished loading the datasets!\n")
    
    return img_train, labels_train, img_test, labels_test

def join_data(X_data, Y_data):
    new_array = []
    if len(X_data) == len(Y_data):
        for i in range(len(X_data)):
            line = np.asarray(X_data[i]).reshape(-1).tolist()
            line.append(Y_data[i])
            
            new_array.append(line)

    return pd.DataFrame(new_array)

def create_data(X_train_notBal, Y_train_notBal,X_test_notBal, Y_test_notBal):
    df1 = join_data(X_train_notBal, Y_train_notBal)
    df2 = join_data(X_test_notBal, Y_test_notBal)

    df = df1.append(df2)
    ds = df.sample(frac=1).reset_index(drop = True)
    
    return ds

def balance_dataset(dataset):
    Y_data = dataset.iloc[:,-1].to_numpy()
    Y_data_count = Counter(Y_data)
    less_number = min(Y_data_count, key = Y_data_count.get)
    quantity = int(Y_data_count[less_number] / 2)
    
    for i in range(1, 7):
        dl = dataset.loc[dataset[784] == i]
        dh = dl.head(quantity)
        dt = dl.tail(quantity)
        df_rec = dt.append(dh)
        if i == 1:
            df = df_rec
        else:
            df = df.append(df_rec)
    
    ds = df.sample(frac=1).reset_index(drop = True)
    
    return ds

def verify_balance(Y_data):
    Y_data_count = Counter(Y_data)
    for i in range(1,6):
        if Y_data_count[i] == Y_data_count[i + 1]:
            pass
        else:
            return False
    return True

def separeteDatasets(dataset):
    diceNumbers = range(1,7)
    classesTest_size = []
    classesTrain_size = []
    classesValidation_size = []

    filter_coef = 1

    for num in diceNumbers:
        dl = dataset.loc[dataset[784] == num]

        total_length = int(len(dl) * filter_coef)

        test_numbers = int(0.2 * total_length)
        train_numbers = int(0.64 * total_length)
        validation_numbers = int(0.16 * total_length)

        classesTest_size.append(test_numbers)
        classesTrain_size.append(train_numbers)
        classesValidation_size.append(validation_numbers)

        df_percTest = dl.head(test_numbers)
        df_percTrain = dl[test_numbers:total_length-validation_numbers]
        df_percValidation = dl.tail(validation_numbers)
        
        if num == 1:
            df_test = df_percTest
            df_train = df_percTrain
            df_validation = df_percValidation
        else:
            df_test = df_test.append(df_percTest)
            df_train = df_train.append(df_percTrain)
            df_validation = df_validation.append(df_percValidation)

    ds_test = df_test.sample(frac=1).reset_index(drop = True)
    ds_train = df_train.sample(frac=1).reset_index(drop = True) 
    ds_validation = df_validation.sample(frac=1).reset_index(drop = True)  
    
    print("------------------[TEST DATASET]-----------------------")
    print("Number of numbers in the testset: " + str(len(ds_test)))
    balance_test = len(set(classesTest_size))
    if balance_test == 1:
        print("\nClasses are BALANCED!, with " + str(classesTest_size[0]) + " sample for class\n")
    else:
        print("\nClasses are UNBALANCED!\n")    

    remove_file('./test.npy')
    print("Created test database")
    np.save('./test', ds_test.to_numpy())
    print("--------------------------------------------------------\n")

    print("------------------[TRAIN DATASET]-----------------------")
    print("Number of numbers in the trainset: " + str(len(ds_train)))
    balance_train = len(set(classesTrain_size))
    if balance_train == 1:
        print("\nClasses are BALANCED!, with " + str(classesTrain_size[0]) + " sample for class\n")
    else:
        print("\nClasses are UNBALANCED!\n")    

    remove_file('./train.npy')
    print("Created train database")
    np.save('./train', ds_train.to_numpy())
    print("--------------------------------------------------------\n")
    
    print("------------------[VALIDATION DATASET]------------------")
    print("Number of numbers in the validationset: " + str(len(ds_validation)))
    balance_validation = len(set(classesValidation_size))
    if balance_validation == 1:
        print("\nClasses are BALANCED!, with " + str(classesValidation_size[0]) + " sample for class\n")
    else:
        print("\nClasses are UNBALANCED!\n")

    remove_file('./validation.npy')
    print("Created validation database")
    np.save('./validation', ds_validation.to_numpy())
    print("--------------------------------------------------------\n")

def create_dataset():
    X_train_notBal, Y_train_notBal, X_test_notBal, Y_test_notBal = load_datasets()
    #Size of train dataset = 36012
    #Size of test dataset  = 6009
    df_sampled = balance_dataset(create_data(X_train_notBal, Y_train_notBal, X_test_notBal, Y_test_notBal))
    if verify_balance(df_sampled.iloc[:,-1].to_numpy()):
        separeteDatasets(df_sampled)

#--------------------------------------------------------------------------------------------

class Digit:
    def __init__(self, data, target):
        self.target = target
        self.width  = int(np.sqrt(len(data)))
        self.image  = data.reshape(self.width, self.width)
        self.features = {'var' : 0,
                         'std' : 0,
                         'mean_grad_M' : 0,
                         'std_grad_M'  : 0,
                         'mean_grad_D' : 0,
                         'std_grad_D'  : 0,
                         'mean_PC_X'   : 0,
                         'std_PC_X'    : 0,
                         'active_PC_X' : 0,
                         'mean_PC_Y'   : 0,
                         'std_PC_Y'    : 0,
                         'active_PC_Y' : 0}
        self.computeFeatures()
    
    def computeFeatures(self):
        # Feature computation
        mag, ang = sobel(self.image)
        pcx, pcy = pixel_count(self.image)
        
        self.features['var'] = np.var(self.image)
        self.features['std'] = np.std(self.image)
        self.features['mean_grad_M'] = np.mean(mag)
        self.features['std_grad_M'] =  np.std(mag)
        self.features['mean_grad_D'] = np.mean(ang)
        self.features['std_grad_D'] =  np.std(ang)
        self.features['mean_PC_X'] =   np.mean(pcx)
        self.features['std_PC_X'] =    np.std(pcx)
        self.features['active_PC_X'] = np.count_nonzero(pcx)
        self.features['mean_PC_Y'] =   np.mean(pcy)
        self.features['std_PC_Y'] =    np.std(pcy)
        self.features['active_PC_Y'] = np.count_nonzero(pcy) 
  
    def __print__(self):
        print("Digit target: "+str(self.target))
        print("Digit features:")
        print(self.features)
        print("Digit image:")
        plt.gray()
        plt.matshow(self.image) 
        plt.show()

def sobel(image):
    w = len(image)
    kernel_x = np.array([ [ 1, 0,-1],
                          [ 2, 0,-2],
                          [ 1, 0,-1] ])

    kernel_y = np.array([ [ 1, 2, 1],
                          [ 0, 0, 0],
                          [-1,-2,-1] ])
    
    grad_x = np.zeros([w - 2, w - 2])
    grad_y = np.zeros([w - 2, w - 2])
    
    for i in range(w - 2):
        for j in range(w - 2):
            grad_x[i, j] = sum(sum(image[i : i + 3, j : j + 3] * kernel_x))
            grad_y[i, j] = sum(sum(image[i : i + 3, j : j + 3] * kernel_y))
            if grad_x[i, j] == 0:
                grad_x[i, j] = 0.000001 
    
    mag = np.sqrt(grad_y ** 2 + grad_x ** 2)
    ang = np.arctan(grad_y / (grad_x + np.finfo(float).eps))
  
    # Gradient computation
  
    return [mag,ang]

def pixel_count(image):
    pc_x = np.zeros(len(image))
    pc_y = np.zeros(len(image))
  
    # Pixel count computation
    for i in range(len(image)):
        pc_x[i] = np.count_nonzero(image[i, :])
        pc_y[i] = np.count_nonzero(image[:, i])

    return [pc_x, pc_y]

class Dataset:
    def __init__(self, array, length):  
        self.array = array
        self.length = length
        self.digits = []
        self.digits = self.createDigits()
        self.raw_features = [[float(f) for f in dig.features.values()] for dig in self.digits]
        self.raw_targets  = [[self.digits[i].target] for i in range(self.length)]
  
    def createDigits(self):
        digits = []
        for row in self.array:
            digits.append(Digit(np.array(row[:-1]), row[-1]))
        return digits

def load_data_set(array):
    dataset = Dataset(array, len(array))
    
    return dataset

def cvt_obj_nparray(dataset):
    X = np.zeros((dataset.length, 12))
    Y = np.zeros((dataset.length,))
    for i, letter in enumerate(dataset.digits):
        Y[i] = letter.target
        for j, feature in enumerate(letter.features):
            X[i, j] = letter.features[feature]
    return X, Y

def create_data_file(filename):
    #Load the database (.npy) files 
    img_array = np.load(filename) 

    print("Creating dataset...")
    data_set = load_data_set(img_array)
    print ("\nFinished creating dataset\n")

    X_array, Y_array = cvt_obj_nparray(data_set)

    return X_array, Y_array

def create_data_list(filename):
    #Load the database (.npy) files 
    img_array = np.load(filename) 

    print("Creating dataset...")
    data_set = load_data_set(img_array)
    print ("\nFinished creating dataset\n")

    return data_set

#Function that normalize the features
def normalize(arr):
    max_line = np.max(arr, axis=0)
    min_line = np.min(arr, axis=0)
    
    arr = (arr - min_line) / (max_line - min_line)
    
    return arr

def create_all_data():
    create_dataset()
    print("Generating TRAIN data...")
    X_train, Y_train = create_data_file('./train.npy')

    print("Generating TEST data...")
    X_test, Y_test = create_data_file('./test.npy')

    print("Generating VALIDATION data...")
    X_validation, Y_validation = create_data_file('./validation.npy')

    X_total = np.concatenate((X_train, X_validation, X_test), axis = 0)
    X_total_norm = normalize(X_total)

    X_train_norm = X_total_norm[0:len(X_train)]
    X_validation_norm = X_total_norm[len(X_train):len(X_train) + len(X_validation)]
    X_test_norm = X_total_norm[-len(X_test):]

    df_test = join_data(X_test_norm, Y_test)
    df_train = join_data(X_train_norm, Y_train)
    df_validation = join_data(X_validation_norm, Y_validation)

    remove_file('./firstAnalyse/test_classes.npy')
    print("Created test database")
    np.save('./firstAnalyse/test_classes', df_test.to_numpy())

    remove_file('./firstAnalyse/train_classes.npy')
    print("Created train database")
    np.save('./firstAnalyse/train_classes', df_train.to_numpy())

    remove_file('./firstAnalyse/validation_classes.npy')
    print("Created validation database")
    np.save('./firstAnalyse/validation_classes', df_validation.to_numpy())

    #return X_train_norm, Y_train, X_test_norm, Y_test, X_validation_norm, Y_validation

def create_train_data_list():
    create_dataset()
    remove_file('./test.npy')
    remove_file('./validation.npy')
    print("Generating TRAIN data...")
    train_list = create_data_list('./train.npy')
    
    return train_list

def create_test_data_list():
    create_dataset()
    remove_file('./train.npy')
    remove_file('./validation.npy')
    print("Generating TEST data...")
    test_list = create_data_list('./test.npy')
    
    return test_list

def create_validation_data_list():
    create_dataset()
    remove_file('./train.npy')
    remove_file('./test.npy')
    print("Generating VALIDATION data...")
    validation_list = create_data_list('./validation.npy')
    
    return validation_list

def create_train_data():
    create_dataset()
    remove_file('./test.npy')
    remove_file('./validation.npy')
    print("Generating TRAIN data...")
    X_train, Y_train = create_data_file('./train.npy')
    
    return X_train, Y_train

def create_test_data():
    create_dataset()
    remove_file('./train.npy')
    remove_file('./validation.npy')
    print("Generating TEST data...")
    X_test, Y_test = create_data_file('./test.npy')
    
    return X_test, Y_test

def create_validation_data():
    create_dataset()
    remove_file('./train.npy')
    remove_file('./test.npy')
    print("Generating VALIDATION data...")
    X_validation, Y_validation = create_data_file('./validation.npy')
    
    return X_validation, Y_validation

def main():
    create_all_data()

if __name__ == "__main__":
    main()