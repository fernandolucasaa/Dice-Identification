# -*- coding: utf-8 -*-
# photosGenerator.py --- Convert taken photos into .npy files

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


import numpy as np
import matplotlib.pyplot as plt
import os
import imgaug as ia
import imgaug.augmenters as iaa
import zipfile
import shutil
import cv2 as cv
from scipy import ndimage
from collections import Counter
import pandas as pd
import sys

# This function verify if a file exists and if so it will remove it.
def remove_file(filename):
    if os.path.isfile(filename):
        print("Removing " + filename)
        os.remove(filename)
    else:
        print(filename + "file does not exist, creating one!")

# This function verify if a folder exists and if so it will remove it.
def remove_folder(foldername):
    if os.path.isdir(foldername):
        print("Removing " + foldername)
        shutil.rmtree(foldername)
    else:
        print(foldername + " folder does not exist, creating one!")

# This function unzip the EMNIST letters database zip folder in a specific folder.
# The unzipped files are binary files of the database
def unzipPhotos(unzipFolder, unzipFile):
    with zipfile.ZipFile(unzipFile, 'r') as zip_file:
        zip_file.extractall(unzipFolder)


# Image processing to follow the format of dataset MNIST
#
# Parameters:
# - image_path: image's path
# - show: 1 to show each image result in the processing
# 
# Return :
# - the processed image (numpy.array)

def image_processing(image_path, show):
    
    # read the image
    # format : <class 'numpy.ndarray'>, shape : (x, y, 3)
    image = cv.imread(image_path, 1)
    # convert the color to gray
    # format : <class 'numpy.ndarray'>, shape : (x, y)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    if show == 1:
        print("Printing the source image in gray scale:")
        plt.imshow(gray, cmap="gray")
        plt.show()
    
    # invert the pixels of the image : 255 (white) -> 0 (black)
    _, binary_invert = cv.threshold(gray, 225, 255, cv.THRESH_BINARY_INV) #OBS: gray is a 2 dimension numpy array
    
    if show == 1:
        print("Printing the binarized invert image")
        plt.imshow(binary_invert, cmap="gray")
        plt.show()
        
    # calculate all the countors of the image
    # the variable contours contains points of the countors
    contours, _ = cv.findContours(binary_invert, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # calculate the area of the contours points 
    for cnt in contours:

        # calcul de la zone des contours
        area = cv.contourArea(cnt)

        # if the contour's area is sufficiently large, it must be the digit
        if area > 50 and area < 700:

            # compute the bounding box
            # (x,y) : top-left coordinate of the rectangle 
            # (w,h) : width, height
            (x, y, w, h) = cv.boundingRect(cnt)

            # affiche les rectangles sur l'image
            #cv.rectangle(gray, (x,y), (x+w,y+h), (255,255,0), 2) 

            # extract the region of interest (ROI)
            diceROI = binary_invert[y-2 : y+h+2, x-2 : x+w+2]
            
            if show == 1:
                print("Printing the region of intest:")
                plt.imshow(diceROI, cmap="gray")
                plt.show()

            break

    # calculate the center of mass, the height and the width of the ROI
    centre_h, centre_w = ndimage.measurements.center_of_mass(diceROI)
    h, w = diceROI.shape

    # insert the ROI in a new image (56x56)
    n = 56
    
    black_image = np.zeros((n,n))
    black_image[int((n-h)/2) : int(((n-h)/2)+h), int((n-w)/2) : int(((n-w)/2)+w)] = diceROI 
    
    if show == 1:
        print("Printing the ROI insertion:")
        plt.imshow(black_image, cmap="gray")
        plt.show()

    # resizing to the 28x28 format
    scale_percent = 50
    
    width = int(black_image.shape[1] * scale_percent / 100)
    height = int(black_image.shape[0] * scale_percent / 100)
    
    dim = (width, height)

    # resize the image
    resized = cv.resize(black_image, dim, interpolation = cv.INTER_AREA)
    
    if show == 1:
        print("Printing the resized image:")
        plt.imshow(resized, cmap="gray")
        plt.show()

    # remove noise
    kernel = np.ones((2,2),np.uint8)
    opening = cv.morphologyEx(resized, cv.MORPH_OPEN, kernel)
    
    processed_image = opening
    
    # second binarization
    #_, binary2 = cv.threshold(processed_image, 100, 255, cv.THRESH_BINARY)
    
    return processed_image

# Process all the photos in a specific folder 
# and return a list with all the processed images
#
# Parameters:
# - folder_path: folder that contains the photos of a same dice's face
#
# Return:
# - list with all processed images

def image_processing_list(folder_path):
    
    # all images in the folder
    image_list = []
    
    # loop over all the elements
    for im in os.listdir(folder_path):
        image_path = folder_path + '/' + im
        image_list.append(image_processing(image_path, 0))
        
    return image_list

# Create a list with all the images adding the target to each image
#
# Parameters:
# - dataset_path: path with all the folders, each of theses folders has images of one
# dice's face
#
# Return:
# - dataset_list: list with all the images with respective targets

def create_data_set(dataset_path):
    
    test_list = []
    train_list = []
    validation_list = []
    
    # loop over all the folders (1 until 6)
    for folder_name in os.listdir(dataset_path):        
        folder_path = dataset_path + '/' + folder_name

        filter_coef = 1

        total_length = int(len(os.listdir(folder_path)) * filter_coef)
        test_numbers = int(0.2 * total_length)
        train_numbers = int(0.64 * total_length)
        validation_numbers = int(0.16 * total_length)

        num_image = 0 

        # loop over all the images
        for im in os.listdir(folder_path):  
            num_image += 1
            image_path = folder_path + '/' + im
            image_processed = image_processing(image_path, 0)
            
            # convert the (28, 28) numpy array to a list (784)
            row = image_processed.reshape(-1)
            row = row.tolist()
            
            # add the target
            target = float(folder_name)
            row.append(target)
            
            # add the complete line
            if num_image <= test_numbers:
                test_list.append(row)
            elif num_image > test_numbers and num_image <= test_numbers + train_numbers:
                train_list.append(row)
            else:
                validation_list.append(row)

    dataset_test = load_data_set(np.asarray(test_list))
    X_test, Y_test = cvt_obj_nparray(dataset_test)

    dataset_train = load_data_set(np.asarray(train_list))
    X_train, Y_train = cvt_obj_nparray(dataset_train)

    dataset_validation = load_data_set(np.asarray(validation_list))
    X_validation, Y_validation = cvt_obj_nparray(dataset_validation)

    return X_test, Y_test, X_train, Y_train, X_validation, Y_validation


# Create a list with all the processed images adding the target to each image
# The processed images are rotated to augmentated the data set
#
# Parameters:
# - dataset_path: path with all the folders, each of theses folders has images of one
# dice's face
# - degree: rotation's degree
# 
# Return:
# - dataset_list: list with all the processed images with respective targets

def create_augmented_data_set_mnist(dataset_path, degree):
    
    dataset_list = []
    
    # loop over all the folders (1 until 6)
    for folder_name in os.listdir(dataset_path):
        folder_path = dataset_path + '/' + folder_name
        
        # loop over all the images
        for im in os.listdir(folder_path):
            image_path = folder_path + '/' + im
            image_processed = image_processing(image_path, 0)
            
            # rotate the processed image
            for rot in range(0, 360, degree):
                
                rotate = iaa.Affine(rotate=rot) # rotate image
                image_rotated = rotate.augment_images([image_processed])[0] # rotated image

                # convert the (28, 28) numpy array to a list (784)
                new_row = image_rotated.reshape(-1).tolist()

                # add the target
                target = float(folder_name)
                new_row.append(target)
            
                # add the complete line
                dataset_list.append(new_row)
    
    return dataset_list

# Create a list with all the processed images adding the target to each image
# The processed images are rotated to augmentated the data set
#
# Parameters:
# - dataset_path: path with all the folders, each of theses folders has images of one
# dice's face
# - degree: rotation's degree
# 
# Return:
# - dataset_list: list with all the processed images with respective targets
def create_augmented_data_set(dataset_path, degree):
    
    test_list = []
    train_list = []
    validation_list = []
    
    # loop over all the folders (1 until 6)
    for folder_name in os.listdir(dataset_path):        
        folder_path = dataset_path + '/' + folder_name

        filter_coef = 1
        
        total_length = int(len(os.listdir(folder_path)) * filter_coef)
        test_numbers = int(0.2 * total_length)
        train_numbers = int(0.64 * total_length)
        validation_numbers = int(0.16 * total_length)

        num_image = 0 

        # loop over all the images
        for im in os.listdir(folder_path):  
            num_image += 1
            image_path = folder_path + '/' + im
            image_processed = image_processing(image_path, 0)

            # rotate the processed image
            for rot in range(0, 360, degree):
                
                rotate = iaa.Affine(rotate=rot) # rotate image
                image_rotated = rotate.augment_images([image_processed])[0] # rotated image
            
                # convert the (28, 28) numpy array to a list (784)
                row = image_processed.reshape(-1)
                row = row.tolist()
                
                # add the target
                target = float(folder_name)
                row.append(target)
                
                # add the complete line
                if num_image <= test_numbers:
                    test_list.append(row)
                elif num_image > test_numbers and num_image <= test_numbers + train_numbers:
                    train_list.append(row)
                else:
                    validation_list.append(row)

    dataset_test = load_data_set(np.asarray(test_list))
    X_test, Y_test = cvt_obj_nparray(dataset_test)

    dataset_train = load_data_set(np.asarray(train_list))
    X_train, Y_train = cvt_obj_nparray(dataset_train)

    dataset_validation = load_data_set(np.asarray(validation_list))
    X_validation, Y_validation = cvt_obj_nparray(dataset_validation)

    return X_test, Y_test, X_train, Y_train, X_validation, Y_validation
#---------------------------------------------------------------------------------------------------#

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
    print ("Finished creating dataset\n")

    X_array, Y_array = cvt_obj_nparray(data_set)

    return X_array, Y_array

#Function that normalize the features
def normalize(arr):
    max_line = np.max(arr, axis=0)
    min_line = np.min(arr, axis=0)
    
    arr = (arr - min_line) / (max_line - min_line)
    
    return arr


def join_data(X_data, Y_data):
    new_array = []
    if len(X_data) == len(Y_data):
        for i in range(len(X_data)):
            line = np.asarray(X_data[i]).reshape(-1).tolist()
            line.append(Y_data[i])
            
            new_array.append(line)

    return pd.DataFrame(new_array)

def balance_dataset(dataset):
    Y_data = dataset.iloc[:,-1].to_numpy()
    Y_data_count = Counter(Y_data)
    less_number = min(Y_data_count, key = Y_data_count.get)
    quantity = int(Y_data_count[less_number] / 2)
        
    for i in range(1, 7):
        dl = dataset.loc[dataset[12] == i]
        if quantity*2 != Y_data_count[less_number]:
            dh = dl.head(quantity+1)
        else:
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

def save_feature_file(X_data_notNorm, Y_data, filename):
    X_data = normalize(X_data_notNorm)
    print("Checking balance of classes...")
    if not(verify_balance(Y_data)):
        print("Classes are UNBALANCED!")
        print(Counter(Y_data)) 
        print("\nBalancing the dataset...")
        ds_data = balance_dataset(join_data(X_data, Y_data))
        print("Re-checking balance of classes...")
        if verify_balance(ds_data.iloc[:,-1].to_numpy()):
            print("Classes are BALANCED!\n")
    else:
        print("Classes are BALANCED!\n")
        ds_data = join_data(X_data, Y_data)

    df_data = ds_data.sample(frac=1).reset_index(drop = True)

    print("Number of digits in the dataset: " + str(len(df_data)))
    print("Number of digits per class: " + str(len(df_data)/6))

    # Creating only one dataset, in this case a test dataset
    remove_file(filename + '.npy')
    print("Created test database\n")
    np.save(filename, df_data.to_numpy())

# This is the main function of this program.    
def main(argv):

    photos_zip_file = './diceFaces/newPhotos.zip'
    final_folder = './diceFaces/newPhotos'
    
    angle = int(argv[0])

    # Some cheking if the files are in the coorect folder.
    if not(os.path.isfile(photos_zip_file)):
        print("Verify that the photos zip file is in the folder!")
        sys.exit()
    elif not(os.path.isdir(final_folder)):
        print("Extracting all photos...")
        unzipPhotos(final_folder, photos_zip_file)
        print("Finished unzipping file\n")


    # The test_photos are the images that will be used in the MNIST machine learning
    test_photos = create_augmented_data_set_mnist(final_folder, angle)

    remove_file('./test_photos.npy')
    print("Created test photos database\n")
    np.save("./test_photos", test_photos)
    
    
    # Creating dataset to compare features in machine learning comparing to MNIST
    X_photos_notNorm, Y_photos = create_data_file('./test_photos.npy')
    save_feature_file(X_photos_notNorm, Y_photos, './firstAnalyse/test_photos_classes')

    # This part is to generate the 3 datasets: Train, Test and Validation 
    #that will be used to study the machine learning problem without MNIST
    X_test_notNorm, Y_test, X_train_notNorm, Y_train, X_validation_notNorm, Y_validation = create_data_set(final_folder)
    print("------------------[TEST PHOTOS DATASET]-----------------------")
    save_feature_file(X_test_notNorm, Y_test, './secondAnalyse/photosFeatures_test')
    print("------------------[TRAIN PHOTOS DATASET]-----------------------")
    save_feature_file(X_train_notNorm, Y_train, './secondAnalyse/photosFeatures_train')
    print("------------------[VALIDATION PHOTOS DATASET]-----------------------")
    save_feature_file(X_validation_notNorm, Y_validation, './secondAnalyse/photosFeatures_validation')

    # Delete all binary unzipped files to reduce the size of the project
    print("Deleting folder with all photos...")
    remove_folder(final_folder)
    print("Folder deleted!\n")
    
    print("Finished program")

if __name__ == "__main__":
    main(sys.argv[1:])
    

