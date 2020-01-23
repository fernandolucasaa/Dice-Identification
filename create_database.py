import numpy as np
import csv
import cv2 as cv
import os
import pandas as pd
import math
import sys

def remove_file(filename):
    if os.path.isfile(filename):
        print("Removing " + filename + '\n')
        os.remove(filename)
    else:
        print(filename + "file does not exist, creating one!\n")

def create_database(image_path):
    
    # set the parameters to read the images
    image_path_chiffre = image_path # path of the file
    chiffre = image_path_chiffre[-2] # number of the dice

    # list to store all the images in the folder
    image_list = []
    for x in os.listdir(image_path_chiffre):
        path_image = image_path_chiffre + x
        im = cv.imread(path_image, 0)
        #im = cv.imread(path_image)
        image_list.append(im)
        
    for im in image_list:    
        
        # resize the image
        width = 28
        height = 28
        dim = (width, height)
        
        resized = cv.resize(im, dim, interpolation = cv.INTER_AREA)
        
        # write the pixel matrix in the test.csv file, the row will have 785 columns (28x28 + 1)
        # where the last column is the dice's number
        A = np.asarray(resized).reshape(-1) # shape the matrix to just one line
        
        row = []
        for result in A:
            row.append(result)
        row.append(float(chiffre)) #the last column is the dice's number

        # open the file in the for appending new information (new row)
        with open('test.csv', "a") as file:
            writer = csv.writer(file)
            writer.writerow(row)


def random_csv(source_file, new_file_train, new_file_test, perc):
    dice_numbers = [1, 2, 3, 4, 5, 6]

    df = pd.read_csv(source_file, header = None)
    
    if perc < 0:
        print("Percentage must be positive!")
        print("Traindataset and testdataset not created!")
        sys.exit()
    elif perc > 1:
        perc = (float(perc) / 100)
    elif perc >= 0 and perc <= 1:
        pass

    print("Percentage of the trainningset: " + str(perc))
    print("Percentage of the trainningset: " + str(1-perc) + "\n")

    for number in dice_numbers:
        dn = df.loc[df[784] == number]
        train_numbers = int(math.ceil(len(dn) * perc))
        test_numbers = len(dn) - train_numbers
        if test_numbers == 0:
            test_numbers = 1

        df_percTrain = dn.head(train_numbers)
        df_percTest = dn.tail(test_numbers)
        if number == 1:
            df_train = df_percTrain 
            df_test = df_percTest
        else:
            df_train = df_train.append(df_percTrain)
            df_test = df_test.append(df_percTest)

    ds_train = df_train.sample(frac=1).reset_index(drop = True)
    ds_test = df_test.sample(frac=1).reset_index(drop = True)  
    
    ds_train.to_csv(new_file_train, header = None, index = False)
    ds_test.to_csv(new_file_test, header = None, index = False)

    remove_file(source_file)

def main(division):

    testFile = 'test.csv'
    trainingFile = 'training_database.csv'
    testingFile = 'testing_database.csv'

    remove_file(testFile)

    chiffre_path1 = os.getcwd() + '/chiffres/Numero1/'
    chiffre_path2 = os.getcwd() + '/chiffres/Numero2/'
    chiffre_path3 = os.getcwd() + '/chiffres/Numero3/'
    chiffre_path4 = os.getcwd() + '/chiffres/Numero4/'
    chiffre_path5 = os.getcwd() + '/chiffres/Numero5/'
    chiffre_path6 = os.getcwd() + '/chiffres/Numero6/' 

    img_paths = [chiffre_path1, chiffre_path2, chiffre_path3, chiffre_path4, chiffre_path5, chiffre_path6]

    print("Creating data base with photos taken...\n")

    for path in img_paths:
        create_database(path)
        print("Data base for number " + str(path[-2]) + " created")

    print("\nData base created\n")

    remove_file(trainingFile)
    remove_file(testingFile)
 
    print("Creating training and testing database randomly...")
    random_csv('test.csv','training_database.csv', 'testing_database.csv', division)  
    print("Training and testing database created\n")


if __name__ == "__main__":
    main(0.8)
