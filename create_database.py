import numpy as np
import csv
import cv2 as cv
import os
import pandas as pd

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


def random_csv(source_file, new_file):
    df = pd.read_csv(source_file, header = None)
    ds = df.sample(frac=1).reset_index(drop = True)  
    ds.to_csv(new_file, header = None, index = False)
    #os.remove('test.csv')

def main():

    if os.path.isfile('test.csv'):
#        answer = input("The 'test.csv' file already exists. Do you want to remove it? yes or no?")
#        if answer == 'yes':
        os.remove('test.csv')
#            print("Test file removed")
#        else:
#        	pass

    chiffre_path1 = os.getcwd() + '/chiffres/Numero1/'
    chiffre_path2 = os.getcwd() + '/chiffres/Numero2/'
    chiffre_path3 = os.getcwd() + '/chiffres/Numero3/'
    chiffre_path4 = os.getcwd() + '/chiffres/Numero4/'
    chiffre_path5 = os.getcwd() + '/chiffres/Numero5/'
    chiffre_path6 = os.getcwd() + '/chiffres/Numero6/' 

    img_paths = [chiffre_path1, chiffre_path2, chiffre_path3, chiffre_path4, chiffre_path5, chiffre_path6]

    print("------------------------")
    print("Creating test data base")

    for path in img_paths:
        create_database(path)
        print("Data base for number " + str(path[-2]) + " created")

    print("Test data base created")
    print("------------------------")

    print("Creating training data base randomly")
    random_csv('test.csv','training_database.csv')
    print("Training data base created")
    print("------------------------")

    print("Creating testing data base randomly")
    random_csv('test.csv','testing_database.csv')
    print("Testing data base created")
    print("------------------------")

'''
if __name__ == "__main__":
    # execute only if run as a script
    main()
'''