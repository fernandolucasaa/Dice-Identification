import numpy as np
#np.random.bit_generator = np.random._bit_generator
import matplotlib.pyplot as plt
import os
import imageio
import imgaug as ia
import imgaug.augmenters as iaa

import sys
import cv2 as cv
import pandas as pd
import numpy as np
import math

def remove_file(filename):
    if os.path.isfile(filename):
        print("Removing " + filename)
        os.remove(filename)
    else:
        print(filename + "file does not exist, creating one!")

def rotate_images(degree):

    print("Creating more images with a rotation of " + str(degree) + " degrees...")

    img1 = os.getcwd() + '/chiffres/Numero1/'
    img2 = os.getcwd() + '/chiffres/Numero2/'
    img3 = os.getcwd() + '/chiffres/Numero3/'
    img4 = os.getcwd() + '/chiffres/Numero4/'
    img5 = os.getcwd() + '/chiffres/Numero5/'
    img6 = os.getcwd() + '/chiffres/Numero6/' 

    img = [img1, img2, img3, img4, img5, img6]

    # resize the image
    width = 28
    height = 28
    dim = (width, height)

    images = []

    for image_path_chiffre in img: 
        dice_number = image_path_chiffre[-2]
        for image_name in os.listdir(image_path_chiffre):

            path_image = image_path_chiffre + image_name
            
            if path_image[-3:] == "JPG" or path_image[-3:] == "jpg":
                image = imageio.imread(path_image)

                for rot in range(0, 360, degree):
                    rotate = iaa.Affine(rotate=rot) # rotate image
                    image_rotated = rotate.augment_images([image])[0]
                    resized = cv.resize(image_rotated, dim, interpolation = cv.INTER_AREA)
                    final_im = (resized[:, :, 0]).reshape(-1)
                    
                    row = []
                    for pixel in final_im:
                        row.append(pixel)
                    row.append(float(dice_number))

                    images.append(row)
                    
    return pd.DataFrame(images)

def random_dataset(source_df, perc1, perc2):
    dice_numbers = [1, 2, 3, 4, 5, 6]
    
    print("Percentage of the trainset: " + str(perc1))
    print("Percentage of the testset: " + str(perc2))
    print("Percentage of the validationset: " + str(1-perc1-perc2) + "\n")

    for number in dice_numbers:
        dn = source_df.loc[source_df[784] == number]
        train_numbers = int(math.ceil(len(dn) * perc1))
        test_numbers = int(math.ceil(len(dn) * perc2))
        validation_numbers = len(dn) - train_numbers - test_numbers

        if validation_numbers <= 0:
            validation_numbers = 1

        df_percTrain = dn.head(train_numbers)
        df_percValidation = dn[(train_numbers):(train_numbers + validation_numbers)]
        df_percTest = dn.tail(test_numbers)
        if number == 1:
            df_train = df_percTrain 
            df_test = df_percTest
            df_validation = df_percValidation
        else:
            df_train = df_train.append(df_percTrain)
            df_test = df_test.append(df_percTest)
            df_validation = df_validation.append(df_percValidation)

    ds_train = df_train.sample(frac=1).reset_index(drop = True)
    ds_test = df_test.sample(frac=1).reset_index(drop = True) 
    ds_validation = df_validation.sample(frac=1).reset_index(drop = True)  
 
    remove_file('train.npy')
    print("Creating train database\n")
    np.save('train',ds_train.to_numpy())

    remove_file('test.npy')
    print("Creating test database\n")
    np.save('test',ds_test.to_numpy())
    
    remove_file('validation.npy')
    print("Creating validation database\n")
    np.save('validation',ds_validation.to_numpy())
    
    # Save in a csv file, used before
    # ds_train.to_csv('training_database.csv', header = None, index = False)
    # ds_test.to_csv('testing_database.csv', header = None, index = False)
    # ds_validation.to_csv('validation_database.csv', header = None, index = False)

def main():
    # degree: the step of the rotation
    # the images will be create in the folder with their originals

    answer = raw_input('Do you want to generate more images? [y/n]\n')
    if answer != 'Y' and answer != 'y':
        sys.exit()

    answer = raw_input('Are you sure? [y/n]\n')
    if answer != 'Y' and answer != 'y':
        sys.exit() 

    degree = raw_input('How many degrees step?\n')
    if int(degree) <= 0:
        sys.exit()     

    proportion_trainset = raw_input('How much do you want to use as trainset?\n')
    if int(proportion_trainset) < 0:
        print("Percentage must be positive!")
        sys.exit()  
    elif int(proportion_trainset) > 1:
    	print("Converting in percentage\n")
        proportion_trainset = (float(proportion_trainset) / 100)
    elif int(proportion_trainset) >= 0 and int(proportion_trainset) <= 1:
        pass

    print("The percentage of the tetset must be smaller than " + str(1 - proportion_trainset))
    proportion_testset = raw_input('How much do you want to use as testset?\n')
    if int(proportion_testset) < 0:
        print("Percentage must be positive!")
        sys.exit()  
    elif int(proportion_testset) > 1:
    	print("Converting in percentage\n")
        proportion_testset = (float(proportion_testset) / 100)
    elif int(proportion_testset) >= 0 and int(proportion_testset) <= 1:
        pass

    image_df = rotate_images(int(degree))
    random_dataset(image_df, proportion_trainset, proportion_testset)

    print("Finished creating databases!")

if __name__ == "__main__":
    main()

'''
#This function test the files created
import numpy as np
from matplotlib import pyplot as plt

img_array_train = np.load('./chiffres/train.npy')
img_array_test = np.load('./chiffres/test.npy')
img_array_validation = np.load('./chiffres/validation.npy')

print img_array_train.shape
print img_array_test.shape
print img_array_validation.shape

def images(array):
    for img in array:
        plt.imshow(img[:-1].reshape(28,28), cmap='gray')
        plt.show()
        
images(img_array_train)
images(img_array_test)
images(img_array_validation)

'''
