import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
import imgaug as ia
import imgaug.augmenters as iaa

img1 = os.getcwd() + '/Numero1/'
img2 = os.getcwd() + '/Numero2/'
img3 = os.getcwd() + '/Numero3/'
img4 = os.getcwd() + '/Numero4/'
img5 = os.getcwd() + '/Numero5/'
img6 = os.getcwd() + '/Numero6/' 

image_path_test = os.getcwd() + '/chiffres'

img = [img1, img2, img3, img4, img5, img6]

for image_path_chiffre in img: 
    for image_name in os.listdir(image_path_chiffre):

        path_image = image_path_chiffre + image_name
        
        if path_image[-3:] == "JPG" or path_image[-3:] == "jpg":
            image = imageio.imread(path_image)

            for rot in range(0, 360, 120):
                rotate = iaa.Affine(rotate=rot) # rotate image
                image_rotated = rotate.augment_images([image])[0]
                imageio.imwrite(image_path_test + image_name[:-4] + '_rot_' + str(rot) + '.jpg', image_rotated)
        else:
            pass


