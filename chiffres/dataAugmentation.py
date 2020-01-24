import numpy as np
#np.random.bit_generator = np.random._bit_generator
import matplotlib.pyplot as plt
import os
import imageio
import imgaug as ia
import imgaug.augmenters as iaa

def main(degree):

    print("Creating more images with a rotation of " + str(degree) + " degrees...")

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

                for rot in range(0, 360, degree):
                    rotate = iaa.Affine(rotate=rot) # rotate image
                    image_rotated = rotate.augment_images([image])[0]
                    imageio.imwrite(path_image + image_name[:-4] + '_rot_' + str(rot) + '.jpg', image_rotated)
            else:
                pass

    print("Finished creating the images!")
# degree: the step of the rotation
# the images will be create in the folder with their originals
'''
if __name__ == "__main__":
    main(30)
'''

