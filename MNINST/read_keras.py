from mnist import MNIST
import os
import csv

faces = [1, 2, 3, 4, 5, 6]

if os.path.isfile('new_test.csv'):
    os.remove('new_test.csv')
else:
    print 'new_test.csv does not exist, so creating one'

try:
    mndata = MNIST(os.getcwd())
    images, labels = mndata.load_training()
    for label in labels:
        if label in faces:            
            new_row = []
            for image in images:
                new_row.append(image)
            new_row.append(label)

            with open('new_test.csv', "a") as file:
                writer = csv.writer(file)
                writer.writerow(new_row)

    print 'DONE SEPARATING'

except:
    print 'MNINST files do not exist!'