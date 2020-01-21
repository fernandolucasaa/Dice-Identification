import csv
import numpy as np
import os

def convert(imgf, labelf, outf, n):
    f = open(imgf, "rb")
    o = open(outf, "w")
    l = open(labelf, "rb")

    f.read(16)
    l.read(8)
    images = []

    for i in range(n):
        image = [ord(l.read(1))]
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image)+"\n")
    f.close()
    o.close()
    l.close()

def generate_mninstset_files(mninst_test_csv, mninst_train_csv, faces):
    mninst_test_filename = 'mnist_test.csv'
    mninst_train_filename = 'mnist_train.csv'

    if os.path.isfile(mninst_test_csv):
        os.remove(mninst_test_csv)
    else:
        print 'New mninst_test.csv does not exist, so creating one'

    if os.path.isfile(mninst_train_csv):
        os.remove(mninst_train_csv)
    else:
        print 'New mninst_train.csv does not exist, so creating one'

    with open(mninst_test_filename, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter = ',', quotechar = '|', quoting = csv.QUOTE_NONNUMERIC)
        with open(mninst_test_csv, "a") as file:
            for row in spamreader:
                if row[0] in faces:
                    new_row = []
                    for result in row[1:]:
                        new_row.append(result)
                    new_row.append(row[0])

                    writer = csv.writer(file)
                    writer.writerow(new_row)

    print 'DONE SEPARATING MNINST TEST'

    with open(mninst_train_filename, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter = ',', quotechar = '|', quoting = csv.QUOTE_NONNUMERIC)
        with open(mninst_train_csv, "a") as file:
            for row in spamreader:
                if row[0] in faces:
                    new_row = []
                    for result in row[1:]:
                        new_row.append(result)
                    new_row.append(row[0])

                    writer = csv.writer(file)
                    writer.writerow(new_row)

    print 'DONE SEPARATING MNINST TRAIN'

def main():

    convert("./data/train-images-idx3-ubyte", "./data/train-labels-idx1-ubyte",
            "mnist_train.csv", 60000)
    convert("./data/t10k-images-idx3-ubyte", "./data/t10k-labels-idx1-ubyte",
            "mnist_test.csv", 10000)
    
    mninst_test_csv = os.getcwd()+'/faces1_3/mnist_testset.csv'
    mninst_train_csv = os.getcwd()+'/faces1_3/mnist_trainset.csv'

    faces = [1.0, 2.0, 3.0]

    print 'GENERATING MNINST FILES FOR NUMBERS: 1, 2 and 3'
    generate_mninstset_files(mninst_test_csv, mninst_train_csv, faces)

    mninst_test_csv = os.getcwd()+'/faces4_6/mnist_testset.csv'
    mninst_train_csv = os.getcwd()+'/faces4_6/mnist_trainset.csv'
    faces = [4.0, 5.0, 6.0]

    print 'GENERATING MNINST FILES FOR NUMBERS: 4, 5 and 6'
    generate_mninstset_files(mninst_test_csv, mninst_train_csv, faces)

main()


