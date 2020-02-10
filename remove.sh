#!/bin/bash
# remove.sh

######################################################
##########        REMOVE BASH      ###################
######################################################


# This bash is executable only for deleting all dataset 
# .npy files, to make possible to update the Github project 

# Firstly we delete the datasets files

trainFile=./train.npy
testFile=./test.npy
validationFile=./validation.npy

echo "Start Deleting numpy files..."
if [ -f "$trainFile" ];then
	rm -r $trainFile
	echo "Deleted train database!"
else
	echo "Train database does not exist!"
fi

if [ -f "$testFile" ];then
	rm -r $testFile
	echo "Deleted test database!"
else
	echo "Test database does not exist!"
fi

if [ -f "$validationFile" ];then
	rm -r $validationFile
	echo "Deleted validation database!"
else
	echo "Validation database does not exist!"
fi

# Next, we delete the classes files

train_classes_file=./firstAnalyse/train_classes.npy
test_classes_file=./firstAnalyse/test_classes.npy
validation_classes_file=./firstAnalyse/validation_classes.npy

if [ -f "$train_classes_file" ];then
	rm -r $train_classes_file
	echo "Deleted train classes database!"
else
	echo "Train database does not exist!"
fi

if [ -f "$test_classes_file" ];then
	rm -r $test_classes_file
	echo "Deleted test classes database!"
else
	echo "Test database does not exist!"
fi

if [ -f "$validation_classes_file" ];then
	rm -r $validation_classes_file
	echo "Deleted validation classes database!"
else
	echo "Validation database does not exist!"
fi

# Finally we deleate the photos files

photos_classes_file=./firstAnalyse/test_photos_classes.npy
photos_file=./test_photos.npy

if [ -f "$photos_classes_file" ];then
	rm -r $photos_classes_file
	echo "Deleted photos classes database!"
else
	echo "Photos classes database does not exist!"
fi

if [ -f "$photos_file" ];then
	rm -r $photos_file
	echo "Deleted photos database!"
else
	echo "Photos database does not exist!"
fi

# And also the new database generated

photosTrain_classes_file=./secondAnalyse/photosFeatures_train.npy
photosTest_classes_file=./secondAnalyse/photosFeatures_test.npy
photosValidation_classes_file=./secondAnalyse/photosFeatures_validation.npy

if [ -f "$photosTrain_classes_file" ];then
	rm -r $photosTrain_classes_file
	echo "Deleted photos train classes database!"
else
	echo "Train photos database does not exist!"
fi

if [ -f "$photosTest_classes_file" ];then
	rm -r $photosTest_classes_file
	echo "Deleted photos test classes database!"
else
	echo "Test photos database does not exist!"
fi

if [ -f "$photosValidation_classes_file" ];then
	rm -r $photosValidation_classes_file
	echo "Deleted photos validation classes database!"
else
	echo "Validation photos database does not exist!"
fi

# And also some new dataset

photosTrain_classes_file=./secondAnalyse/train.npy
photosTest_classes_file=./secondAnalyse/test.npy
photosValidation_classes_file=./secondAnalyse/validation.npy

if [ -f "$photosTrain_classes_file" ];then
	rm -r $photosTrain_classes_file
	echo "Deleted photos train database!"
else
	echo "Train photos database does not exist!"
fi

if [ -f "$photosTest_classes_file" ];then
	rm -r $photosTest_classes_file
	echo "Deleted photos test database!"
else
	echo "Test photos database does not exist!"
fi

if [ -f "$photosValidation_classes_file" ];then
	rm -r $photosValidation_classes_file
	echo "Deleted photos validation classes database!"
else
	echo "Validation photos does not exist!"
fi

echo "All files removed!"