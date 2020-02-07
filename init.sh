#!/bin/bash
# init.sh

######################################################
##########      INITIALIZATION BASH      #############
######################################################


# This is a initilization bash that generates all dataset that are used in the study.
# Moreover it generates seconds datasets that can be used for further studies.


genFile=./firstAnalyse/generateDataset.py
imgFile=./diceFaces/photosGenerator.py

echo "Start executing algorithms to create the dataset..."

echo "Creating datasets from MNIST..."
if [ -f "$genFile" ];then
	python $genFile
else
	echo "Python script does not exist!"
fi

echo "Creating datasets from photos taken..."
if [ -f "$imgFile" ];then
	python $imgFile 1
else
	echo "Python script does not exist!"
fi

echo "All data created!"
echo "Ready to use jupyter notebook!"