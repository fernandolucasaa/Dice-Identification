#!/bin/bash
# init.sh

genFile=./firstAnalyse/generateDataset.py

echo "Start executing algorithms to create the dataset..."

if [ -f "$genFile" ];then
	python $genFile
else
	echo "Python script does not exist!"
fi

echo "Ready to use jupyter notebook!"

#echo "Creating rotated photo letters..."
#cd ../application/photos/
#python ./dataAugmentation.py 8
#echo "Letters created!"
#echo "All data created!"
