# Dice-Identification
This project uses jupyter notebook to show the results and to use the classification algorithms.

If it is your first time using python, you have to install it. For doing so, you can follow the tutorial below.

1. Installing Python (https://www.python.org/)

To check if python is installed on your computer, you must open a terminal and write "python". If you see the python environment, then it is installed, otherwise to install you must write the command "sudo apt get install python2.6" (any version of python2, because then you can upgrade it). So, to update the python, use the command "apt install -only -upgrade python". Finally, to install "pip" which is a python library installation utility, use the command "sudo apt install python-pip".

Next, in this project it was used the Jupyter Lab as an interface to better visualize the results. So to install it you can follow the tutorial below.

2. Installing Jupyter Lab (https://jupyter.org/)

The process to install jupyter lab is shown in the jupyter website (https://jupyter.org/install). 
With conda, install with the command: "conda install -c conda-forge jupyterlab"
Next, use the command "conda install -c conda-forge notebook" to install the notebook
With pip, install with the command: "pip install jupyterlab"
Next, use the command "pip install notebook" to install the notebook

Finally to execute and use this project. Here are some guidelines to understand the project organization. 

3. Viewing and using the notebooks created in jupyter lab:

	IMPORTANT: Before opening any jupyter notebook, you must execute the bash script "init.sh" to generate all datasets used in the classificators.
	
	3.1. Lunch the jupyter notebook (in the command line type "jupyter notebook" in the project folder downloaded from Github) 

	3.2. The file "main.ipynb" analyse four classification methods: KNN, Decision Tree, Random Forest and SVM, all of them using the MNIST handwritten digits database (http://yann.lecun.com/exdb/mnist/) and also the photos taken from a pico camera.

	3.3. The file "knn.ipynb" analyse just the KNN algorithm, its purpose is to analyse its hyperparameter to chose the best hyperparameter value and use it to classify the handwritten digits from the photos taken.

	.4. The file "decisionTree.ipynb" analyse just the Decision Tree algorithm, its purpose is to analyse its hyperparameter to chose the best hyperparameter value and use it to classify the handwritten digits from the photos taken.

	3.5. The file "randomForest.ipynb" analyse just the Random Forest algorithm, its purpose is to analyse its hyperparameter to chose the best hyperparameter value and use it to classify the handwritten digits from the photos taken.

	3.6. The file "svm.ipynb" analyse just the Support Vector Machine algorithm, its purpose is to analyse its hyperparameter to chose the best hyperparameter value and use it to classify the handwritten digits from the photos taken.

