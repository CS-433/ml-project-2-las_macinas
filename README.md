# LAS MACINAS - MACHINE LEARNING PROJECT 2
Constance Gontier, Gianna Crovetto, Hendrik Hilsberg

## Tutorial

To run with our weights saved and our testing dataset cleaned and augmented: 
1. Download train an test sets [link](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs/dataset_files) and put them in the data file
2. Download the files in "logfiles"
3. Just launch "run.py"

To train and run the model: 
1. Download train an test sets [link](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs/dataset_files) and put them in the data file
2. Create a file named logfile 
3. In "config.py", set the MODE to "Train" 
4. Launch "train.py" 
5. In "config.py", set the MODE to "Test"
6. In "run.py", decomment the lines 25 to 27 and comment line 28
7. Launch "run.py" 
8. The desired output will be in /logfiles/submission.csv


## File structure
The underdash i (_i) means there is one file per PRI jet number(0,1,2,3)  

├── config.py  
├── cross_validation.py  
├── data_augmented.py  
├── helpers.py  
├── implementations.py  
├── loss.py  
├── models.py  
├── preprocessing.py  
├── run.py  
└── train.py  
├── data  
│   ├── <sample-submission.csv>  
│   ├── <train.csv>  
│   └── <test.csv>   
├── logfiles  
│   ├── col_i.csv  
│   ├── cross_array_i.csv  
│   ├── data_test_augmented_i.npy  
│   ├── data_train_augmented_i.npy  
│   ├── sample-submission.csv  
│   └── weights_i.csv  
└── README.md



## File description
*config.py* contains constants, modulable modes and paths  
*cross_validation.py* contains the cross validation algorithm  
*data_augmented.py* contains our data augmentation techniques  
*helpers.py* contains some helper functions  
*implementations.py* contains our regression functions  
*loss.py* contains everything related to losses and gradients  
*models.py* contains the model selection  
*preprocessing.py* contains our preprocessing steps  
*run.py* contains functions to get a prediction and accuracy of the trained model  
*train.py* contains functions to train ou model  
