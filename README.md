# LAS MACINAS - MACHINE LEARNING PROJECT 2
Constance Gontier, Gianna Crovetto, Hendrik Hilsberg

## Tutorial

To run with our weights saved and our testing dataset cleaned and augmented: 
1. Download train an test sets [link](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation/dataset_files) and put them in the data file
2. Download the files in "logfiles"
3. Just launch "run.py"

To train and run the model: 
1. Download train an test sets [link](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation/dataset_files) and put them in the data file
2. Create a file named logfile 
3. In "config.py", set the MODE to "TRAIN" 
4. Launch "training.py" 
5. In "config.py", set the MODE to "TEST"
6. Launch "run.py" 
7. The desired output will be in /logfiles/submission.csv


## File structure

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
*training.py* contains functions to train ou model  
