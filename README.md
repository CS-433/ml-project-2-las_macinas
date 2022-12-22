# Machine Learning Project 2 - Road Segmentation 

For the EPFL course Machine Learning CS-433, we developed a binary classifier to identify roads in satellite images and distinguish them from the background. We compared various CNN and U-Net architectures to determine the best model for this task. Our approach also included image pre-processing steps to improve the accuracy of the classifier. Our results showed that the U-Net model achieved an F-1 score of 0.94. This project demonstrates the effectiveness of using U-Net model for road segmentation in satellite imagery.

The structure of the image dataset and their groundtruth can be seen in the following image:

![Alt text](/data-label.png?raw=true "Data-Label")


## Tutorial

To run with our weights saved and our testing dataset cleaned and augmented: 
1. Download train an test sets [link](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation/dataset_files) and put them in the data file
2. Just launch "run.py"

To train and run the model to have predictions: 
1. Download train an test sets [link](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation/dataset_files) and put them in the data file
2. Create a folder named models and another folder submission 
3. In "constants.py", set the TRAINING_MODE to True and TESTING_MODE to True 
4. Launch "run.py" 
7. The desired predictions will be in /submission/submission.csv

## Structure

<pre>
.  
├── data                    # Dataset for training/ testing the model  
├── report                  # Report of the project  
├── src                     # Source files
│    └── submissions        # Folder containing the prediction
│    └── models             # Savestate of the best models    
│    └── run.py             # Run the model : training and testing
└── README.md  
</pre>

## Dataset

To run the model or to train it, the data has to be structured in a specific way with respect to the root directory where the 'run.py' file is situated. 

<pre>
.  
├── ...  
├── data  
│   ├── training  
│           └── groundtruth         # Groundtruth of the training images (400x400)  
│           └── images              # Training images (400x400 RGB)  
│   └── test                        # Test images (608x608 RGB)  
└── ...  
</pre>

## Model used

![Net](https://user-images.githubusercontent.com/26313021/151218916-fc29920a-4dd3-43a0-9a9c-f678f34cfc08.png)

## Run the code 
In order to run the best model that was implemented the following command has to be sent in a terminal: i
```
python run.py
```
This has to be done with the data organised as mentioned above.

## Results
With the optimal model, one of the Unet versions the following results are obtained:


|           | Validation F1-score | Validation accuracy   | Test F1-score | Test Accuracy |
|:---------:|:-------------------:|:---------------------:|:-------------:|:-------------:|
| U-Net     |        0.94         |         -----         |     0.899     |     0.945     |

The prediction on an image is :

![Alt text](/prediction.png?raw=true "Image and Predcition")
## Authors - LAS MACINAS

- Constance Gontier [@constig](https://github.com/constig)
- Gianna Crovetto [@crovetto](https://github.com/crovetto)
- Hendrik Hilsberg [@hhilsber](https://github.com/hhilsber)
