# Path of datasets
DATA_DIR = "../data/training/"
TRAIN_DATA_FILENAME = DATA_DIR + "images/"
TRAIN_LABELS_FILENAME = DATA_DIR + "groundtruth/"
TEST_DATA_FILENAME = '../data/test_set_images/'
BEST_MODEL_FILENAME = 'models/best_model.pth'
SUBMISSION_FILENAME = 'submissions/submission_best_model.csv'

# Mode desired
TRAINING_MODE = False
TESTING_MODE = True

# Training dataset parameters
TRAIN_SIZE = 400 
CROP_SIZE = 256
TRAINING_SIZE = 100

# Data augmentation parameters
SIGMAS = [0.1*360/0.5, 0.25*1/2, 0.25*1/2]

# Testing dataset parameters
TESTING_SIZE = 50
PATCH_SIZE = 16
SHAPE_TESTING = 608

# Labels threshold to determine class
FOREGROUND_TRESH = 0.25

# Model parameters
BATCH_SIZE = 16
NUM_EPOCHS = 100


