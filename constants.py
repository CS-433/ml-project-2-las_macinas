# Path of datasets
DATA_DIR = "../ML_clean/data/training/"
TRAIN_DATA_FILENAME = DATA_DIR + "images/"
TRAIN_LABELS_FILENAME = DATA_DIR + "groundtruth/"
TEST_DATA_FILENAME = '../ML_clean/data/test_set_images/'
BEST_MODEL_FILENAME = 'models/best_model_c32_p01_mode1.pth'
SUBMISSION_FILENAME = 'submissions/submission_tuesday_morning.csv'

# Mode desired
TRAINING_MODE = True
TESTING_MODE = not(TRAINING_MODE)

# Training dataset parameters
TRAIN_SIZE = 400 
CROP_SIZE = 256
TRAINING_SIZE = 100

# Testing dataset parameters
TESTING_SIZE = 50
PATCH_SIZE = 16
SHAPE_TESTING = 608

# Labels threshold to determine class
FOREGROUND_TRESH = 0.25

# Model parameters
BATCH_SIZE = 16
NUM_EPOCHS = 50

