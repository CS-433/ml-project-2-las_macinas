"""
Function to produce our best results
Possibility to choose to re train the model or load our weight
Possibility to test our model
"""
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import numpy as np

from models import Unet 
from constants import *
from extract_dataset import extract_dataset, RoadDataset
from training import run_training
from testing import testing

if TRAINING_MODE == True:
  # Extract data and labels 
  training_data, training_labels = extract_dataset(augment=True)

   # Split in training and validation set
  train_data, validation_data, train_labels, validation_labels = train_test_split(training_data, training_labels, test_size=0.2, random_state=42)

  # Build training and validation datasets
  train_set = RoadDataset(train_data, train_labels)
  validation_set = RoadDataset(validation_data, validation_labels)

  # Build dataloader
  train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
  val_loader = DataLoader(validation_set, 1, shuffle=False)

  # Set parameters to train 
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = Unet()
  print('Start training')
  best_model = run_training(model=model, num_epochs=NUM_EPOCHS, train_loader=train_loader, val_loader=val_loader , device=device)

if TESTING_MODE == True:
  # Load model
  if TRAINING_MODE == False:
    model = Unet()
    model.load_state_dict(torch.load(BEST_MODEL_FILENAME, map_location=torch.device('cpu')))
  else:
    model = best_model

  # Test the model and save predictions
  print('Start testing')
  data_test, prediction = testing(model)
  print('Submission saved')






