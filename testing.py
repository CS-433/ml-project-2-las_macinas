 
import sys
import urllib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
import torch.nn as nn

from extract_dataset import extract_data
from constants import *

def testing(model):
  
  # Extract testing images
  data_test  = extract_data(TEST_DATA_FILENAME, num_images=TESTING_SIZE, mode='TEST')
  patched_image = torch.from_numpy(data_test).float()
  model_cpu = model.cpu()

  # Compute predictions and convert to np array
  prediction = model_cpu(patched_image.permute(0, 3, 1, 2))
  prediction = prediction.detach().numpy()

  ids = []
  count = 0

  # Convert pixelwise predictions to patches predictions
  for img_number in range(0, TESTING_SIZE):
      print(img_number)
      for j in range(0, SHAPE_TESTING, PATCH_SIZE):
          for i in range(0, SHAPE_TESTING, PATCH_SIZE):
              mean_value = np.mean(prediction[img_number, i:i+PATCH_SIZE, j:j+PATCH_SIZE])
              if mean_value > FOREGROUND_TRESH:
                  label = 1
              else:
                  label = 0
              ids.append("{:03d}_{}_{},{}".format((img_number+1), j, i, label))
              count += 1

  with open(SUBMISSION_FILENAME, 'w') as f:
      f.write('id,prediction\n')
      f.writelines('{}\n'.format(s) for s in ids)











