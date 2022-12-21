 
import sys
import urllib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
#from torchvision import datasets, transforms
#from torch.utils.data import DataLoader, Dataset

#from Htraining import *
from crop_img import *
from Hmodels import *

def testing():
    filename = 'data/test_set_images/'

    data_test  = extract_data(filename, num_images=TESTING_SIZE, mode='TEST')
    patched_image = torch.from_numpy(data_test).float()

    #model = Unet()
    #model.load_state_dict(torch.load('models/best_model_coco.pth'))
    #patched_image = patched_image.to(device)

    best_model_cpu = best_model.cpu()
    prediction = best_model_cpu(patched_image.permute(0, 3, 1, 2))
    prediction = prediction.detach().numpy()

    print('prediction:', prediction.shape)
    submission_filename = 'submission_morning.csv'

    ids = []
    count = 0

    for img_number in range(0, TESTING_SIZE):
        print(img_number)
        for j in range(0, SHAPE_TESTING, PATCH_SIZE):
            for i in range(0, SHAPE_TESTING, PATCH_SIZE):
                mean_value = np.mean(prediction[img_number, i:i+PATCH_SIZE, j:j+PATCH_SIZE])
                if mean_value > 0.25:
                    label = 1
                else:
                    label = 0
                ids.append("{:03d}_{}_{},{}".format((img_number+1), j, i, label))
                count += 1

    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        f.writelines('{}\n'.format(s) for s in ids)











