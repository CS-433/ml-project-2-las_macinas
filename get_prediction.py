 
import sys
import urllib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

from training import *
from load_data import *

TESTING_SIZE = 50
PATCH_SIZE = 16
SHAPE_TESTING = 608
'''
def get_prediction_with_groundtruth(filename, image_idx):

    imageid = "satImage_%.3d" % image_idx
    image_filename = filename + imageid + ".png"
    img = mpimg.imread(image_filename)

    img_prediction = get_prediction(img)
    cimg = concatenate_images(img, img_prediction)

    return 
'''

'''
def testing(model, test_loader):

    index = 0
    output_patches = []
    computed_images = []


    for batch in test_loader: # a batch corresponds to all patches from a single image
        
        output = model(batch)
        #output = np.rint(output.detach().numpy()).squeeze()
        #output_patches.append(output)
        #image = unpatchify(output.reshape(6, 6, 120, 120), (720, 720)).T
        computed_images.append(output[:608, :608])
        
    return computed_images #computed_images

'''

def mask_to_submission_strings(image_filename):
    """Reads a single image and outputs the strings that should go into the submission file"""
    img_number = int(re.search(r"\d+", image_filename).group(0))
    im = mpimg.imread(image_filename)
    patch_size = 16

    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
           
            label = patch_to_label(patch)
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))



def test():
    filename = 'data/test_set_images/'
    data_test  = extract_data(filename, num_images=50, mode='TEST')
    print(data_test.shape)

    patched_image = torch.from_numpy(data_test).float()
    #print(patched_image.shape)

    model = SimpleNet()
    model.load_state_dict(torch.load('models/last_model.pth'))

    # TEST SET ET TEST LOADER A APPLIQUER A NOS IMAGES
    prediction = model(patched_image.permute(0, 3, 1, 2))
    #print(prediction.shape)

    submission_filename = 'dummy_submission.csv'

    ids = []
    count = 0

    for img_number in range(1, TESTING_SIZE+1):
        print(img_number)
        for i in range(0, SHAPE_TESTING, PATCH_SIZE):
            for j in range(0, SHAPE_TESTING, PATCH_SIZE):
                label = torch.round(prediction[count], decimals=0)
                label = int(label)
                #f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(fn))
                ids.append("{:03d}_{}_{},{}".format(img_number, i, j, label))
                count += 1

    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        f.writelines('{}\n'.format(s) for s in ids)


test()










