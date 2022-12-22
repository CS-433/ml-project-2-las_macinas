"""
Functions to extract training dataset / testing dataset.
"""

import os
import matplotlib.image as mpimg
import numpy as np
import torch
from torch.utils.data import Dataset

from constants import *
from data_augmentation import augment_data

def extract_dataset(augment=False):
    """
    Load images and crop each of them into 5 images and extract corresponding labels.
    """
    print('Loading 100 images and crop them.')
    train_data = extract_data(TRAIN_DATA_FILENAME, TRAINING_SIZE, 'TRAIN')
    train_labels = extract_labels(TRAIN_LABELS_FILENAME, TRAINING_SIZE)

    if augment == True:
      print('Augment data.')
      train_data, train_labels = augment_data(train_data, train_labels)
    
    return train_data, np.float32(train_labels)

def extract_data(filename, num_images, mode):
    """
    Extract the images into a 4D tensor [image index, y, x, channels].
    If TRAIN mode: crop each image into 5 images that overlap (4 corners and middle image).
    """
    imgs = []
    for i in range(1, num_images + 1):
        if mode == 'TRAIN': 
            imageid = "satImage_%.3d" % i
        elif mode == 'TEST':
            imageid = 'test_%.1d' %i + '/test_%.1d' %i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            img = mpimg.imread(image_filename)
            imgs.append(img)
        else:
            print("File " + image_filename + " does not exist")

    # Crop an image into 5 images
    num_images = len(imgs)
    if mode == 'TRAIN':
        img_patches = [img_crop(imgs[i]) for i in range(num_images)]
        data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]
    elif mode == 'TEST':
        data = imgs
    return np.asarray(data)


# Extract images from a given image with overlap 
def img_crop(im):
    """
    Take one image and extract 5 images of size CROP_SIZE x CROP_SIZE.
    """
    list_patches = []
    is_2d = len(im.shape) < 3

    # Select 4 images on the corners
    for i in range(0, CROP_SIZE, (TRAIN_SIZE-CROP_SIZE)):
        for j in range(0, CROP_SIZE, (TRAIN_SIZE-CROP_SIZE)):
            if is_2d:
                im_patch = im[j:j+CROP_SIZE, i:i+CROP_SIZE]
            else:
                im_patch = im[j:j+CROP_SIZE, i:i+CROP_SIZE, :]
            list_patches.append(im_patch)

    # Select image in the center of image
    pxl_start = int((TRAIN_SIZE-CROP_SIZE)/2)
    pxl_end = int((TRAIN_SIZE+CROP_SIZE)/2)
    if is_2d:
        im_patch = im[pxl_start:pxl_end, pxl_start:pxl_end]
    else:
        im_patch = im[pxl_start:pxl_end, pxl_start:pxl_end, :]
    list_patches.append(im_patch)
    return list_patches

# Extract label images
def extract_labels(filename, num_images):
    """
    Extract the labels into a CROP_SIZE x CROP_SIZE matrix.
    """
    gt_imgs = []
    # Load groundtruth images 
    for i in range(1, num_images + 1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            img = mpimg.imread(image_filename)
            gt_imgs.append(img)
        else:
            print("File " + image_filename + " does not exist")
    num_images = len(gt_imgs)

    # Crop into 5 gt images
    gt_patches = [img_crop(gt_imgs[i]) for i in range(num_images)]
    data = np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    labels = [gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))]
    labels = np.asarray(labels)

    # Associate 0 or 1 labels for each corresponding pixel
    labels_matrix = np.zeros((len(data), CROP_SIZE, CROP_SIZE))
    for img in range(len(data)):
        for i in range(CROP_SIZE):
            for j in range(CROP_SIZE):
                if labels[img, i, j] > FOREGROUND_TRESH:
                    labels_matrix[img, i, j] = 1
                else:
                    labels_matrix[img, i, j] = 0
    return labels_matrix

class RoadDataset(Dataset):
    """
    Class to save the training dataset 
    """
    def __init__(self, data, labels):
        # Store images and groundtruths
        self.images, self.gd_truth = data, labels

    def __len__(self): 
        # Returns len (used for data loaders) 
        return len(self.images)

    def __getitem__(self, idx): 
        # Convert to pytorch tensors
        cropped_image = torch.from_numpy(self.images[idx]).float()
        ground_truth = torch.from_numpy(np.asarray(self.gd_truth[idx]))

        return cropped_image, ground_truth