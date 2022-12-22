"""
Functions to augment our dataset with transformations
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from scipy.ndimage import rotate
import torch
import torchvision.transforms.functional as t


from constants import *

def augment_data(images, labels):
  """
  Perform data augmentation with multiple operations (randomly do different ones)
  """
  augmented_images = []
  augmented_labels = []

  for i in range(len(images)):
    # 0) Conserve authentic image
    augmented_images.append(images[i])
    augmented_labels.append(labels[i])
    
    # 1) augmentation: apply random hue/contrast/brightness/saturation
    # Apply consecutively random hue, contrast, brightness, saturation
    image_hsv = hsv_transform(images[i])
    label_hsv = labels[i]
        

    # 2) augmentation: rotation
    # Choose randomly the angle of rotation
    angles = [-60, -45, -30, 30, 45, 60]
    angle = random.choice(angles)

    # Choose randomly if change hsv or not
    random_value = np.random.randint(0, 3)
    if random_value == 0:
      img_hsv = hsv_transform(images[i])
    else:
      img_hsv = images[i]

    # Apply rotation and crop with same size
    image_rotate = rotate(images[i], angle=angle, mode ='wrap', reshape=False)
    label_rotate = rotate(labels[i], angle=angle, mode ='wrap', reshape=False)
    for k in range(CROP_SIZE):
      for l in range(CROP_SIZE):
        if label_rotate[k, l] > FOREGROUND_TRESH:
          label_rotate[k, l] = 1
        else:
          label_rotate[k, l] = 0

    # 3) augmentation: flip randomly left/right or up/down
    random_value = np.random.randint(0, 1)
    if random_value == 0:
      image_flip = np.flipud(images[i])
      label_flip = np.flipud(labels[i])
    else:
      image_flip = np.fliplr(images[i])
      label_flip = np.fliplr(labels[i])
    
    # 4) augmentation: image processing (blurring and edge detection)
    # Setting parameter values
    t_lower = 120  # Lower Threshold
    t_upper = 150  # Upper threshold

    image_blur = cv2.GaussianBlur((images[i]*255).astype(np.uint8), (5,5),0)
    label_blur = labels[i]
    image_edge = cv2.Canny(image_blur, t_lower, t_upper, 3)
    image_edge= cv2.cvtColor(image_edge,cv2.COLOR_GRAY2RGB)
    label_edge = labels[i]

    # 4) More augmentation for images with a lot of road: color transformation
    if np.mean(images[i]) > 0.28:
      angles_2 = [120, 150, 210]
      angle_2 = random.choice(angles_2)

      # Apply rotation and crop with same size
      image_rotate_2 = rotate(images[i], angle=angle_2, mode ='wrap', reshape=False)
      label_rotate_2 = rotate(labels[i], angle=angle_2, mode ='wrap', reshape=False)
      for k in range(CROP_SIZE):
        for l in range(CROP_SIZE):
          if label_rotate_2[k, l] > FOREGROUND_TRESH:
            label_rotate_2[k, l] = 1
          else:
            label_rotate_2[k, l] = 0

      if random_value == 1:
        image_flip_2 = np.flipud(images[i])
        label_flip_2 = np.flipud(labels[i])
      else:
        image_flip_2 = np.fliplr(images[i])
        label_flip_2 = np.fliplr(labels[i])

      augmented_images.append(image_rotate_2)
      augmented_labels.append(label_rotate_2)
      augmented_images.append(image_flip_2)
      augmented_labels.append(label_flip_2)

    augmented_images.append(image_hsv)
    augmented_labels.append(label_hsv)
    augmented_images.append(image_rotate)
    augmented_labels.append(label_rotate)
    augmented_images.append(image_flip)
    augmented_labels.append(label_flip)
    augmented_images.append(image_blur)
    augmented_labels.append(label_blur)
    augmented_images.append(image_edge)
    augmented_labels.append(label_edge)

  return np.asarray(augmented_images), np.asarray(augmented_labels)

def RGB_to_HSV(img_rgb):
  """
  Convert an RGB image to HSV format
  """
  # Split channels
  R = img_rgb[:, :, 0]                        
  G = img_rgb[:, :, 1]
  B = img_rgb[:, :, 2]
  
    # Compute max, min & chroma
  v_max = np.max(img_rgb, axis=2)            
  v_min = np.min(img_rgb, axis=2)
  C = v_max - v_min                                      
  
  # Check if hue can be computed
  hue_defined = C > 0                                    
  
  # Computation of hue depends on max
  r_is_max = np.logical_and(R == v_max, hue_defined)     
  g_is_max = np.logical_and(G == v_max, hue_defined)
  b_is_max = np.logical_and(B == v_max, hue_defined)
  
  # Compute hue
  H = np.zeros_like(v_max)                               
  H_r = ((G[r_is_max] - B[r_is_max]) / C[r_is_max]) % 6
  H_g = ((B[g_is_max] - R[g_is_max]) / C[g_is_max]) + 2
  H_b = ((R[b_is_max] - G[b_is_max]) / C[b_is_max]) + 4
  
  H[r_is_max] = H_r
  H[g_is_max] = H_g
  H[b_is_max] = H_b
  H *= 60
  
  # Compute value
  V = v_max                                              
  
  # Compute saturation
  sat_defined = V > 0
  S = np.zeros_like(v_max)                               
  S[sat_defined] = C[sat_defined] / V[sat_defined]
  
  return np.dstack((H, S, V))

def HSV_to_RGB(img_hsv):
  """
  Convert an HSV image to RGB format
  """
  # Split attributes
  H = img_hsv[:, :, 0]                                           
  S = img_hsv[:, :, 1]
  V = img_hsv[:, :, 2]
  
  # Compute chroma
  C = V * S                                                  
  
  # Normalize hue
  H_ = H / 60.0    
  # Compute value of 2nd largest color                                          
  X  = C * (1 - np.abs(H_ % 2 - 1))                          
  
  # Store color orderings
  H_0_1 = np.logical_and(0 <= H_, H_<= 1)                    
  H_1_2 = np.logical_and(1 <  H_, H_<= 2)
  H_2_3 = np.logical_and(2 <  H_, H_<= 3)
  H_3_4 = np.logical_and(3 <  H_, H_<= 4)
  H_4_5 = np.logical_and(4 <  H_, H_<= 5)
  H_5_6 = np.logical_and(5 <  H_, H_<= 6)
  
  # Compute relative color values
  R1G1B1 = np.zeros_like(img_hsv)                                
  Z = np.zeros_like(H)
  
  R1G1B1[H_0_1] = np.dstack((C[H_0_1], X[H_0_1], Z[H_0_1]))  
  R1G1B1[H_1_2] = np.dstack((X[H_1_2], C[H_1_2], Z[H_1_2]))
  R1G1B1[H_2_3] = np.dstack((Z[H_2_3], C[H_2_3], X[H_2_3]))
  R1G1B1[H_3_4] = np.dstack((Z[H_3_4], X[H_3_4], C[H_3_4]))
  R1G1B1[H_4_5] = np.dstack((X[H_4_5], Z[H_4_5], C[H_4_5]))
  R1G1B1[H_5_6] = np.dstack((C[H_5_6], Z[H_5_6], X[H_5_6]))
  
  # Adding the value correction
  m = V - C
  img_rgb = R1G1B1 + np.dstack((m, m, m))                        
  
  return img_rgb

def hsv_transform(img):
  """
  Handle transformation by doing an HSV change
  """
  image_hsv = RGB_to_HSV(img)

  hue = (image_hsv[:,:,0] + random.uniform(-SIGMAS[0],SIGMAS[0])) % 360                                           # Modify Hue 
  sat =  np.minimum(image_hsv[:,:,1] * (1+random.uniform(-SIGMAS[1],SIGMAS[1])), np.ones_like(image_hsv[:,:,1]))  # Modify saturation
  val =  np.minimum(image_hsv[:,:,2] * (1+ random.uniform(-SIGMAS[2],SIGMAS[2])), np.ones_like(image_hsv[:,:,1])) # Modify value (brightness)

  # Stack to a new HSV array
  image_hsv_modify = np.dstack((hue, sat, val)) 
  # Convert it to RGB      
  image_rgb = HSV_to_RGB(image_hsv_modify)            
  
  return image_rgb                     








