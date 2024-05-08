import os
from os.path import exists
import pandas as pd
import numpy as np
import cv2

# Importing function to extract features
from extract_features import  extract_features

#-------------------
# Main script
#-------------------

# Defining function to process images
def process_images(file_data, path_image, path_mask, feature_names):
    
    # Defining where we will store the features
    file_features = 'features/features.xlsx'

    # Read meta-data into a Pandas dataframe
    df = pd.read_csv(file_data)

    # Extract image IDs and labels from the data.
    image_id = list(df['img_id'])
    mask_id = [id.replace('.png', '_mask.png') for id in df['img_id']]


    num_features = len(feature_names)
    features = []
    valid_image_ids = []

    # Loop through all images (limited by num_images)
    num_images = len(image_id)
    for i in np.arange(num_images):

        # Define filenames related to this image
        file_image = path_image + os.sep + image_id[i]
        file_image_mask = path_mask + os.sep + mask_id[i]

        # Check if both the image and mask files exist
        if exists(file_image) and exists(file_image_mask):
            # Read the image and mask
            im = cv2.imread(file_image)
            mask = cv2.imread(file_image_mask, cv2.IMREAD_GRAYSCALE)
            if im.shape[:2] != mask.shape[:2]:
                print(f"Skipping image {image_id[i]}: Image and mask dimensions do not match.")
                continue
            # Check if the mask has any contour other than just a black background
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                # Skip this image if no contours are found in the mask
                continue
            
            # Measure features
            x = extract_features(im, mask)

            # Store in the list we created before
            features.append(x)

            # Keep track of the valid image ID
            valid_image_ids.append(image_id[i])

    # Create DataFrame from the features list and add image IDs
    df_features = pd.DataFrame(features, columns=feature_names)
    df_features['image_id'] = valid_image_ids
    return df_features

# Defining paths to metadata, images and their masks
file_data = '..' + os.sep + 'data' + os.sep + 'metadata.csv'
path_image = '..' + os.sep + 'data' + os.sep + 'images' + os.sep + 'images_train_new'
path_mask = '..' + os.sep + 'data' + os.sep + 'images' + os.sep + 'masks_train'    

# Defining the path to store the features
file_features = 'features/features_original.csv'
feature_names = ['assymetry', 'colours', 'dots and globules', 'compactness']

#Processing images and saving annotations to csv
df_features=process_images(file_data, path_image, path_mask,feature_names)
df_features.to_csv(file_features, index=False)