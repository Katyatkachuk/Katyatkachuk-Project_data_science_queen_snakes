import pickle
import numpy as np
import cv2  
import os

from extract_features import extract_features

def classify(img, mask):
    """
    Classifies an image based on extracted features and a pre-trained classifier.

    Args:
        img (np.array): The image data as a NumPy array.
        mask (np.array): The mask data as a NumPy array.

    Returns:
        tuple: A tuple containing the predicted label and the probabilities for each class.
    """

    # img = cv2.resize(img, (256, 256))
    # mask = cv2.resize(mask, (256, 256))

    # Extract features using the function defined in extract_features.py
    features = extract_features(img, mask).reshape(1, -1)  

    # Load the trained classifier
    try:
        with open(r"Queen_Snakes_classifier.sav", 'rb') as file:
            classifier = pickle.load(file)
    except FileNotFoundError:
        print("Model file not found. Please ensure the model path is correct.")
        return None, None
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return None, None

    # Predict the label and the posterior probability for the given image
    pred_label = classifier.predict(features)
    pred_prob = classifier.predict_proba(features)

    # Print the results
    print('Predicted label is:', pred_label)
    print('Predicted probability is:', pred_prob)

    return pred_label, pred_prob

# Example usage:

path_image = '..' + os.sep + 'data' + os.sep + 'images' + os.sep + 'images_evaluate/PAT_31_43_129.png'
path_mask = '..' + os.sep + 'data' + os.sep + 'images' + os.sep + 'masks_evaluate/PAT_31_43_129_mask.png'   

img = cv2.imread(path_image)
mask = cv2.imread(path_mask, cv2.IMREAD_GRAYSCALE)
label, probability = classify(img, mask)

