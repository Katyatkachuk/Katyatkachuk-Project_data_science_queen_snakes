import pickle
import numpy as np
import cv2  # Used for image operations if necessary

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
        with open(r"C:\Users\tettret\OneDrive - DFDS\Desktop\ITU\Data Science Project\Project_data_science_queen_snakes-3\Classifier_final\Classifier\fyp2024\classifier_7.sav", 'rb') as file:
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

img = cv2.imread(r"C:\Users\tettret\OneDrive - DFDS\Desktop\ITU\Data Science Project\Project_data_science_queen_snakes-3\Classifier_final\Classifier\data\images\images_original\PAT_31_43_129.png")
mask = cv2.imread(r"C:\Users\tettret\OneDrive - DFDS\Desktop\ITU\Data Science Project\Project_data_science_queen_snakes-3\Classifier_final\Classifier\data\images\masks_original\PAT_31_43_129_mask.png", cv2.IMREAD_GRAYSCALE)
label, probability = classify(img, mask)


