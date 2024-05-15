import os
import pickle
import pandas as pd
from extract_features import process_images

#Defining path to metadata and testing set
features_path = '..' + os.sep + 'data' + os.sep + 'metadata.csv'
path_image = '..' + os.sep + 'data' + os.sep + 'images' + os.sep + 'images_evaluate'
path_mask = '..' + os.sep + 'data' + os.sep + 'images' + os.sep + 'masks_evaluate'   

#Define features to use in classification
feature_names = ['asymmetry', 'colours', 'dots and globules', 'compactness']
features_df = process_images(features_path, path_image, path_mask, feature_names)

# Load metadata
metadata_df = pd.read_csv(features_path)
combined_df = features_df.merge(metadata_df[['img_id', 'diagnostic', 'patient_id']], left_on='image_id', right_on='img_id', how='left')

# Define the target variable 'y' where '1' represents cancerous conditions and '0' represents non-cancerous conditions
cancerous_conditions = ['BCC', 'MEL', 'SCC']
combined_df['target'] = combined_df['diagnostic'].apply(lambda x: 1 if x in cancerous_conditions else 0)

# Load model and predict probabilities
def predict_probabilities(model_filename, X):
    try:
        with open(model_filename, 'rb') as model_file:
            classifier = pickle.load(model_file)
        print(f"Loaded classifier from {model_filename}")
        probabilities = classifier.predict_proba(X)
        return probabilities
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return None

#Define pass to the model
model_filename = r"Queen_Snakes_classifier.sav"


# Create a list to save probabilities with patient id and labels
results = []

probabilities = predict_probabilities(model_filename, features_df[feature_names].values)
if probabilities is not None:
    # Iterate over each prediction and corresponding image ID
    for idx, prob in enumerate(probabilities):
        results.append({
            'Patient ID': combined_df.iloc[idx]['patient_id'],
            'Image ID': features_df.iloc[idx]['image_id'],
            'Probability Non-cancerous': prob[0],  # Assuming 0 is non-cancerous
            'Probability Cancerous': prob[1],     # Assuming 1 is cancerous
            'Actual Label': combined_df.iloc[idx]['target']  # Ensure target column exists
        })

# Create a DataFrame from results
results_df = pd.DataFrame(results)

# Define the Excel writer and the path for the Excel file
excel_path = 'predictions_new.xlsx'


# Write DataFrame to an Excel file
results_df.to_excel(excel_path, index=False)


print(f"Predictions have been saved to {excel_path}")
