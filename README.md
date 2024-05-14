# Bachelor of Science in Data Science, ITU of Copenhagen 2023/24 - First Year Project
## Project's Title: Medical Imaging (by queen_snakes)

### Project Description:
The skin lesion classifier project aims to develop a computer-based system for accurately identifying and classifying skin lesions. Using machine learning techniques, we focuse on training classifiers to distinguish between cancerous and non-cancerous lesions based on dermatoscopic images. By automating the diagnostic process traditionally performed by dermatologists, our goal is to improve the efficiency and accuracy of skin cancer detection, eventually contributing to better patient outcomes.

### Contents:
1. How to Install and Run the Project.
2. How to Use the Project.
3. Project content.
4. Credits.

#### 1. How to Install and Run the Project:
 To install and run the Skin Imaging project, follow these steps:

1. Clone the project repository from GitHub URL.
2. Navigate to the project directory.
3. Install the required dependencies using pip install -r requirements.txt.
4. Run the main script to train and evaluate the classifier.

#### 2. How to Use the Project:
 Once the project is installed and running, follow these steps to use the Skin Lesion Classifier:

1. Navigate to the classifier_full.ipynb in the fyp2024 folder and open it.
2. Provide the paths to the folders of dermatoscopic images of skin lesions and their masks as input to evaluate the classifier.
3. Run the main script (3rd cell) to evaluate the classifier.
4. Run the main script (4th cell) to see the comfusion matrix visualisation.
5. Run the main script (5th cell) to see the probability of cancer in the generated predictions_new.xlsx.file.
 
 The classifier will analyze the images and provide predictions on whether the lesions are cancerous or non-cancerous.
Review the classification results and any associated metrics to assess the performance of the classifier.

#### 3. Project content:
   - New_masks where each image is annotated by at least two people;
   - Queen_snakes_masks - masks annotated first time by one person;
   - data - contains matadata file and a folder with images and masks for evaluation a classifier;
   - fyp2024 - contains 10 classifiers with their folds, feature file for evaluation a classifier, full classider source file, predictions_new.xlsx.file with probability of cancer results and a requirements.txt file;
   - .gitignore - excludes the datasets for training a classifier;
   - Queen_snakes_annotation_comments.md with our findings about the annotation process;
   - Queen_snakes_anotations_manual where only our manual measurements are stored;
   - Queen_snakes_imageids.csv with two columns: annotator ID, and filename of an annotation;
   - Queen_snakes_measurements_with_code_results where our manual measurements and features results are stored;
   - Queen_snakes_summary.md with our findings about the data;
   - details_img.xlsx contains merged data with additional details.

#### 4. Credits
 Contributors:
   
   q1 - Tetiana Tretiak (tetr@itu.dk)

   q2 - Kateryna Tkachuk (ktka@itu.dk)

   q3 - Mariia Zviahintseva (mazv@itu.dk)

