from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, classification_report
import pickle
import pandas as pd
import numpy as np
import os


# Defining paths to metadata, images and their masks
file_data = '..' + os.sep + 'data' + os.sep + 'metadata.csv'
file_features = 'features/features_original.csv'
# Read metadata
metadata_df = pd.read_csv(file_data)
df_features = pd.read_csv(file_features)
feature_names = ['asymmetry', 'colours', 'dots and globules', 'compactness']
# Merging created annotations with patient details
combined_df = df_features.merge(metadata_df[['img_id', 'diagnostic', 'patient_id']], left_on='image_id', right_on='img_id', how='left')
if combined_df.isnull().values.any():
    raise ValueError("NaN values detected after merge! Check the data integrity.")

# Preparing the dataset
#defining canserous diagnosis
combined_df['target'] = np.logical_or(combined_df['diagnostic'].values == 'BCC', combined_df['diagnostic'].values == 'MEL', combined_df['diagnostic'].values == 'SCC') 
patient_id = combined_df['patient_id'].values
y = combined_df['target'].values
X = combined_df[feature_names].values


# Preparing cross-validation
num_folds = 5
group_kfold = GroupKFold(n_splits=num_folds)

# Initialize classifiers with appropriate names
classifiers = [
    KNeighborsClassifier(1),
    KNeighborsClassifier(5),
    RandomForestClassifier(n_estimators=100, random_state=42),
    GradientBoostingClassifier(n_estimators=100, random_state=42),
    AdaBoostClassifier(n_estimators=100, random_state=42),
    DecisionTreeClassifier(random_state=42),
    make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight='balanced')),
    make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3)),
    GaussianNB()
]

classifier_names = [
    "KNeighbors (n=1)", "KNeighbors (n=5)", 
    "RandomForest", "GradientBoosting", 
    "AdaBoost", "DecisionTree", 
    "LogisticRegression (Std)", "LogisticRegression (Std, Balanced)",
    "SGDClassifier (Std)", "GaussianNB"
]

# Initializing accuracy storage
acc_val = np.empty((num_folds, len(classifiers)))

# Preparing the directory for saving classifier files
classifiers_save_path = 'Trained_classifiers'  
if not os.path.exists(classifiers_save_path):
    os.makedirs(classifiers_save_path)

# Performing cross-validation
for j, clf in enumerate(classifiers):
    fold_accuracies = []
    for i, (train_index, val_index) in enumerate(group_kfold.split(X, y, patient_id)):
        x_train, y_train = X[train_index], y[train_index]
        x_val, y_val = X[val_index], y[val_index]
        
        # Fitting and predicting
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_val)
        acc = accuracy_score(y_val, y_pred)
        fold_accuracies.append(acc)
        
        # Saving the classifier after training on this fold
        fold_filename = f'{classifier_names[j]}_fold_{i}.sav'
        fold_path = os.path.join(classifiers_save_path, fold_filename)
        pickle.dump(clf, open(fold_path, 'wb'))
    
    acc_val[:, j] = fold_accuracies

# Calculating average accuracy for each classifier
average_acc = np.mean(acc_val, axis=0)
for idx, acc in enumerate(average_acc):
    print(f'Classifier {classifier_names[idx]}: average accuracy={acc:.3f}')

# Saving and evaluating each classifier on the full dataset
eval_results = {}
for idx, clf in enumerate(classifiers):
    
    classifier_filename = f'{classifier_names[idx]}.sav'
    full_classifier_path = os.path.join(classifiers_save_path, classifier_filename)
    
    # Saving classifier
    with open(full_classifier_path, 'wb') as f:
        pickle.dump(clf, f)
    
    # Loading classifier
    with open(full_classifier_path, 'rb') as f:
        loaded_clf = pickle.load(f)
    
    # Predicting on the full dataset and calculating evaluation metrics
    y_pred = loaded_clf.predict(X)
    acc = accuracy_score(y, y_pred)
    clf_report = classification_report(y, y_pred)
    
    eval_results[classifier_names[idx]] = {'accuracy': acc, 'report': clf_report}


# Displaying evaluation results
for clf_name, results in eval_results.items():
    print(f"Results for {clf_name}:")
    print(f"Accuracy: {results['accuracy']}")
    #print(f"Classification Report:\n{results['report']}\n")