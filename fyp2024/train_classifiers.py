from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd
import numpy as np
import os


file_data = '..' + os.sep + 'data' + os.sep + 'metadata.csv'
file_features = 'features/features_original.csv'
feature_names = ['assymetry', 'colours', 'dots and globules', 'compactness']

metadata_df = pd.read_csv(file_features)
df_features=pd.read_csv

#Merging created annotations with patiend details
combined_df = df_features.merge(metadata_df[['img_id', 'diagnostic', 'patient_id']], left_on='image_id', right_on='img_id', how='left')
if combined_df.isnull().values.any():
    raise ValueError("NaN values detected after merge! Check the data integrity.")

# Preparing the dataset

# X - features evaluation
X = combined_df[feature_names].to_numpy() 
# Y - patient diagnistic assuming that 'BCC' , 'MEL' and 'SCC' are cancerous
y = np.logical_or(combined_df['diagnostic'].values == 'BCC', combined_df['diagnostic'].values == 'MEL', combined_df['diagnostic'].values == 'SCC')
patient_id = combined_df['patient_id'].values

# Preparing cross-validation
num_folds = 5
group_kfold = GroupKFold(n_splits=num_folds)

# Initializing classifiers
classifiers = [
    KNeighborsClassifier(1), #classifier 0
    KNeighborsClassifier(5), #classifier 1
    RandomForestClassifier(n_estimators=100, random_state=42), #classifier 2
    GradientBoostingClassifier(n_estimators=100, random_state=42), #classifier 3
    AdaBoostClassifier(n_estimators=100, random_state=42), #classifier 4
    DecisionTreeClassifier(random_state=42), #classifier 5
    make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)), #classifier 6
    make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight='balanced')), #classifier 7
    make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3)), #classifier 8
    GaussianNB() #classifier 9
]

# Initializing accuracy storage
acc_val = np.empty((num_folds, len(classifiers)))
classifier_names = []

# Performing cross-validation
for j, clf in enumerate(classifiers):
    classifier_name = (clf.named_steps['svc'].__class__.__name__ if 'pipeline' in str(clf)
                       else clf.__class__.__name__)
    classifier_names.append(classifier_name)
    fold_accuracies = []
    
    for i, (train_index, val_index) in enumerate(group_kfold.split(X, y, patient_id)):
        x_train, y_train = X[train_index], y[train_index]
        x_val, y_val = X[val_index], y[val_index]
        
        # Fit and predict
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_val)
        acc = accuracy_score(y_val, y_pred)
        fold_accuracies.append(acc)
        
        # Save the classifier after training on this fold
        fold_filename = f'classifier_{j}_fold_{i}.sav'
        pickle.dump(clf, open(fold_filename, 'wb'))
    
    acc_val[:, j] = fold_accuracies

# Calculating average accuracy for each classifier
average_acc = np.mean(acc_val, axis=0)
for idx, acc in enumerate(average_acc):
    print(f'Classifier {idx + 1} ({classifier_names[idx]}): average accuracy={acc:.3f}')

# Saving and evaluating each classifier on the full dataset
eval_results = {}
for idx, clf in enumerate(classifiers):

    classifier_name = classifier_names[idx]
    classifier_filename = f'classifier_{idx}.sav'
    
    # Saving the classifier
    pickle.dump(clf, open(classifier_filename, 'wb'))
    
    # Load the classifier
    loaded_clf = pickle.load(open(classifier_filename, 'rb'))
    
    # Predict on the full dataset and calculate evaluation metrics
    y_pred = loaded_clf.predict(X)
    acc = accuracy_score(y, y_pred)
    clf_report = classification_report(y, y_pred)
    
    eval_results[classifier_name] = {'accuracy': acc, 'report': clf_report}

# Display evaluation results
for clf_name, results in eval_results.items():
    print(f"Results for {clf_name}:")
    print(f"Accuracy: {results['accuracy']}")