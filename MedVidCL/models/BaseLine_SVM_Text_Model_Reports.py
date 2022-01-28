import json
import os
import sys
import re
import unicodedata
import argparse

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.calibration import CalibratedClassifierCV
import pickle

from sklearn.svm import SVC

# # Set Env Variables
parser = argparse.ArgumentParser()
parser.add_argument("--source_dir", type=str, default="../data/text/", help="Directory where source JSON files with text are located")
args = parser.parse_args()

# ### Method to Create Sequence Label for Medical Instructional vs Medical Non-Instructional vs Non-medical videos

def label_vid(row):
    # Check if row is NaN value or not
    if "Medical Non-instructional" in row:
        return 0
    elif "Medical Instructional" in row:
        return 1
    elif "Non-medical" in row:
        return 2
    else:
        return -1


# ## Method to Create and Hyperparameter-Tune an SVM Model 

def train_and_tune_SVM(full_dataset, train_dataset_length, val_dataset_length, test1_dataset_length, hyperparam_svm = False):
    # Create features to train SVM based on dataset's video sub_titles
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')

    features = tfidf.fit_transform(full_dataset['text']).toarray()
    labels = full_dataset.labels

    # Create train+val and test1 sets
    X_train = features[:(train_dataset_length+val_dataset_length)]
    X_test_1 = features[(train_dataset_length+val_dataset_length):]
    y_train = labels[:(train_dataset_length+val_dataset_length)]
    y_test_1 = labels[(train_dataset_length+val_dataset_length):]
    
    print("Length of training+val dataset: ", len(X_train))
    print("Length of test dataset: ", len(X_test_1))
    
    # Create train and val sets
    X_train_train = X_train[:train_dataset_length]
    X_train_val = X_train[train_dataset_length:]
    y_train_train = y_train[:train_dataset_length]
    y_train_val = y_train[train_dataset_length:]
    
    print("Length of training dataset: ", len(X_train_train))
    print("Length of val dataset: ", len(X_train_val))
    
    # Hyperparameter Tuning for LinearSVC
    if not hyperparam_svm:

        # Tune the regularization (C) and tolerance (tol) values
        best_f1 = 0
        best_regularization_value = 1.0
        best_tolerance_value = 1e-4
        for regularization_value in [0.5, 0.75, 1.0, 1.25, 1.5]:
            for tolerance_value in [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]:
                print("Regularization Parameter: ", regularization_value)
                print("Tolerance Value: ", tolerance_value)
                
                svm_v2 = LinearSVC(tol = tolerance_value, C = regularization_value, random_state=42)
                svm_v2.fit(X_train_train, y_train_train)

                clf = CalibratedClassifierCV(base_estimator=svm_v2, cv='prefit') 
                clf.fit(X_train_train, y_train_train)

                # Predict on validation test set to check hyperparameters
                y_pred = clf.predict(X_train_val)
                current_f1 = f1_score(y_train_val, y_pred, average='macro')
                print("Current F1:\n", current_f1)
                if current_f1 > best_f1:
                    best_f1 = current_f1
                    best_regularization_value = regularization_value
                    best_tolerance_value = tolerance_value
                
        print("Best Tolerance Value: ", best_tolerance_value)
        print("Best Regularization Value: ", best_regularization_value)
        print("Best F1", best_f1)
        
        svm_v2 = LinearSVC(tol = best_tolerance_value, C = best_regularization_value, random_state=42)
        svm_v2.fit(X_train_train, y_train_train)

        clf = CalibratedClassifierCV(base_estimator=svm_v2, cv='prefit') 
        clf.fit(X_train_train, y_train_train)

        y_pred = clf.predict(X_test_1)
        print("Classification Report with tuned LinearSVC on Test Set:\n", classification_report(y_test_1, y_pred, target_names=['0', '1', '2'], digits=4))

    # Hyperparameter Tuning for SVM
    else:

        # Tune the regularization (C), tolerance (tol), and kernel values
        best_f1 = 0
        best_regularization_value = 1.0
        best_tolerance_value = 1e-3
        best_kernel_value = 'rbf'
        for regularization_value in [0.1, 1, 10, 100, 1000]:
            for tolerance_value in [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]:
                for kernel_value in ['poly', 'rbf', 'sigmoid']:
                    print("Regularization Parameter: ", regularization_value)
                    print("Tolerance Value: ", tolerance_value)
                    print("Kernel Value: ", kernel_value)

                    svm_v2 = SVC(tol = tolerance_value, C = regularization_value, kernel = kernel_value, random_state=42)
                    svm_v2.fit(X_train_train, y_train_train)

                    # Predict on validation test set to check hyperparameters
                    y_pred = svm_v2.predict(X_train_val)
                    current_f1 = f1_score(y_train_val, y_pred, average='macro')
                    print("Current F1:\n", current_f1)
                    if current_f1 > best_f1:
                        best_f1 = current_f1
                        best_regularization_value = regularization_value
                        best_tolerance_value = tolerance_value
                        best_kernel_value = kernel_value
        
        print("Best Kernel Value: ", best_kernel_value)
        print("Best Tolerance Value: ", best_tolerance_value)
        print("Best Regularization Value: ", best_regularization_value)
        print("Best F1", best_f1)
        
        svm_v2 = SVC(tol = best_tolerance_value, C = best_regularization_value, kernel = best_kernel_value, random_state=42)
        svm_v2.fit(X_train_train, y_train_train)

        y_pred = svm_v2.predict(X_test_1)
        print("Classification Report with tuned SVM on Test Set:\n", classification_report(y_test_1, y_pred, target_names=['0', '1', '2'], digits=4))

# # Import Datasets 

# Import all new datasets
new_train_df = pd.read_json(args.source_dir + "train.json")
new_eval_df = pd.read_json(args.source_dir + "val.json")
new_test1_df = pd.read_json(args.source_dir + "test.json")

# Rename column names in new datasets to match old datasets
new_train_df=new_train_df.rename(columns = {'video_sub_title':'text', 'video_title':'title', 'label':'labels', 'video_id':'YouTube_ID'})
new_eval_df=new_eval_df.rename(columns = {'video_sub_title':'text', 'video_title':'title', 'label':'labels', 'video_id':'YouTube_ID'})
new_test1_df=new_test1_df.rename(columns = {'video_sub_title':'text', 'video_title':'title', 'label':'labels', 'video_id':'YouTube_ID'})

# Change labels of new datasets to match the old one
new_train_df['labels'] = new_train_df['labels'].apply(label_vid)
new_eval_df['labels'] = new_eval_df['labels'].apply(label_vid)
new_test1_df['labels'] = new_test1_df['labels'].apply(label_vid)


# Create new full dataframes for testing all possible models
df = pd.concat([new_train_df, new_eval_df, new_test1_df], ignore_index=True)


# ## Try Hyperparameter Tuning to Improve LinearSVC and SVM
train_and_tune_SVM(df, len(new_train_df), len(new_eval_df), len(new_test1_df))

train_and_tune_SVM(df, len(new_train_df), len(new_eval_df), len(new_test1_df), hyperparam_svm = True)


