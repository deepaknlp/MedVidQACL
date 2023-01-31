import logging
import json
import os
import sys
import re
import unicodedata
import math
import argparse
import random
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, classification_report, average_precision_score
import pickle

import torch
from transformers import (AutoTokenizer, 
                          AutoModelForSequenceClassification, 
                          Trainer, 
                          TrainingArguments,
                          EarlyStoppingCallback,
                          EvalPrediction
                         )
from datasets import load_metric, Dataset


# # Set Env Variables

parser = argparse.ArgumentParser()
parser.add_argument("--source_dir", type=str, default="../data/text/Med-Instr-Hierarchical/", help="Directory where source JSON files with text are located")
parser.add_argument("--early_stopping_patience", type=int, default=3, help="Number of epochs to wait before terminating training if no improvement")
parser.add_argument("--evaluation_metric", type=str, default="eval_Fweighted", help="Metric to use to track and choose best model")
parser.add_argument("--seed", type=int, default=42, help="Seed to use to initialize random numbers")
args = parser.parse_args()

# Set reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
os.environ['PYTHONHASHSEED']=str(args.seed)


# # Creating One-Hot-Encoded Binary Labels

# In[32]:


# Create id's based on new labels
old_labels_to_new_labels_map = pd.read_csv(args.source_dir + 'level_2_to_level_1_category_label_map.csv')
new_labels = list(set(old_labels_to_new_labels_map['level_1_category_label'].tolist()))
old_labels = list(set(old_labels_to_new_labels_map['level_2_category_label'].tolist()))
id2label = dict(enumerate(new_labels))
label2id = {v: k for k, v in id2label.items()}
old_to_new_label_dict = dict(zip(old_labels_to_new_labels_map['level_2_category_label'], old_labels_to_new_labels_map['level_1_category_label']))

def label_vid(queries):
    # Convert old labels to new labels
    new_queries = [old_to_new_label_dict[old_label] for old_label in queries]
    # Create binary labels
    binary_labels = [1. if id2label[i] in new_queries else 0. for i in range(len(new_labels))]
    return binary_labels


# # Import Datasets

datasets = {}

# Import JSON files first
json_filenames = [pos_json for pos_json in os.listdir(args.source_dir) if pos_json.endswith('.json')]
for json_filename in json_filenames:
    datasets[json_filename] = pd.read_json(args.source_dir + '/' + json_filename)
    # Rename columns
    datasets[json_filename] = datasets[json_filename].rename(columns = {'video_title':'title', 'video_id':'YouTube_ID', 'video_sub_title':'text'})
    # Change labels of new datasets to match the old one
    datasets[json_filename]['labels'] = datasets[json_filename]['level_2_category_labels'].apply(label_vid)
    # Remove duplicates
    datasets[json_filename].drop_duplicates(subset=['YouTube_ID'], inplace=True, ignore_index=True)
    # Keep only certain columns
    datasets[json_filename] = datasets[json_filename][["text", "labels", "YouTube_ID"]]
    

# Inspect Amount of datapoints and unique labels in separate dataframes
for filename in list(datasets.keys()):
    print(filename + ": " + str(len(datasets[filename])))


# # Train Baseline Transformer Models

def train_and_test_Transformer_Model(model_name = "google/bigbird-roberta-base", max_token_length = 1024, batch_size = 8):
    # Set variables
    specific_model = model_name.split('/')[-1]
    checkpoint_dir = 'checkpoints/' + specific_model + '_runs'

    # Method to Tokenize Text
    def encode(examples):
        tokenized_input = tokenizer(
            examples['text'], 
            padding = 'max_length', 
            truncation=True, 
            max_length = max_token_length)

        return tokenized_input
    
    # Get Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create & Tokenize dictionary of pandas datasets converted to Transformer Datasts
    ddatasets = {}
    for filename in list(datasets.keys()):
        ddatasets[filename] = Dataset.from_pandas(datasets[filename])
        ddatasets[filename] = ddatasets[filename].map(encode, batched=True)
        ddatasets[filename].set_format('torch')
        
    # Initialize Model
    model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                problem_type="multi_label_classification",
                num_labels=len(new_labels),
                id2label=id2label,
                label2id=label2id)
    
    # Define Metrics
    def multi_label_metrics(predictions, labels):
        # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
        sigmoid = torch.nn.Sigmoid()
        y_pred = sigmoid(torch.Tensor(predictions))
        # finally, compute metrics
        y_true = labels
        sum_precision = 0
        number_of_nan_free_precision_scores_for_labels = 0
        
        # Find precision score over all nan-free labels and average them
        for i in range(y_true.shape[1]):
            precision = average_precision_score(y_true[:, i], y_pred[:, i])
            if not np.isnan(precision):
                sum_precision += precision
                number_of_nan_free_precision_scores_for_labels += 1
        avgPrecision = sum_precision / number_of_nan_free_precision_scores_for_labels
        
        # Find f1 metrics
        best_threshold = 0
        best_f1_weighted = 0
        for threshold in np.arange(0, 0.5, 0.01):
            preds = np.zeros(y_pred.shape)
            preds[np.where(y_pred >= threshold)] = 1

            f1_macro = f1_score(y_true, preds, average='macro')
            f1_micro = f1_score(y_true, preds, average='micro')
            f1_weighted = f1_score(y_true, preds, average='weighted')
            
            if f1_weighted > best_f1_weighted:
                best_f1_weighted = f1_weighted
                best_threshold = threshold
            

        # return as dictionary
        metrics = {'Avgprecision': avgPrecision,
                  'F1-Macro': f1_macro,
                  'F1-Micro': f1_micro,
                  'Fweighted': best_f1_weighted,
                  'F1-Threshold': best_threshold}
        return metrics
    
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, 
                tuple) else p.predictions
        result = multi_label_metrics(
            predictions=preds, 
            labels=p.label_ids)
        return result
    
    
    # Set Training Arguments
    training_args = TrainingArguments(
        do_train = True,
        evaluation_strategy = "epoch",
        load_best_model_at_end=True,
        logging_strategy = 'epoch',
        metric_for_best_model = args.evaluation_metric,
        num_train_epochs = 10,
        output_dir = checkpoint_dir,
        overwrite_output_dir = True,
        per_device_eval_batch_size = batch_size,
        per_device_train_batch_size = batch_size,
        remove_unused_columns=True,
        report_to="none",
        save_strategy = 'epoch',
        save_total_limit = 1,
        seed = args.seed,
    )

    # Train model
    trainer = Trainer(
        model=model, 
        args=training_args, 
        tokenizer = tokenizer,
        train_dataset=ddatasets['train.json'], 
        eval_dataset=ddatasets['val.json'],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    trainer.train()
    
    # Get Average Precision Score
    def get_avg_prec_score_and_f1(trainer, test_dataset):
        probs, label_ids1, *_ = trainer.predict(test_dataset)
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(probs))
        
        sum_precision = 0
        number_of_nan_free_precision_scores_for_labels = 0
        
        # Find precision score over all nan-free labels and average them
        for i in range(label_ids1.shape[1]):
            precision = average_precision_score(label_ids1[:, i], probs[:, i])
            if not np.isnan(precision):
                sum_precision += precision
                number_of_nan_free_precision_scores_for_labels += 1
        results = sum_precision / number_of_nan_free_precision_scores_for_labels
        
        print("Average Precision Score on Test Set:", results)
    
        preds1 = np.zeros(probs.shape)
        preds1[np.where(probs >= 0.17)] = 1
        
        f1_macro = f1_score(label_ids1, preds1, average='macro')
        f1_micro = f1_score(label_ids1, preds1, average='micro')
        f1_weighted = f1_score(label_ids1, preds1, average='weighted')
        
        print("F1-Score Macro on Test Set: {:.4f}".format(f1_macro))
        print("F1-Score Micro on Test Set: {:.4f}".format(f1_micro))
        print("F1-Score Weighted on Test Set: {:.4f}".format(f1_weighted))
        return (label_ids1, probs, preds1)
        #return results
        
    
    report_v1 = get_avg_prec_score_and_f1(trainer, ddatasets['test.json'])
    
    # Save Model & Tokenizer
    trainer.save_model('models/best_run_' + specific_model)
    tokenizer.save_pretrained('models/best_run_' + specific_model)
    
    return report_v1

## Train various models

# BigBird-Base
print("BigBird-Base Train&Testing")
report = train_and_test_Transformer_Model("google/bigbird-roberta-base", 1024, batch_size = 4)

# Save probs and labels and preds1
with open('multilabel_text_report_13_labels.pickle', 'wb') as f:
    pickle.dump(report, f)


# ## Check Best Threshold

# Save probs and labels and preds1
with open('multilabel_text_report_13_labels.pickle', 'rb') as f:
    report = pickle.load(f)

best_val_threshold = 0.22
print("Threshold:", best_val_threshold)
y_pred = np.zeros(report[1].shape)
y_pred[np.where(report[1] >= best_val_threshold)] = 1

f1_macro = f1_score(report[0], y_pred, average='macro')
f1_micro = f1_score(report[0], y_pred, average='micro')
f1_weighted = f1_score(report[0], y_pred, average='weighted')

print("F1-Score Macro on Test Set: {:.4f}".format(f1_macro))
print("F1-Score Micro on Test Set: {:.4f}".format(f1_micro))
print("F1-Score Weighted on Test Set: {:.4f}".format(f1_weighted))


# ## Check Label-Wise Metrics

y_preds = np.zeros(report[1].shape)
y_preds[np.where(report[1] >= best_val_threshold)] = 1

class_report = classification_report(y_true = report[0], 
                                              y_pred = y_preds, 
                                              target_names = new_labels, 
                                              digits = 4)
print(class_report)

