import logging
import json
import os
import sys
import re
import unicodedata
import math
import argparse
import random

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pickle

import torch
from transformers import (AutoTokenizer, 
                          AutoModelForSequenceClassification, 
                          Trainer, 
                          TrainingArguments,
                          EarlyStoppingCallback,
                         )
from datasets import load_metric, Dataset


# # Set Env Variables
parser = argparse.ArgumentParser()
parser.add_argument("--source_dir", type=str, default="../data/text/", help="Directory where source JSON files with text are located")
parser.add_argument("--early_stopping_patience", type=int, default=3, help="Number of epochs to wait before terminating training if no improvement")
parser.add_argument("--evaluation_metric", type=str, default="eval_f1", help="Metric to use to track and choose best model")
parser.add_argument("--seed", type=int, default=42, help="Seed to use to initialize random numbers")
args = parser.parse_args()

# Set reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
os.environ['PYTHONHASHSEED']=str(args.seed)

# # Method to Create Sequence Label for Medical Instructional vs Medical Non-Instructional vs Non-medical videos

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


# # Import Datasets 

datasets = {}

# Import JSON files first
json_filenames = [pos_json for pos_json in os.listdir(args.source_dir) if pos_json.endswith('.json')]
for json_filename in json_filenames:
    datasets[json_filename] = pd.read_json(args.source_dir + '/' + json_filename)
    # Rename columns
    datasets[json_filename] = datasets[json_filename].rename(columns = {'video_sub_title':'text', 'video_title':'title', 'label':'labels', 'video_id':'YouTube_ID'})
    # Change labels of new datasets to match the old one
    datasets[json_filename]['labels'] = datasets[json_filename]['labels'].apply(label_vid)
    # Remove duplicates
    datasets[json_filename].drop_duplicates(subset=['text'], inplace=True, ignore_index=True)
    

# Inspect Amount of datapoints and unique labels in separate dataframes
for filename in list(datasets.keys()):
    print(filename + ": " + str(len(datasets[filename])))
    print(filename + ' Label Amounts:\n', datasets[filename].labels.value_counts())


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
        
    # Initialize Model
    model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=len(set(datasets["train.json"]['labels'])),
            )
    
    # Define Metrics
    metric = load_metric("f1")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = predictions.argmax(axis=-1)
        return metric.compute(predictions=predictions, references=labels, average='macro')
    
    
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
    
    # Get Classification Reports
    def get_classification_report(trainer, test_dataset):
        preds1, label_ids1, *_ = trainer.predict(test_dataset)
        preds1 = preds1.argmax(axis=-1)
        report1 = classification_report(y_true = test_dataset['labels'],
                                        y_pred = list(map(int, preds1)),
                                        labels = list(set(test_dataset['labels'])),
                                        digits = 4)
        print(report1)
        return report1
    
    report = get_classification_report(trainer, ddatasets['test.json'])
    
    # Save Model & Tokenizer
    trainer.save_model('models/best_run_' + specific_model)
    tokenizer.save_pretrained('models/best_run_' + specific_model)


# ## Train various models

# BigBird-Base
print("BigBird-Base Train&Testing")
train_and_test_Transformer_Model("google/bigbird-roberta-base", 1024, batch_size = 4)

# Bert-Base-Uncased
print("Bert-Base-Uncased Train&Testing")
train_and_test_Transformer_Model("bert-base-uncased", 512)

# Roberta-Base
print("Roberta-Base Train&Testing")
train_and_test_Transformer_Model("roberta-base", 512)

