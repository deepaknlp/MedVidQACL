import logging
import json
import os
import sys
import re
import unicodedata
import math
import time
import copy
from typing import Tuple
import random
import argparse
import pickle
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, f1_score, classification_report, average_precision_score

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from torchtext.legacy.data import Field, TabularDataset, BucketIterator
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import tensorflow_ranking as tfr
import keras.backend as K
import tensorflow_addons as tfa


# # Set Env Variables

parser = argparse.ArgumentParser()
parser.add_argument("--source_dir", type=str, default="../data/text/Med-Instr-Hierarchical/", help="Directory where source JSON files with text are located")
parser.add_argument("--i3d_feature_dir", type=str, default="../features/I3D/", help="Directory where I3D-extracted video features are located")
parser.add_argument("--vit_feature_dir", type=str, default="../features/ViT/", help="Directory where ViT-extracted video features are located")
parser.add_argument("--seed", type=int, default=42, help="Seed to use to initialize random numbers")
parser.add_argument("--batch_size", type=int, default=16, help="Batch Size to train the models with")
parser.add_argument("--epochs", type=int, default=50, help="Number of iterations to spend training the model")
parser.add_argument("--lstm_learning_rate", type=float, default=4e-4, help="Learning Rate for LSTM Model")
parser.add_argument("--transformer_learning_rate", type=float, default=5e-5, help="Learning Rate for Transformer Model")
args = parser.parse_args()

VIT_MAX_SEQ_LENGTH = 20
VIT_NUM_FEATURES = 768

# Create directory to hold models
if not os.path.isdir('./multimodal_tmp'):
    os.mkdir('./multimodal_tmp')

# Set reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
os.environ['PYTHONHASHSEED']=str(args.seed)
tf.random.set_seed(args.seed)

# Set device to GPU if available
is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# # Transformer Video Classifier

class PositionalEmbedding(layers.Layer):
    '''Create PositionalEmbedding class for Transformer from Keras'''
    def __init__(self, sequence_length, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim
        )
        self.sequence_length = sequence_length
        self.output_dim = output_dim

    def call(self, inputs):
        # The inputs are of shape: `(batch_size, frames, num_features)`
        length = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_positions = self.position_embeddings(positions)
        return inputs + embedded_positions

    def compute_mask(self, inputs, mask=None):
        mask = tf.reduce_any(tf.cast(inputs, "bool"), axis=-1)
        return mask

class TransformerEncoder(layers.Layer):
    '''Create TransformerEncoder class from Keras'''
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        
        # Create standard BERT-Based Transformer model
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.1
        )
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation=tf.nn.gelu), layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization(epsilon=1e-12)
        self.layernorm_2 = layers.LayerNormalization(epsilon=1e-12)

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]

        attention_output = self.attention(inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


# # Creating One-Hot-Encoded Binary Labels

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


# # Method to Import Data of Different Types

def import_datasets():
    
    datasets = {}
    torch_features = {}
    # Import JSON files first
    json_filenames = [pos_json for pos_json in os.listdir(args.source_dir) if pos_json.endswith('.json')]
    video_filenames = [filename.replace(".json", "_vit_v2.json") for filename in json_filenames]
    for video_filename, json_filename in zip(video_filenames, json_filenames):
        datasets[json_filename] = pd.read_json(args.source_dir + '/' + json_filename)
        # Rename columns
        datasets[json_filename] = datasets[json_filename].rename(columns = {'video_sub_title':'text', 'video_title':'title', 'video_id':'YouTube_ID'})
        # Change labels of new datasets to match the old one
        datasets[json_filename]['labels'] = datasets[json_filename]['level_2_category_labels'].apply(label_vid)
        # Remove duplicates
        datasets[json_filename].drop_duplicates(subset=['YouTube_ID'], inplace=True, ignore_index=True)
        # Send to CSV
        #datasets[json_filename][['labels', 'text']].to_csv(args.source_dir+json_filename[:-5]+'.csv', index=False)
        # Add feature numpy array
        datasets[json_filename]['features'] = datasets[json_filename]['YouTube_ID'].apply(lambda x: np.load(args.vit_feature_dir + video_filename[:-5] + '/' + x + '.npy'))
        # Convert all numpy arrays to float32
        datasets[json_filename]['features'] = datasets[json_filename]['features'].apply(lambda x: x.astype('float32'))
        # Convert all numpy arrays to Pytorch tensors
        datasets[json_filename]['torch_features'] = datasets[json_filename]['features'].apply(lambda x: torch.Tensor(x))
        # Change each feature column to a list
        torch_features[json_filename] = torch.nn.utils.rnn.pad_sequence(datasets[json_filename]['torch_features'].to_list(), batch_first=True, padding_value=0)
    
    return datasets, torch_features


# # Import Datasets

# Import ViT-extracted, 20-frame-based datasets
vit_datasets, vit_torch_features = import_datasets()


# ## Preprocess all Datasets

vit_transformer_datasets = {}
vit_transformer_labels = {}
json_filenames = [pos_json for pos_json in os.listdir(args.source_dir) if pos_json.endswith('.json')]
for json_filename in json_filenames:

    # Develop ViT datasets for Transformers
    vit_transformer_datasets[json_filename] = vit_torch_features[json_filename].detach().numpy()
    vit_transformer_labels[json_filename] = np.stack(vit_datasets[json_filename]['labels'].values)


# # Prepare Text Dataset for Transformers

tf_train_docs = list(zip(vit_datasets['train.json']['text'], vit_datasets['train.json']['labels'])) 
tf_val_docs = list(zip(vit_datasets['val.json']['text'], vit_datasets['val.json']['labels'])) 
tf_test_docs = list(zip(vit_datasets['test.json']['text'], vit_datasets['test.json']['labels'])) 

## Hyperparameters for tokenizer
vocab_size = 30000
maxlen = 512


# Prepare training set
## texts vs. labels
train_texts = np.array([t for (t, l) in tf_train_docs])
train_labels = np.array([l for (t, l) in tf_train_docs])
## tokenizer
tokenizer = keras.preprocessing.text.Tokenizer(num_words=vocab_size)
## fit tokenizer
tokenizer.fit_on_texts(train_texts)
## text to sequences
train_texts_to_int = tokenizer.texts_to_sequences(train_texts)
## pad sequences
train_texts_to_int_pad = keras.preprocessing.sequence.pad_sequences(train_texts_to_int,
                                                              maxlen=maxlen,
                                                              truncating='post',
                                                              padding='post')

# Prepare validation set
## Perform same vectorization on validation set
val_texts = np.array([t for (t,l) in tf_val_docs])

val_texts_to_int_pad = keras.preprocessing.sequence.pad_sequences(
    tokenizer.texts_to_sequences(val_texts),
    maxlen=maxlen,
    truncating='post',
    padding='post'
)
    
val_labels = np.array([l for (t, l) in tf_val_docs])

# Prepare test set
## Perform same vectorization on testing set
test_texts = np.array([t for (t,l) in tf_test_docs])

test_texts_to_int_pad = keras.preprocessing.sequence.pad_sequences(
    tokenizer.texts_to_sequences(test_texts),
    maxlen=maxlen,
    truncating='post',
    padding='post'
)
    
test_labels = np.array([l for (t, l) in tf_test_docs])

total_vocab_size = len(tokenizer.word_index) + 1


# # Train Baseline Transformer Models

# Calculate mean average precision score
def mAP(y_true, y_pred):
    sum_precision = tf.constant(0, dtype=tf.float64)
    number_of_nan_free_precision_scores_for_labels = tf.constant(0, dtype=tf.float64)
    
    for i in range(y_true.shape[1]):
        precision = tf.py_function(average_precision_score, (y_true[:, i], y_pred[:, i]), tf.double)
        if not tf.math.is_nan(precision):
            sum_precision += precision
            number_of_nan_free_precision_scores_for_labels += tf.constant(1, dtype=tf.float64)
    return sum_precision / number_of_nan_free_precision_scores_for_labels


def f1_weighted_for_py(y_true, y_pred):
    return f1_score(y_true, y_pred, average='weighted')


# Calculate f1-weighted over thresholds
def f1Weighted(y_true, y_pred):
    best_threshold = tf.constant(0, dtype=tf.float64)
    best_f1_weighted = tf.constant(0, dtype=tf.float64)
    for eval_threshold in np.arange(0.01, 0.5, 0.01):
        y_pred_numpy = y_pred.numpy()
        preds = np.zeros(y_pred_numpy.shape)
        preds[np.where(y_pred_numpy >= eval_threshold)] = 1
        preds = tf.convert_to_tensor(preds)

        result = tf.py_function(f1_weighted_for_py, (y_true, preds), tf.double)
        
        if result > best_f1_weighted:
            best_f1_weighted = result
            best_threshold = eval_threshold
    return best_f1_weighted


# Method to train Transformer Model 
def get_compiled_model(max_seq_length, num_features, vocab_size, maxlen, learning_rate, dense_dim=12):
    # Set parameters for Transformer Encoder architecture
    # Video Parameters
    sequence_length = max_seq_length
    embed_dim = num_features
    dense_dim = dense_dim
    num_heads = 12
    
    # Text parameters
    text_embed_dim = 32  # Embedding size for each token
    text_num_heads = 2  # Number of attention heads
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer
    
    classes = train_labels.shape[1]
    
    video_inputs = keras.Input(shape=(None, None))
    x = PositionalEmbedding(
        sequence_length, embed_dim, name="frame_position_embedding"
    )(video_inputs)
    x = TransformerEncoder(embed_dim, dense_dim, num_heads, name="transformer_layer")(x)
    # Add simple Max-pooling layer
    x = layers.GlobalMaxPooling1D()(x)
    # Send to Dropout Layer
    x = layers.Dropout(0.1)(x)
    
    ## Using Sequential API
    text_inputs = layers.Input(shape=(maxlen,))
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, text_embed_dim)
    y = embedding_layer(text_inputs)
    transformer_block = TransformerBlock(text_embed_dim, text_num_heads, ff_dim)
    y = transformer_block(y)
    y = layers.GlobalAveragePooling1D()(y)
    y = layers.Dropout(0.1)(y)
    y = layers.Dense(12, activation="relu")(y)
    y = layers.Dropout(0.1)(y)
    # Concatenate the two outputs here and then send through softmax
    z = tf.keras.layers.Concatenate(axis=-1)([x, y])
    
    # Send to Fully Connected layer to get probabilities for each of the 16 classes
    outputs = layers.Dense(classes, activation="sigmoid")(z)
    model = keras.Model([video_inputs, text_inputs], outputs)
    
    # Compile model with Adam optimizer and loss function
    model.compile(
        run_eagerly=True, optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_2=0.999), loss="binary_crossentropy", metrics=[mAP, 
                                                                                                                                              f1Weighted,
                                                                                                                            tfa.metrics.F1Score(num_classes=classes,
                                                                                                                                                average='macro',
                                                                                                                                                threshold=0.22,
                                                                                                                                                name = "f1_macro"),
                                                                                                                            tfa.metrics.F1Score(num_classes=classes,
                                                                                                                                                average='micro',
                                                                                                                                                threshold=0.22,
                                                                                                                                                name = "f1_micro")]
    )
    return model


def train_and_test_transformer(datasets, labels, filepath, max_seq_length, num_features, vocab_size, maxlen, learning_rate=1e-5, dense_dim=12, epochs=args.epochs):
    # Create callbacks
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath, monitor = "val_f1Weighted", mode = "max", save_weights_only=True, save_best_only=True, verbose=1
    )
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_f1Weighted', patience=10, mode="max", restore_best_weights=True)
    
    # Train model with early stopping based on validation dataset
    model = get_compiled_model(max_seq_length, num_features, vocab_size, maxlen, learning_rate, dense_dim)
    history = model.fit(
        [datasets['train.json'], train_texts_to_int_pad],
        labels['train.json'],
        validation_data=([datasets['val.json'], val_texts_to_int_pad], labels['val.json']),
        epochs=epochs,
        callbacks=[checkpoint, early_stopping],
    )
    
    # Load in model
    model.load_weights(filepath)
    # Test on Set 1
    test1_preds = model.predict([datasets['test.json'], test_texts_to_int_pad])
    test1_labels = labels['test.json']

    #return report
    return (test1_labels, test1_preds)

# Train and Test Transformer model on ViT Dataset
report = train_and_test_transformer(vit_transformer_datasets, vit_transformer_labels, "./multimodal_tmp/transformer_vit_multimodal_classifier", VIT_MAX_SEQ_LENGTH, VIT_NUM_FEATURES, vocab_size, maxlen, args.transformer_learning_rate)


# Save report
with open('multilabel_multimodal_report_13_labels.pickle', 'wb') as f:
    pickle.dump(report, f)


# ## Check Best Threshold

# Load report
with open('multilabel_multimodal_report_13_labels.pickle', 'rb') as f:
    report = pickle.load(f)


eval_threshold = 0.22

f1_macro = tfa.metrics.F1Score(num_classes=len(new_labels), threshold=eval_threshold, average="macro")
f1_macro.update_state(report[0], report[1])
result = f1_macro.result()
print("Test Set 1 F1-Macro Score:", result)

f1_micro = tfa.metrics.F1Score(num_classes=len(new_labels), threshold=eval_threshold, average="micro")
f1_micro.update_state(report[0], report[1])
result2 = f1_micro.result()
print("Test Set 1 F1-Micro Score:", result2)

f1_weighted = tfa.metrics.F1Score(num_classes=len(new_labels), threshold=eval_threshold, average="weighted")
f1_weighted.update_state(report[0], report[1])
result3 = f1_weighted.result()
print("Test Set 1 F1-Weighted Score:", result3)

result4 = mAP(report[0], report[1])
print("Test Set 1 mAP Score:", result4)


# ## Check Label-Wise Metrics

y_preds = np.zeros(report[1].shape)
y_preds[np.where(report[1] >= eval_threshold)] = 1

class_report = classification_report(y_true = report[0], 
                                              y_pred = y_preds, 
                                              target_names = new_labels, 
                                              digits = 4)
print(class_report)

