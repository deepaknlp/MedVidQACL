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

import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, f1_score, classification_report

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf


# # Set Env Variables
parser = argparse.ArgumentParser()
parser.add_argument("--source_dir", type=str, default="../data/text/", help="Directory where source JSON files with text are located")
parser.add_argument("--i3d_feature_dir", type=str, default="../data/features/I3D/", help="Directory where I3D-extracted video features are located")
parser.add_argument("--vit_feature_dir", type=str, default="../data/features/ViT/", help="Directory where ViT-extracted video features are located")
parser.add_argument("--seed", type=int, default=42, help="Seed to use to initialize random numbers")
parser.add_argument("--batch_size", type=int, default=16, help="Batch Size to train the models with")
parser.add_argument("--epochs", type=int, default=50, help="Number of iterations to spend training the model")
parser.add_argument("--lstm_learning_rate", type=float, default=4e-4, help="Learning Rate for LSTM Model")
parser.add_argument("--transformer_learning_rate", type=float, default=1e-5, help="Learning Rate for Transformer Model")
args = parser.parse_args()

I3D_MAX_SEQ_LENGTH = 2
VIT_MAX_SEQ_LENGTH = 20
I3D_NUM_FEATURES = 1024
VIT_NUM_FEATURES = 768

# Create directory to hold models
if not os.path.isdir('./tmp'):
    os.mkdir('./tmp')

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


# # LSTM Video Classifier

class LSTM_MultiClassifier(nn.Module):
    '''Create LSTM_MultiClassifier class from PyTorch'''
    def __init__(self, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        super(LSTM_MultiClassifier, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        # Create standard LSTM model with a classification head
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_dim, output_size)
        
    def forward(self, x, hidden):
        # Standard training method for LSTM model
        lstm_out, hidden = self.lstm(x, hidden)
        
        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = out[:,-1]
        return out, hidden
    
    def init_hidden(self, batch_size):
        # Initialize the weights of the model
        hidden = (torch.rand(self.n_layers, batch_size, self.hidden_dim).to(device),
                  torch.rand(self.n_layers, batch_size, self.hidden_dim).to(device))
        return hidden


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


# # Method to Create Sequence Label for Instructional vs Non-Instructional vs Non-Medical

def label_vid(row):
    if "Medical Non-instructional" in row:
        return 0
    elif "Medical Instructional" in row:
        return 1
    elif "Non-medical" in row:
        return 2
    else:
        return -1


# # Method to Import Data of Different Types

def import_datasets(vit_datatype = True):
    
    datasets = {}
    torch_features = {}
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
        # Add feature numpy array
        if vit_datatype:
            datasets[json_filename]['features'] = datasets[json_filename]['YouTube_ID'].apply(lambda x: np.load(args.vit_feature_dir + json_filename[:-5] + '/' + x + '.npy'))
        else:
            datasets[json_filename]['features'] = datasets[json_filename]['YouTube_ID'].apply(lambda x: np.load(args.i3d_feature_dir + json_filename[:-5] + '/' + x + '.npy'))
        # Convert all numpy arrays to float32
        datasets[json_filename]['features'] = datasets[json_filename]['features'].apply(lambda x: x.astype('float32'))
        # Convert all numpy arrays to Pytorch tensors
        datasets[json_filename]['torch_features'] = datasets[json_filename]['features'].apply(lambda x: torch.Tensor(x))
        # Change each feature column to a list
        torch_features[json_filename] = torch.nn.utils.rnn.pad_sequence(datasets[json_filename]['torch_features'].to_list(), batch_first=True, padding_value=0)
    
    return datasets, torch_features


# # Import Raw Datasets 

# Import I3D-extracted, 20-frame-based datasets
datasets, torch_features = import_datasets(vit_datatype = False)
# Import ViT-extracted, 20-frame-based datasets
vit_datasets, vit_torch_features = import_datasets()


# # Methods to train Baseline Models

# Method to train LSTM Model 

def train_and_test_lstm(dataset, filepath, max_seq_length, num_features, batch_size, learning_rate = 1e-4, hidden_dim=128, epochs=args.epochs):
    # Set up standard LSTM Classifier to classify 3 classes
    model = LSTM_MultiClassifier(3, num_features, hidden_dim, 2, 0.2)
    # Send to GPU
    model = model.to(device)
    # Initialize with loss function and Adam optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    
    # Set variables for training parameters and early stopping
    epochs = epochs
    counter = 0
    print_every = 1000
    clip = 5
    valid_loss_min = np.Inf
    early_stopping_count = 0
    early_stopping_patience = 5

    model.train()
    for i in range(epochs):
        # Early Stopping
        if early_stopping_count >= early_stopping_patience:
            print("Early Stopping Reached. Ending Training")
            break
        early_stopping_count += 1
        
        # Set up initial weights
        h_init = model.init_hidden(batch_size)
        
        for inputs, labels in dataset['train.json']:
            # Perform a training step
            counter += 1
            h_init = tuple([e.data for e in h_init])
            inputs, labels = inputs.to(device), labels.to(device)
            model.zero_grad()
            output, h = model(inputs, h_init)
            loss = criterion(output, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            if counter%print_every == 0:
                # Perform a validation step every 1000 training examples
                val_losses = []
                model.eval()
                for inp, lab in dataset['val.json']:
                    inp, lab = inp.to(device), lab.to(device)
                    out, val_h = model(inp, h_init)
                    val_loss = criterion(out, lab)
                    val_losses.append(val_loss.item())

                model.train()
                # Log Performance metrics
                print("Epoch: {}/{}...".format(i+1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.6f}...".format(loss.item()),
                      "Val Loss: {:.6f}".format(np.mean(val_losses)))
                if np.mean(val_losses) <= valid_loss_min:
                    torch.save(model.state_dict(), filepath)
                    print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,np.mean(val_losses)))
                    valid_loss_min = np.mean(val_losses)
                    early_stopping_count = 0
                    
    # Loading the best model
    model.load_state_dict(torch.load(filepath))

    # Test
    test_targets = []
    test_preds = []
    
    model.eval()
    for inputs, labels in dataset['test.json']:
        if batch_size == 1:
            test_targets.append(int(labels))
        else:
            test_targets.extend(labels.tolist())
        inputs, labels = inputs.to(device), labels.to(device)
        output, h = model(inputs, h_init)
        if batch_size == 1:
            test_preds.append(int(output.argmax(axis=-1)))
        else:
            test_preds.extend(output.argmax(axis=-1).tolist())

    test_report = classification_report(y_true = test_targets,
                          y_pred = test_preds,
                          labels = [0, 1, 2],
                          digits = 4)
    print("Test Classification Report:\n", test_report)

# Method to train Transformer Model 

def get_compiled_model(max_seq_length, num_features, learning_rate, dense_dim=12):
    # Set parameters for Transformer Encoder architecture
    sequence_length = max_seq_length
    embed_dim = num_features
    dense_dim = dense_dim
    num_heads = 12
    classes = 3
    
    inputs = keras.Input(shape=(None, None))
    x = PositionalEmbedding(
        sequence_length, embed_dim, name="frame_position_embedding"
    )(inputs)
    x = TransformerEncoder(embed_dim, dense_dim, num_heads, name="transformer_layer")(x)
    # Add simple Max-pooling layer
    x = layers.GlobalMaxPooling1D()(x)
    # Send to Dropout Layer
    x = layers.Dropout(0.1)(x)
    # Send to Fully Connected layer to get probabilities for each of the 3 classes
    outputs = layers.Dense(classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    
    # Compile model with Adam optimizer and loss function
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_2=0.999), loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def train_and_test_transformer(datasets, labels, filepath, max_seq_length, num_features, learning_rate=1e-5, dense_dim=12, epochs=args.epochs):
    # Create callbacks
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath, save_weights_only=True, save_best_only=True, verbose=1
    )
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Train model with early stopping based on validation dataset
    model = get_compiled_model(max_seq_length, num_features, learning_rate, dense_dim)
    history = model.fit(
        datasets['train.json'],
        labels['train.json'],
        validation_data=(datasets['val.json'], labels['val.json']),
        epochs=epochs,
        callbacks=[checkpoint, early_stopping],
    )
    
    # Load in model
    model.load_weights(filepath)
    # Test on Set 1
    test_preds = model.predict(datasets['test.json']).argmax(axis=-1)
    test_report = classification_report(y_true = labels['test.json'],
                          y_pred = test_preds,
                          labels = [0, 1, 2],
                          digits = 4)
    print("Test Classification Report:\n", test_report)

    return model


# ## Preprocess all Datasets

tensor_datasets = {}
dls = {}

vit_tensor_datasets = {}
vit_dls = {}

transformer_datasets = {}
transformer_labels = {}

vit_transformer_datasets = {}
vit_transformer_labels = {}

json_filenames = [pos_json for pos_json in os.listdir(args.source_dir) if pos_json.endswith('.json')]
for json_filename in json_filenames:
    # Develop I3D datasets for LSTM
    tensor_datasets[json_filename] = TensorDataset(torch_features[json_filename], torch.from_numpy(datasets[json_filename]['labels'].values))
    dls[json_filename] = DataLoader(tensor_datasets[json_filename], shuffle=False, batch_size=args.batch_size, drop_last=True)

    # Develop ViT datasets for LSTM
    vit_tensor_datasets[json_filename] = TensorDataset(vit_torch_features[json_filename], torch.from_numpy(vit_datasets[json_filename]['labels'].values))
    vit_dls[json_filename] = DataLoader(vit_tensor_datasets[json_filename], shuffle=False, batch_size=args.batch_size, drop_last=True)

    # Develop I3D datasets for Transformers
    transformer_datasets[json_filename] = torch_features[json_filename].detach().numpy()
    transformer_labels[json_filename] = datasets[json_filename]['labels'].values

    # Develop ViT datasets for Transformers
    vit_transformer_datasets[json_filename] = vit_torch_features[json_filename].detach().numpy()
    vit_transformer_labels[json_filename] = vit_datasets[json_filename]['labels'].values

# Train and test LSTM model on I3D Dataset
train_and_test_lstm(dls, './tmp/lstm_i3d_trained_state_dict.pt', I3D_MAX_SEQ_LENGTH, I3D_NUM_FEATURES, args.batch_size, args.lstm_learning_rate)

# Train and test LSTM model on VIT Dataset
train_and_test_lstm(vit_dls, './tmp/lstm_vit_trained_state_dict.pt', VIT_MAX_SEQ_LENGTH, VIT_NUM_FEATURES, args.batch_size, args.lstm_learning_rate)

# Train and Test Transformer model on I3D Dataset
trained_model = train_and_test_transformer(transformer_datasets, transformer_labels, "./tmp/transformer_i3d_video_classifier", I3D_MAX_SEQ_LENGTH, I3D_NUM_FEATURES, args.transformer_learning_rate)

# Train and Test Transformer model on ViT Dataset
vit_trained_model = train_and_test_transformer(vit_transformer_datasets, vit_transformer_labels, "./tmp/transformer_vit_video_classifier", VIT_MAX_SEQ_LENGTH, VIT_NUM_FEATURES, args.transformer_learning_rate)
