# The MedVidCL Dataset

These are the benchmark experiments for the MedVidCL dataset in our paper "A Dataset for Medical Instructional Video Classification and Question Answering" 

Please install [Anaconda](https://www.anaconda.com/distribution/) to create a conda environment as follow before Data Preparation or Training and Testing:
```shell script
# preparing environment
conda create -n medvidcl python=3.9
conda activate medvidcl
pip install --upgrade pip
pip install -r requirements.txt
```

# JSON Files

Each json file (train.json/val.json/test.json) contains the following fields:

- video_id:
Unique identifier for each YouTube video
- video_link:
Link to download YouTube video
- video_title:
Title of the video
- label:
Category of the video



This can be loaded into python as:

>>> import json
>>> with open('train.json', 'r') as rfile:
>>>     data_items = json.load(rfile)


Due to copyright issues, we cannot directly share the videos as a part of this dataset. The videos can be downloaded using pytube library (https://github.com/pytube/pytube):

>>> from pytube import YouTube
>>> YouTube('https://youtu.be2lAe1cqCOXo').streams.first().download()

# Data Preparation

1) Download the video features from [here](https://bionlp.nlm.nih.gov/), unzip the file and place the contents of `MedVidCL/I3D` in `data/features/I3D`
2) Download the video features from [here](https://bionlp.nlm.nih.gov/), unzip the file and place the contents of `MedVidCL/ViT` in `data/features/ViT`
3) To extract the subtitles (i.e., text) of each YouTube video, change directory to the `prepare` directory and run the following command:

``python Text_Extraction.py --target_dir ../data/text
``

If you want to prepare your own video features, change directory to the `MedVidCL` directory and please follow these steps:

## I3D Extraction
1) Download the pre-trained RGB model from [here](https://github.com/piergiaj/pytorch-i3d/blob/master/models/rgb_imagenet.pt) and place it in `data` directory
2) set the pythonpath
```shell script
export PYTHONPATH=$PYTHONPATH/path/to/the/medvidcl/directory
```
3) Run the following command

``python prepare/Extract_MedVidCL_I3D.py --dataset_dir path/to/data/with/video/ids --video_dir path/to/store/videos --images_dir path/to/store/images --save_dir path/to/store/extracted/features
``

## ViT Extraction
1) set the pythonpath
```shell script
export PYTHONPATH=$PYTHONPATH/path/to/the/medvidcl/directory
```
2) Run the following command

``python prepare/Extract_MedVidCL_ViT.py --dataset_dir path/to/data/with/video/ids --video_dir path/to/store/videos --images_dir path/to/store/images --save_dir path/to/store/extracted/features
``

# Dataset statistics
Training dataset:
{'Medical Non-instructional': 2394, 'Non-medical': 1034, 'Medical Instructional': 789}

Validation Dataset:
{'Medical Non-instructional': 100, 'Non-medical': 100, 'Medical Instructional': 100}

Test Dataset:
{'Medical Instructional': 600, 'Medical Non-instructional': 500, 'Non-medical': 500}

# Training and Testing BaseLine Models

1) After Data Preparation, change directory to the `models` directory

## Text SVM Models
2) Run the following command:

``python BaseLine_SVM_Text_Model_Reports.py
``

## Text Transformer Models
2) Run the following command:

``python BaseLine_Transformer_Text_Model_Reports.py
``

## Video Models
2) Run the following command:

``python BaseLine_Video_Model_Reports.py
``

## MultiModal Models
2) Run the following command:

``python BaseLine_MultiModal_Model_Reports.py
``
