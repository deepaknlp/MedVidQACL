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

# Data Preparation
1) Download the MedVidCL dataset from [OSF repository](https://doi.org/10.17605/OSF.IO/PC594) and place train.json/val.json/test.json in the `MedVidCL` directory
2) Download the video features from [here](https://bionlp.nlm.nih.gov/), unzip the file and place the contents of `MedVidCL/I3D` in `data/features/I3D`
3) Download the video features from [here](https://bionlp.nlm.nih.gov/), unzip the file and place the contents of `MedVidCL/ViT` in `data/features/ViT`
4) To extract the subtitles (i.e., text) of each YouTube video, change directory to the `prepare` directory and run the following command:

``python Text_Extraction.py --target_dir ../data/text
``

If you want to prepare your own video features, change directory to the `MedVidCL` directory and please follow these steps:

## I3D Extraction
1) Download the MedVidCL dataset from the [OSF repository]() and place train.json/val.json/test.json in the `MedVidCL` directory
2) Download the pre-trained RGB model from [here](https://github.com/piergiaj/pytorch-i3d/blob/master/models/rgb_imagenet.pt) and place it in `data` directory
3) Set the pythonpath
```shell script
export PYTHONPATH=$PYTHONPATH/path/to/the/medvidcl/directory
```
4) Run the following command

``python prepare/Extract_MedVidCL_I3D.py --dataset_dir path/to/data/with/video/ids --video_dir path/to/store/videos --images_dir path/to/store/images --save_dir path/to/store/extracted/features
``

## ViT Extraction
1) Set the pythonpath
```shell script
export PYTHONPATH=$PYTHONPATH/path/to/the/medvidcl/directory
```
2) Run the following command

``python prepare/Extract_MedVidCL_ViT.py --dataset_dir path/to/data/with/video/ids --video_dir path/to/store/videos --images_dir path/to/store/images --save_dir path/to/store/extracted/features
``

# Training and Testing BaseLine Models

1) After Data Preparation, change directory to the `models` directory

## To Train Monomodal (Language) SVM Models
2) Run the following command:

``python BaseLine_SVM_Text_Model_Reports.py
``

## To Train Monomodal (Language) Transformer Models
2) Run the following command:

``python BaseLine_Transformer_Text_Model_Reports.py
``

## To Train Monomodal (Vision) Models
2) Run the following command:

``python BaseLine_Video_Model_Reports.py
``

## To Train MultiModal (Language + Vision) Models
2) Run the following command:

``python BaseLine_MultiModal_Model_Reports.py
``
To get the best result of the Transformer Model on the I3D dataset, run the following command
``python BaseLine_MultiModal_Model_Reports.py --transformer_learning_rate 1e-5
``
