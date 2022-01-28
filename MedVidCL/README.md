# A Dataset for Medical Instructional Video Classification and Question Answering

These are the benchmark experiments reported for MedVidCL dataset in our paper "A Dataset for Medical Instructional Video Classification and Question Answering" 

Please install [Anaconda](https://www.anaconda.com/distribution/) to create a conda environment as follow before Data Preparation or Training and Testing:
```shell script
# preparing environment
conda create -n medvidcl python=3.9
conda activate medvidcl
pip install --upgrade pip
pip install -r requirements.txt
```

## Data Preparation
1) Download the MedVidCL dataset from [OSF repository](https://doi.org/10.17605/OSF.IO/PC594) and place train.json/val.json/test.json in the `MedVidCL` directory
2) Download the video features from [here](https://bionlp.nlm.nih.gov/VideoFeatures.zip), unzip the file and place the contents of `MedVidCL/I3D` in `data/features/I3D` and  `MedVidCL/ViT` in `data/features/ViT`
3) To extract the subtitles (i.e., text) of each YouTube video run the following command:

```
cd prepare
python Text_Extraction.py --target_dir ../data/text
```

If you want to prepare your own video features, please follow these steps:
1) Download the MedVidCL dataset from the [OSF repository](https://doi.org/10.17605/OSF.IO/PC594) and place train.json/val.json/test.json in the `MedVidCL` directory
2) Download the pre-trained RGB model from [here](https://github.com/piergiaj/pytorch-i3d/blob/master/models/rgb_imagenet.pt) and place it in `data` directory
3) Set the PYTHONPATH
```shell script
export PYTHONPATH=$PYTHONPATH/path/to/the/MedVidCL/directory
```
4) Run the following command
#### I3D Extraction
```
cd ../ 
python prepare/Extract_MedVidCL_I3D.py --dataset_dir path/to/data/with/video/ids --video_dir path/to/store/videos --images_dir path/to/store/images --save_dir path/to/store/extracted/features
```
#### ViT Extraction
```
cd ../
python prepare/Extract_MedVidCL_ViT.py --dataset_dir path/to/data/with/video/ids --video_dir path/to/store/videos --images_dir path/to/store/images --save_dir path/to/store/extracted/features
```

## Training and Testing Benchmark Models


### To Train Monomodal (Language) SVM Models
Run the following command:

```
cd ../models
python BaseLine_SVM_Text_Model_Reports.py
```

### To Train Monomodal (Language) Transformer Models
Run the following command:

```
cd ../models
python BaseLine_Transformer_Text_Model_Reports.py
```

### To Train Monomodal (Vision) Models
Run the following command:

```
cd ../models
python BaseLine_Video_Model_Reports.py
```

## To Train MultiModal (Language + Vision) Models
Run the following command:

```
cd ../models
python BaseLine_MultiModal_Model_Reports.py
```
To get the best result of the Transformer Model on the I3D dataset, run the following command
```
python BaseLine_MultiModal_Model_Reports.py --transformer_learning_rate 1e-5
```
