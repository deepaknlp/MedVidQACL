# A Dataset for Medical Instructional Video Classification and Question Answering

These are the benchmark experiments reported for MedVidQA dataset in our paper "A Dataset for Medical Instructional Video Classification and Question Answering" 


Please install [Anaconda](https://www.anaconda.com/distribution/) to create a conda environment as follow:
```shell script
# preparing environment
conda env create -f environment.yml
conda activate medvidqa
```

## Data Preparation
1) Download the MedVidQA dataset from [OSF repository](https://doi.org/10.17605/OSF.IO/PC594) and place train.json/val.json/test.json in `data/dataset/medvidqa` directory
2) Download the video features from [here](https://bionlp.nlm.nih.gov/), unzip the file and place the content of `MedVidQA/I3D` in `data/features/medvidqa`
3) Download the word embeddings from [here](http://nlp.stanford.edu/data/glove.840B.300d.zip) and place it to `data/word_embedding`

If you want to prepare your own video features, please follow these steps:
1) Download the pre-trained RGB model from [here](https://github.com/piergiaj/pytorch-i3d/blob/master/models/rgb_imagenet.pt) and place it in `data` directory
2) set the pythonpath
```shell script
export PYTHONPATH=$PYTHONPATH/path/to/the/medvidqa/directory
```
3) Run the following command

``python prepare/extract_medvidqa.py
``


## Training and Test

```shell script
export PYTHONPATH=$PYTHONPATH/path/to/the/medvidqa/directory
python main.py --mode train
python main.py --mode test
```

## Credit
This code repo is adapted from this [repo](https://github.com/IsaacChanghau/VSLNet).
