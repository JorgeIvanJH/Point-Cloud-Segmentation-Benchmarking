# Point-Cloud-Segmentation-Benchmarking

This repo contains the first approach to standardize training an all the selected models for pointcloud segmentation, and the data processing code leveraged to turn the raw object and full real-life scenes .pcd pointcloud files of the MiniMarket dataset into more manageable HDF5 files apt for training segmentation models, and compatible with the data input requirements of the TorchPoints3D on which support for the MiniMarket dataset was added.

## Conda Environment

The configuration file for the conda environmet required to run the scripts in this repository is found in "Point-Cloud-Segmentation-Benchmarking\environment.yaml". To create the environment, run:

```bash
conda env create -f environment.yml
```

and the environment called "pointnet" wih python 3.7 with all libraries required will be created. To activate run:

```bash
conda activate pointnet
```

## Data Processing Pipeline

Parameters to build training, validation, and testing datasets are taken from "Point-Cloud-Segmentation-Benchmarking\config.yaml", specificly in the "DATA:" section. Description of the meaning of each parameters is specified there in the form of comments.

### Generate Artificial Scenes Datasets

The main code to generate artificial scenes for training and validation is located in "Point-Cloud-Segmentation-Benchmarking\data\processing\generate_trainvalid_dataset.py", there, the .pcf files are read, and for each object a hdf5 file is created containing all the pointclouds, sampled at the NUM_POINTS_PER_OBJECT specified in the config.yaml. Then, the creation of the segmentation dataset automatically follows, taking the HDF5 file of the object recognized (fully or partially) by the string in "TARGET_OBJECT_NAME" as the object if interest, and between MIN_NUM_OBJECTS and MIN_NUM_OBJECTS number of alien objects per scene, each scene involving NUM_ORIENTATIONS number of rotations of its objets involved. The raw MiniMarket directory is specified in RAW_DATASET_DIR and the final output in POINTCLOUD_SAVE_DIR.

To execute this code just run:

```bash
python generate_trainvalid_dataset.py
```

note: The original MiniMarket dataset can be downloaded from: https://www.kaggle.com/datasets/83896356f3cefb84a1256545154992a94d8ed5495c49b901bff8471c30daaacc 

### Generate Real-Life Scenes Datasets

Similar to artificial scenes, the parameter configrations are taken from the same .yaml file, and is executed from "Point-Cloud-Segmentation-Benchmarking\data\processing\generate_test_dataset.py". The HDF5 file containing the manually set labels (named "all_test_scenes.h5") is specified in RAW_TEST_SET_DIR, and the save directory in POINTCLOUD_SAVE_DIR. This file has 10 scenes only, from which those involved in the creation of the dataaset through augmentation techniques are specified in the scenes_included list (e.g. scenes_included = [1] was used to create the test set, involving the scene 1 only).

To execute this code just run:

```bash
python generate_test_dataset.py
```

#### Creation of Ground Truth in Test Scenes

A walkthrough of the creation of the grund truth (the "all_test_scenes.h5" file) of the test scenes is recorded in the following video: https://www.youtube.com/watch?v=1oW0zgDS-q8&ab_channel=JorgeIv%C3%A1nJaramilloHerrera . This video relates to the same script, where lines previous to the generation of the dataset deal with the cleaning of null values, centering, and matching of the labels in json format generated and downloaded from https://app.supervisely.com/ 

### Visualize Created Datasets

One by one, the scenes in the generated HDF5 are plotted and shown, both with its original colours, and the masks, in different plots, when running:

```bash
python Point-Cloud-Segmentation-Benchmarking\utils.py
```

With the specified file in the directory in DATASET_DIR. Along with the plot, asssert validation on the matrix sizes of each pointcloud are done as well.


## Model Training (not used in the report)

The first approach to train the segmentation models was implemented in this same repository, however it was later switched to that based on TochPoints3D. However, here we still present how it worked:

The script "Point-Cloud-Segmentation-Benchmarking\train.py" reads the config.yaml file, overiding any additional argparse parameters specified for the TRAIN and MODEL sections in it, such as the number of epochs, learning rate, batch size, etc.

The models architectures for PointNet and PointNet++ are stored in "Point-Cloud-Segmentation-Benchmarking\models\architectures", and the performance metrics, and final trained model were stored in "Point-Cloud-Segmentation-Benchmarking\experiments", in a folder named accordingly to the model being trained, when running

```bash
python train.py
```
