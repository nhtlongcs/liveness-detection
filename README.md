# Liveness detection

## Table of Contents
<!-- table of content of this file -->
- [Table of Contents](#table-of-contents)
- [Problem statement](#problem-statement)
- [Environment setup guide](#environment)
- [Data preparation](#data-preparation)
- [Model training and evaluation](#training-evaluation-and-inference)
- [Contribution guide](#contribution-guide)


## Problem statement

In verification services related to face recognition (such as eKYC and face access control), the key question is whether the input face video is real (from a live person present at the point of capture), or fake (from a spoof artifact or lifeless body). Liveness detection is the AI problem to answer that question.

In this challenge, participants will build a liveness detection model to classify if a given facial video is real or spoofed.

- Input: a video of selfie/portrait face with a length of 1-5 seconds (you can use any frames you like).

- Output: Liveness score in [0...1] (0 = Fake, 1 = Real).

Example Output: Predict.csv

| fname        | liveness_score |
|--------------|----------------|
| VideoID.mp4  | 0.10372        |
| ,,,          | ...            |
| ,,,          | ...            |
| ,,,          | ...            |
## Environment

For necessary packages, please refer to environment.yml. You can create a conda environment with the following command:

```bash
conda env create -f environment.yml 
```

Alternatively, you can use the docker image provided by us. Please refer to the [Dockerfile](Dockerfile) for more details.


## Data preparation

Raw data is available at 
- Train dataset: [https://dl-challenge.zalo.ai/liveness-detection/train.zip](https://dl-challenge.zalo.ai/liveness-detection/train.zip)
- Public test: [https://dl-challenge.zalo.ai/liveness-detection/public_test.zip](https://dl-challenge.zalo.ai/liveness-detection/public_test.zip)

The downloaded data should be extracted to the `data` folder

```text
|-- this-repo
    |-- data
        |-- train
            |-- labels.csv
            |-- videos
                |-- VideoID.mp4
                |-- VideoID.mp4
                |-- ...

        |-- public_test
            |-- videos
                |-- VideoID.mp4
                |-- VideoID.mp4
                |-- ...
|-- ...
```

Preprocessing is done in a single script `preprocess_train_data.sh` which requires dataset directory as first argument. It will execute the steps below:
- Extract frames from videos
- Crop faces from frames using YoloV3
- Rename the columns of `labels.csv` to `filename` and `label` and save it to `labels_video.csv`
- Split the dataset into train and validation sets
- Generate `labels_keyframes_*.csv` files which contain the labels for each keyframe
- Generate `labels_bbox_*.csv` files which contain the labels for each bounding box

> *(IMPORTANT NOTE)* This step is only required if you want to re-preprocess the data. If you want to use the preprocessed data, you can skip this step and download the preprocessed data from [GoogleDrive](https://drive.google.com/drive/u/2/folders/1Fx_UW9Ic-crr7Z57Z-3fv9tZVm8vl2EW) then extract it to the `data/train` folder.

The script will generate the following files:

```text
|-- data
    |-- train
        |-- labels.csv                          // original labels.csv
        |-- labels_video_train.csv              // labels for each video (train set)
        |-- labels_video_test.csv               // labels for each video (validation set)
        |-- labels_video.csv                    // labels for each video (whole dataset)
        |-- labels_keyframes_train.csv          // labels for each keyframe (train set)
        |-- labels_keyframes_test.csv           // labels for each keyframe (validation set)
        |-- labels_keyframes.csv                // labels for each keyframe (whole dataset)
        |-- labels_bbox_train.csv               // labels for each bounding box (train set)
        |-- labels_bbox_test.csv                // labels for each bounding box (validation set)
        |-- labels_bbox.csv                     // labels for each bounding box (whole dataset)
        |-- videos
            |-- VideoID.mp4
            |-- VideoID.mp4
            |-- ...
        |-- frames                              // extracted frames directory
            |-- VideoID-FrameID.jpg
            |-- ...
        |-- faces                               // extracted faces data directory
            |-- bbox                            // extracted faces bbox directory
                |-- VideoID-FrameID.txt         // each line is a bounding box (CLS_NAME, CONF, X, Y, W, H)
                |-- ...
            |-- crop                            // extracted croped faces directory
                |-- VideoID-FrameID.jpg
                |-- ...
            |-- viz                             // bbox visualization directory
                |-- VideoID-FrameID.jpg
                |-- ...
            |-- ...
```

For sanity check, you can run the following command comparing the extracted frames and faces with the original videos and labels. The script will compare the matched frames and faces with the original videos and labels. If there is any mismatch, the script will print the mismatched frames and faces. It will also print the number of matched videos, frames, and faces in the train and validation sets.

```bash
python preprocessing/sanity_check.py // Not working yet
```

## Training, evaluation, and inference

Provided scripts are in `scripts` folder. All scripts have the same interface which requires the following arguments:
- `-c` or `--config`: path to the config file
- `-o` or `--opt`: additional options to override the config file (e.g. `--opt extractor.name=efficientnet`)
For example, checkout the provided config files in `configs` folder and training instructions in `train.ipynb` notebook. Same for evaluation and inference in `predict.ipynb` notebook.

## Contribution guide

If you want to contribute to this repo, please follow steps below:

1. Fork your own version from this repository
1. Checkout to another branch, e.g. `fix-loss`, `add-feat`.
1. Make changes/Add features/Fix bugs
1. Add test cases in the `test` folder and run them to make sure they are all passed (see below)
1. Create and describe feature/bugfix in the PR description (or create new document)
1. Push the commit(s) to your own repository
1. Create a pull request on this repository

```bash
pip install pytest
python -m pytest tests/
```

Expected result:

```bash
============================== test session starts ===============================
platform darwin -- Python 3.7.12, pytest-7.1.1, pluggy-1.0.0
rootdir: /Users/nhtlong/workspace/zaloai2022/
collected 10 items

tests/test_env.py ...                                                      [ 30%]
tests/test_utils.py .                                                      [ 40%]
tests/test_dataset.py .                                                    [ 50%]
tests/test_eval.py .                                                       [ 60%]
tests/test_extractor.py ...                                                [ 90%]
tests/test_model.py .                                                      [100%]
```
