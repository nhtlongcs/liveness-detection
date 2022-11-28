# Liveness detection

A strong baseline for liveness detection. The challenge is a part of the [ZaloAI Challenge Series](https://challenge.zalo.ai/), a series of challenges organized by ZaloAI to promote AI research in Vietnam. The source code could be used for similar tasks, such as face anti-spoofing or detecting fake videos.

## Table of Contents
<!-- table of content of this file -->
- [Table of Contents](#table-of-contents)
- [Problem statement](#problem-statement)
- [Environment setup guide](#environment)
- [Data preparation](#data-preparation)
- [Model training and evaluation](#training-evaluation-and-inference)
- [Build docker image](#build-docker-image)
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


## Features

Currently, the following features are supported:

- [x] Training and evaluation code for liveness detection, frame-level classification / face-level classification.
- [x] Support for training on multiple GPUs.
- [x] Automatic mixed precision training.
- [x] Auto find best learning rate.
- [x] Support EfficientNet, ViT, etc.
- [x] Manage experiments with [Weights & Biases](https://wandb.ai/site).
- [x] Code management with registry, config, and logging.
- [x] Dockerfile for deployment.
- [x] Unit tests.
- [x] Packaging available.
- [x] Support semi-supervised learning on external unlabeled data.

In the future, we will add more features, such as:

- [ ] Command-line interface.
- [ ] New config system with [Hydra](https://hydra.cc/).
- [ ] New package management system with [Poetry](https://python-poetry.org/).
- [ ] Cloud training with [DVC](https://dvc.org/) and [AWS](https://aws.amazon.com/).
- [ ] Testing with [CML](https://cml.dev/).
- [ ] Support ensemble models, such as [Stacking](https://scikit-learn.org/stable/modules/ensemble.html#stacking), [Blending](https://scikit-learn.org/stable/modules/ensemble.html#blending) or [Model Soup](https://github.com/mlfoundations/model-soups).
- [ ] Support fine-tuning as a module.
- [ ] More learning strategies, such as [Decoupled Knowledge Distillation](https://github.com/megvii-research/mdistiller)
- [ ] Hyperparameter optimization with [Optuna](https://optuna.org/).

If you have any suggestions, please feel free to open an issue or pull request.

## Environment

For necessary packages, please refer to environment.yml. You can create a conda environment with the following command:

```bash
conda env create -f environment.yml 
conda activate zaloai
```

Alternatively, you can use the docker image provided by us. Please refer to the [Dockerfile](Dockerfile) for more details.


## Data preparation
Train dataset: 1168 videos of faces with facemask, in which 598 are real and 570 are fake.

Public test: 350 videos of faces with facemask, without label file.

Public test 2: 486 videos of faces with facemask, without label file.

Private test: 839 videos of faces with facemask, without label file.


Raw data is available at 
- Train dataset: [https://dl-challenge.zalo.ai/liveness-detection/train.zip](https://dl-challenge.zalo.ai/liveness-detection/train.zip)
- Public test: [https://dl-challenge.zalo.ai/liveness-detection/public_test.zip](https://dl-challenge.zalo.ai/liveness-detection/public_test.zip)

- **Public test 2**: [https://dl-challenge.zalo.ai/liveness-detection/public_test_2.zip](https://dl-challenge.zalo.ai/liveness-detection/public_test_2.zip)

The downloaded data should be extracted to the `data` folder. To see the sample data structure, please refer to the [data_sample](data_sample) folder.

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

## Build docker image 
For deployment/training purpose, docker is an ready-to-use solution.

To build docker image:
```bash
$ cd <this-repo>
$ DOCKER_BUILDKIT=1 docker build -t infection:latest .
```
To start docker container in interactive mode:
```bash
# With device is the GPU device number, and shm-size is the shared memory size 
# should be larger than the size of the model
$ docker run --rm --name infection --gpus device=0,1 --shm-size 16G -it -v $(pwd)/:/home/workspace/src/ infection:latest /bin/bash
```
To use docker container to run predict script with input data folder:
```bash
# sudo docker run –v [path to test data]:/data –v [current dir]:/result [docker name]
$ docker run --gpus device=0,1 -v /home/username/data:/data -v /home/username/result/:/result/ infection /bin/bash predict.sh
```
To use docker container with jupyter notebook: (not working yet)
```bash
$ docker run --gpus device=0,1 -p 9777:9777 -v /home/username/data:/data -v /home/username/result/:/result/ infection /bin/bash jupyter.sh
```
Other useful docker commands:
```bash
# Attach to the running container
$ docker attach <container_name> 
# list all containers
$ docker ps -a
# list all images
$ docker images 
# stop a container
$ docker stop <container_id>
# remove a container
$ docker rm <container_id>
```

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
============================== test session starts ==============================
platform linux -- Python 3.9.13, pytest-7.2.0, pluggy-1.0.0
rootdir: /home/nhtlong/playground/zalo-ai/liveness-detection
plugins: anyio-3.6.1, order-1.0.1
collected 12 items                                                                                                                                                                                    

tests/test_args.py ...                                                      [ 25%]
tests/test_data.py .                                                        [ 33%]
tests/test_env.py .                                                         [ 41%]
tests/units/test_evaluation.py .                                            [ 50%]
tests/units/test_extractor.py ..                                            [ 66%]
tests/units/test_image_folder_from_csv_ds.py .                              [ 75%]
tests/units/test_model.py .                                                 [ 83%]
tests/units/test_train_and_resume.py .                                      [ 91%]
tests/units/test_load_and_predict.py .                                      [100%]
...
======================= 12 passed, 10 warnings in 30.95s =========================
```

To run code-format

```bash
pip install pre-commit
pre-commit install
```
And every time you commit, the code will be formatted automatically. Or you can run `pre-commit run -a` to format all files.

Expected result:
```bash
$ git add scripts/ cli/
$ git commit -m "rename"         

[WARNING] Unstaged files detected.                                                               
[INFO] Stashing unstaged files to /home/nhtlong/.cache/pre-commit/...
yapf........................................................................Passed
[INFO] Restored changes from /home/nhtlong/.cache/pre-commit/...
[main f552910] rename                                                                            
 4 files changed, 4 insertions(+), 8 deletions(-)                                                
 rename {scripts => cli}/make_soup.py (100%)                                                     
 rename {scripts => cli}/predict.py (91%)                                                        
 rename {scripts => cli}/train.py (97%)                                                          
 rename {scripts => cli}/validate.py (100%)                                  
 ```

 ## Acknowledgement

 The base solution is inspired by [this discussion](https://www.kaggle.com/competitions/deepfake-detection-challenge/discussion/145721).