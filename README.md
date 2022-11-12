# Liveness detection

## Table of Contents

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

## Data preparation

Raw data is available at 
- Train dataset: [https://dl-challenge.zalo.ai/liveness-detection/train.zip](https://dl-challenge.zalo.ai/liveness-detection/train.zip)
- Public test: [https://dl-challenge.zalo.ai/liveness-detection/public_test.zip](https://dl-challenge.zalo.ai/liveness-detection/public_test.zip)

The downloaded data should be extracted to the `data` folder

```bash
|-- this-repo
    |-- data
        |-- train
            |-- VideoID.mp4
            |-- VideoID.mp4
            |-- ...
            |-- labels.csv

        |-- public_test
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

The script will generate the following files:

```bash
|-- data
    |-- train
        |-- VideoID.mp4
        |-- VideoID.mp4
        |-- ...
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
python preprocessing/sanity_check.py
```
