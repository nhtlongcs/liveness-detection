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
# python preprocessing/sanity_check.py <data-dir>
python preprocessing/sanity_check.py data/train/
```

Or you can use provided module functions to check the data.

```python
# ensure that data is valid, please wait for a while
from preprocessing.sanity_checks import check_all
check_all(osp.join(ROOT_DIR,"data/train/"))
check_all(osp.join(ROOT_DIR,"data/public/"))
```