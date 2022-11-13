echo "example command: ./preprocess_train_data.sh /home/username/this-repo/train/"
DATA_TRAIN_DIR=$1 # Path to the directory containing the training data

VIDEO_DIR=$DATA_TRAIN_DIR/videos # Path to the directory containing the videos
KEYFRAME_DIR=$DATA_TRAIN_DIR/keyframes
FACE_DIR=$DATA_TRAIN_DIR/faces

mkdir -p $KEYFRAME_DIR
mkdir -p $FACE_DIR

# Extract frames from videos
sh preprocessing/extract_keyframes.sh $VIDEO_DIR $KEYFRAME_DIR
python preprocessing/rename_header.py $DATA_TRAIN_DIR/label.csv $DATA_TRAIN_DIR/labels_video.csv
python preprocessing/split.py $DATA_TRAIN_DIR/labels_video.csv
python preprocessing/generate_keyframes_labels.py $KEYFRAME_DIR $DATA_TRAIN_DIR/labels_video.csv $DATA_TRAIN_DIR/labels_keyframes.csv
python preprocessing/generate_keyframes_labels.py $KEYFRAME_DIR $DATA_TRAIN_DIR/labels_video_train.csv $DATA_TRAIN_DIR/labels_keyframes_train.csv
python preprocessing/generate_keyframes_labels.py $KEYFRAME_DIR $DATA_TRAIN_DIR/labels_video_test.csv $DATA_TRAIN_DIR/labels_keyframes_test.csv
PYTHONPATH=preprocessing/yolov3/ python preprocessing/yolov3/image_detect.py --input $KEYFRAME_DIR --output $FACE_DIR \
--model_def preprocessing/weight/yolov3_mask.cfg \
--weights_path preprocessing/weight/yolov3_ckpt_35.pth \
--class_path preprocessing/weight/mask_dataset.names

python preprocessing/generate_bbox_labels.py $FACE_DIR/crop $DATA_TRAIN_DIR/labels_video.csv $DATA_TRAIN_DIR/labels_bbox.csv
python preprocessing/generate_bbox_labels.py $FACE_DIR/crop $DATA_TRAIN_DIR/labels_video_train.csv $DATA_TRAIN_DIR/labels_bbox_train.csv
python preprocessing/generate_bbox_labels.py $FACE_DIR/crop $DATA_TRAIN_DIR/labels_video_test.csv $DATA_TRAIN_DIR/labels_bbox_test.csv