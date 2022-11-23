echo "example command: ./preprocess_train_data.sh /home/username/this-repo/train/"
echo "This script only generates labels files in case keyframes and faces extraction has been done."
DATA_TRAIN_DIR=$1 # Path to the directory containing the training data

VIDEO_DIR=$DATA_TRAIN_DIR/videos # Path to the directory containing the videos
KEYFRAME_DIR=$DATA_TRAIN_DIR/keyframes
FACE_DIR=$DATA_TRAIN_DIR/faces

mkdir -p $KEYFRAME_DIR
mkdir -p $FACE_DIR

# Extract frames from videos
python preprocessing/rename_header.py $DATA_TRAIN_DIR/label.csv $DATA_TRAIN_DIR/labels_video.csv
python preprocessing/split.py $DATA_TRAIN_DIR/labels_video.csv
python preprocessing/generate_keyframes_labels.py $KEYFRAME_DIR $DATA_TRAIN_DIR/labels_video.csv $DATA_TRAIN_DIR/labels_keyframes.csv
python preprocessing/generate_keyframes_labels.py $KEYFRAME_DIR $DATA_TRAIN_DIR/labels_video_train.csv $DATA_TRAIN_DIR/labels_keyframes_train.csv
python preprocessing/generate_keyframes_labels.py $KEYFRAME_DIR $DATA_TRAIN_DIR/labels_video_test.csv $DATA_TRAIN_DIR/labels_keyframes_test.csv

python preprocessing/generate_bbox_labels.py $FACE_DIR/crop $DATA_TRAIN_DIR/labels_video.csv $DATA_TRAIN_DIR/labels_bbox.csv
python preprocessing/generate_bbox_labels.py $FACE_DIR/crop $DATA_TRAIN_DIR/labels_video_train.csv $DATA_TRAIN_DIR/labels_bbox_train.csv
python preprocessing/generate_bbox_labels.py $FACE_DIR/crop $DATA_TRAIN_DIR/labels_video_test.csv $DATA_TRAIN_DIR/labels_bbox_test.csv