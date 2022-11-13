echo "example command: ./preprocess_train_data.sh /home/username/this-repo/public/"
DATA_PUBLIC_DIR=$1 # Path to the directory containing the training data

VIDEO_DIR=$DATA_PUBLIC_DIR/videos # Path to the directory containing the videos
KEYFRAME_DIR=$DATA_PUBLIC_DIR/keyframes
FACE_DIR=$DATA_PUBLIC_DIR/faces

mkdir -p $KEYFRAME_DIR
mkdir -p $FACE_DIR

# Extract frames from videos
python preprocessing/generate_raw_labels_from_folder.py $VIDEO_DIR $DATA_PUBLIC_DIR/labels_video.csv
sh preprocessing/extract_keyframes.sh $VIDEO_DIR $KEYFRAME_DIR
python preprocessing/generate_keyframes_labels.py $KEYFRAME_DIR $DATA_PUBLIC_DIR/labels_video.csv $DATA_PUBLIC_DIR/labels_keyframes.csv
PYTHONPATH=preprocessing/yolov3/ python preprocessing/yolov3/image_detect.py --input $KEYFRAME_DIR --output $FACE_DIR \
--model_def preprocessing/weight/yolov3_mask.cfg \
--weights_path preprocessing/weight/yolov3_ckpt_35.pth \
--class_path preprocessing/weight/mask_dataset.names

python preprocessing/generate_bbox_labels.py $FACE_DIR/crop $DATA_PUBLIC_DIR/labels_video.csv $DATA_PUBLIC_DIR/labels_bbox.csv