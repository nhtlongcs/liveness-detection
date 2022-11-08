DATAPATH=$1
CUDA_VISIBLE_DEVICES=0  python src/predictor.py \
                        -c configs/inference.yml \
                        -o data.text.json_path=$DATAPATH/test_queries.json \
                        data.track.json_path=$DATAPATH/test_tracks.json \
                        data.track.image_dir=$DATAPATH/extracted_frames/ \
                        data.track.motion_path=$DATAPATH/motion_map
