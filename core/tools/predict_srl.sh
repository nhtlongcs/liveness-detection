# ./tools/predict_srl.sh  configs/srl/col_infer.yml \
#                         artifacts/color/model.ckpt \
#                         configs/srl/veh_infer.yml \
#                         artifacts/vehicle/model.ckpt \
#                         data/meta/relation/test_neighbors.json \
#                         data/meta/extracted_frames \
#                         out_test_neighbors

COLOR_CFG_PATH=$1
COLOR_CKPT_PATH=$2

VEHCL_CFG_PATH=$3
VEHCL_CKPT_PATH=$4

JSON_PATH=$5
IMAGE_DIR=$6
SAVE_DIR=$7
BATCH_SIZE=4
mkdir -p $SAVE_DIR
echo saving to folder $SAVE_DIR
COLOR_SAVEPATH=$SAVE_DIR/color_prediction.json
VEHCL_SAVEPATH=$SAVE_DIR/vehicle_prediction.json

echo saving to $COLOR_SAVEPATH
echo saving to $VEHCL_SAVEPATH

python scripts/srl/predict.py   -c $COLOR_CFG_PATH \
                                -o  global.pretrained=$COLOR_CKPT_PATH \
                                    global.save_path=$COLOR_SAVEPATH \
                                    global.batch_size=$BATCH_SIZE \
                                    data.json_path=$JSON_PATH \
                                    data.image_dir=$IMAGE_DIR

python scripts/srl/predict.py   -c $VEHCL_CFG_PATH \
                                -o  global.pretrained=$VEHCL_CKPT_PATH \
                                    global.save_path=$VEHCL_SAVEPATH \
                                    global.batch_size=$BATCH_SIZE \
                                    data.json_path=$JSON_PATH \
                                    data.image_dir=$IMAGE_DIR 

