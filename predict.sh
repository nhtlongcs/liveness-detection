ROOT=./
WORKSPACE=./
DATA_DIR=$ROOT/data
RESULT_DIR=$ROOT/results


CFG_PATH=$WORKSPACE/configs/sup/frameb7.yml
CKPT_PATH=$WORKSPACE/checkpoints/FrameClfB7-StrongAug0.9724.ckpt
PUBLIC_CFG_PATH=$WORKSPACE/configs/public/frameb7.yml
PUBLIC_OUTNAME=$WORKSPACE/public_eff.csv
PUBLIC_SUB_OUTNAME=$RESULT_DIR/submission.csv

python scripts/predict.py   -c  $PUBLIC_CFG_PATH \
                            -o  global.pretrained=$CKPT_PATH \
                                global.save_path=$PUBLIC_OUTNAME

python competition/strategy/confident.py --input=$PUBLIC_OUTNAME --output=$PUBLIC_SUB_OUTNAME