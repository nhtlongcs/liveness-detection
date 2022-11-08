DATAPATH=$1
TRACK_JSON=$2

REL_PATH=$DATAPATH/meta/relation
mkdir $REL_PATH

AUX_JSON_PATH=$REL_PATH/neighbor_tracks.json
AUX_MAPPING_JSON_PATH=$REL_PATH/neighbor_mapping.json

SUBSTRING=${TRACK_JSON##*/}
REL_JSON_PATH=$REL_PATH/${SUBSTRING%.*}_relation.json

echo Extracting auxiliary tracks ...
python scripts/relation/gen_aux_tracks.py \
    -i $DATAPATH \
    -o $AUX_JSON_PATH && echo DONE || echo Run FAILED, please check

echo Generate neighbor mapping ...
python scripts/relation/gen_neighbor_mapping.py \
    -i $DATAPATH \
    -o $AUX_MAPPING_JSON_PATH && echo DONE || echo Run FAILED, please check

echo Refine neighbor ...
python scripts/relation/refine_neighbor.py \
    --image_dir $DATAPATH/meta/extracted_frames \
    --tracks_json $TRACK_JSON \
    --aux_tracks_json $AUX_JSON_PATH \
    --aux_tracks_mapping_json $AUX_MAPPING_JSON_PATH \
    -o $REL_JSON_PATH && echo DONE || echo Run FAILED, please check