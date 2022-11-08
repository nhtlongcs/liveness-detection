DATAPATH=$1
# Generate augment data for training (Optional)
echo Augmenting data ...
python scripts/data/nlpaug_uts_v2.py $DATAPATH/train_tracks.json && echo DONE || echo Run FAILED, please check
python scripts/data/nlpaug_uts_v2.py $DATAPATH/test_queries.json && echo DONE || echo Run FAILED, please check

# Extract train and test's queries into separated parts following the English PropBank Semantic Role Labeling rules.
# SRL PART
# python scripts/srl/extraction.py <input_data_path> <output_metadata_srl_path>
echo Extracting SRL data ...
python scripts/srl/extraction.py $DATAPATH $DATAPATH && echo DONE || echo Run FAILED, please check

SRL_PATH=$DATAPATH/srl
mkdir $SRL_PATH
mkdir $SRL_PATH/action
mkdir $SRL_PATH/color
mkdir $SRL_PATH/veh

echo Extracting SRL data [action] ...
python scripts/srl/action_prep.py $SRL_PATH $SRL_PATH/action && echo DONE || echo Run FAILED, please check
echo Extracting SRL data [color] ...
python scripts/srl/color_prep.py $DATAPATH $SRL_PATH $DATAPATH/extracted_frames $SRL_PATH/color && echo DONE || echo Run FAILED, please check
echo Extracting SRL data [vehicle] ...
python scripts/srl/veh_prep.py $DATAPATH $SRL_PATH $DATAPATH/extracted_frames $SRL_PATH/veh && echo DONE || echo Run FAILED, please check

# Please check this part
mkdir $SRL_PATH/postproc
echo Extracting SRL data [postproc] ...
python scripts/srl/extract_postproc.py $SRL_PATH  $SRL_PATH/postproc && echo DONE || echo Run FAILED, please check
