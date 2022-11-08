RETRIEVAL_RESULT=$1
SRL_CSV=$2
TRACK_DIR=$3

TEST_RELATION=$4
TEST_NEIGHBOR=$5
CLS_DIR=$6
SAVE_DIR=$7

TEST_ACTION=$CLS_DIR/test_tracks/action_prediction.json
# # DEBUG
# RETRIEVAL_RESULT='data/result/refinement/sub_34_fix.json'
# SRL_CSV='data/result/srl_direct/aic22_test_notOther_10Apr.csv'
# TRACK_DIR='data/result/test_relation_action_f1'

# TEST_RELATION='data/result/test_relation.json'
# TEST_NEIGHBOR='data/result/test_neighbors.json'
# TEST_ACTION='data/result/test_action_tuned_10Apr_1225/test_action_f1.json'

# SAVE_DIR='./tmp'

echo parse relation result and save to $TRACK_DIR
python scripts/refinement/parse_relation.py \
    $TEST_RELATION $TEST_NEIGHBOR $TEST_ACTION $TRACK_DIR

echo run refinement, save result to $SAVE_DIR
python scripts/refinement/main.py \
    $RETRIEVAL_RESULT $SRL_CSV $TRACK_DIR $SAVE_DIR $CLS_DIR
