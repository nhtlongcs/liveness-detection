import os 
import os.path as osp 
import json 
import pandas as pd
from tqdm import tqdm
import sys

from external.refinement.srl_reader import SrlReader
from external.refinement.refine_wo_cls import (
    get_priority_list_by_action, 
    calculate_subject_score, 
    calculate_relation_score, 
    calculate_action_score,
    sort_by_score, calculate_relation_score_no_class
)

# Input
# TEST_CSV = 'data/result/srl_direct/aic22_test_notOther_10Apr.csv' #'data/result/srl/postproc-2/srl_test_postproc.csv'
# RAW_RESULT = 'data/result/refinement/sub_34_fix.json' # raw retrieval result (submission file from model)
# TRACK_DIR = 'data/result/test_relation_action_f1'

RAW_RESULT = sys.argv[1] # raw retrieval result (submission file from model)
TEST_CSV = sys.argv[2]
TRACK_DIR = sys.argv[3]
SAVE_DIR = sys.argv[4]
CLS_DIR = sys.argv[5]

# Meatadata: Classification results 
TEST_SUB_VEH_CLS = osp.join(CLS_DIR,'test_tracks/vehicle_prediction.json' )
TEST_SUB_COL_CLS = osp.join(CLS_DIR,'test_tracks/color_prediction.json')
TEST_NEI_VEH_CLS = osp.join(CLS_DIR,'neighbors/vehicle_prediction.json')
TEST_NEI_COL_CLS = osp.join(CLS_DIR,'neighbors/color_prediction.json')

def save_json(data, fpath):
    with open(fpath, 'w') as f:
        json.dump(data, f, indent=2)
    

def main():
    srl_data = SrlReader(TEST_CSV)
    ret_result = json.load(open(RAW_RESULT, 'r'))
    sub_veh_cls = json.load(open(TEST_SUB_VEH_CLS, 'r'))
    sub_col_cls = json.load(open(TEST_SUB_COL_CLS, 'r'))
    nei_veh_cls = json.load(open(TEST_NEI_VEH_CLS, 'r'))
    nei_col_cls = json.load(open(TEST_NEI_COL_CLS, 'r'))

    final_result, action_result, vehcol_result = {}, {}, {}
    count_svo = 0
    for query_id in tqdm(srl_data.query_data):
        query_data = srl_data.query_data[query_id]
        query_sub_col = query_data['subject_color_label']
        query_sub_veh = query_data['subject_vehicle_label']
        
        raw_result = ret_result[query_id]

        score_map = {}
        for track_id in raw_result:
            score_map[track_id] = {'subject': 0.0, 'relation': 0.0, 'action': 0.0}


        # 1. Prioritize by action score
        query_actions = query_data['actions']
        calculate_action_score(raw_result, TRACK_DIR, query_actions, score_map)
        action_result[query_id] = sort_by_score(score_map, raw_result, 'action')

        # 2. Calculate att scores (vehcol + relation) in each list 

        ## 2.1 calculate subject veh + col score
        calculate_subject_score(query_sub_veh, query_sub_col, sub_veh_cls, sub_col_cls, score_map)
        vehcol_result[query_id] = sort_by_score(score_map, raw_result, 'subject')

        ## 2.2 calculate relation score
        if query_data['is_svo'] == True:
            count_svo += 1
            calculate_relation_score(query_data, nei_veh_cls, nei_col_cls, TRACK_DIR, score_map)
            # calculate_relation_score_no_class(query_data, nei_veh_cls, nei_col_cls, TRACK_DIR, score_map)

        # 4. Sort on total_score to obtain final result
        final_result[query_id] = sort_by_score(score_map, raw_result, 'total_score')
    
    print(f'n svo: {count_svo}')
    ### comment these lines when finish debugging
    fname = RAW_RESULT.split('/')[-1]
    
    
    save_path = osp.join(SAVE_DIR, fname.replace('.json', '_full_Class.json'))
    print('saved full class submission to:', save_path)
    save_json(final_result, save_path)

    save_path = osp.join(SAVE_DIR, fname.replace('.json', '_action.json'))
    print('saved action only submission to:', save_path)
    save_json(action_result, save_path)

    save_path = osp.join(SAVE_DIR, fname.replace('.json', '_vehcol.json'))
    print('saved vehcol only submission to:', save_path)
    save_json(vehcol_result, save_path)


if __name__ == '__main__':
    main()
