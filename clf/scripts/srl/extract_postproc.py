import os, json, cv2, sys
import os.path as osp
import pandas as pd
from tqdm import tqdm

from external.extraction.textual.query import Query
from external.extraction.textual.gather_utils import (
    get_label_info, get_label_vector, setup_info, 
    get_label_vector_with_split, get_rep_class
)
from external.extraction.paths import (
    COLOR_GROUP_JSON, VEHICLE_GROUP_JSON, ACTION_GROUP_JSON,
    COLOR_GROUP_REP_JSON, VEHICLE_GROUP_REP_JSON, ACTION_GROUP_REP_JSON,
)

SRL_DIR = sys.argv[1]
SAVE_DIR = sys.argv[2]

TRAIN_SRL_JSON = osp.join(SRL_DIR, 'srl_train_tracks.json')
TEST_SRL_JSON = osp.join(SRL_DIR, 'srl_test_queries.json')
srl_json = {"train": TRAIN_SRL_JSON, "test": TEST_SRL_JSON}

veh_info, col_info, act_info = {}, {}, {}
setup_info(veh_info, VEHICLE_GROUP_JSON)
setup_info(col_info, COLOR_GROUP_JSON)
setup_info(act_info, ACTION_GROUP_JSON)


def parse(mode: str):
    mode_json = srl_json[mode]
    srl_data = json.load(open(mode_json, 'r')) 
    list_ids = list(srl_data.keys())
    is_test = (mode == 'test')

    list_res = []
    query_no_sub_veh = []
    query_no_sub_col = []
    
    for raw_key in tqdm(list_ids):
        query_dict = {}
        query = Query(srl_data[raw_key], raw_key)
        srl_data[raw_key] = query.get_query_content_update()
                
        is_svo = False 
        if ('follow' in query.relation_actions) or ('followed' in query.relation_actions):
            is_svo = True

        
        is_sub_veh, is_sub_col = True, True
        subject_vehicle_label = get_label_vector(query.subject_vehicle, veh_info['num_classes'], veh_info['label_map'], is_test)
        subject_color_label = get_label_vector(query.subject_color, col_info['num_classes'], col_info['label_map'], is_test)
        
        if subject_vehicle_label is None:
            query_no_sub_veh.append(raw_key)
            is_sub_veh = False
        if subject_color_label is None:
            query_no_sub_col.append(raw_key)
            is_sub_col = False
        

        query_dict['query_id'] = raw_key
        query_dict['captions'] = query.get_list_captions()
        query_dict['cleaned_captions'] = query.get_list_cleaned_captions()

        query_dict['subject_vehicle'] = query.subject_vehicle
        query_dict['subject_color'] = list(set(get_rep_class(COLOR_GROUP_REP_JSON, query.subject_color)))
        
        query_dict['is_sub_veh'] = is_sub_veh
        query_dict['is_sub_col'] = is_sub_col

        query_dict['action'] = list(set(get_rep_class(ACTION_GROUP_REP_JSON, query.actions)))
        query_dict['relation_action'] = query.relation_actions
        query_dict['is_svo'] = is_svo

        query_dict['subject_vehicle_label'] = subject_vehicle_label
        query_dict['subject_color_label'] = subject_color_label
        query_dict['action_label'] = get_label_vector(query.actions, act_info['num_classes'], act_info['label_map'], is_test)

        query_dict['object_vehicle'] = query.object_vehicle
        query_dict['object_color'] = query.object_color
        query_dict['object_vehicle_label'] = get_label_vector_with_split(query.object_vehicle, veh_info['num_classes'], veh_info['label_map'])
        query_dict['object_color_label'] = get_label_vector_with_split(query.object_color, col_info['num_classes'], col_info['label_map'])
        
        query_dict['is_follow'] = query.is_follow
        
        list_res.append(query_dict)        

    df_res = pd.DataFrame(list_res)
    return df_res


def main():
    for mode in ["train", "test"]:
        print("=" * 10 + f" Parse result in {mode} " + "=" * 10)
        mode_save_dir = osp.join(SAVE_DIR, f"{mode}_srl")
        os.makedirs(mode_save_dir, exist_ok=True)
        df_mode = parse(mode)
        df_mode.to_csv(osp.join(SAVE_DIR, f"{mode}_srl.csv"), index=False)
        
if __name__ == "__main__":
    main()

