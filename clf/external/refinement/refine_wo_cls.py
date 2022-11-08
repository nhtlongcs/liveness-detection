from typing import Dict, List
import os 
import os.path as osp 
import json
import numpy as np

from .att_score import score_att


def calculate_relation_score_no_class(
    query_data: Dict,
    track_nei_veh: Dict, 
    track_nei_col: Dict,
    relation_dir: str,
    score_map: Dict
):
    for track_id in score_map:
        track_relation_json = osp.join(relation_dir, f'{track_id}.json')
        track_relation_info = json.load(open(track_relation_json, 'r'))

        # Get follow score 
        relations = ['follow', 'followed_by']
        final_score = 0
        for rel in relations:
            if query_data.get(rel) is None:
                continue
            
            if len(track_relation_info[rel]) > 0:
                final_score += 0.5

        score_map[track_id]['relation'] = final_score
    pass


def calculate_relation_score(
    query_data: Dict,
    track_nei_veh: Dict, 
    track_nei_col: Dict,
    relation_dir: str,
    score_map: Dict
):
    for track_id in score_map:
        track_relation_json = osp.join(relation_dir, f'{track_id}.json')
        track_relation_info = json.load(open(track_relation_json, 'r'))

        # Get follow score 
        relations = ['follow', 'followed_by']
        final_score = 0
        for rel in relations:
            if query_data.get(rel) is None:
                continue

            list_query_vehs = query_data[rel]['object_vehicle_label']
            list_query_cols = query_data[rel]['object_color_label']
            list_veh_scores, list_col_scores = [], []

            
            # for query_veh in list_query_vehs:
            #     for obj_id in track_relation_info[rel]:
            #         obj_veh = track_nei_veh[obj_id]
            #         list_veh_scores.append(score_att(query_veh, obj_veh, mode='vehicle'))

            # for query_col in list_query_cols:
            #     for obj_id in track_relation_info[rel]:
            #         obj_col = track_nei_col[obj_id]
            #         list_col_scores.append(score_att(query_col, obj_col, mode='color'))

            list_scores = []
            for query_veh, query_col in zip(list_query_vehs, list_query_cols):
                for obj_id in track_relation_info[rel]:
                    obj_veh = track_nei_veh[obj_id]
                    obj_col = track_nei_col[obj_id]
                    list_scores.append(
                        score_att(query_veh, obj_veh, mode='vehicle') + score_att(query_col, obj_col, mode='color')
                    )

            obj_score = max(list_scores) if len(list_scores) > 0 else 0
            final_score += obj_score

            # veh_score = max(list_veh_scores) if len(list_veh_scores) > 0 else 0
            # col_score = max(list_col_scores) if len(list_col_scores) > 0 else 0
            # rel_score = veh_score + col_score
            # final_score += rel_score

        score_map[track_id]['relation'] = final_score


def calculate_subject_score(
    query_sub_veh: List, 
    query_sub_col: List, 
    track_sub_veh: Dict, 
    track_sub_col: Dict,
    score_map: Dict
):
    """
    Args:
        query_sub_veh (List): one-hot vector of subject vehicle type (get from srl)
        query_sub_col (List): one-hot vector of subject color type (get from srl)
        track_sub_veh (Dict): one-hot vector of subject vehicle type (get from Vehicle Classifier)
        track_sub_col (Dict): one-hot vector of subject color type (get from Vehicle Classifier)
        score_map (Dict): _description_
    """
    query_sub_veh, query_sub_col = np.array(query_sub_veh), np.array(query_sub_col)

    for track_id in score_map:
        sub_veh = np.array(track_sub_veh[track_id])
        sub_col = np.array(track_sub_col[track_id]) 
        
        track_veh_score = score_att(query_sub_veh, sub_veh)
        track_col_score = score_att(query_sub_col, sub_col)

        score_map[track_id]['subject'] = track_veh_score + track_col_score
        # score_map[track_id]['subject'] = track_veh_score 
        # score_map[track_id]['subject'] = track_col_score
    pass

def get_priority_list_by_action(raw_list, track_dir, query_actions):
    list_a, list_b, list_c = [], [], []
    is_turn = ('turn' in query_actions)
    is_stop = ('stop' in query_actions)

    for track_id in raw_list:
        track_path = f'{track_dir}/{track_id}.json'
        track_data = json.load(open(track_path, 'r'))
        
        if is_turn and is_stop:
            if track_data['is_stop'] and track_data['is_turn']:
                list_a.append(track_id)
            elif track_data['is_stop'] or track_data['is_turn']:
                list_b.append(track_id)
            else:
                list_c.append(track_id) 

        elif is_stop and not is_turn:
            if track_data['is_stop']:
                list_a.append(track_id)
            else:
                list_b.append(track_id)
        
        elif is_turn and not is_stop:
            if track_data['is_turn']:
                list_a.append(track_id)
            else:
                list_b.append(track_id)

        else:
            if (not track_data['is_turn']) and (not track_data['is_stop']):
                list_a.append(track_id)
            else:
                list_b.append(track_id)


    return list_a, list_b, list_c


def calculate_action_score(raw_list, track_dir, query_actions, score_map):
    is_turn = ('turn' in query_actions)
    is_stop = ('stop' in query_actions)

    for track_id in raw_list:
        track_path = f'{track_dir}/{track_id}.json'
        track_data = json.load(open(track_path, 'r'))
        
        if is_turn:
            if track_data['is_turn']:
                score_map[track_id]['action'] += 1

        if is_stop:
            if track_data['is_stop']:
                score_map[track_id]['action'] += 0.5


def sort_by_score(score_map, list_track_ids, score_name = 'total_score'):
    list_to_sort = []
    for track_id in list_track_ids:
        if score_name == 'total_score':
            score = 0.0
            for att in score_map[track_id]:
                score += score_map[track_id][att]
        else:
            score = score_map[track_id][score_name]
        
        list_to_sort.append((track_id, score))

    list_to_sort = sorted(list_to_sort, key = lambda x: x[1], reverse=True)
    result = [val[0] for val in list_to_sort]
    return result 

