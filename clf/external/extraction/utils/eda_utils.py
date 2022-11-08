import os, json
import os.path as osp 
import pandas as pd 
import cv2 
from tqdm import tqdm

def stat_annot_vehicle(srl_dict):
    res = {'query_id': [], 'vehicles': [], 'n_vehicles': [], 'details': []}
    for qid in srl_dict.keys():
        stat_vehs = {}
        for cid in srl_dict[qid].keys():
            veh = srl_dict[qid][cid]['main_subject']
            if stat_vehs.get(veh) is None:
                stat_vehs[veh] = 1
            else:
                stat_vehs[veh] += 1
        
        res['query_id'].append(qid)
        res['vehicles'].append(list(stat_vehs.keys()))
        res['n_vehicles'].append(len(stat_vehs))
        res['details'].append(stat_vehs)

    return pd.DataFrame.from_dict(res), res
    
def refine_vehicle_with_level(veh_json):
    
    pass

def convert_vehicle(srl_dict, convert_map):
    """Convert vehicle
    Args:
        convert_map ([type]): [description]
        srl_dict ([type]): [description]
    """
    count = 0
    new_srl_dict = srl_dict.copy()
    for qid in new_srl_dict.keys():
        for cid in new_srl_dict[qid].keys():
            org_veh = new_srl_dict[qid][cid]['main_subject']
            if convert_map.get(org_veh) is not None:
                count += 1
                new_srl_dict[qid][cid]['main_subject'] = convert_map[org_veh]
    
    print(f'Convert {count} captions')
    return new_srl_dict

def get_subject_vehicle(srl_dict):
    """
    srl_dict: srl result
    """
    list_vehicles = []
    for qid in srl_dict.keys():
        for capid in srl_dict[qid].keys():
            list_vehicles.append(srl_dict[qid][capid]['main_subject'])
    
    return list(set(list_vehicles))

def vehicle_stat(srl_dict):
    result = {'vehicle': []}
    for qid in srl_dict.keys():
        for capid in srl_dict[qid].keys():
            result['vehicle'].append(srl_dict[qid][capid]['main_subject'])
    
    res_df = pd.DataFrame.from_dict(result)
    return res_df

def color_stat(srl_dict):
    result = {'color': [], 'adv': [], 'combine': []}
    for qid in srl_dict.keys():
        for capid in srl_dict[qid].keys():
            for srl_ele in srl_dict[qid][capid]['srl']:
                # print(srl_ele)
                if srl_ele['is_main_subject'] == True and len(srl_ele.get('subject_color')) > 0:
                    adv = srl_ele.get('subject_color')[0].get('adv')
                    color = srl_ele.get('subject_color')[0].get('color')
                    combine=None
                    if adv is None:
                        combine = f'{color}'
                    else:
                        combine = f'{adv}_{color}'
                    result['color'].append(color)
                    result['combine'].append(combine)
                    result['adv'].append(adv)
                
    res_df = pd.DataFrame.from_dict(result)
    return res_df
    

def create_query_vehicle_df(srl_dict):
    df_dict = {'query_id': [], 'vehicles': []}
    for qid in srl_dict.keys():
        list_vehs = []
        for capid in srl_dict[qid].keys():
            list_vehs.append(srl_dict[qid][capid]['main_subject'])
        list_vehs = list_vehs #list(set(list_vehs))
        
        df_dict['query_id'].append(qid)
        df_dict['vehicles'].append(list_vehs)
    
    return pd.DataFrame.from_dict(df_dict)

def count_components(list_appear_vehs):
    res = {}
    for veh in list_appear_vehs:
        if res.get(veh) is None:
            res[veh] = 1
        else:
            res[veh] += 1

    return res

def show_value_count_horizontal(data_df, col='combine'):
    a = data_df[col].value_counts()
    a = pd.DataFrame.from_dict({col: a.index, 'counts': a.values})
    print(a.T)
    pass

def show_unique_values(data_df, col):
    unique_vals = data_df[col].unique().tolist()
    print(f'{col} unique values: {unique_vals}')
    pass

def get_important_vehicles(count_df):
    res = []
    list_vals = count_df[count_df['n_vehicles'] == 1]['vehicles'].values.tolist()
    for val in list_vals:
        if isinstance(val, str):
            val = eval(val)
        res.extend(val)

    single_veh_df = pd.DataFrame.from_dict({'vehicles': res})
    return single_veh_df