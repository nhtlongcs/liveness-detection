import os, cv2, json
import os.path as osp 
import pandas as pd 

def color_stat(srl_dict):
    result = { 'color': [], 'adv': [], 'combine': []}
    for qid in srl_dict.keys():
        for capid in srl_dict[qid].keys():
            for srl_ele in srl_dict[qid][capid]['srl']:
                # print(srl_ele)
                if srl_ele['is_main_subject'] == True and len(srl_ele.get('subject_color')) > 0:
                    adv = srl_ele.get('subject_color')[0].get('adv')
                    color = srl_ele.get('subject_color')[0].get('color')
                    combine = None
                    if adv is None:
                        combine = f'{color}'
                    else:
                        combine = f'{adv}_{color}'
                    result['color'].append(color)
                    result['combine'].append(combine)
                    result['adv'].append(adv)
                
    res_df = pd.DataFrame.from_dict(result)
    return res_df