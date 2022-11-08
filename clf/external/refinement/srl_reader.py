import pandas as pd
import numpy as np

VALID_RELATION = ['follow', 'followed_by']
BASIC_VEHICLE = [] #['vehicle', 'car']
BASIC_COLOR = [] #[None]

def get_array_from_str(data: str, is_np = True):
    list_data = eval(data)
    # list_data = ast.literal_eval(data)
    if is_np:
        return np.array(list_data)
    else:
        return list_data
    
def get_text_info(df_srl: pd.DataFrame, key_query: str):
    field_infos = [
        'subject_vehicle_label', 'subject_color_label',
        'relation_action', 'action', 
        'object_vehicle', 'object_color',
        'object_vehicle_label', 'object_color_label'
    ]
    text_infos = {}
    for field in field_infos:
        info_str = df_srl[df_srl['query_id'] == key_query][field].values[0]
        text_infos[field] = get_array_from_str(info_str, is_np=False)
    text_infos['is_svo'] = df_srl[df_srl['query_id'] == key_query]['is_svo'].values[0]

    return text_infos

def refine_relation_action_infos(list_actions: list, list_object_vehicle: list, list_object_color: list):
    action_map = {}
    for i, action in enumerate(list_actions):
        if action not in VALID_RELATION:
            continue
        if action_map.get(action) is None:
            action_map[action] = {
                'col_ids': [], 'veh_ids': [], 'vehicle': [], 'color': []
            }

        if list_object_vehicle[i] not in action_map[action]['vehicle']:
            action_map[action]['veh_ids'].append(i)
            action_map[action]['vehicle'].append(list_object_vehicle[i])
        
        # if list_object_color[i] not in action_map[action]['color']:
            action_map[action]['col_ids'].append(i)
            action_map[action]['color'].append(list_object_color[i])

    for action in VALID_RELATION:
        if action_map.get(action) is not None:
            list_important_vehicles = [(i, veh) for (i, veh) in enumerate(action_map[action]['vehicle']) \
                                        if veh not in BASIC_VEHICLE]
            if len(list_important_vehicles) > 0:
                action_map[action]['veh_ids'] = []
                action_map[action]['vehicle'] = []
                for (i, veh) in list_important_vehicles:
                    action_map[action]['veh_ids'].append(i)
                    action_map[action]['vehicle'].append(veh)
        
            list_important_colors = [(i, col) for (i, col) in enumerate(action_map[action]['color']) \
                                        if col not in BASIC_COLOR]
            if len(list_important_colors) > 0:
                action_map[action]['col_ids'] = []
                action_map[action]['color'] = []
                for (i, col) in list_important_colors:
                    action_map[action]['col_ids'].append(i)
                    action_map[action]['color'].append(col)

    return action_map


class SrlReader(object):
    def __init__(self, srl_csv: str) -> None:
        self.query_data = {}
        self._setup(srl_csv)

    def _setup(self, srl_csv: str):
        df_srl = pd.read_csv(srl_csv)
        list_query_ids = df_srl['query_id'].values.tolist()

        for query_id in list_query_ids:
            self.query_data[query_id] = {}

            query_infos = get_text_info(df_srl, query_id)
            self.query_data[query_id]['actions'] = query_infos['action']

            # Set subject infos
            self.query_data[query_id]['subject_vehicle_label'] = query_infos['subject_vehicle_label']
            self.query_data[query_id]['subject_color_label'] = query_infos['subject_color_label']
            
            # Set relation info
            self.query_data[query_id]['is_svo'] = query_infos['is_svo']
            action_map = refine_relation_action_infos(
                query_infos['relation_action'], query_infos['object_vehicle'], query_infos['object_color']
            )
            for relation in VALID_RELATION:
                if action_map.get(relation) is None:
                    self.query_data[query_id][relation] = None
                    continue 
                
                choose_vehicle_ids = action_map[relation]['veh_ids']
                choose_color_ids = action_map[relation]['col_ids']
                self.query_data[query_id][relation] = {
                    'object_vehicle': [query_infos['object_vehicle'][i] for i in choose_vehicle_ids],
                    'object_color': [query_infos['object_color'][i] for i in choose_color_ids],
                    'object_vehicle_label': [query_infos['object_vehicle_label'][i] for i in choose_vehicle_ids],
                    'object_color_label': [query_infos['object_color_label'][i] for i in choose_color_ids],
                }
            pass
        pass

