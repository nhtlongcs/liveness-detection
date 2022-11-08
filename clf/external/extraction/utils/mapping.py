import json 

def json_load(json_path: str):
    data = None
    with open(json_path, 'r') as f:
        data = json.load(f)

    return data

def setup_reverse_map(map_dict: dict):
    order_id_map = {}
    for item in map_dict:
        order_id_map[item['key']] = item['order'] 
    return order_id_map

def get_map_dict(TRAIN_TRACK_MAP_JSON,TEST_TRACK_MAP_JSON,TEST_QUERY_MAP_JSON):
    test_query_map = json_load(TEST_QUERY_MAP_JSON)
    test_query_map = setup_reverse_map(test_query_map)
    train_track_map = json_load(TRAIN_TRACK_MAP_JSON)
    train_track_map = setup_reverse_map(train_track_map)
    if TEST_TRACK_MAP_JSON is None:
        return train_track_map, None, test_query_map
    test_track_map = setup_reverse_map(test_track_map)
    test_track_map = json_load(TEST_TRACK_MAP_JSON)
    return train_track_map, test_track_map, test_query_map