import json
from external.detect_stop_turn.stop_detector import StopDetector
from external.detect_stop_turn.turn_detector import TurnDetector, TurnState
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Detect stop-turn action of the motion of test dataset')
parser.add_argument('-g', '--test_track_json', type=str,
                    help="Path to tesk track json ")
parser.add_argument('-o', '--output_json', type=str,
                    help="Path to save result")


TURN_CONFIG_TUNE = {
    'precision': {
        'eps' : 0.045, 
        'skip_frame' : 6, 
        'is_normalize': 0 
    },
    'recall': {
        'eps' : 0.01, 
        'skip_frame' : 2, 
        'is_normalize': 0 
    },
    'f1': {
        'eps' : 0.025, 
        'skip_frame' : 6, 
        'is_normalize': 0 
    },
}

STOP_CONFIG_TUNE = {
    'precision': {
        'k' : 8, 
        'delta' : 14, 
        'alpha': 0.05
    },
    'recall': {
        'k' : 1, 
        'delta' : 5, 
        'alpha': 0.2 
    },
    'f1': {
        'k' : 4, 
        'delta' : 12, 
        'alpha': 0.1 
    },
}


TURN_CONFIG = TURN_CONFIG_TUNE['f1']
STOP_CONFIG = STOP_CONFIG_TUNE['f1']

def run(args):
    stop_det = StopDetector(
        **STOP_CONFIG
    )

    turn_det = TurnDetector(
        **TURN_CONFIG
    )

    with open(args.test_track_json, 'r') as f:
        data = json.load(f)

    # Turn results
    norm_area_dict = {}
    
    vertical_views = {}
    straight_percent_dict = {}
    speed_dict = {}

    result_dict = {
        'stop': [],
        'turn_left': [],
        'turn_right': [],
        'straight_views': []
    }

    for track_id, track_value in tqdm(data.items()):
        boxes = track_value['boxes']

        # Detect stop
        is_stop = stop_det.process(boxes)
        if is_stop:
            result_dict['stop'].append(track_id)
            
        # Detect turn
        is_turn, turn_state, norm_area, list_points = turn_det.process(boxes)
        is_vertical_view, straight_percent, list_angles, speed_record = turn_det.find_vertical_views(list_points)

        vertical_views[track_id] = list_angles
        straight_percent_dict[track_id] = straight_percent
        speed_dict[track_id] = speed_record
        norm_area_dict[track_id] = norm_area
        
        if is_turn:
            if turn_state == TurnState.LEFT:
                result_dict['turn_left'].append(track_id)
            else:
                result_dict['turn_right'].append(track_id)

            if is_vertical_view:
                result_dict['straight_views'].append(track_id)
    
    print('Number of track which turns either left: ', len(result_dict['turn_left']))
    print('Number of track which turns either right: ', len(result_dict['turn_right']))
    print('Number of track which stops: ', len(result_dict['stop']))
    print('Number of straight-view track: ', len(result_dict['straight_views']))

    with open(args.output_json, 'w') as f:
        json.dump(result_dict, f, indent=4)
    
if __name__ == '__main__':
    args = parser.parse_args()
    run(args)