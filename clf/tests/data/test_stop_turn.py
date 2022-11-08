import pytest
import json
from external.detect_stop_turn.stop_detector import StopDetector
from external.detect_stop_turn.turn_detector import TurnDetector, TurnState

@pytest.mark.parametrize("test_track_json", ["data_sample/meta/test_tracks.json"])
def test_stop_turn(tmp_path, test_track_json):
    stop_det = StopDetector(
        k = 5,
        delta = 10, # as paper
        alpha=0.15 # increase means more retrieved
    )

    turn_det = TurnDetector(
        eps = 0.035,
        skip_frame = 5,
        is_normalize=0 
    )

    with open(test_track_json, 'r') as f:
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

    for track_id, track_value in data.items():
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