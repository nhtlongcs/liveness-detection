"""
Script for generating all neighbor tracks based on annotation from AIC22 dataset
Read in the auxiliary tracks and the main tracks and decide which is the neighbor to which
"""

import os.path as osp
import json
import pandas as pd
from tqdm import tqdm
from external.relation.frame_utils import get_frame_ids_by_names, get_camera_id_by_name
from scripts.relation.constants import Constants
import argparse

parser = argparse.ArgumentParser('Generate neighbor mapping')
parser.add_argument("-i", "--data_path", type=str, help='Path to root')
parser.add_argument("-o", "--output_json", type=str, help='Output file')
args = parser.parse_args()

CONSTANT = Constants(args.data_path)
NUM_FRAMES_THRESHOLD = 5 # filter out tracks which appear less than threshold
OUTPATH = args.output_json

FOLDER_NAME = ['train', 'validation'] #because AIC22 structure folder this way
CAM_IDS = [
    CONSTANT.TEST_CAM_IDS, 
    CONSTANT.TRAIN_CAM_IDS, 
] 

TRACKS_JSON = [
    CONSTANT.TEST_TRACKS_JSON, 
    CONSTANT.TRAIN_TRACKS_JSON, 
]

ANNO = "{AIC22_ORI_ROOT}/{FOLDER_NAME}/{CAMERA}/gt/gt.txt"

def generate_neighbor_tracks_mapping(camera_id, folder_name, track_json):

    csv_path = ANNO.format(CAMERA=camera_id, FOLDER_NAME=folder_name, AIC22_ORI_ROOT=CONSTANT.AIC22_ORI_ROOT)
    if not osp.isfile(csv_path):
        return {}
    df = pd.read_csv(csv_path)

    df.columns = [
        'frame_id', 
        'track_id', 
        'x', 'y', 'w', 'h', 
        'conf', 'unk1', 'unk2', 'unk3'
    ]

    with open(track_json, 'r') as f:
        data = json.load(f)

    main_track_ids = list(data.keys())

    neighbor_mapping = {}
    for main_track_id in main_track_ids:

        frame_names = data[main_track_id]['frames']
        main_boxes = data[main_track_id]['boxes']

        # The camera id of the track
        current_camera_id = get_camera_id_by_name(frame_names[0])
        if current_camera_id != camera_id:
            continue

        # All the frames that main track appears
        frame_ids = get_frame_ids_by_names(frame_names)

        neighbor_mapping[main_track_id] = []
        # tracks that appear at same  frame with main track
        aux_appearances = {}
        for (frame_id, main_box) in zip(frame_ids, main_boxes):
            aux_df = df[df.frame_id == frame_id]
            for _, row in aux_df.iterrows():

                track_id, x, y, w, h = row[1:6]
                unique_track_id = f"{camera_id.replace('/', '_')}_{track_id}"

                main_box[0] += main_box[2]
                main_box[1] += main_box[3]
                other_box = [x,y,x+w,y+h]

                # Store the neighbor track candidates
                if unique_track_id not in aux_appearances.keys():
                    aux_appearances[unique_track_id] = []
                aux_appearances[unique_track_id].append(other_box)

        # filter out tracks which appear less than threshold
        aux_tracks = {k:v for k,v in aux_appearances.items() if len(v) >= NUM_FRAMES_THRESHOLD}
        
        for aux_track_id in aux_tracks.keys():
            neighbor_mapping[main_track_id].append(aux_track_id)

    return neighbor_mapping
        
def run():

    final_dict = {}

    for cam_split, folder_name, track_json in zip(CAM_IDS, FOLDER_NAME, TRACKS_JSON):
        for camera_id in tqdm(cam_split):
            camera_neighbor_dict = generate_neighbor_tracks_mapping(camera_id, folder_name, track_json)
            final_dict.update(camera_neighbor_dict)

    with open(OUTPATH, 'w') as f:
        json.dump(final_dict, f, indent=4)

if __name__ == '__main__':
    run()