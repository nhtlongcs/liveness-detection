"""
Script for generating all neighbor tracks based on annotation from AIC22 dataset
Read in tracking results and convert into same format as AIC22 tracks
"""

import os.path as osp
import json
import pandas as pd
from tqdm import tqdm
from scripts.relation.constants import Constants

import argparse
parser = argparse.ArgumentParser('Generate auxiliary tracks')

parser.add_argument("-i", "--data_path", type=str, help='Path to root')
parser.add_argument("-o", "--output_json", type=str, help='Output file')
args = parser.parse_args()


CONSTANT = Constants(args.data_path)

OUTPATH = args.output_json
CAM_IDS = [CONSTANT.TEST_CAM_IDS, CONSTANT.TRAIN_CAM_IDS] 
FOLDER_NAME = ['train', 'validation'] #because AIC22 structure folder this way
ANNO = "{AIC22_ORI_ROOT}/{FOLDER_NAME}/{CAMERA}/gt/gt.txt"

def generate_unique_neighbor_tracks(camera_id, folder_name):
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

    neighbor_dict = {}
    track_ids = list(df.track_id)
    for track_id in track_ids:
        track_df = df[df.track_id == track_id]

        unique_track_id = f"{camera_id.replace('/', '_')}_{track_id}"

        neighbor_dict[unique_track_id] = {
            'frames': [],
            'boxes': []
        }

        for _, row in track_df.iterrows():
            frame_id, _, x, y, w, h = row[:6]
            neighbor_dict[unique_track_id]['frames'].append(
                f"./{folder_name}/{camera_id}/img1/{str(frame_id).zfill(6)}.jpg"
            )

            neighbor_dict[unique_track_id]['boxes'].append([x, y, w, h])

    print(f"Number of auxiliary tracks: {len(neighbor_dict.keys())}")
    return neighbor_dict
        
def run():

    final_dict = {}

    for cam_split, folder_name in zip(CAM_IDS, FOLDER_NAME):
        for camera_id in tqdm(cam_split):
            camera_neighbor_dict = generate_unique_neighbor_tracks(camera_id, folder_name)
            final_dict.update(camera_neighbor_dict)

    with open(OUTPATH, 'w') as f:
        json.dump(final_dict, f, indent=4)

if __name__ == '__main__':
    run()