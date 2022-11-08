"""
Script for visualizing tracking annotation from AIC22 dataset
"""

import json
import argparse
import os
import os.path as osp
import cv2
from tqdm import tqdm
import pandas as pd
from external.relation.drawing import visualize_one_frame
from external.relation.bb_utils import xywh_to_xyxy_lst
from external.relation.frame_utils import get_frame_ids_by_names
from tools.visualization.constants import Constants

parser = argparse.ArgumentParser(description='Streamlit visualization')
parser.add_argument('-i', '--root_dir', type=str,
                    help="Path to root dir")
parser.add_argument('-o', '--out_dir', type=str,
                    help="Path to output folder to save videos")
parser.add_argument('-t', '--track_json', type=str,
                    help="Path to json files contains main tracks information")
parser.add_argument('-r', '--relation_json', type=str,
                    help="Path to json files contains relation information")
parser.add_argument('-a', '--aux_json', type=str,
                    help="Path to json files contains auxiliary tracks information")
args = parser.parse_args()

CONSTANT = Constants(args.root_dir)
OUTDIR = args.out_dir
TRACKS_JSON = args.track_json
RELATION_TRACKS_JSON = args.relation_json
AUX_TRACKS_JSON = args.aux_json

os.makedirs(OUTDIR, exist_ok=True)

def visualize_neighbors():

    with open(TRACKS_JSON, 'r') as f:
        main_data = json.load(f)

    with open(RELATION_TRACKS_JSON, 'r') as f:
        neighbor_mapping = json.load(f)

    with open(AUX_TRACKS_JSON, 'r') as f:
        aux_data = json.load(f)


    main_track_ids = list(neighbor_mapping.keys())

    for main_track_id in tqdm(main_track_ids):

        # Main track info
        main_boxes = xywh_to_xyxy_lst(main_data[main_track_id]['boxes'])
        main_frame_names = main_data[main_track_id]['frames']

        # Init video writer
        tmp_path = osp.join(CONSTANT.EXTRACTED_FRAMES_DIR, main_frame_names[0][2:])
        img = cv2.imread(tmp_path)
        height, width = img.shape[:-1]

        if len(main_frame_names) > 10:
            fps = 10
        else:
            fps = max(len(main_frame_names)//2, 1)

        writer = cv2.VideoWriter(
            osp.join(OUTDIR, main_track_id+'.mp4'),   
            cv2.VideoWriter_fourcc(*'mp4v'), 
            fps, 
            (width, height))

        main_frame_ids = get_frame_ids_by_names(main_frame_names)
        main_frame_ids.sort()

        # Neighbor infos        
        neighbors = neighbor_mapping[main_track_id]
        followed_byed_ids = neighbors['followed_by']
        follow_ids = neighbors['follow']

        # FOR EASIEST WAY, WE USE DATAFRAME TO STORE ALL TRACKS, THEN VISUALIZE THIS DATAFRAME #
        
        ## Create list of dicts to generate dataframe later
        track_lists = []

        # Visualize main track first
        for main_box, main_frame_id in zip(main_boxes, main_frame_ids):
            track_lists.append({
                'frame_id': main_frame_id,
                'track_id': -1, 
                'x1': main_box[0], 'y1': main_box[1], 
                'x2':  main_box[2], 'y2':  main_box[3],
                'color': [0, 255, 0]
            })


        if len(followed_byed_ids) > 0:
            # Visualize followed by
            followed_byed_neighbors = [(id, aux_data[id]) for id in followed_byed_ids]

            for neighbor_track_id, neighbor in followed_byed_neighbors:
                neighbor_boxes = xywh_to_xyxy_lst(neighbor['boxes'])
                neighbor_frames = neighbor['frames']
                neighbor_frames_ids = get_frame_ids_by_names(neighbor_frames)
                neighbor_frames_ids.sort()
                intersected = [
                    (box, id) for id, box in zip(neighbor_frames_ids, neighbor_boxes)
                    if id in main_frame_ids
                ]

                for neighbor_box, neighbor_frame_id in intersected:
                    track_lists.append({
                        'frame_id': neighbor_frame_id,
                        'track_id': neighbor_track_id + '- followed by', 
                        'x1': neighbor_box[0], 'y1': neighbor_box[1], 
                        'x2':  neighbor_box[2], 'y2':  neighbor_box[3],
                        'color': [0, 255, 255]
                    })

        if len(follow_ids) > 0:
            # Visualize following
            follow_neighbors = [(id, aux_data[id]) for id in follow_ids]

            for neighbor_track_id, neighbor in follow_neighbors:
                neighbor_boxes = xywh_to_xyxy_lst(neighbor['boxes'])
                neighbor_frames = neighbor['frames']
                neighbor_frames_ids = get_frame_ids_by_names(neighbor_frames)
                neighbor_frames_ids.sort()
                intersected = [
                    (box, id) for id, box in zip(neighbor_frames_ids, neighbor_boxes)
                    if id in main_frame_ids
                ]

                for neighbor_box, neighbor_frame_id in intersected:
                    track_lists.append({
                        'frame_id': neighbor_frame_id,
                        'track_id': neighbor_track_id + '- following', 
                        'x1': neighbor_box[0], 'y1': neighbor_box[1], 
                        'x2':  neighbor_box[2], 'y2':  neighbor_box[3],
                        'color': [0, 0, 255]
                    })


        # Write to video
        track_df = pd.DataFrame(track_lists)

        for frame_id, frame_name in zip(main_frame_ids, main_frame_names):
            # Frame image
            img = cv2.imread(
                osp.join(CONSTANT.EXTRACTED_FRAMES_DIR, frame_name[2:]))

            # All tracks in that frame
            frame_df = track_df[track_df.frame_id==frame_id]
            img = visualize_one_frame(img, frame_df)

            writer.write(img)

if __name__ == '__main__':
    visualize_neighbors()