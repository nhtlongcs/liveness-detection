"""
Script for refining all neighbor tracks based on raw neighbor mapping json generated ny gem_neighbor_mapping.py
Read in the neighbor tracks and the main tracks and refine the relationshop
"""

import os
import os.path as osp
import cv2
import json
import numpy as np
from tqdm import tqdm
from external.relation.bb_utils import refine_boxes, xywh_to_xyxy_lst, get_attention_mask, check_attention_mask
from external.relation.frame_utils import get_frame_ids_by_names
from external.relation.track_utils import (
    check_is_neighbor_tracks, check_same_tracks, get_relation_between_tracks
)

import argparse
parser = argparse.ArgumentParser('Generate auxiliary tracks')

parser.add_argument("-i", "--tracks_json", type=str, help='Track json file')
parser.add_argument("-d", "--image_dir", type=str, help='Path to extracted frames')
parser.add_argument("-o", "--output_json", type=str, help='Output file')
parser.add_argument("--aux_tracks_json", type=str, help='Auxiliary json file')
parser.add_argument("--aux_tracks_mapping_json", type=str, help='Auxiliary mapping json file')
parser.add_argument("--top_k_neighbors", type=int, default=2, help='Top k neighbors each relation')
args = parser.parse_args()


TRACKS_JSON = args.tracks_json
OUTPATH = args.output_json
AUX_TRACKS_MAPPING = args.aux_tracks_mapping_json
AUX_TRACKS = args.aux_tracks_json
FRAME_DIR = args.image_dir

def run():
    with open(TRACKS_JSON, 'r') as f:
        main_tracks = json.load(f)

    with open(AUX_TRACKS, 'r') as f:
        aux_tracks = json.load(f)

    with open(AUX_TRACKS_MAPPING, 'r') as f:
        aux_tracks_mapping = json.load(f)

    relation_graph = {}

    main_track_ids = list(main_tracks.keys())
    for main_track_id in tqdm(main_track_ids):

        relation_graph[main_track_id] ={
            'follow': [],
            'followed_by': [],
        }

        main_boxes = main_tracks[main_track_id]['boxes']
        main_boxes = xywh_to_xyxy_lst(main_boxes)
        main_frame_names = main_tracks[main_track_id]['frames']

        if main_track_id not in aux_tracks_mapping.keys():
            continue
            
        aux_track_ids = aux_tracks_mapping[main_track_id]

        # Interpolate main track boxes
        main_frame_ids = get_frame_ids_by_names(main_frame_names)
        main_start_id = main_frame_ids[0]

        main_refined_boxes = refine_boxes(main_frame_ids, main_boxes)
        main_frame_ids = [i for i in range(main_start_id, main_start_id+len(main_refined_boxes))]
        
        # Generate attention mask
        tmp_frame = cv2.imread(osp.join(FRAME_DIR, main_frame_names[0]))
        frame_h, frame_w, _ = tmp_frame.shape
        attention_mask = get_attention_mask(main_refined_boxes, frame_w, frame_h, expand_ratio=0)

        # Interpolate aux track boxes
        neighbor_candidates = []
        for aux_track_id in aux_track_ids:
            aux_boxes = aux_tracks[aux_track_id]['boxes']
            aux_boxes = xywh_to_xyxy_lst(aux_boxes)
            aux_frame_names = aux_tracks[aux_track_id]['frames']
            aux_frame_ids = get_frame_ids_by_names(aux_frame_names)
            aux_frame_ids.sort()

            aux_refined_boxes = refine_boxes(aux_frame_ids, aux_boxes)
            aux_start_id = aux_frame_ids[0]
            aux_frame_ids = [i for i in range(aux_start_id, aux_start_id+len(aux_refined_boxes))]

            # Only sampling aux frames and boxes within main track, meaning aux track and main track appear at the same frame
            intersect_frame_ids = list(set(main_frame_ids).intersection(set(aux_frame_ids)))
            intersect_frame_ids.sort()
            main_intersect_boxes = [box for (box, id) in zip(main_refined_boxes, main_frame_ids) if id in intersect_frame_ids]
            aux_intersect_boxes = [box for (box, id) in zip(aux_refined_boxes, aux_frame_ids) if id in intersect_frame_ids]

            # Check if both tracks are the same, both tracks have been aligned
            if check_same_tracks(main_intersect_boxes, aux_intersect_boxes, iou_mean_threshold=0.3):
                continue
            
            # Check if neighbors track is inside attention mask
            if not check_attention_mask(
                attention_mask, 
                aux_intersect_boxes,
                attention_thresh=0.3,
                min_rate=0.2 
            ):
                continue

            # Check if both tracks are related (near to each other), optional
            # if not check_is_neighbor_tracks(main_intersect_boxes, aux_intersect_boxes, dist_mean_threshold=300):
            #     continue
            
            # Finally, two determine relation between two neighbor tracks
            relation, priority_level = get_relation_between_tracks(main_intersect_boxes, aux_intersect_boxes)
            if relation in ['follow', 'followed_by']:
                neighbor_candidates.append((aux_track_id, relation, priority_level))
        
        # Store result
        neighbor_candidates.sort(key=lambda tup: tup[-1]) # ranking by priority level, smaller means more qualified
        for candidate_id, relation, _ in neighbor_candidates:
            if len(relation_graph[main_track_id][relation]) < args.top_k_neighbors:
                relation_graph[main_track_id][relation].append(candidate_id)

    with open(OUTPATH, 'w') as f:
        json.dump(relation_graph, f, indent=4)

if __name__ == '__main__':
    run()