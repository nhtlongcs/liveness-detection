from tqdm import tqdm 
import pandas as pd 
from pathlib import Path 
import logging

def check_video_ids(video_label_ls, bbox_label_ls, keyframe_label_ls):
    for vid_lbl_path, bb_lbl_path, keyframe_lbl_path in zip(video_label_ls, bbox_label_ls, keyframe_label_ls):
        vid_lbl_df = pd.read_csv(vid_lbl_path)
        bb_lbl_df = pd.read_csv(bb_lbl_path)
        keyframe_lbl_df = pd.read_csv(keyframe_lbl_path)

        video_ids = vid_lbl_df['filename'].values
        video_ids = set([int(vid_id.split('.')[0]) for vid_id in video_ids])
        video_ids_bb = set(bb_lbl_df['video_id'].values.tolist())
        video_ids_keyframe = set(keyframe_lbl_df['video_id'].values.tolist())

        video_ids_bb_missing = video_ids - video_ids_bb
        video_ids_keyframe_missing = video_ids - video_ids_keyframe
        if len(video_ids_bb_missing) > 0:
            logging.warning(f"Video IDs missing from {bb_lbl_path.name}: {video_ids_bb_missing}")
        if len(video_ids_keyframe_missing) > 0:
            logging.warning(f"Video IDs missing from {keyframe_lbl_path.name}: {video_ids_keyframe_missing}")

        video_ids_bb_ukn = video_ids_bb - video_ids
        video_ids_keyframe_ukn = video_ids_keyframe - video_ids

        for vid_id in video_ids_bb_ukn:
            logging.info(f"Unknown video ID {vid_id} in {bb_lbl_path.name}")
        for vid_id in video_ids_keyframe_ukn:
            logging.info(f"Unknown video ID {vid_id} in {keyframe_lbl_path.name}")
        assert len(video_ids_bb_ukn) == len(video_ids_keyframe_ukn) == 0, f"Unknown video IDs found in bbox and keyframe label files, {len(video_ids_bb_ukn)}, {len(video_ids_keyframe_ukn)}"

# check labels in generated labels with original labels
def check_labels(video_label_ls, bbox_label_ls, keyframe_label_ls):
    for vid_lbl_path, bb_lbl_path, keyframe_lbl_path in zip(video_label_ls, bbox_label_ls, keyframe_label_ls):
        vid_lbl_df = pd.read_csv(vid_lbl_path)
        bb_lbl_df = pd.read_csv(bb_lbl_path)
        keyframe_lbl_df = pd.read_csv(keyframe_lbl_path)

        for lbl_df in [vid_lbl_df, bb_lbl_df, keyframe_lbl_df]:
            labels = set(lbl_df['label'].values.tolist())
            assert labels.intersection({0, 1}) == labels, f"Unknown labels found in {lbl_df.name}, {labels}"

        bb_video_ids = set(bb_lbl_df['video_id'].values.tolist())

        for vid_id in bb_video_ids:
            bb_lbls = bb_lbl_df.groupby('video_id').get_group(vid_id)['label'].values.tolist()
            assert len(set(bb_lbls)) == 1, f"Labels must be the same for all bounding boxes in a video, found {set(bb_lbls)} for video {vid_id}"
            
            raw_video_lbl = vid_lbl_df[vid_lbl_df['filename'] == f'{vid_id}.mp4']['label'].values[0]
            assert raw_video_lbl == bb_lbls[0], f"Label mismatch between raw video and bounding box labels for video {vid_id}"

        keyframe_video_ids = set(keyframe_lbl_df['video_id'].values.tolist())
        for vid_id in keyframe_video_ids:
            keyframe_lbls = keyframe_lbl_df.groupby('video_id').get_group(vid_id)['label'].values.tolist()
            assert len(set(keyframe_lbls)) == 1, f"Labels must be the same for all keyframes in a video, found {set(keyframe_lbls)} for video {vid_id}"
            
            raw_video_lbl = vid_lbl_df[vid_lbl_df['filename'] == f'{vid_id}.mp4']['label'].values[0]
            assert raw_video_lbl == keyframe_lbls[0], f"Label mismatch between raw video and keyframe labels for video {vid_id}"

# check labels in generated labels with existing keyframes / bounding boxes
def check_file_exists(data_dir, video_label_ls, bbox_label_ls, keyframe_label_ls):
    video_file_dir = data_dir/'videos'
    bbox_file_dir = data_dir/'faces/crop'
    keyframe_file_dir = data_dir/'keyframes'
    for vid_lbl_path, bb_lbl_path, keyframe_lbl_path in zip(video_label_ls, bbox_label_ls, keyframe_label_ls):
        vid_lbl_df = pd.read_csv(vid_lbl_path)
        bb_lbl_df = pd.read_csv(bb_lbl_path)
        keyframe_lbl_df = pd.read_csv(keyframe_lbl_path)

        for i, row in vid_lbl_df.iterrows():
            video_filename = row['filename']
            video_path = video_file_dir/video_filename
            assert video_path.exists(), f"Video file {video_path} does not exist"

        for i, row in bb_lbl_df.iterrows():
            bbox_filename = row['filename']
            bbox_path = bbox_file_dir/bbox_filename
            assert bbox_path.exists(), f"Bounding box file {bbox_path} does not exist"
        
        for i, row in keyframe_lbl_df.iterrows():
            keyframe_filename = row['filename']
            keyframe_path = keyframe_file_dir/keyframe_filename
            assert keyframe_path.exists(), f"Keyframe file {keyframe_path} does not exist"

def check_all(data_dir: str):
    data_dir = Path(data_dir)
    logging.getLogger().setLevel(logging.INFO)

    video_label_ls = sorted(list(data_dir.glob('labels_video*.csv')))
    bbox_label_ls = sorted(list(data_dir.glob('labels_bbox*.csv')))
    keyframe_label_ls = sorted(list(data_dir.glob('labels_keyframes*.csv')))

    assert len(video_label_ls) == len(bbox_label_ls) == len(keyframe_label_ls), f"Number of video, bbox, and keyframe label files do not match, {len(video_label_ls)}, {len(bbox_label_ls)}, {len(keyframe_label_ls)}"
    logging.info("Checking video ids in generated labels. Step 1/3")
    check_video_ids(video_label_ls, bbox_label_ls, keyframe_label_ls)
    logging.info("Checking labels in generated labels with original labels. Step 2/3")
    check_labels(video_label_ls, bbox_label_ls, keyframe_label_ls)
    logging.info("Checking labels in generated labels with existing keyframes / bounding boxes. Step 3/3")
    check_file_exists(data_dir, video_label_ls, bbox_label_ls, keyframe_label_ls)

if __name__ == '__main__':
    check_all('data/train')