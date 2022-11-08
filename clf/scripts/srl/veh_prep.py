import os, json, cv2
import os.path as osp
import pandas as pd
from tqdm import tqdm
import numpy as np
import sys

from external.extraction.textual.query import Query
from external.extraction.paths import VEHICLE_GROUP_JSON

pd.set_option("display.max_columns", None)

TYPE = "veh"
BOX_FIELD = "boxes"
EXTRCTED_FRMS_DIR = None
NUM_CLS, VEH_MAP = None, None

def init():
    data_group = json.load(open(VEHICLE_GROUP_JSON, "r"))
    num_classes = len(data_group.keys())

    id_map = {}  # {'group-1': 0}
    for k in data_group.keys():
        i = int(k.split("-")[1]) - 1
        id_map[k] = i

    veh_map = {}  # {'suv': 2}
    for k in data_group.keys():
        i = id_map[k]
        for veh in data_group[k]:
            veh_map[veh] = i
    print(veh_map)
    return num_classes, veh_map


def create_ohe_vector(list_vehicles, use_fraction=False):
    y = np.zeros(NUM_CLS)
    flag = True  # Check if exist at least one valid vehicle or not
    for veh in list_vehicles:
        if VEH_MAP.get(veh) is None:
            print(f"invalid veh: {veh}")
            continue
        flag = False
        if use_fraction:
            y[VEH_MAP[veh]] += 1
        else:
            y[VEH_MAP[veh]] = 1

    if flag:
        return np.ones(NUM_CLS)

    if use_fraction:
        y /= np.sum(y)

    return y


def get_list_boxes(data_track, query_id, save_dir=None):
    list_boxes = data_track[query_id][BOX_FIELD]
    n = len(list_boxes)
    ids2use = [0, n // 3, 2 * n // 3, n - 1]

    res = {"paths": [], "width": [], "height": []}
    for i in ids2use:
        img_path = osp.join(EXTRCTED_FRMS_DIR, data_track[query_id]["frames"][i])
        cv_img = cv2.imread(img_path)
        x, y, w, h = list_boxes[i]
        cv_box = cv_img[y : y + h, x : x + w, :]
        res["width"].append(w)
        res["height"].append(h)

        if save_dir is not None:
            box_save_path = osp.join(save_dir, f"{query_id}_{i}.png")
            res["paths"].append(box_save_path)

            if not osp.isfile(box_save_path):
                cv2.imwrite(box_save_path, cv_box)

    return res


def parse_to_csv(data_srl, data_track, mode="train", fraction=True,save_dir = './'):
    df_dict = {
        "query_id": [],
        "box_id": [],
        "width": [],
        "height": [],
        "labels": [],
        "vehicles": [],
        # "vehicle_code": [],
        "paths": [],
    }
    box_id = 0
    train_vis_dir = osp.join(save_dir, "veh_imgs")
    os.makedirs(train_vis_dir, exist_ok=True)

    fail_query_ids = []
    for query_id in tqdm(data_srl.keys()):
        query_content = data_srl[query_id]
        query = Query(query_content, query_id)
        query.subjects.sort()

        query_veh_labels = create_ohe_vector(query.subjects, use_fraction=fraction)
        if query_veh_labels is None:
            fail_query_ids.append(query_id)
            continue

        res = get_list_boxes(data_track, query_id, train_vis_dir)
        for i in range(len(res["width"])):
            box_id += 1
            df_dict["query_id"].append(query_id)
            df_dict["labels"].append(query_veh_labels.tolist())
            df_dict["vehicles"].append(query.subjects)

            # Box information
            df_dict["box_id"].append(box_id)
            df_dict["width"].append(res["width"][i])
            df_dict["height"].append(res["height"][i])
            df_dict["paths"].append(res["paths"][i])

    df_final = pd.DataFrame.from_dict(df_dict)
    csv_save_path = osp.join(save_dir, f"{mode}.csv")
    if fraction is True:
        csv_save_path = osp.join(save_dir, f"{mode}_fraction.csv")

    df_final.to_csv(csv_save_path, index=False)
    print(f"Extract dataframe to {csv_save_path}")
    print(f"Fail queries: {fail_query_ids}")
    return df_final


def parse_to_csv_test(data_srl, data_track, mode="test", fraction=True,save_dir = './'):
    df_dict = {
        "query_id": [],
        "queries": [],
        "vehicles": [],
        "labels": [],
    }
    box_id = 0

    fail_query_ids = []
    for query_id in tqdm(data_srl.keys()):
        query_content = data_srl[query_id]
        query = Query(query_content, query_id)
        query.subjects.sort()

        query_veh_labels = create_ohe_vector(query.subjects, use_fraction=fraction)
        if query_veh_labels is None:
            fail_str = f"{query_id}: {query.subjects}"
            fail_query_ids.append(fail_str)
            continue

        df_dict["queries"].append(query.get_list_captions_str())
        df_dict["query_id"].append(query_id)
        df_dict["labels"].append(query_veh_labels.tolist())
        df_dict["vehicles"].append(query.subjects)

    df_final = pd.DataFrame.from_dict(df_dict)
    csv_save_path = osp.join(save_dir, f"{TYPE}_{mode}.csv")
    if fraction is True:
        csv_save_path = osp.join(save_dir, f"{TYPE}_{mode}_fraction.csv")

    df_final.to_csv(csv_save_path, index=False)
    print(f"Extract dataframe to {csv_save_path}")
    print(f"Fail queries: {fail_query_ids}")
    return df_final


def main():
    
    global EXTRCTED_FRMS_DIR 
    track_dir = sys.argv[1]
    meta_data_dir = sys.argv[2]
    EXTRCTED_FRMS_DIR = sys.argv[3]
    out_dir =  sys.argv[4]


    TRAIN_SRL_JSON = osp.join(meta_data_dir, "srl_train_tracks.json")
    TEST_SRL_JSON = osp.join(meta_data_dir, "srl_test_queries.json")
    TRAIN_TRACK_JSON = osp.join(track_dir, "train_tracks.json")
    TEST_TRACK_JSON = osp.join(track_dir, "test_queries.json")
    # Train tracks
    print("RUN TRAIN")
    train_srl = json.load(open(TRAIN_SRL_JSON))
    train_track = json.load(open(TRAIN_TRACK_JSON))
    parse_to_csv(train_srl, train_track, "train", fraction=True, save_dir=out_dir)

    # Test tracks
    print("RUN TEST")
    train_srl = json.load(open(TEST_SRL_JSON))
    train_track = json.load(open(TEST_TRACK_JSON))
    parse_to_csv_test(train_srl, train_track, "test", fraction=True,save_dir=out_dir)
    parse_to_csv_test(train_srl, train_track, "test", fraction=False,save_dir = out_dir)



if __name__ == "__main__":
    NUM_CLS, VEH_MAP = init()
    main()
