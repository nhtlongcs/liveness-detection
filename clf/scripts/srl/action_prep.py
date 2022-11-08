import os
import json
import os.path as osp
from typing import List
import numpy as np
import pandas as pd
from external.extraction.textual.query import Query
from external.extraction.paths import ACTION_GROUP_JSON
from tqdm import tqdm
from pathlib import Path
import sys
pd.set_option("display.max_columns", None)
PRINT_CSV = True
TYPE = "action"
NUM_CLS, VEH_MAP = None, None


def init():
    data_group = json.load(open(ACTION_GROUP_JSON, "r"))
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
            print(f"invalid action: {veh}")
            continue
        flag = False
        if use_fraction:
            y[VEH_MAP[veh]] += 1
        else:
            y[VEH_MAP[veh]] = 1

    if flag:
        return None

    if use_fraction:
        y /= np.sum(y)

    return y


def parse_to_csv(data_srl, mode="test", use_fraction=True, is_csv=True,save_dir='./'):
    df_dict = {
        "query_id": [],
        "captions": [],
        "actions": [],
        "labels": [],
    }

    fail_query_ids = []
    for query_id in tqdm(data_srl.keys()):
        query_content = data_srl[query_id]
        query = Query(query_content, query_id)

        query.actions.sort()
        query_veh_labels = create_ohe_vector(query.actions, use_fraction)
        if query_veh_labels is None:
            fail_str = f"{query_id}: {query.actions}"
            fail_query_ids.append(fail_str)
            continue

        df_dict["query_id"].append(query_id)
        df_dict["labels"].append(query_veh_labels.tolist())
        df_dict["actions"].append(query.actions)
        list_caps = [c.caption for c in query.list_caps]
        df_dict["captions"].append("\n".join(list_caps))

    df_final = None
    if is_csv:
        df_final = pd.DataFrame.from_dict(df_dict)
        csv_save_path = osp.join(save_dir, f"{TYPE}_{mode}_ohe.csv")
        if use_fraction is True:
            csv_save_path = osp.join(save_dir, f"{TYPE}_{mode}_fraction.csv")

        df_final.to_csv(csv_save_path, index=False)

        print(f"save result to {save_dir} directory")
        print(f"{len(fail_query_ids)} fail queries: {fail_query_ids}")
    return df_final


def prepare_action_metadata(srl_path, save_dir, use_fraction=True):
    assert osp.exists(srl_path), f"{srl_path} not found"
    os.makedirs(save_dir, exist_ok=True)
    data_srl = json.load(open(srl_path))
    mode = osp.basename(srl_path).split("_")[1]
    use_fraction = use_fraction if isinstance(use_fraction, List) else [use_fraction]
    for type in use_fraction:
        parse_to_csv(data_srl, mode, use_fraction=type, is_csv=PRINT_CSV, save_dir=save_dir)

def main():
    srl_dir = sys.argv[1]
    out_dir = sys.argv[2]
    prepare_action_metadata(osp.join(srl_dir , "srl_train_tracks.json"), out_dir,use_fraction=[True,False])
    prepare_action_metadata(osp.join(srl_dir , "srl_test_queries.json"), out_dir,use_fraction=[True,False])


if __name__ == "__main__":
    NUM_CLS, VEH_MAP = init()
    main()
    # python action_prep.py meta_data/srl_train_tracks.json/.. meta_data/
    