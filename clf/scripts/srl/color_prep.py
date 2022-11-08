import os, json, cv2, sys
import os.path as osp
import pandas as pd
from tqdm import tqdm
import numpy as np


from external.extraction.textual.query import Query
from external.extraction.paths import COLOR_GROUP_JSON

pd.set_option("display.max_columns", None)

IS_TEST = True
BOX_FIELD = "boxes"
TYPE = "col"
EXTRCTED_FRMS_DIR = None
PRINT_CSV = True

NUM_CLS, VEH_MAP = None, None

def init():
    data_group = json.load(open(COLOR_GROUP_JSON, "r"))
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
            print(f"invalid color: {veh}")
            continue
        flag = False
        if use_fraction:
            y[VEH_MAP[veh]] += 1
        else:
            y[VEH_MAP[veh]] = 1

    if flag:
        if IS_TEST:
            return np.ones(NUM_CLS)
        else:
            return None

    if use_fraction:
        y /= np.sum(y)

    return y


def get_list_boxes(data_track, query_id, labels, save_dir=None, save_labels=True):
    list_boxes = data_track[query_id][BOX_FIELD]
    n = len(list_boxes)
    ids2use = [0, n // 3, 2 * n // 3, n - 1]
    res = {"paths": [], "width": [], "height": []}
    for i in ids2use:
        img_path = osp.join(EXTRCTED_FRMS_DIR, data_track[query_id]["frames"][i])
        assert osp.exists(img_path), f"{img_path} not found"
        assert img_path.endswith(".jpg"), f"{img_path} is not a jpg file"
        cv_img = cv2.imread(img_path)
        x, y, w, h = list_boxes[i]
        cv_box = cv_img[y : y + h, x : x + w, :]
        res["width"].append(w)
        res["height"].append(h)

        if save_dir is not None:
            box_save_path = osp.join(save_dir, f"{query_id}_{i}.png")
            res["paths"].append(box_save_path)
            if save_labels:
                box_label_save_dir = save_dir + "_label"
                os.makedirs(box_label_save_dir, exist_ok=True)
                box_label_save_path = osp.join(box_label_save_dir, f"{query_id}_{i}.txt")
                if not osp.isfile(box_label_save_path):
                    with open(box_label_save_path, "w") as f:
                        f.write(' '.join(labels))
            if not osp.isfile(box_save_path):
                cv2.imwrite(box_save_path, cv_box)

    return res


def parse_to_csv_test(
    data_srl, data_track, mode="test", use_fraction=True, is_csv=True,save_dir='./'
):
    df_dict = {"query_id": [], "labels": [], "colors": []}

    fail_query_ids = []

    for query_id in tqdm(data_srl.keys()):
        query_content = data_srl[query_id]
        query = Query(query_content, query_id)

        query._refine_colors()
        query.colors.sort()

        query_veh_labels = create_ohe_vector(query.colors, use_fraction)
        if query_veh_labels is None:
            fail_str = f"{query_id}: {query.colors}"
            fail_query_ids.append(fail_str)
            continue

        df_dict["query_id"].append(query_id)
        df_dict["labels"].append(query_veh_labels.tolist())
        df_dict["colors"].append(query.colors)

    df_final = None
    if is_csv:
        df_final = pd.DataFrame.from_dict(df_dict)
        csv_save_path = osp.join(save_dir, f"{TYPE}_{mode}.csv")
        if use_fraction is True:
            csv_save_path = osp.join(save_dir, f"{TYPE}_{mode}_fraction.csv")

        df_final.to_csv(csv_save_path, index=False)

        print(f"save result to {save_dir} directory")
        print(f"Fail queries: {fail_query_ids}")
    return df_final


def parse_to_csv(data_srl, data_track, mode="train", use_fraction=True, is_csv=True,save_dir='./'):
    df_dict = {
        "query_id": [],
        "box_id": [],
        "width": [],
        "height": [],
        "labels": [],
        "colors": [],
        "paths": [],
    }

    fail_query_ids = []
    box_id = 0

    train_vis_dir = osp.join(save_dir, "veh_imgs")
    os.makedirs(train_vis_dir, exist_ok=True)

    for query_id in tqdm(data_srl.keys()):
        query_content = data_srl[query_id]
        query = Query(query_content, query_id)
        query.colors.sort()

        col_before = query.colors
        query._refine_colors()
        query_veh_labels = create_ohe_vector(query.colors, use_fraction)
        col_after = query.colors

        if query_veh_labels is None:
            print(f"fail id: {query_id}")
            print(f"before: {col_before}")
            print(f"after: {col_after}")
            fail_query_ids.append(query_id)
            continue
        res = get_list_boxes(data_track, query_id, query.colors, train_vis_dir, save_labels=False)
        for i in range(len(res["width"])):
            box_id += 1
            df_dict["query_id"].append(query_id)
            df_dict["labels"].append(query_veh_labels.tolist())
            df_dict["colors"].append(query.colors)

            df_dict["box_id"].append(box_id)
            df_dict["width"].append(res["width"][i])
            df_dict["height"].append(res["height"][i])
            df_dict["paths"].append(res["paths"][i])
        # break

    df_final = None
    if is_csv:
        df_final = pd.DataFrame.from_dict(df_dict)
        csv_save_path = osp.join(save_dir, f"{TYPE}_{mode}.csv")
        if use_fraction is True:
            csv_save_path = osp.join(save_dir, f"{TYPE}_{mode}_fraction.csv")

        df_final.to_csv(csv_save_path, index=False)

        print(f"save result to {save_dir} directory")
        print(f"Fail queries: {fail_query_ids}")
    return df_final

def prepare_color_metadata(track_path, srl_path, save_dir, parse_func=None, mode="train"):
    assert osp.exists(srl_path), f"{srl_path} not found"
    assert osp.exists(track_path), f"{track_path} not found"
    assert parse_func is not None, "parse_func cannot be None"

    os.makedirs(save_dir, exist_ok=True)
    data_srl = json.load(open(srl_path))
    data_track = json.load(open(track_path))

    parse_func(data_srl, data_track, mode, use_fraction=True, is_csv=PRINT_CSV,save_dir=save_dir)
    parse_func(data_srl, data_track, mode, use_fraction=False, is_csv=PRINT_CSV,save_dir=save_dir)

def main():
    global EXTRCTED_FRMS_DIR 
    track_dir = sys.argv[1]
    srl_data_dir = sys.argv[2]
    EXTRCTED_FRMS_DIR = sys.argv[3]
    out_dir =  sys.argv[4]
    print("RUN TRAIN")
    prepare_color_metadata(osp.join(track_dir,'train_tracks.json'),osp.join(srl_data_dir,'srl_train_tracks.json'),out_dir,mode="train",parse_func=parse_to_csv)
    print("RUN TEST")
    prepare_color_metadata(osp.join(track_dir,'test_queries.json'),osp.join(srl_data_dir,'srl_test_queries.json'),out_dir,mode="test",parse_func=parse_to_csv_test)
if __name__ == "__main__":
    NUM_CLS, VEH_MAP = init()
    main()
