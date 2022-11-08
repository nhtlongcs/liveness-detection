import cv2
import os
import json
from tqdm import tqdm
import os.path as osp

from ..paths import (
    VEHICLE_GROUP_REP,
    VEHICLE_VOCAB,
    LIST_REDUNDANT_VEHICLES,
    COLOR_VOCAB,
    ACTION_VOCAB,
)


def remove_redundant_actions(list_actions: list):
    res = [c for c in list_actions if c in ACTION_VOCAB]
    return res


def remove_redundant_colors(list_colors: list):
    res = [c for c in list_colors if c in COLOR_VOCAB]
    return res


def remove_redundant_subjects(list_subjects: list):
    res = [s for s in list_subjects if s not in LIST_REDUNDANT_VEHICLES]
    return res


def convert_to_representation_subject(list_subjects: list):
    map_dict = {}
    for k, v in VEHICLE_GROUP_REP.items():
        for veh in v:
            map_dict[veh] = k

    res = []
    fail = []
    for s in list_subjects:
        if map_dict.get(s) is None:
            fail.append(s)
            continue
        res.append(map_dict[s])

    if len(fail):
        print(fail)
    return res


def is_list_in_list(list_values, list_to_check):
    res = True
    for val in list_to_check:
        if val not in list_values:
            return False
    return True


def get_vehicle_name_map(vehicle_vocab: dict):
    """Convert from {"group-1": SUV} to {"SUV": 1}"""
    id_map = {}  # {'group-1': 0}
    for k in vehicle_vocab.keys():
        i = int(k.split("-")[1]) - 1
        id_map[k] = i

    veh_map = {}  # {'suv': 2}
    for k in vehicle_vocab.keys():
        i = id_map[k]
        for veh in vehicle_vocab[k]:
            veh_map[veh] = i

    return veh_map


def dump_json(data_dict, json_path, verbose=False):
    with open(json_path, "w") as f:
        json.dump(data_dict, f, indent=2)

    if verbose:
        print(f"Save result to {json_path}")


def scan_images(list_img_path):
    for img_path in tqdm(list_img_path):
        pass
    pass


def refine_list_colors(list_colors, unique=True):
    """Refine list colors to achieve valid classes

    Args:
        list_colors ([type]): List of colors parsed from SRL tools
        unique (bool, optional): keep unique values only. Use for ohe label
    """
    new_list = []
    new_list = remove_redundant_colors(list_colors)
    if is_list_in_list(new_list, ["light_gray"]):
        new_list = new_list.remove("light_gray")
        if new_list is None or len(new_list) == 0:
            new_list = ["gray"]
        else:
            new_list.append("gray")

    if is_list_in_list(new_list, ["dark_gray"]):
        new_list = new_list.remove("dark_gray")
        if new_list is None or len(new_list) == 0:
            new_list = ["gray"]
        else:
            new_list.append("gray")

    if unique:
        new_list = list(set(new_list))
    return new_list


def refine_list_subjects(list_subjects, unique=True, is_subject=True):
    """Refine list colors to achieve valid classes

    Args:
        list_colors ([type]): List of colors parsed from SRL tools
        unique (bool, optional): keep unique values only. Use for ohe label
        is_subject: flag denotes this list is main subjects
                    --> remove common classes (car, vehicle, etc.)
    """
    new_list = list_subjects
    if is_subject:
        # 1. Remove redundant subjects (car, vehicle)
        new_list = remove_redundant_subjects(new_list)
        # 2. Convert all subjects to their representation name of each groups
        new_list = convert_to_representation_subject(new_list)

    if unique:
        new_list = list(set(new_list))

        # 3. Handle ambiguous annotations
        # [SUV, bus-truck] = [bus-truck]
        if is_list_in_list(new_list, ["suv", "bus-truck"]):
            new_list = ["suv", "pickup"]

        # [jeep, SUV, ...] = [Jeep, SUV]
        elif is_list_in_list(new_list, ["jeep", "suv"]):
            new_list = ["jeep", "suv"]

        # [jeep, pickup] = jeep
        elif is_list_in_list(new_list, ["jeep", "pickup"]):
            new_list = ["jeep"]

        # [sedan, suv, van], [sedan, van] = [suv, van]
        elif is_list_in_list(new_list, ["sedan", "suv", "van"]) or is_list_in_list(
            new_list, ["sedan", "van"]
        ):
            new_list = ["suv", "van"]

        # [pickup, truck] = [pickup]
        elif is_list_in_list(new_list, ["pickup", "bus-truck"]):
            new_list = ["pickup"]

        # [pickup, sedan, suv] = [pickup, suv]
        elif (
            is_list_in_list(new_list, ["sedan", "suv", "pickup"])
            or is_list_in_list(new_list, ["van", "suv", "pickup"])
            or is_list_in_list(new_list, ["van", "pickup"])
        ):
            new_list = ["suv", "pickup"]

    return new_list
