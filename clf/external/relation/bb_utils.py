from typing import List
import numpy as np

def bb_intersection_over_union(boxA, boxB):
    """Calculate IOU
    box in (x1, y1, x2, y2)"""
    
    assert boxA[2] >= boxA[0] and boxA[3] >= boxA[1], "wrong format"
    assert boxB[2] >= boxB[0] and boxB[3] >= boxB[1], "wrong format"

	# determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def interpolate_box(cur_idx, start_idx, end_idx, start_box, end_box):
    """
    Interpolate box at current frame index
    """
    end_ratio = (end_idx-cur_idx)/(end_idx-start_idx)
    start_ratio = (cur_idx-start_idx)/(end_idx-start_idx)
    
    box1, box2 = start_box, end_box
    if isinstance(start_box, list):
        box1 = np.array(start_box) 
    if isinstance(end_box, list):
        box2 = np.array(end_box) 
    
    cur_box = start_ratio*box2 + end_ratio*box1 
    cur_box = cur_box.tolist()
    return cur_box

def generate_boxes(start_idx, end_idx, start_box, end_box):
    """
    Interpolate boxes at between timestamp
    """
    res = [] 
    for i in range(start_idx+1, end_idx):
        res.append(interpolate_box(i, start_idx, end_idx, start_box, end_box))
    return res

def refine_boxes(list_fids, list_boxes):
    """
    Interpolate missing boxes based on missing frame ids 
    """
    N = len(list_fids)
    res = []
    latest_idx = 0
    for i in range(N-1):
        if list_fids[i+1] - list_fids[i] > 1:
            if bb_intersection_over_union(list_boxes[i+1], list_boxes[i]) < 0.2:
                new_boxes = generate_boxes(
                    list_fids[i], list_fids[i+1], list_boxes[i], list_boxes[i+1]
                )
                
                res += (list_boxes[latest_idx : i+1] + new_boxes)
                latest_idx = i+1
                pass # Interpolate box
    
    res += list_boxes[latest_idx:]
    return res

def get_box_center(box: list):
    return (box[0]+box[2]/2, box[1]+box[3]/2)

def get_velocity(center_0, center_1):
    return (center_1[0]-center_0[0], center_1[1]-center_0[1])

def xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x+w, y+h]

def xywh_to_xyxy_lst(boxes):
    new_boxes = []
    for box in boxes:
        new_boxes.append(xywh_to_xyxy(box))
    return new_boxes


### Attention mask


def expand_boxes(list_xyxy_boxes: list, n: int=4, skip_frame: int=1):
    # xyxy

    list_boxes = []
    for box in list_xyxy_boxes:
        list_boxes.append(
            [box[0], box[1], box[2]-box[0], box[3] - box[1]]
        )


    first_box, last_box = list_boxes[0], list_boxes[-1]
    list_center = [get_box_center(box) for box in list_boxes]
    first_velocity = get_velocity(list_center[skip_frame], list_center[0])
    last_velocity = get_velocity(list_center[-1 - skip_frame], list_center[-1])

    # Expand head
    new_head_boxes = []
    cur_box = first_box
    cur_center = list_center[0]
    for _ in range(n):
        w, h = cur_box[-2:]
        cur_center = (cur_center[0] + first_velocity[0], cur_center[1] + first_velocity[1])
        new_head_boxes.append([cur_center[0], cur_center[1], w, h])

    # Expand trail
    new_trail_boxes = []
    cur_box = last_box
    cur_center = list_center[-1]
    for _ in range(n):
        w, h = cur_box[-2:]
        cur_center = (cur_center[0] + last_velocity[0], cur_center[1] + last_velocity[1])
        new_trail_boxes.append([cur_center[0], cur_center[1], w, h])

    res = new_head_boxes[::-1] + list_boxes + new_trail_boxes

    res = xywh_to_xyxy_lst(res)
    return res

def check_attention_mask(attn_mask: np.ndarray, boxes: np.array, attention_thresh: float =0.3, min_rate: float = 0.2):
    # min rate: percentage of frames length in which the track is in attention mask
    chosen_ids = [] # list of boxes that inside mask
    for i, box in enumerate(boxes):
        x, y, x2, y2 = box
        x, y, x2, y2 = int(x),  int(y),  int(x2),  int(y2)
        w = x2 - x
        h = y2 - y

        area = w*h 
        mask_area = attn_mask[y: y+h+1, x: x+w+1]
        mask_area[np.where(mask_area <= attention_thresh)] = 0
        overlap_ratio = np.sum(mask_area)/area
        
        if overlap_ratio > attention_thresh:
            chosen_ids.append(i)
    return len(chosen_ids) > min_rate * len(boxes)

def get_attention_mask(list_boxes: List, frame_w, frame_h, expand_ratio=0.35, n_expand=2, skip_frame=3):
    # boxes xywh
    mask = np.zeros((frame_h, frame_w))
    if len(list_boxes) >= skip_frame:
        exp_boxes = expand_boxes(list_boxes, n_expand)
    else:
        exp_boxes = list_boxes
    for box in exp_boxes:
        x1, y1, x2, y2 = get_mask_area(box, frame_w, frame_h, expand_ratio)
        mask[y1:y2, x1:x2] = 1.0
    return mask

def get_valid_coor(x, delta, xmax, xmin=0):
    x_new = x+delta
    x_new = min(xmax, max(xmin, x_new))
    return x_new

def get_mask_area(box, W, H, ratio=0.5):
    # xyxy
    x, y, x2, y2 = box
    w = x2 - x
    h = y2 - y 
    exp_w = w*ratio
    exp_h = h*ratio

    new_x1 = int(get_valid_coor(x, -exp_w, W-1, 0))
    new_x2 = int(get_valid_coor(x, w+exp_w, W-1, 0))
    new_y1 = int(get_valid_coor(y, -exp_h, H-1, 0))
    new_y2 = int(get_valid_coor(y, h+exp_h, H-1, 0))

    return new_x1, new_y1, new_x2, new_y2