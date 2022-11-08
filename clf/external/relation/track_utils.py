import numpy as np
from .bb_utils import (
    bb_intersection_over_union, get_box_center
)

from .relation_utils import (
    PositionState, FollowState, FollowRelationCounter,
    VELOCITY_SKIP_FRAME, DISTANCE_THRES, FOLLOW_STATE_THRES,
    MAX_TRAJ_THRES_LEVEL, THRES_LEVEL, POS_THRES,
    calculate_velocity_vector, calculate_distance_vector, 
    length_vector, cosine_similarity
)

def check_same_tracks(
    track_boxes1, track_boxes2, 
    iou_mean_threshold: float = 0.5):
    """
    Check two aligned tracks are the same.
    Return true if mean iou threshold for every boxes is over the threshold
    """

    iou_scores = []
    for (box1, box2) in zip(track_boxes1, track_boxes2):
        iou_score = bb_intersection_over_union(box1, box2)
        iou_scores.append(iou_score)
    return np.mean(iou_scores) >= iou_mean_threshold

def check_is_neighbor_tracks(
    track_boxes1, track_boxes2, 
    dist_mean_threshold: float = 200):
    """
    Check two aligned tracks are the neighbors.
    Return true if distance between locations lower than threhold 
    (means tracks are close to each other)
    """

    distance_list = []
    for (box1, box2) in zip(track_boxes1, track_boxes2):
        center1 = get_box_center(box1)
        center2 = get_box_center(box2)
        dist = np.sqrt(
            (center1[0]-center2[0])*(center1[0]-center2[0]) + (center1[1]-center2[1])*(center1[1]-center2[1])
        )
        distance_list.append(dist)

    return np.mean(distance_list) <= dist_mean_threshold


def get_relation_between_tracks(track_boxes1, track_boxes2):
    """
    Adapt from HCMUS-AIC22
    Get the relation betweeen two tracks.
    Return state
    """
    num_boxes = len(track_boxes1)

    if num_boxes <= 2:
        return FollowState.NO_RELATION, -1

    if num_boxes <= 9:
        skip_frame = VELOCITY_SKIP_FRAME // 2
    else:
        skip_frame = VELOCITY_SKIP_FRAME

    # Mean distances between tracks
    velocity_vect_a = calculate_velocity_vector(
        track_boxes1, skip_frame=skip_frame)
    velocity_vect_b = calculate_velocity_vector(
        track_boxes2, skip_frame=skip_frame)
    distance_vect_ab = calculate_distance_vector(
        track_boxes1, track_boxes2, skip_frame=skip_frame)
    avg_distance = np.mean([length_vector(v) for v in distance_vect_ab])

    # Mean cosine similarity between tracks
    n = len(distance_vect_ab)
    cosine_va_ab = [cosine_similarity(velocity_vect_a[i], distance_vect_ab[i]) for i in range(n)]
    # cosine_vb_ab = [cosine_similarity(velocity_vect_b[i], distance_vect_ab[i]) for i in range(n)]
    cosine_va_vb = [cosine_similarity(velocity_vect_a[i], velocity_vect_b[i]) for i in range(n)]
    avg_cos = np.mean(cosine_va_ab)
    
    isFollow = FollowState.NO_RELATION

    priority_level = 0 #used to ranking the "nearest" neighbor
    # Loop through threshold
    for traj_thres in MAX_TRAJ_THRES_LEVEL:
        for v_thres in THRES_LEVEL:

            # check position (A behind B or B behind A)
            position_state = []
            for i in range(n):
                x_a, y_a = velocity_vect_a[i]
                if x_a**2 + y_a**2 <= DISTANCE_THRES:
                    position_state.append(PositionState.NO_RELATION)
                    continue

                x_b, y_b = velocity_vect_b[i]
                if x_b**2 + y_b**2 <= DISTANCE_THRES:
                    position_state.append(PositionState.NO_RELATION)
                    continue

                if cosine_va_ab[i] >= POS_THRES:
                    position_state.append(PositionState.A_BEHIND_B)
                elif cosine_va_ab[i] <= -POS_THRES:
                    position_state.append(PositionState.B_BEHIND_A)
                else:
                    position_state.append(PositionState.NO_RELATION)

            # check relation (A folow B or B follow A or no relation)
            follow_state = []
            follow_state_counter = FollowRelationCounter()
            for i in range(n):
                if position_state[i] == PositionState.NO_RELATION:
                    continue
                if position_state[i] == PositionState.A_BEHIND_B:
                    if (not (cosine_va_vb[i] >= v_thres)) or abs(cosine_va_ab[i]) < traj_thres:
                        follow_state.append(FollowState.NO_RELATION)
                        follow_state_counter.update(FollowState.NO_RELATION)
                    else:
                        follow_state.append(FollowState.A_FOLLOW_B)
                        follow_state_counter.update(FollowState.A_FOLLOW_B)

                if position_state[i] == PositionState.B_BEHIND_A:
                    # if abs(cosine_vb_ab[i]) < FOLLOW_COSINE_THRES:
                    if (not (cosine_va_vb[i] >= v_thres)) or abs(cosine_va_ab[i]) < traj_thres:
                        follow_state.append(FollowState.NO_RELATION)
                        follow_state_counter.update(FollowState.NO_RELATION)                
                    else:
                        follow_state.append(FollowState.B_FOLLOW_A)
                        follow_state_counter.update(FollowState.B_FOLLOW_A)
            
            if (follow_state_counter.find_longest_state(FollowState.B_FOLLOW_A) >= FOLLOW_STATE_THRES and 
                follow_state_counter.find_longest_state(FollowState.A_FOLLOW_B) >= FOLLOW_STATE_THRES):
                isFollow = FollowState.NO_RELATION
            else:
                isFollow = follow_state_counter.get_famous_value()
            
            if isFollow in (
                FollowState.A_FOLLOW_B,
                FollowState.B_FOLLOW_A
            ):
                return FollowState.RELATION_NAME[isFollow], priority_level
            else:
                priority_level += 1

    return FollowState.RELATION_NAME[isFollow], priority_level