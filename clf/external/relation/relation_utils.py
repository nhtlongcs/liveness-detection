import numpy as np
import math

class PositionState:
    NO_RELATION = 0
    A_BEHIND_B = 1
    B_BEHIND_A = 2

class FollowState:
    NO_RELATION = 0
    A_FOLLOW_B = 1
    B_FOLLOW_A = 2
    RELATION_NAME = {
        0: 'no_relation', 1: 'follow', 2: 'followed_by'
    }

NUM_COUNT_THRES = 3
SUBSEQUENCE_THRES = 2
VELOCITY_SKIP_FRAME = 4
DISTANCE_THRES = 4.5
FOLLOW_STATE_THRES = 2
POS_THRES = math.cos(math.pi/10)

THRES_LEVEL = [
    math.cos(math.pi/60),
    math.cos(math.pi/36),
    math.cos(math.pi/18),
    math.cos(math.pi/15),
    math.cos(math.pi/12),
    math.cos(math.pi/10),
    math.cos(math.pi/9),
    ]

MAX_TRAJ_THRES_LEVEL = [
    math.cos(math.pi/36),
    math.cos(math.pi/20),
    math.cos(math.pi/15),
    math.cos(math.pi/12),
    math.cos(math.pi/10),
    math.cos(math.pi/9),
    ]


def minus_vector(vector_a, vector_b):
    return [vector_b[0] - vector_a[0], vector_b[1] - vector_a[1]]

def calculate_velocity_vector(coor: list, skip_frame=2, smooth_frame=2):
    if skip_frame > len(coor):
        skip_frame = 1
    vel_list = [minus_vector(coor[i], coor[i+skip_frame]) for i in range(len(coor) - skip_frame)]
    
    return vel_list

def calculate_distance_vector(coor_a, coor_b, skip_frame=2):
    if skip_frame > len(coor_a):
        skip_frame = 1
    dis_list = [minus_vector(coor_a[i], coor_b[i]) for i in range(len(coor_a) - skip_frame)]
    return dis_list

def cosine_similarity(vect_a, vect_b):
    if isinstance(vect_a, list):
        vect_a = np.array(vect_a)
    if isinstance(vect_b, list):
        vect_b = np.array(vect_b)
    return np.dot(vect_a, vect_b)/(np.linalg.norm(vect_a)*np.linalg.norm(vect_b)) #default: L2 norm

def length_vector(vector):
    return math.sqrt(vector[0]**2 + vector[1]**2)

class FollowRelationCounter(object):
    def __init__(self):
        self.counter = {}
        self.trace = []
        self.total = 0
        self.famous_value = None
        self.max_count = 1
        pass
    
    def update(self, value):
        if self.counter.get(value) is None:
            self.counter[value] = 1
        else:
            self.counter[value] += 1

        if self.counter[value] > self.max_count:
            self.max_count = self.counter[value]
            self.famous_value = value 

        self.total += 1
        self.trace.append(value)

    def find_longest_state(self, state_val):
        count = 0
        ans = 0
        for val in self.trace:
            if val == state_val:
                count += 1
                ans = max(ans, count)
            else:
                count = 0
        return ans

    def get_famous_value(self):
        a_fl_b = self.counter.get(FollowState.A_FOLLOW_B, None)
        b_fl_a = self.counter.get(FollowState.B_FOLLOW_A, None)
        ans = FollowState.NO_RELATION
        if a_fl_b is None and b_fl_a is None:
            pass
        elif a_fl_b is None:
            if b_fl_a >= NUM_COUNT_THRES:
                ans = FollowState.B_FOLLOW_A
        elif b_fl_a is None:
            if a_fl_b >= NUM_COUNT_THRES:
                ans = FollowState.A_FOLLOW_B
        else:
            if a_fl_b >= b_fl_a:
                ans = FollowState.A_FOLLOW_B
            else:
                ans = FollowState.B_FOLLOW_A

        if ans != FollowState.NO_RELATION:
            if self.find_longest_state(ans) < SUBSEQUENCE_THRES:
                ans = FollowState.NO_RELATION
        
        return ans