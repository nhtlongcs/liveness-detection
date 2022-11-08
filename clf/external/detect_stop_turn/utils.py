import numpy as np 
import math

NONE_COSINE = 2

class Point(object):
    def __init__(self, x, y) -> None:
        super().__init__()
        self.x = x 
        self.y = y

def cal_distance(box_a, box_b):
    xa, ya, _, _ = box_a
    xb, yb, _, _ = box_b
    return np.sqrt((xa-xb)**2 + (ya-yb)**2)

def xyxy_to_xywh(coor):
    x1, y1, x2, y2 = coor
    return [(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1]

def cross(a: Point, b: Point):
    return a.x*b.y - a.y*b.x
    
def euclid_distance(a: Point, b: Point):
    return np.sqrt((a.x-b.x)**2 + (a.y-b.y)**2)

def sqr_distance(a: Point, b: Point):
    return (a.x-b.x)**2 + (a.y-b.y)**2

def algebra_area(a: Point, b: Point, c: Point):
    return (cross(a, b) + cross(b, c) + cross(c, a))


def minus_vector(vector_a, vector_b):
    return [vector_b[0] - vector_a[0], vector_b[1] - vector_a[1]]


def calculate_velocity_vector(coor: list, skip_frame=2):
    if skip_frame > len(coor):
        skip_frame = 1
    vel_list = [minus_vector(coor[i], coor[i+skip_frame]) for i in range(len(coor) - skip_frame)]
    
    return vel_list


def calculate_change_angle_speed(list_angle: list, skip_frame=2):
    if skip_frame > len(list_angle):
        skip_frame = 1
    vel_list = [abs(list_angle[i] - list_angle[i+skip_frame]) for i in range(len(list_angle) - skip_frame)]
    
    return vel_list

def cosine_similarity(vect_a, vect_b):
    # if np.linalg.norm(vect_a) < 1 or np.linalg.norm(vect_b) < 1:
    #     return NONE_COSINE

    if isinstance(vect_a, list):
        vect_a = np.array(vect_a)
    if isinstance(vect_b, list):
        vect_b = np.array(vect_b)
    return np.dot(vect_a, vect_b)/(np.linalg.norm(vect_a)*np.linalg.norm(vect_b)) #default: L2 norm
