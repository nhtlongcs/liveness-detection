from external.detect_stop_turn.utils import (
    Point, NONE_COSINE,
    euclid_distance, algebra_area, xyxy_to_xywh, cal_distance, sqr_distance,
    calculate_velocity_vector, cosine_similarity, calculate_change_angle_speed,
    minus_vector
)
import math
import numpy as np
import copy

def cmp(a, b):
    return (a > b) - (a < b)

class TurnState:
    NO_TURN = 0
    LEFT = 1
    RIGHT = 2
    pass

class AngleState:
    STRAIGHT = -1
    INCONCLUSIVE = 0
    TURN = 1
    pass

class TurnDetector(object):
    def __init__(self, eps=0.035, skip_frame = 1, is_normalize = False) -> None:
        super().__init__()
        self.eps = eps
        self.skip_frame = skip_frame
        self.is_normalize = is_normalize

    def _get_direction(self, org_list_points: list):
        is_turn = False
        turn_type = TurnState.NO_TURN

        N = len(org_list_points[::self.skip_frame])
        if N <= 8:
            return is_turn, turn_type, 0

        list_points = copy.deepcopy(org_list_points) 
        if self.is_normalize:
            list_points = self.normalize(list_points)
        coor = [[point.x, point.y] for point in list_points]

        if self.is_normalize:
            angle_state = self.check_angle_start_end(list_points)
            if angle_state == AngleState.TURN:
                return True, turn_type, 0
    
        list_points = list_points[::self.skip_frame]

        total_area = 0.0       

        src_dst_dist = sqr_distance(list_points[0], list_points[-1])
        for i in range(2, N):
            total_area += algebra_area(
                list_points[0], list_points[i-1], list_points[i]
            )
            pass
        
        norm_area = total_area/2.0/(src_dst_dist) #*math.log(N, src_dst_dist)

        if abs(norm_area) > self.eps:
            angle_state_start = self.check_angle_start(org_list_points)
            angle_state_start_end = self.check_angle_start_end(org_list_points)

            if angle_state_start == AngleState.STRAIGHT and angle_state_start_end == AngleState.STRAIGHT:
                if not self.is_intersection(org_list_points):
                    return False, turn_type, norm_area

            is_turn = True
            if norm_area < 0:
                turn_type = TurnState.LEFT
            else:
                turn_type = TurnState.RIGHT
        
        return is_turn, turn_type, norm_area


    def process(self, list_boxes: list):
        N = len(list_boxes)
        # list_boxes = [xyxy_to_xywh(box) for box in list_boxes]
        # list_points = [Point(x, -y) for box in list_boxes] # Reverse y axis
        list_points = []
        for box in list_boxes:
            x, y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            # x, y = (box[0] + box[2]), (box[1] + box[3])
            list_points.append(Point(x, -y))

        is_turn, turn_state, norm_area = self._get_direction(list_points)
        is_vertical_view = self.find_vertical_views(list_points)
        return is_turn, turn_state, norm_area, list_points 


# --------------------------------------------------------------------------------
# HELPER FUNCTIONS
    def normalize(self, list_points):
        list_x = [point.x for point in list_points]
        max_x = max(list_x)
        norm_x = [i/max_x for i in list_x]

        list_y = [(point.y) for point in list_points]

        max_y = max(list_y)
        norm_y = [i/max_y for i in list_y]

        for i in range(len(list_points)):
            list_points[i].x = norm_x[i]
            list_points[i].y = norm_y[i]
        
        return list_points

    def check_angle_start_end(self, list_points):
        if len(list_points) <= 5:
            return AngleState.INCONCLUSIVE
        coor = [[point.x, point.y] for point in list_points]
        vec_start, vec_end = self.calculate_velocity_from_start_end(coor)
        angle = cosine_similarity(vec_start, vec_end)
        if abs(angle) <= math.cos(math.pi/4.5):
            return AngleState.TURN
        elif abs(angle) >= math.cos(math.pi/30):
            return AngleState.STRAIGHT
        else:
            return AngleState.INCONCLUSIVE

    def calculate_velocity_from_start_end(self, coor, skip_frame=2, count=5):
        while len(coor)/2 < skip_frame * count:
            skip_frame -= 1
        if skip_frame == 0:
            skip_frame = 1
            while len(coor)/2 < skip_frame * count:
                count -= 1
        vel_list_start = [minus_vector(coor[i], coor[skip_frame + i]) for i in range(skip_frame, skip_frame * (count + 1), skip_frame)]
        vel_list_end = [minus_vector(coor[i], coor[skip_frame + i]) for i in range(-skip_frame, -skip_frame * (count + 1), -skip_frame)]
        vec_start = [sum(row[i] for row in vel_list_start) for i in range(len(vel_list_start[0]))]
        vec_end = [sum(row[i] for row in vel_list_end) for i in range(len(vel_list_end[0]))]
        return vec_start, vec_end

    def is_between(self, a, b, c):
        # Check if slope of a to c is the same as a to b ;
        # that is, when moving from a.x to c.x, c.y must be proportionally
        # increased than it takes to get from a.x to b.x .

        # Then, c.x must be between a.x and b.x, and c.y must be between a.y and b.y.
        # => c is after a and before b, or the opposite
        # that is, the absolute value of cmp(a, b) + cmp(b, c) is either 0 ( 1 + -1 )
        #    or 1 ( c == a or c == b)
        return ((b.x - a.x) * (c.y - a.y) == (c.x - a.x) * (b.y - a.y) and 
                abs(cmp(a.x, c.x) + cmp(b.x, c.x)) <= 1 and
                abs(cmp(a.y, c.y) + cmp(b.y, c.y)) <= 1)

    def is_intersection(self, list_points):
        for i in range(1, len(list_points) - 1):
            if self.is_between(list_points[0], list_points[-1], list_points[i]):
                return True
        count_above = 0
        count_below = 0
        coor = [[point.x, point.y] for point in list_points]

        for i in range(1, len(list_points) - 1):
            v1 = minus_vector(coor[0], coor[-1])   # Vector 1
            v2 = minus_vector(coor[i], coor[-1])   # Vector 1
            xp = v1[0]*v2[1] - v1[1]*v2[0]  # Cross product
            if xp > 0:
                count_above += 1
            elif xp < 0:
                count_below += 1
        return count_below >= 3 and count_above >= 3

    def check_angle_start(self, list_points):
        if len(list_points) <= 5:
            return AngleState.INCONCLUSIVE
        coor = [[point.x, point.y] for point in list_points]
        vec_start, vec_end = self.calculate_velocity_from_start(coor)
        angle = cosine_similarity(vec_start, vec_end)
        if abs(angle) <= math.cos(math.pi/4.5):
            return AngleState.TURN
        elif abs(angle) >= math.cos(math.pi/30):
            return AngleState.STRAIGHT
        else:
            return AngleState.INCONCLUSIVE

    def calculate_velocity_from_start(self, coor, skip_frame=3, count=3):
        while len(coor)/2 < skip_frame * count:
            skip_frame -= 1
        if skip_frame == 0:
            skip_frame = 1
            while len(coor)/2 < skip_frame * count:
                count -= 1
        vel_list_start = [minus_vector(coor[0], coor[i]) for i in range(0, skip_frame * (count + 1), skip_frame)]
        vel_list_end = [minus_vector(coor[i], coor[-1]) for i in range(0, -skip_frame * (count + 1), -skip_frame)]
        vec_start = [sum(row[i] for row in vel_list_start) for i in range(len(vel_list_start[0]))]
        vec_end = [sum(row[i] for row in vel_list_end) for i in range(len(vel_list_end[0]))]
        return vec_start, vec_end

    # Tim cam doc
    def find_vertical_views(self, list_points, track_id = None):
        thres = math.cos(math.pi/(180/40)) # Cos(80)
        speed_record = {}

        is_straight = False
        is_sure = False

        vx = np.array([1, 0], dtype=np.float)
        # Remove stop points
        first_point = list_points[0]
        list_coor = [[first_point.x, first_point.y]]
        prev_point = first_point
        for point in list_points[1:]:
            if euclid_distance(prev_point, point) < 2:
                continue
            else:
                prev_point = point
                list_coor.append([point.x, point.y])
            pass
        # list_coor = [[point.x, point.y] for point in list_points]
        list_vel_vects = calculate_velocity_vector(list_coor, skip_frame=5)
        valid_vects = []
        for vector in list_vel_vects:
            if np.linalg.norm(np.array(vector)) != 0.0:
                valid_vects.append(vector)
        list_vel_vects = valid_vects
        list_angle = [abs(cosine_similarity(vx, vel_vector)) for vel_vector in list_vel_vects]
        list_angle = [angle for angle in list_angle if angle != NONE_COSINE]
        
        N = len(list_angle)
        if N < 5:
            return True, 1.11, [], speed_record

        c = 0
        for angle in list_angle:
            if angle <= thres:
                c += 1
        
        straight_percent = c/N
        if straight_percent > 0.85:
            is_straight = True
            if straight_percent > 0.9:
                is_sure = True

        c = 0
        change_angle_speed = calculate_change_angle_speed(list_angle, skip_frame=2)
        mean_speed = np.mean(change_angle_speed)
        speed_record['mean_speed'] = mean_speed
        speed_record['angle_speed'] = change_angle_speed
        for angle_speed in change_angle_speed:
            if angle_speed > 5*mean_speed:
                c += 1
        if c > 2 and is_sure == False:
            is_straight = False
        
        return is_straight, straight_percent, list_angle, speed_record       
        
        
        