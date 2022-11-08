import numpy as np

from external.detect_stop_turn.utils import (
    xyxy_to_xywh, cal_distance
)

class StopDetector(object):
    def __init__(self, k=5, delta=5, alpha=0.15) -> None:
        """[summary]

        Args:
            k ([type]): skip_frame used to calculate velocity at each timestamp
            delta ([type]): param used to smoothing velocity magnitude
            alpha ([type]): param used to determine stop event
        """
        super().__init__()
        self.skip_frame = k
        self.delta = delta
        self.alpha = alpha

    def process(self, list_boxes: list):
        """Determine target vehicle stops or not
        Args:
            list_boxes (list): of xyxy box
        Returns:
            result (bool)
        """
        list_boxes = [xyxy_to_xywh(box) for box in list_boxes]
        N = len(list_boxes)
        distances = [
            cal_distance(list_boxes[i], list_boxes[i+self.skip_frame]) for i in range(N-self.skip_frame)
        ]
        distances = self.smooth(distances)
        mean_distance = np.mean(distances)
        # print(f'mean distance: {mean_distance}')
        # print(distances)
        for i in range(N-self.skip_frame):
            if (distances[i] < mean_distance*self.alpha):
                return True
        
        return False

    def smooth(self, distances: list):
        N = len(distances)
        if N <= self.delta:
            return distances

        res = [0]*N
        for i in range(N-self.delta):
            s = 0
            for j in range(self.delta):
                s += distances[i+j]
            res[i] = s/self.delta
        
        for i in range(N-self.delta, N):
            res[i] = res[N-self.delta-1]
        
        return res

