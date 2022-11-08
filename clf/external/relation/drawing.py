import cv2
from .bb_utils import *

def draw_arrow(image, start_point, end_point, color):
    start_point = tuple(start_point)
    end_point = tuple(end_point)
    image = cv2.line(image, start_point, end_point, color, 3)
    image = cv2.circle(image, end_point, 8, color, -1)
    return image

def draw_start_last_points(ori_im, start_point, last_point, color=(0, 255, 0)):
    return draw_arrow(ori_im, start_point, last_point, color)

def draw_one_box(img, box, key=None, color=None, line_thickness=None):
    tl = line_thickness or int(round(0.001 * max(img.shape[0:2])))  # line thickness

    coord = [box[0], box[1], box[2], box[3]]
    c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3]))
    img = cv2.rectangle(img, c1, c2, color, thickness=tl*2)
    if key is not None:
        header = f'{key}'
        tf = max(tl - 2, 1)  # font thickness
        t_size = cv2.getTextSize(f'{key}', 0, fontScale=float(tl) / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0] + 15, c1[1] - t_size[1] - 3
        img = cv2.rectangle(img, c1, c2, color, -1)  # filled
        img = cv2.putText(img, header, (c1[0], c1[1] - 2), 0, float(tl) / 3, [0, 0, 0],
                    thickness=tf, lineType=cv2.FONT_HERSHEY_SIMPLEX)    
    return img

def visualize_one_frame(img, df):
    """
    Draw boxes for 1 frame
    """    
    anns = [
        i for i in zip(
            df.track_id, 
            df.x1, 
            df.x2, 
            df.y1, 
            df.y2, 
            df.color)
    ]

    for (track_id, x1, x2, y1, y2, color) in anns:
        box = [x1, y1, x2, y2]
        img = draw_one_box(
                img, 
                box, 
                key=f'id: {track_id}',
                color=color)
        
    return img