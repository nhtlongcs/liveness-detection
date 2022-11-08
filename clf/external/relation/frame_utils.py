# Example path  "./train/S01/c003/img1/000028.jpg"

def get_frame_ids_by_names(frame_names):
    """
    Get frame id by its names
    """
    frame_ids = [
        int(i.split('/')[-1][:-4])
        for i in frame_names
    ]
    return frame_ids

def get_camera_id_by_name(frame_name):
    """
    Get camera id by its name
    """
    camera_id = '/'.join(frame_name.split('/')[2:4])
    return camera_id