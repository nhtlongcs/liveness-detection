import detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg

def get_default_cfg(model_id: str = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_id))
    cfg.MODEL.MODEL_ID = model_id
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_id)
    
    return cfg
