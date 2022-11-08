import torch 
import numpy as np 
import cv2 


def tensor2numpy(extract_preds, feats):
    extract_preds.pred_boxes = extract_preds.pred_boxes.detach().cpu().numpy()
    extract_preds.pred_classes = extract_preds.pred_classes.detach().cpu().numpy()
    
    feats = feats.detach().cpu().numpy()
    
    return extract_preds, feats
