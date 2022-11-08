import numpy as np

def thresholding(preds, thres):
    preds[np.where(preds >= thres)] = 1.0
    preds[np.where(preds < thres)] = 0.0
    pass

def thresholding_prediction(preds, thres = 0.3):
    res = np.copy(preds)
    final_thres = None
    if isinstance(thres, list):
        thres = sorted(thres, reverse=True)    
        for score in thres:
            if np.max(res) >= score:
                final_thres = score 
                break
        if final_thres is None:
            final_thres = np.max(res)
        
    if isinstance(thres, float):
        final_thres = thres

    thresholding(res, final_thres)
    return res

def check_label_by_predict(label, chosen_idx):
    for i in chosen_idx:
        if label[i] > 0:
            return True
    return False

def calculate_score(label: np.ndarray, preds: np.ndarray):
    tp = (label*preds).sum()
    recall = tp/(label.sum())

    pred_p = preds.sum()
    if pred_p == 0:
        precision = 0.0
    else:
        precision = tp/(preds.sum())

    if (precision == 0) and (recall == 0):
        f2 = 0.0
    else:
        f2 = (2*precision*recall)/(precision + recall)

    score = {'tp': tp, 'precision': precision, 'recall': recall, 'f2': f2}
    return score
    
def score_att(label, predict, mode='color'):
    """
    Args:
        label ([type]): array of label from text (1, 0 values only)
        predict ([type]): array of prediction from classifier (1, 0 values only)
        mode (str, optional): 'vehicle' or 'color'.

    Returns:
        score (float)
    """
    MAX_POINT = 1.5
    MIN_POINT = 0.0
    if mode == 'color':
        MAX_POINT = 1.25
        MIN_POINT = 0 #-1.25
    else:
        MAX_POINT = 1.25
        MIN_POINT = 0 #-1.2

    if np.sum(label) == len(label):
        return MAX_POINT

    if np.sum(predict) == 0:
        return MAX_POINT
        
    # label: from query, predict: classifier from track
    score = calculate_score(np.array(label), np.array(predict))
    
    if score['tp'] > 0:
        return MAX_POINT
    else:
        return MIN_POINT
