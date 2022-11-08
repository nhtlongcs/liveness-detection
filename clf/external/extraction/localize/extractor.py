from PIL import Image 
import torch 
import cv2

import detectron2 
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

import detectron2.data.transforms as T

class Extractor(object):
    def __init__(self, cfg, threshold=0.5):
        # init model
        model = build_model(cfg)
        checkpointer = DetectionCheckpointer(model)
        checkpointer.load(cfg.MODEL.WEIGHTS)
        self.model = model

        self.infer_aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], 
            cfg.INPUT.MAX_SIZE_TEST
        )
    
    def set_eval(self):
        self.model.eval()

    def __call__(self, inp):
        return self.backbone(inp)

    def inference_image(self, cv_img, get_feats=False):
        """Inference single cv2 image
        Args:
            get_feats: return bboxes' feature or not

        Return:
            tuple: (pred_instances, feats)
            - pred_instances: .pred_boxes [N, 4] (xyxy), .pred_classes: [N]
            - feats: [N, 1024] torch Tensor, feature of N detected boxes
        """
        with torch.no_grad():
            H, W, C = cv_img.shape
            img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            img = self.infer_aug.get_transform(img).apply_image(img)
            img = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
            inputs = [{
                'image': img, 'height': H, 'width': W
            }]
            
            feats=None

            if get_feats:
                images = self.model.preprocess_image(inputs)
                features = self.model.backbone(images.tensor)
                proposals, _ = self.model.proposal_generator(images, features, None) #RPN
                features_ = [features[f] for f in self.model.roi_heads.box_in_features]

                box_features = self.model.roi_heads.box_pooler(features_, [x.proposal_boxes for x in proposals])
                box_features = self.model.roi_heads.box_head(box_features)  # features of all 1k candidates
                predictions = self.model.roi_heads.box_predictor(box_features)
                pred_instances, pred_inds = self.model.roi_heads.box_predictor.inference(predictions, proposals)
                pred_instances = self.model.roi_heads.forward_with_given_boxes(features, pred_instances)

                # output boxes, masks, scores, etc
                pred_instances = self.model._postprocess(pred_instances, inputs, images.image_sizes)  # scale box to orig size
                # features of the proposed boxes
                feats = box_features[pred_inds]

            else:
                pred_instances = self.model(inputs)

            return pred_instances[0]['instances'], feats
            

