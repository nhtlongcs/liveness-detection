import cv2
import torch
import random
import torchvision
import numpy as np
from PIL import Image
from typing import Any, List, Optional, Tuple, Union
from .colors import color_list
from .utils import (draw_text_cv2)


class Visualizer():
    r"""Visualizer class that do all the visualization stuffs
        """

    def __init__(self):
        self.image: Optional[np.ndarray] = None
        self.class_names = None

    def set_image(self, image: np.ndarray) -> None:
        """
        Set the current image
        """
        self.image = image

        if self.image.dtype == 'uint8':
            self.image = self.image / 255.0

    def set_classnames(self, class_names: List[str]) -> None:
        self.class_names = class_names

    def get_image(self) -> np.ndarray:
        """
        Get the current image
        """
        if self.image.dtype == 'uint8':
            self.image = np.clip(self.image, 0, 255)
        elif self.image.dtype == 'float32' or self.image.dtype == 'float64':
            self.image = np.clip(self.image, 0.0, 1.0)
            self.image = (self.image * 255).astype(np.uint8)

        return self.image

    def save_image(self, path: str) -> None:
        """
        Save the image
        """
        cv2.imwrite(path, self.get_image()[:, :, ::-1])

    def draw_label(self,
                   label: str,
                   font: Any = cv2.FONT_HERSHEY_SIMPLEX,
                   fontScale: int = 2,
                   fontColor: Tuple = (0, 0, 1),
                   thickness: int = 3,
                   outline: Tuple = (0, 0, 0),
                   offset: int = 50):
        """
        Draw text on the image then return

        font: Any  
            cv2 font style            
        fontScale: int  
            size of text       
        fontColor: Tuple  
            color of text     
        thickness: int     
            text thickness    
        outline:   Tuple     
            outline the text, leave None to have disable  
        offset:    `int`
            offset to position of text from the bottom
        """

        assert self.image is not None
        if self.class_names is not None:
            label = self.class_names[label]

        h, w, c = self.image.shape
        white_canvas = np.ones((h + offset, w, c))
        white_canvas[:h, :w, :c] = self.image
        bottomLeftCornerOfText = (int(w / 6), h + 10)

        draw_text_cv2(white_canvas,
                      str(label),
                      bottomLeftCornerOfText,
                      fontFace=font,
                      fontScale=fontScale,
                      color=fontColor,
                      outline_color=outline,
                      thickness=thickness)

        self.image = white_canvas.copy()

    def draw_bbox(self, boxes, labels=None) -> None:
        assert self.image is not None

        tl = int(round(0.001 * max(self.image.shape[:2])))  # line thickness

        tup = zip(boxes, labels) if labels is not None else boxes

        for item in tup:
            if labels is not None:
                box, label = item
                color = color_list[int(label)]
            else:
                box, label = item, None

            coord = [box[0], box[1], box[2], box[3]]
            c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[2]),
                                                      int(coord[3]))
            cv2.rectangle(self.image, c1, c2, color, thickness=tl * 2)

            if label is not None:
                if self.class_names is not None:
                    label = self.class_names[label]
                tf = max(tl - 2, 1)  # font thickness
                s_size = cv2.getTextSize(f'{label}',
                                         0,
                                         fontScale=float(tl) / 3,
                                         thickness=tf)[0]
                c2 = c1[0] + s_size[0] + 15, c1[1] - s_size[1] - 3
                cv2.rectangle(self.image, c1, c2, color, -1)  # filled
                cv2.putText(self.image,
                            f'{label}', (c1[0], c1[1] - 2),
                            0,
                            float(tl) / 3, [0, 0, 0],
                            thickness=tf,
                            lineType=cv2.FONT_HERSHEY_SIMPLEX)

    def _tensor_to_numpy(self, image: torch.Tensor) -> np.ndarray:
        """
        Convert torch image to numpy image (C, H, W) --> (H, W, C)
        """
        return image.numpy().squeeze().transpose((1, 2, 0))

    def make_grid(self,
                  batch: List[torch.Tensor],
                  nrow: Optional[int] = None,
                  normalize: bool = False) -> torch.Tensor:
        """
        Make grid from batch of image
            batch: `List[torch.Tensor]`
                batch of tensor image with shape (C,H,W)
            nrow: `Optional[int]`
                width size of grid
            normalize: `bool`
                whether to normalize the grid in range [0,1]
            return: `torch.Tensor`
                grid image with shape [H*ncol, W*nrow, C]
        """

        if nrow is None:
            nrow = int(np.ceil(np.sqrt(len(batch))))

        batch = torch.stack(batch, dim=0)  # (N, C, H, W)
        grid_img = torchvision.utils.make_grid(batch,
                                               nrow=nrow,
                                               normalize=normalize)

        return grid_img.permute(1, 2, 0)

    def denormalize(self,
                    image: Union[torch.Tensor, np.ndarray],
                    mean: List[float] = [0.485, 0.456, 0.406],
                    std: List[float] = [0.229, 0.224, 0.225]) -> np.ndarray:
        """
        Denormalize an image and return
        image: `torch.Tensor` or `np.ndarray`
            image to be denormalized
        """
        mean = np.array(mean)
        std = np.array(std)

        if isinstance(image, torch.Tensor):
            img_show = self._tensor_to_numpy(image)
        else:
            img_show = image.copy()

        img_show = (img_show * std + mean)
        img_show = np.clip(img_show, 0, 1)
        return img_show
