import wandb
import os
import cv2
import json
import numpy as np
import os.path as osp
from pytorch_lightning.callbacks import Callback
from torchvision.transforms import functional as TFF
from src.utils.visualization.visualizer import Visualizer

class VisualizerCallback(Callback):

    def __init__(self, motion_path, gt_json_path, query_results_json) -> None:
        super().__init__()
        self.motion_path = motion_path
        self.gt_json_path = gt_json_path
        self.query_results_json = query_results_json
        self.visualizer = Visualizer()

    def on_sanity_check_start(self, trainer, pl_module):
        """Called when the validation batch ends."""
        val_batch = next(iter(pl_module.val_dataloader()))
        train_batch = next(iter(pl_module.train_dataloader()))
        print("Visualizing dataset...")
        self.visualize_instance_gt(train_batch, val_batch, trainer.logger)
        self.visualize_motion_gt(train_batch, val_batch, trainer.logger)
        self.visualize_text_gt(train_batch, val_batch, trainer.logger)

    def on_validation_epoch_start(self, trainer, pl_module):
        # Save mapping for visualization
        os.makedirs('temps', exist_ok=True)

    def on_validation_epoch_end(self, trainer, pl_module):

        """
        After finish validation
        """

        print('Visualizing predictions...')

        with open(self.query_results_json, 'r') as f:
            results = json.load(f)
        mapping = list(results.keys())

        indexes = [i for i in np.random.randint(0, len(mapping), size=5)]
        my_table = []
        columns = ['id', 'queries', 'prediction', 'groundtruth']
        for index in indexes:
            query_id = mapping[index]
            pred_ids = [i for i in results[query_id]['pred_ids']]
            target_ids = [i for i in results[query_id]['target_ids']]
            colors = [[0,1,0] if id in target_ids else [1,0,0] for id in pred_ids]
            scores = results[query_id]['scores']
            
            # Predictions
            pred_batch = []
            pred_images = [show_motion(i, self.motion_path) for i in pred_ids]
            for idx, (img_show, prob, color) in enumerate(zip(pred_images, scores, colors)):
                self.visualizer.set_image(img_show)
                self.visualizer.draw_label(
                    f"C: {prob:.4f}", 
                    fontColor=color, 
                    fontScale=0.5,
                    thickness=1,
                    outline=None,
                    offset=30
                )
                pred_img = self.visualizer.get_image()
                img_show = TFF.to_tensor(pred_img)
                pred_batch.append(img_show)
            pred_grid_img = self.visualizer.make_grid(pred_batch)

            # Ground truth
            gt_batch = []
            target_images = [show_motion(i, self.motion_path) for i in target_ids]
            for idx, img_show in enumerate(target_images):
                img_show = TFF.to_tensor(img_show)
                gt_batch.append(img_show)
            gt_grid_img = self.visualizer.make_grid(gt_batch)

            # Query texts
            texts = show_texts(query_id, self.gt_json_path)
            query = "\n".join([f"{i}. "+text for i, text in enumerate(texts)])
            record = [
                query_id, 
                query, 
                wandb.Image(pred_grid_img.permute(2,0,1)), 
                wandb.Image(gt_grid_img.permute(2,0,1))
            ]
            my_table.append(record)

        trainer.logger.log_table(
            "val/prediction", data=my_table, columns=columns
        )

    def visualize_instance_gt(self, train_batch, val_batch,  logger):
        """
        Visualize dataloader for sanity check 
        """

        # Train batch
        images = train_batch["images"]

        batch = []
        for idx, inputs in enumerate(images):
            img_show = self.visualizer.denormalize(inputs)
            img_cam = TFF.to_tensor(img_show)
            batch.append(img_cam)
        grid_img = self.visualizer.make_grid(batch)

        logger.log_image(
            key='sanity/train_instances', 
            images=[wandb.Image(grid_img.permute(2,0,1))], 
        )

        # Validation batch
        images = val_batch["images"]

        batch = []
        for idx, inputs in enumerate(images):
            img_show = self.visualizer.denormalize(inputs)
            img_cam = TFF.to_tensor(img_show)
            batch.append(img_cam)
        grid_img = self.visualizer.make_grid(batch)

        logger.log_image(
            key='sanity/val_instances', 
            images=[wandb.Image(grid_img.permute(2,0,1))],
        )


    def visualize_motion_gt(self, train_batch, val_batch, logger):
        """
        Visualize dataloader for sanity check 
        """
        # Train batch
        images = train_batch["motions"]

        batch = []
        for idx, inputs in enumerate(images):
            img_show = self.visualizer.denormalize(inputs)
            img_cam = TFF.to_tensor(img_show)
            batch.append(img_cam)
        grid_img = self.visualizer.make_grid(batch)

        logger.log_image(
            key='sanity/train_motions', 
            images=[wandb.Image(grid_img.permute(2,0,1))],
        )

        # Validation batch
        images = val_batch["motions"]

        batch = []
        for idx, inputs in enumerate(images):
            img_show = self.visualizer.denormalize(inputs)
            img_cam = TFF.to_tensor(img_show)
            batch.append(img_cam)
        grid_img = self.visualizer.make_grid(batch)

        logger.log_image(
            key='sanity/val_motions', 
            images=[wandb.Image(grid_img.permute(2,0,1))], 
        )

    def visualize_text_gt(self, train_batch, val_batch, logger):
        """
        Visualize dataloader for sanity check 
        """

        # Train batch
        texts = train_batch["texts"]
        batch = "\n".join([f"{i}. "+text for i, text in enumerate(texts)])

        logger.log_text(
          key="sanity/train_queries", 
          columns=['texts'],
          data=[[batch]])

        # Validation batch
        texts = val_batch["texts"]
        batch = "\n".join([f"{i}. "+text for i, text in enumerate(texts)])

        logger.log_text(
          key="sanity/val_queries", 
          columns=['texts'],
          data=[[batch]])


def show_motion(track_id, motion_dir):
    motion_image = cv2.imread(osp.join(motion_dir, track_id+'.jpg'))
    motion_image = cv2.resize(motion_image, (200,200))
    motion_image = cv2.cvtColor(motion_image, cv2.COLOR_BGR2RGB)
    return motion_image

def show_texts(track_id, json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    texts = data[track_id]['nl']
    return texts