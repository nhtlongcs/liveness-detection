from __future__ import division

from models import Darknet
from utils.utils import load_classes, weights_init_normal
from utils.datasets import ListDataset
from utils.parse_config import parse_data_config
from validate import evaluate

from terminaltables import AsciiTable

import os
import time
import argparse

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# from utils.logger import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",
                        type=int,
                        default=100,
                        help="number of epochs")
    parser.add_argument("--batch_size",
                        type=int,
                        default=6,
                        help="size of each image batch")
    parser.add_argument("--gradient_accumulations",
                        type=int,
                        default=2,
                        help="number of gradient accums before step")
    parser.add_argument("--model_def",
                        type=str,
                        default="config/yolov3_mask.cfg",
                        help="path to model definition file")
    parser.add_argument("--data_config",
                        type=str,
                        default="config/mask_dataset.data",
                        help="path to data config file")
    parser.add_argument("--pretrained_weights",
                        type=str,
                        default="weights/yolov3.weights",
                        help="if specified starts from checkpoint model")
    parser.add_argument(
        "--n_cpu",
        type=int,
        default=8,
        help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size",
                        type=int,
                        default=416,
                        help="size of each image dimension")
    parser.add_argument("--checkpoint_interval",
                        type=int,
                        default=1,
                        help="interval between saving model weights")
    parser.add_argument("--evaluation_interval",
                        type=int,
                        default=1,
                        help="interval evaluations on validation set")
    parser.add_argument("--compute_map",
                        default=True,
                        help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training",
                        default=True,
                        help="allow for multi-scale training")
    opt = parser.parse_args()
    print(opt)

    # logger = Logger("logs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    model.apply(weights_init_normal)

    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights)

    # Get dataloader
    dataset = ListDataset(train_path,
                          augment=True,
                          multiscale=opt.multiscale_training)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    optimizer = torch.optim.Adam(model.parameters())

    # to get mAP
    to_get_mAP = None

    for epoch in range(opt.epochs):
        model.train()
        start_time = time.time()
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = Variable(imgs.to(device))
            targets = \
                Variable(targets.to(device), requires_grad=False)

            loss, outputs = model(imgs, targets)
            loss.backward()

            if batches_done % opt.gradient_accumulations:
                optimizer.step()
                optimizer.zero_grad()

            log_str = "---- [Epoch %d/%d, Batch %d/%d] ----" % (
                epoch, opt.epochs, batch_i, len(dataloader))
            log_str += f"Total loss {loss.item()}"
            print(log_str)

            model.seen += imgs.size(0)

        if epoch % opt.evaluation_interval == 0:
            try:
                print("\n---- Evaluating Model ----")
                # Evaluate the model on the validation set
                precision, recall, AP, f1, ap_class = evaluate(
                    model,
                    path=valid_path,
                    iou_thres=0.5,
                    conf_thres=0.5,
                    nms_thres=0.5,
                    img_size=opt.img_size,
                    batch_size=4,
                )
                evaluation_metrics = [
                    ("val_precision", precision.mean()),
                    ("val_recall", recall.mean()),
                    ("val_mAP", AP.mean()),
                    ("val_f1", f1.mean()),
                ]
                # logger.list_of_scalars_summary(evaluation_metrics, epoch)

                # Print class APs and mAP
                ap_table = [["Index", "Class name", "AP"]]
                for i, c in enumerate(ap_class):
                    ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
                print(AsciiTable(ap_table).table)
                print(f"---- mAP {AP.mean()}")
                to_get_mAP = AP.mean()
            except:
                to_get_mAP = 999999999999

        if epoch % opt.checkpoint_interval == 0:
            torch.save(
                model.state_dict(),
                "checkpoints/23-04-2020__02-35/yolov3_ckpt_{0}__'{1}'__'{2}'.pth"
                .format(epoch, loss.item(), to_get_mAP))
