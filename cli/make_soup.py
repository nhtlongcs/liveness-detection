# https://github.com/mlfoundations/model-soups

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from typing import OrderedDict


def main(args):
    config = read_json(args.config)
    device = torch.device('cuda') if args.cuda else torch.device('cpu')
    model = getattr(module_arch,
                    config['arch']['type'])(device=device,
                                            **config['arch']['args'])

    models_path = {
        # put all the trained model weight and corresponding performance here
        "model1.pth": 58.80149226295106,
        "model2.pth": 59.04275354876265,
        "model2.pth": 59.22375592753993,
        "model2.pth": 58.57664191323542
    }

    models_path = {
        k: v
        for k, v in sorted(
            models_path.items(), key=lambda item: item[1], reverse=True)
    }
    print(models_path)

    print("######## Average Soup ############")
    # average soup
    avg_soup = None
    for model_path, model_mrr in models_path.items():
        state_dict = torch.load(model_path)
        # handle model saved in DataParallel mode
        if list(state_dict['state_dict'].keys())[0].startswith('module'):
            model_state_dict = OrderedDict(
                {k[7:]: v
                 for k, v in state_dict['state_dict'].items()})
        else:
            model_state_dict = state_dict['state_dict']

        if avg_soup == None:
            avg_soup = model_state_dict
            continue

        for key in model_state_dict.keys():
            avg_soup[key] = (avg_soup[key] + model_state_dict[key]) / 2.0

    print("Saved to {}/checkpoint-soup.pth...".format(args.record_path))
    if not os.path.exists(args.record_path):
        os.makedirs(args.record_path)
    torch.save(avg_soup, '{}/checkpoint-soup-new.pth'.format(args.record_path))
    print("Done")
    print("######## Average Soup ############")


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-c',
                      '--config',
                      default=None,
                      type=str,
                      help='config file path (default: None)')
    args.add_argument('-r',
                      '--record_path',
                      default="../record/model_soup/",
                      type=str)
    args.add_argument('--cpu', dest='cuda', action='store_false')
    args.set_defaults(cuda=True)
    args = args.parse_args()

    main(args)
