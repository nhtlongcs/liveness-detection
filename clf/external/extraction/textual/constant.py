import os, json 
import os.path as osp


LIST_REDUNDANT_VEHICLES = ['volvo', 'chevrolet', 'vehicle', 'car']
FOLLOW = "follow"
FOLLOW_BY = "followed by"

OPPOSITE = {
    FOLLOW: FOLLOW_BY,
    FOLLOW_BY: FOLLOW
}

HAS_FOLLOW = 2
NO_FOLLOW = -1
NO_CONCLUSION = 1

