import os
import json
import random

RATIO = 0.2
SHUFFLE = True
random.seed(1812)
import sys
from pathlib import Path

path = Path(sys.argv[1])
out_dir = path.parent / 'split'
out_dir.mkdir(exist_ok=True)
assert path.exists(), "json file not found"

with open(path) as f:
    tracks_train = json.load(f)

keys = list(tracks_train.keys())
random.shuffle(keys)

train_data = dict()
val_data = dict()
val_len = max(int(len(keys) * RATIO), 1)

for key in keys[:val_len]:
    val_data[key] = tracks_train[key]
for key in keys[val_len:]:
    train_data[key] = tracks_train[key]
out_train_path = os.path.join(out_dir, "train.json")
out_val_path = os.path.join(out_dir,"val.json")

with open(out_train_path, "w") as f, open(out_val_path, "w") as g:
    json.dump(train_data, f, indent=4)
    json.dump(val_data, g, indent=4)
