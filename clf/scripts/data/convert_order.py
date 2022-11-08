import json
import sys
import os.path as osp


def json_save(data, save_path, verbose=True):
    with open(save_path, "w") as f:
        json.dump(data, f, indent=2)
    if verbose:
        print(f"Saved data to {save_path}")


# change path
data_path = sys.argv[1]
save_path = sys.argv[2]

train_tracks = json.load(open(osp.join(data_path, "train_tracks.json")))
test_tracks = json.load(open(osp.join(data_path, "test_tracks.json")))
test_queries = json.load(open(osp.join(data_path, "test_queries.json")))

count = 0
ans = []
for key in train_tracks:
    ans.append({"key": key, "order": count})
    count += 1
train_tracks_json = ans

json_save(
    data=train_tracks_json, save_path=osp.join(save_path, "train_tracks_order.json"),
)


count = 0
ans = []
for key in test_tracks:
    ans.append({"key": key, "order": count})
    count += 1
test_tracks_json = ans
json_save(
    data=test_tracks_json, save_path=osp.join(save_path, "test_tracks_order.json")
)

count = 0
ans = []
for key in test_queries:
    ans.append({"key": key, "order": count})
    count += 1
test_queries_json = ans
json_save(
    data=test_queries_json, save_path=osp.join(save_path, "test_queries_order.json"),
)
