# run `python -m spacy download en_core_web_sm` before running this script
import json
import sys

import spacy
from tqdm import tqdm

with open(sys.argv[1]) as f:
    train = json.load(f)

nlp = spacy.load("en_core_web_sm")
track_ids = list(train.keys())
for id_ in tqdm(track_ids):
    for nl_key in ['nl','nl_other_views']:
        new_text = ""
        for i, text in enumerate(train[id_][nl_key]):
            doc = nlp(text)
            for chunk in doc.noun_chunks:
                nb = chunk.text
                if "subjects" not in train[id_].keys():
                    train[id_]["subjects"] = []
                train[id_]["subjects"].append(nb)
                break

with open(sys.argv[1].split(".")[-2] + "_nlpaug_v2.json", "w") as f:
    json.dump(train, f, indent=4)
