## generate motion map

```
python scripts/data/motion_map.py $DATAPATH
```

## Generate augment data for training (Optional)

```
# python -m spacy download en_core_web_sm # uncomment if you need to install spacy
python scripts/data/nlpaug_uts_v2.py $DATAPATH/train_tracks.json
python scripts/data/nlpaug_uts_v2.py $DATAPATH/test_queries.json
```

## Split data, train and test data into train and test data (Optional). By running the following commands, you can split the data into train and test data in same folder.

```
python scripts/data/split.py $DATAPATH/train_tracks.json
```
