# Track video generation

- These scripts are for visualizing tracks, each video will represent a full track

## Generate main tracks
```
python vis_tracks.py \
   -i <root_data_dir> \
   -o <output_dir> \
   -t <track json>
```

## Generate main tracks with relation
```
python vis_relation.py
   -i <root_data_dir> \
   -o <output_dir> \
   -t <track json> \
   -r <relation json>
   -a <auxilary tracks json>
```