# Relation graph extraction

## Preparation
- Download raw dataset from AIC22 and processed metadata by us
- Structure the data folder as follow:

```
this repo
│   data
└───AIC22_Track2_NL_Retrieval
│   └───train
│   └───validation
└───meta  
│   └───bk_map
│   └───extracted_frames
│   └───motion_map
│   └───relation
│   └───split
│   └───srl
│   └───track_visualization   
```

## Relation graph

- (**IMPORTANT**) Changes constants in `scripts/relation/constants.py`
- Run `scripts/relation/gen_aux_tracks.py` to generate auxiliary tracks from tracking annotations.
- Run `scripts/relation/gen_neighbor_mapping` to determine which auxiliary tracks related to the which in the main tracks
- Run `scripts/relation/refine_neighbor.py` to determine the relation between the main tracks and all its neighbors.