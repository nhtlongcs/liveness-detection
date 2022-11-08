# Streamlit: Visualization is so lit!

## Install 
- Install [Streamlit](https://docs.streamlit.io/en/stable/)
```
pip install streamlit
```

## Data preparation
- Generate track videos using scripts in `tools/visualization/video_gen`
- Prepare required files which are specified in `constants.py` and change paths. 


- Folder structure
```
this repo
└─── data
      └───AIC22_Track2_NL_Retrieval
      │   └───train
      │   └───validation
      └───meta  
      │   └───bk_map
      │   └───extracted_frames                  # generated from `tools/extract_vdo2frms_AIC.sh`
      │   └───motion_map
      │   └───split
      │   └───action                            # generated from `scripts/action/stop_turn_det.py`
      │   │   train_stop_turn.json
      │   │   test_stop_turn.json
      │   └───relation                          # generated from `tools/preproc_relation.sh`
      │   │   train_tracks_relation.json
      │   │   test_tracks_relation.json
      │   └───srl                               # generated from `tools/preproc_srl.sh`
      │   │   srl_train_tracks.json
      │   │   srl_test_queries.json
      │   └───track_visualization               # generated from `tools/visualization/video_gen`
      │   │   └───relation 
      │   │   │   └───test-convert 
      │   │   │   └───train-val-convert 
      │   train_tracks.json
      │   test_tracks.json
      │   test_queries.json
      └───results                               # generated from `tools/predict_srl.sh`
            └───classification
                  └───neighbors
                  │   color_prediction.json
                  │   vehicle_prediction.json
                  └───test_tracks
                  │   color_prediction.json
                  │   action_prediction.json
                  │   vehicle_prediction.json

```

## How to run
- To run streamlit with arguments, use `--` before flags
- To visualize prediction before submission 
```
streamlit run app.py -- \
   -i <root data dir> \
   -s <["train"] or ["test"]> \
   --result_folder <folder contains submission-ready json files>
```