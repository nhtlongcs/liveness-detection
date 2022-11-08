# SRL Handler Module

This module handles the following tasks:

- Read the output result from SRL Extraction step.
- Extract vehicle type, color, action label for training tracks
- Produce vehicle boxes used for Classifier module.

## Module organization

```bash
|- this-repo
    |- data
    |- external
        |- extraction
            |- heuristic
            |- localize (not working yet)
            |- textual
    |- scripts
        |- srl
            |- veh-prep.py
            |- color-prep.py
            |- action-prep.py
            |- README.md
            |- ...
```

## Run extraction

Extract train and test's queries into separated parts following the English PropBank Semantic Role Labeling rules.

```
$ python scripts/srl/extraction.py <input_data_path> <output_metadata_srl_path>
```

Make a output folder named `srl` in the same directory of the input data path. Include 'action', 'color' and 'vehicle' folders.

```
$ python scripts/srl/action_prep.py <srl_dir> <srl_dir>/action

$ python scripts/srl/color_prep.py <data_inp_path> <srl_dir> <data_inp_path>/extracted_frames <srl_dir>/color

$ python scripts/srl/veh_prep.py <data_inp_path> <srl_dir> <data_inp_path>/extracted_frames <srl_dir>/veh
```
