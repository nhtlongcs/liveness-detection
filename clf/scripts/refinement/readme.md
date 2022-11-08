# 1. Gather Video (Track) result

Gather relation/classification/action (stop-turn) results from different modules for each track to a single file.

**Input:**

- data/result/test_relation.json
- data/result/test_neighbors.json
- data/result/test_stop_turn.json

**Output:**

- save folder (example in data/result/test_relation)

<!-- Outdir at 'data/result/test_relation' -->

```bash
python external/refinement/parse_relation.py
```

# 2. Refinement

Input:

- srl.csv (output from script/srl/extract_postproc.py)
- test_track_dir : output of step 2
- retrieval result: submission format

Output:

- final retrieval result: submission format

```bash
python external/refinement/main.py
```
