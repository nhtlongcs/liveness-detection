# AIC2022-VER (WIP)

Repo of our source code at the AI City challenge 2022, Track 2: natural language-based vehicle retrieval task. (updating)

For reproducibility, we also provide a colab notebook [![notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nhtlongcs/AIC2022-VER/blob/main/guides/reproduce.ipynb) that contains the code for reproducing the results.

## Development environment

Before using this repo, please use the environment setup as below.

Pre-installation

Install conda according to the instructions on the homepage
Before installing the repo, we need to install the CUDA driver version >=10.2.

```
$ conda env create -f environment.yml
$ conda activate hcmus
$ pip install -r requirements.txt
$ pip install -e .
```

## Prepare data

Create a symbolic link to the data directory in the data directory of the project.

<!-- MAC OSX FAQ https://discussions.apple.com/thread/7423765 -->

```bash
$ cd /Users/your_short_username/path/to/where/you/want/to/put/the/symlink
$ ln -s /Volumes/HDD_name/path/to/where/you/are/storing/the/moved/files    symbolic_link_name_you_want_to_use
```

Ensure your data folder structure as same as our `data_sample` before running the code.

```bash
$ ./tools/extract_vdo2frms_AIC.sh ./data/AIC22_Track2_NL_Retrieval/ ./data/meta/extracted_frames/
$ cp ./data/AIC22_Track2_NL_Retrieval/*.json ./data/meta/
$ ./tools/preproc_motion.sh ./data/meta
$ ./tools/preproc_srl.sh ./data/meta
```

For detail, please take a look at [extract data notebook](guides/extract_data.ipynb)

For testing purpose, you can use the command above with data_dir is `./data_sample/meta`

Reading detail document of preprocessing part can be found in the [srl part](external/extraction/README.md) and [basic part](scripts/data/README.md) (adapted from hcmus team and alibaba team source code).

## Inference

We provide a simple inference script for inference purpose.
With artifacts/ is the directory where you store the trained classification model.

```bash
$ ./tools/infer.sh ./data/meta/
```

For detail, please take a look at `Predictor` class in `src/predictor.py` or [inference notebook](guides/inference.ipynb)

## Training

Updating

## Deployment (not working yet)

For deployment/training purpose, docker is an ready-to-use solution.

To build docker image:

```bash
$ cd <this-repo>
$ DOCKER_BUILDKIT=1 docker build -t aic22:latest .
```

To start docker container:

```bash
$ docker run --rm --name aic-t2 --gpus device=0 --shm-size 16G -it -v $(pwd)/:/home/workspace/src/ aic22:latest /bin/bash
```

With device is the GPU device number, and shm-size is the shared memory size (should be larger than the size of the model).

To attach to the container:

```bash
$ docker attach aic-t2
```

## Contribution guide

If you want to contribute to this repo, please follow steps below:

1. Fork your own version from this repository
1. Checkout to another branch, e.g. `fix-loss`, `add-feat`.
1. Make changes/Add features/Fix bugs
1. Add test cases in the `test` folder and run them to make sure they are all passed (see below)
1. Create and describe feature/bugfix in the PR description (or create new document)
1. Push the commit(s) to your own repository
1. Create a pull request on this repository

```bash
pip install pytest
python -m pytest tests/
```

Expected result:

```bash
============================== test session starts ===============================
platform darwin -- Python 3.7.12, pytest-7.1.1, pluggy-1.0.0
rootdir: /Users/nhtlong/workspace/aic/aic2022
collected 10 items

tests/test_args.py ...                                                     [ 30%]
tests/test_utils.py .                                                      [ 40%]
tests/uts/test_dataset.py .                                                [ 50%]
tests/uts/test_eval.py .                                                   [ 60%]
tests/uts/test_extractor.py ...                                            [ 90%]
tests/uts/test_model.py .                                                  [100%]
```
