import pytest
import os

from tests import DATASETS_PATH
import os.path as osp
import subprocess


def run(cmd: str):
    process = subprocess.Popen(
        cmd.split(), stderr=subprocess.PIPE, stdout=subprocess.PIPE
    )
    stdout, stderr = process.communicate()
    exit_code = process.wait()
    return exit_code, stderr


def gen_motion_map(DATAPATH):
    command = f"python scripts/data/motion_map.py {DATAPATH}"
    status, output = run(command)
    assert status == 0, f"{output}\nRun test motion map failed, please check"


def gen_nlpaug(DATAPATH):
    command = "python -m spacy download en_core_web_sm"
    status, output = run(command)
    assert status == 0, f"{output}\nRun test augment nlp failed, please check"

    command = f"python scripts/data/nlpaug_uts.py {DATAPATH}/train_tracks.json"
    status, output = run(command)
    assert status == 0, f"{output}\nRun test augment nlp failed, please check"

    command = f"python scripts/data/nlpaug_uts.py {DATAPATH}/test_queries.json"
    status, output = run(command)

    assert status == 0, f"{output}\nRun test augment nlp failed, please check"


def split(DATAPATH):
    command = f"python scripts/data/split.py {DATAPATH}/train_tracks.json"
    status, output = run(command)

    assert status == 0, f"{output}\nRun test split failed, please check"


def extract_srl(DATAPATH):
    command = f"python scripts/srl/extraction.py {DATAPATH} {DATAPATH}"
    status, output = run(command)

    assert status == 0, f"{output}\nRun test extract srl failed, please check"


def extract_srl_prep(DATAPATH):
    SRL_PATH = f"{DATAPATH}/srl"
    os.makedirs(SRL_PATH, exist_ok=True)
    os.makedirs(f"{SRL_PATH}/action", exist_ok=True)
    os.makedirs(f"{SRL_PATH}/color", exist_ok=True)
    os.makedirs(f"{SRL_PATH}/veh", exist_ok=True)

    command = f"python scripts/srl/action_prep.py {SRL_PATH} {SRL_PATH}/action"
    status, output = run(command)

    assert status == 0, f"{output}\nRun test extract srl failed, please check"
    command = f"python scripts/srl/color_prep.py {DATAPATH} {SRL_PATH} {DATAPATH}/extracted_frames {SRL_PATH}/color"
    status, output = run(command)

    assert status == 0, f"{output}\nRun test extract srl failed, please check"
    command = f"python scripts/srl/veh_prep.py {DATAPATH} {SRL_PATH} {DATAPATH}/extracted_frames {SRL_PATH}/veh"
    status, output = run(command)

    assert status == 0, f"{output}\nRun test extract srl failed, please check"


@pytest.mark.order("first")
def test_preproc(tmp_path):
    DATAPATH = osp.join(DATASETS_PATH, "meta")
    gen_motion_map(DATAPATH)
    gen_nlpaug(DATAPATH)
    split(DATAPATH)
    extract_srl(DATAPATH)
    extract_srl_prep(DATAPATH)
