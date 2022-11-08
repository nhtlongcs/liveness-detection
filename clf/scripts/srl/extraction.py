# Usage
import json
from external.extraction.textual import SRL
from pathlib import Path
import sys


def extract_textual_metadata(json_path, out_dir):
    assert Path(json_path).exists(), "json file not found, ensure the path {} is correct".format(json_path)
    srl_test = SRL(path=json_path)
    filename = "srl_" + Path(json_path).name
    out_path = Path(out_dir)
    out_path.mkdir(exist_ok=True)
    out_path = out_path / filename
    with open(str(out_path), "w") as f:
        ans_test = srl_test.extract_data(srl_test.data)
        json.dump(ans_test, f, indent=2)
        f.close()


def main():
    data_dir = Path(sys.argv[1])
    meta_data_dir = Path(sys.argv[2]) / "srl"
    extract_textual_metadata(data_dir / "train_tracks.json", meta_data_dir)
    extract_textual_metadata(data_dir / "test_queries.json", meta_data_dir)


if __name__ == "__main__":
    main()
