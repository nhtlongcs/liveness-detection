"""
Modified from https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.4/tools/program.py
"""
from typing import Optional
from argparse import ArgumentParser, RawDescriptionHelpFormatter
import yaml
import json
from src.utils.loading import load_yaml


class Config(dict):
    """Single level attribute dict, NOT recursive"""

    def __init__(self, yaml_path):
        super(Config, self).__init__()

        config = load_yaml(yaml_path)
        super(Config, self).update(config)

    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError("object has no attribute '{}'".format(key))

    def save_yaml(self, path):
        print(f"Saving config to {path}...")
        with open(path, "w") as f:
            yaml.dump(dict(self), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load_yaml(cls, path):
        print(f"Loading config from {path}...")
        return cls(path)

    def __repr__(self) -> str:
        return str(json.dumps(dict(self), sort_keys=False, indent=4))


class Opts(ArgumentParser):
    def __init__(self, cfg: Optional[str] = None):
        super(Opts, self).__init__(formatter_class=RawDescriptionHelpFormatter)
        self.add_argument(
            "-c", "--config", default=cfg, help="configuration file to use"
        )
        self.add_argument(
            "-o", "--opt", nargs="+", help="override configuration options"
        )

    def parse_args(self, argv=None):
        args = super(Opts, self).parse_args(argv)
        assert args.config is not None, "Please specify --config=configure_file_path."
        args.opt = self._parse_opt(args.opt)

        config = Config(args.config)
        config = self.override(config, args.opt)
        return config

    def _parse_opt(self, opts):
        config = {}
        if not opts:
            return config
        for s in opts:
            s = s.strip()
            k, v = s.split("=")
            config[k] = yaml.load(v, Loader=yaml.Loader)
        return config

    def override(self, global_config, overriden):
        """
        Merge config into global config.
        Args:
            config (dict): Config to be merged.
        Returns: global config
        """
        print("Overriding configurating")
        for key, value in overriden.items():
            if "." not in key:
                if isinstance(value, dict) and key in global_config:
                    global_config[key].update(value)
                else:
                    if key in global_config.keys():
                        global_config[key] = value
                    print(f"'{key}' not found in config")
            else:
                sub_keys = key.split(".")
                assert (
                    sub_keys[0] in global_config
                ), "the sub_keys can only be one of global_config: {}, but get: {}, please check your running command".format(
                    global_config.keys(), sub_keys[0]
                )
                cur = global_config[sub_keys[0]]
                for idx, sub_key in enumerate(sub_keys[1:]):
                    if idx == len(sub_keys) - 2:
                        if sub_key in cur.keys():
                            cur[sub_key] = value
                        else:
                            print(f"'{key}' not found in config")
                    else:
                        cur = cur[sub_key]
        return global_config
