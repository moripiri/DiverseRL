import argparse
import ast
from typing import Any, Dict

import yaml


def set_config(args: argparse.Namespace) -> Dict[str, Any]:
    if args.config_path is not None:
        with open(args.config_path, "r") as f:
            config = yaml.safe_load(f)
            config['config_path'] = args.config_path
    else:
        config = vars(args)

    return config

class StoreDictKeyPair(argparse.Action):
    """
    ref: https://stackoverflow.com/questions/29986185/python-argparse-dict-arg
    """

    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values.split(" "):
            k, v = kv.split("=")
            try:
                v = ast.literal_eval(v)
            except:
                pass
            my_dict[k] = v

        setattr(namespace, self.dest, my_dict)
