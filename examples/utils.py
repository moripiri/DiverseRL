import argparse
import ast


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
