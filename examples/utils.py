import argparse


class StoreDictKeyPair(argparse.Action):
    """
    ref: https://stackoverflow.com/questions/29986185/python-argparse-dict-arg
    """

    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values.split(","):
            k, v = kv.split("=")
            my_dict[k] = v
        setattr(namespace, self.dest, my_dict)
