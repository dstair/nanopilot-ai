#!/usr/bin/env python3
"""
Scripts to train a model using tensorflow (Keras) or PyTorch.

Usage:
    train.py [--tubs=tubs] (--model=<model>)
    [--type=(linear|inferred|tensorrt_linear|tflite_linear)]
    [--comment=<comment>]
    [--myconfig=<filename>]

Options:
    -h --help              Show this screen.
    --myconfig=filename    Specify myconfig file to use.
                           [default: drone_config.py]

Use .pth extension for PyTorch training, .h5 or other for Keras/TF training.
"""

import os
from docopt import docopt
import donkeycar as dk


def main():
    args = docopt(__doc__)
    cfg = dk.load_config(config_path=os.path.join(os.path.dirname(__file__), 'config.py'), myconfig=args['--myconfig'])
    tubs = args['--tubs']
    model = args['--model']
    model_type = args['--type']
    comment = args['--comment']

    if model and model.endswith('.pth'):
        from torch_train import train as torch_train
        tub_paths = [t.strip() for t in tubs.split(',')] if tubs else [cfg.DATA_PATH]
        torch_train(cfg, tub_paths, model)
    else:
        from donkeycar.pipeline.training import train
        train(cfg, tubs, model, model_type, comment)


if __name__ == "__main__":
    main()
