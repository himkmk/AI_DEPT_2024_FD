import os
from typing import Callable, Iterable, List, Tuple
import argparse


def get_model_dir(args: argparse.Namespace) -> str:
    """
    Obtain the directory to save/load the model
    """
    path = os.path.join(
        'model_ckpt',
        f'exp_{args.exp}'
    )
    return path