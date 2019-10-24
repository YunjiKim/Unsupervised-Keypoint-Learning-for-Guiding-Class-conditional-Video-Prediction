from pathlib import Path

import yaml

from . import model, training, data


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def touch_dir(target_dir):
    """
    Creates directory if not exists.
    :param target_dir:
    :return:
    """
    Path(target_dir).mkdir(parents=True, exist_ok=True)
