# from .sem import SEM, sem_run, sem_run_with_boundaries
# Store the version here so:
# 1) we don't load dependencies by storing it in __init__.py
# 2) we can import it in setup.py for the same reason
# 3) we can import it into your module module
# __version__ = '1.2.7'

# This code make keras models pickable
import pickle

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.python.keras.saving import saving_utils


def unpack(model, training_config, weights):
    restored_model = deserialize(model)
    if training_config is not None:
        restored_model.compile(
            **saving_utils.compile_args_from_training_config(
                training_config
            )
        )
    restored_model.set_weights(weights)
    return restored_model


# Hotfix function
def make_keras_picklable():
    def __reduce__(self):
        model_metadata = saving_utils.model_metadata(self)
        training_config = model_metadata.get("training_config", None)
        model = serialize(self)
        weights = self.get_weights()
        return (unpack, (model, training_config, weights))

    cls = Model
    cls.__reduce__ = __reduce__

import logging
import os
import coloredlogs
logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get('LOGLEVEL', logging.INFO))
# must have a handler, otherwise logging will use lastresort
c_handler = logging.StreamHandler()
LOGFORMAT = '%(name)s - %(levelname)s - %(message)s'
# c_handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
# c_handler.setFormatter(coloredlogs.ColoredFormatter(LOGFORMAT))
c_handler.setFormatter(logging.Formatter(LOGFORMAT))
logger.addHandler(c_handler)

# logger.info(f'{__file__}: Make Model picklable')
# Run the function
make_keras_picklable()

