import gin
from icdar.icdar_data import ICDARTFDataset
from icdar.icdar_iterator import CIDARIterator
from model.east_model import EASTModel
from engines.experiments import Experiments

@gin.configurable
def get_experiment_root_directory(value):
    return value
