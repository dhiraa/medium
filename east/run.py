import gin
import time
import argparse
from tqdm import tqdm

import tensorflow as tf
from icdar.icdar_data import ICDARTFDataset
from icdar.icdar_iterator import CIDARIterator
from model.east_model import EASTModel
from engines.experiments import Experiments

def test_iterator(data_iterator):
    i = 0
    for features, label in tqdm(data_iterator):
        for key in features.keys():
            print("Batch {} =>  Shape of feature : {} is {}".format(i, key, features[key].shape))
            i = i + 1


def main(args):
    print(' -' * 35)
    print('Running Experiment:')
    print(' -' * 35)
    dataset = ICDARTFDataset()
    dataset.run()

    iterator = CIDARIterator()
    # test_iterator(iterator.train_input_fn())

    model = EASTModel()
    print(model)

    experiment = Experiments(dataset=dataset, iterator=iterator, model=model)
    experiment.run(None)

    print(' -' * 35)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-cf", "--config_file", required=True,
                    help="Google gin config file path")
    args = vars(ap.parse_args())
    gin.parse_config_file(args['config_file'])
    main(args)