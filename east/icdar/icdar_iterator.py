import tensorflow as tf
import glob
import os
import gin
from tqdm import tqdm 


def get_tf_records_count(files):
    total_records = -1
    for file in tqdm(files, desc="tfrecords size: "):
        total_records += sum(1 for _ in tf.data.TFRecordDataset(file))
    return total_records


@gin.configurable
class CIDARIterator:
    """
    """

    def __init__(self,
                 data_dir=gin.REQUIRED,
                 num_threads=1,
                 batch_size=16,
                 prefetch_size=16,
                 dataset=None):
        """
        ......
        """
    
        self._data_dir = data_dir
        self._num_threads = num_threads
        self._batch_size = batch_size
        self._prefetch_size = prefetch_size
        
        self._num_train_examples = -1
        
        #TODO find a right way to get this
        files = glob.glob(os.path.join(self._data_dir, "train/*.tfrecords"))
        self._num_train_examples = get_tf_records_count(files=files)

    @property
    def num_train_examples(self):
        return self._num_train_examples


    def dataset_to_iterator(self, dataset):
        # Create an iterator
        iterator = dataset.make_one_shot_iterator()

        # Create your tf representation of the iterator
        image, label = iterator.get_next()

        # # Bring your picture back in shape
        # image = tf.reshape(image, [-1, 256, 256, 1])
        #
        # # Create a one hot array for your labels
        # label = tf.one_hot(label, NUM_CLASSES)

        return image, label

    def train_input_fn(self):
        """
        Returns an data set iterator function that can be used in estimator
        :return:
        """
        return self._get_train_input_fn()

    def val_input_fn(self):
        """
        Returns an data set iterator function that can be used in estimator
        :return:
        """
        return self._get_val_input_fn()

    def test_input_fn(self):
        """
        Returns an data set iterator function that can be used in estimator
        :return:
        """
        return self._get_test_input_fn()

    def decode(self, serialized_example):
        # 1. define a parser
        features = tf.io.parse_single_example(
            serialized_example,
            # Defaults are not specified since both keys are required.
            features={
                'images': tf.io.FixedLenFeature([512 * 512 * 3], tf.float32),
                'score_maps': tf.io.FixedLenFeature([128 * 128 * 1], tf.float32),
                'geo_maps': tf.io.FixedLenFeature([128 * 128 * 5], tf.float32),
                'training_masks': tf.io.FixedLenFeature([128 * 128 * 1], tf.float32),
            })

        image = tf.reshape(
            tf.cast(features['images'], tf.float32), shape=[512, 512, 3])
        score_map = tf.reshape(
            tf.cast(features['score_maps'], tf.float32), shape=[128, 128, 1])
        geo_map = tf.reshape(
            tf.cast(features['geo_maps'], tf.float32), shape=[128, 128, 5])
        training_masks = tf.reshape(
            tf.cast(features['training_masks'], tf.float32), shape=[128, 128, 1])

        return {"images": image, "score_maps": score_map, "geo_maps": geo_map, "training_masks": training_masks}, training_masks

    def _get_train_input_fn(self):
        """
        Inheriting class must implement this
        :return: dataset
        """
        files = glob.glob(os.path.join(self._data_dir, "train/*.tfrecords"))
                          
        # self._num_train_examples = get_tf_records_count(files=files)
        # TF dataset APIs
        dataset = tf.data.TFRecordDataset(files, num_parallel_reads=self._num_threads)
        # Map the generator output as features as a dict and labels
        dataset = dataset.map(self.decode)

        dataset = dataset.batch(batch_size=self._batch_size, drop_remainder=False)
        dataset = dataset.prefetch(self._prefetch_size)
        # dataset = dataset.cache(filename=os.path.join(self.iterator_dir, "train_data_cache"))
        print("Dataset output sizes are: ")
        print(dataset)

        return dataset

    def _get_val_input_fn(self):
        """
        Inheriting class must implement this
        :return: callable
        """
        files = glob.glob(os.path.join(self._data_dir, "val/*.tfrecords"))
        # TF dataset APIs
        dataset = tf.data.TFRecordDataset(files, num_parallel_reads=self._num_threads)
        # Map the generator output as features as a dict and labels
        dataset = dataset.map(self.decode)

        dataset = dataset.batch(
            batch_size=self._batch_size, drop_remainder=False)
        dataset = dataset.prefetch(self._prefetch_size)
        print("Dataset output sizes are: ")
        print(dataset)

        return dataset

    def _get_test_input_function(self):
        """
        Inheriting class must implement this
        :return: callable
        """
        files = glob.glob(os.path.join(self._data_dir, "test/*.tfrecords"))
        # TF dataset APIs
        dataset = tf.data.TFRecordDataset(files, num_parallel_reads=self._num_threads)
    
        # Map the generator output as features as a dict and labels
        dataset = dataset.map(self.decode)

        dataset = dataset.batch(
                batch_size=self._hparams.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(self._hparams.prefetch_size)
        print("Dataset output sizes are: ")
        print(dataset)

        return dataset

    def serving_input_receiver_fn(self):
        inputs = {
            "images": tf.placeholder(tf.float32, [None, None, None, 3]),
        }
        return tf.estimator.export.ServingInputReceiver(inputs, inputs)

