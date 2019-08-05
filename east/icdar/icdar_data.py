#!/usr/bin/env python
# coding: utf-8

import os
import gin
import tensorflow as tf
from tqdm import tqdm
from icdar.icdar_utils import *


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _mat_feature(mat):
    return tf.train.Feature(float_list=tf.train.FloatList(value=mat.flatten()))


@gin.configurable
class ICDARTFDataset:
    def __init__(self,
                 data_dir=gin.REQUIRED,
                 out_dir=gin.REQUIRED,
                 max_image_large_side=1280,
                 max_text_size=800,
                 min_text_size=5,
                 min_crop_side_ratio=0.1,
                 geometry="RBOX",
                 number_images_per_tfrecords=8):

        self._data_dir = data_dir

        self._train_out_dir = out_dir + "/train/"
        self._val_out_dir = out_dir + "/val/"
        self._test_out_dir = out_dir + "/test/"

        make_dirs(self._train_out_dir)
        make_dirs(self._val_out_dir)
        make_dirs(self._test_out_dir)

        self._geometry = geometry
        self._min_text_size = min_text_size
        self._max_image_large_side = max_image_large_side
        self._max_text_size = max_text_size
        self._min_crop_side_ratio = min_crop_side_ratio
        self._number_images_per_tfrecords = number_images_per_tfrecords

        self.run()

    def _get_features(self, image_mat, score_map_mat, geo_map_mat, training_masks_mat):
        """
        """
        return {
            "images": _mat_feature(image_mat),
            "score_maps": _mat_feature(score_map_mat),
            "geo_maps": _mat_feature(geo_map_mat),
            "training_masks": _mat_feature(training_masks_mat)
        }

    def write_tf_records(self, images, file_path_name):
        num_of_files_skipped = 0

        if os.path.exists(file_path_name):
            print("Found ", file_path_name, "already! Hence skipping")
            return

        with tf.io.TFRecordWriter(file_path_name) as writer:
            for image_file in images:
                ret = image_2_data(image_file_path=image_file,
                                   geometry=self._geometry,
                                   min_text_size=self._min_text_size,
                                   min_crop_side_ratio=self._min_crop_side_ratio)
                try:
                    image_mat, score_map_mat, geo_map_mat, training_masks_mat = ret
                except:
                    num_of_files_skipped += 1
                    continue
                features = tf.train.Features(
                    feature=self._get_features(image_mat, score_map_mat, geo_map_mat, training_masks_mat))
                example = tf.train.Example(features=features)
                writer.write(example.SerializeToString())

        print("Number of files skipped : ", num_of_files_skipped)

    def prepare_data(self, data_path, out_path):

        print("Serializing data found in ", data_path)

        images = get_images(data_path)

        index = 0
        for i in tqdm(range(0, len(images), self._number_images_per_tfrecords), desc="prepare_data: "):
            self.write_tf_records(images=images[i:i + self._number_images_per_tfrecords],
                                  file_path_name=out_path + "/" + str(index) + ".tfrecords")
            index += 1

    def run(self):
        self.prepare_data(data_path=self._data_dir + "/train/", out_path=self._train_out_dir)
        self.prepare_data(data_path=self._data_dir + "/val/", out_path=self._val_out_dir)
        self.prepare_data(data_path=self._data_dir + "/test/", out_path=self._test_out_dir)
