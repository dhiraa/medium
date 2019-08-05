# Copyright 2018 The Shabda Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Experiments class that allows easy plug n play of modules
"""
import logging
import os
import shutil
import time

import gin

import tensorflow as tf
from tqdm import tqdm

from print_helper import *
from engines.executor import Executor

# get TF logger
log = logging.getLogger('tensorflow')
log.setLevel(logging.DEBUG)

CGREEN2 = '\33[92m'
CEND = '\33[0m'


@gin.configurable
class Experiments(object):
    """
    Experiments uses dataset, data iterator & model factory classes and import them
    dynamically based on the string.
    This allows the user to choose the modules dynamically and run the experiments without ever writing the
    code when we need mix and experiment dataset and modules.
    """

    def __init__(self,
                 dataset,
                 iterator,
                 model,
                 num_epochs=5,
                 save_checkpoints_steps=50,
                 keep_checkpoint_max=5,
                 save_summary_steps=25,
                 log_step_count_steps=10,
                 clear_model_data=False,
                 plug_dataset=True,
                 mode='train',
                 batch_size=8):
        
        self.mode = mode

        self.num_epochs = num_epochs
        self._dataset = dataset
        self._data_iterator = iterator
        self._model = model
        print(self._model)
        self.save_checkpoints_steps = save_checkpoints_steps
        self.keep_checkpoint_max = keep_checkpoint_max
        self.save_summary_steps = save_summary_steps
        self.log_step_count_steps = log_step_count_steps
        self.clear_model_data = clear_model_data
        self.plug_dataset = plug_dataset
        self.batch_size = batch_size

    def _init_tf_config(self):
        run_config = tf.compat.v1.ConfigProto()
        run_config.gpu_options.allow_growth = True
        # run_config.gpu_options.per_process_gpu_memory_fraction = 0.50
        run_config.allow_soft_placement = True
        run_config.log_device_placement = False
        model_dir = self._model.model_dir

        if self.clear_model_data:
            if os.path.exists(model_dir):
                shutil.rmtree(model_dir)

        self._run_config = tf.estimator.RunConfig(session_config=run_config,
                                                  save_checkpoints_steps=self.save_checkpoints_steps,
                                                  keep_checkpoint_max=self.keep_checkpoint_max,
                                                  save_summary_steps=self.save_summary_steps,
                                                  model_dir=model_dir,
                                                  log_step_count_steps=self.log_step_count_steps)
        return run_config


    def run(self, args):
        num_samples = self._data_iterator.num_train_examples
        print_info("Number of trianing samples : {}".format(num_samples))
        batch_size = self.batch_size
        num_epochs = self.num_epochs
        mode = self.mode
        self._init_tf_config()

        if mode == "test_iterator":
            self.test_iterator()

        executor = Executor(model=self._model, data_iterator=self._data_iterator, config=self._run_config)

        if mode in ["train", "retrain"]:
            for current_epoch in tqdm(range(num_epochs), desc="Epoch"):
                current_max_steps = (num_samples // batch_size) * (current_epoch + 1)
                print("\n\n Training for epoch {} with steps {}\n\n".format(current_epoch, current_max_steps))
                executor.train(max_steps=None)
                print("\n\n Evaluating for epoch\n\n", current_epoch)
                executor.evaluate(steps=200)
                executor.export_model(self._model.model_dir + "/exported/")

        elif mode == "predict":
            self._data_iterator.predict_on_test_files(executor=executor)

        elif mode == "predict_instance":
            self._data_iterator.predict_on_instance(executor=executor, file_path=args.test_file_path)
        else:
            print_error("Given mode is not avaialble!")

