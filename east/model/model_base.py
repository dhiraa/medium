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
Base class for models.
"""
import os

# pylint: disable=too-many-arguments

__all__ = [
    "ModelBase"
]


class ModelBase(object):
    """Base class inherited by all model classes.

    A model class implements interfaces that are compatible with
    :tf_main:`TF Estimator <estimator/Estimator>`. In particular,
    :meth:`_build` implements the
    :tf_main:`model_fn <estimator/Estimator#__init__>` interface; and
    :meth:`get_input_fn` is for the :attr:`input_fn` interface.

    .. document private functions
    .. automethod:: _build
    """

    def __init__(self,
                 experiment_name,
                 model_root_directory=os.path.join(os.path.expanduser("~"), "vitaFlow/", "default_model_dir")):
        self._experiment_name = experiment_name
        self._model_root_directory = model_root_directory

    def __call__(self, features, labels, params, mode, config=None):
        """
        Used for the :tf_main:`model_fn <estimator/Estimator#__init__>`
        argument when constructing
        :tf_main:`tf.estimator.Estimator <estimator/Estimator>`.
        """
        return self._build(features, labels, params, mode, config=config)

    @property
    def model_dir(self):
        """
        Returns model directory `model_root_directory`/`experiment_name`/VanillaGAN
        :return:
        """
        return os.path.join(self._model_root_directory,
                            self._experiment_name,
                            type(self).__name__)

    def _build_layers(self, features, mode):
        raise NotImplementedError

    def _get_loss(self, labels, logits):
        raise NotImplementedError

    def _get_optimizer(self, loss):
        raise NotImplementedError

    def _get_eval_metrics(self, predictions, labels):
        raise NotImplementedError

    def _build(self, features, labels, params, mode, config=None):
        """
        Used for the :tf_main:`model_fn <estimator/Estimator#__init__>`
        argument when constructing
        :tf_main:`tf.estimator.Estimator <estimator/Estimator>`.
        """
        raise NotImplementedError



