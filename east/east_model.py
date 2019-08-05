# """ResNet50 model for Keras.
#
# # Reference:
#
# - [Deep Residual Learning for Image Recognition](
#     https://arxiv.org/abs/1512.03385)
#
# Adapted from code contributed by BigMoyan.
# """
# import os
# import warnings
#
# import tensorflow as tf
# import numpy as np
# from keras_applications.imagenet_utils import _obtain_input_shape
#
# WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
#                 'releases/download/v0.2/'
#                 'resnet50_weights_tf_dim_ordering_tf_kernels.h5')
# WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
#                        'releases/download/v0.2/'
#                        'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
#
# layers = tf.keras.layers
# models = tf.keras.models
# keras_utils = tf.keras.utils
#
#
# class EastModel:
#
#     def __init__(self,
#                  input_shape,
#                  images,
#                  input_score_maps,
#                  input_geo_maps,
#                  input_training_masks):
#
#         self.input_shape = input_shape
#         self.images = images
#         self.input_score_maps = input_score_maps
#         self.input_geo_maps = input_geo_maps
#         self.input_training_masks = input_training_masks
#
#         self.img_input = None
#
#
#     def identity_block(self, input_tensor, kernel_size, filters, stage, block):
#         """The identity block is the block that has no conv layer at shortcut.
#
#         # Arguments
#             input_tensor: input tensor
#             kernel_size: default 3, the kernel size of
#                 middle conv layer at main path
#             filters: list of integers, the filters of 3 conv layer at main path
#             stage: integer, current stage label, used for generating layer names
#             block: 'a','b'..., current block label, used for generating layer names
#
#         # Returns
#             Output tensor for the block.
#         """
#         filters1, filters2, filters3 = filters
#         bn_axis = 3
#
#         #
#         # if backend.image_data_format() == 'channels_last':
#         #     bn_axis = 3
#         # else:
#         #     bn_axis = 1
#
#         conv_name_base = 'res' + str(stage) + block + '_branch'
#         bn_name_base = 'bn' + str(stage) + block + '_branch'
#
#         x = layers.Conv2D(filters1, (1, 1),
#                           kernel_initializer='he_normal',
#                           name=conv_name_base + '2a')(input_tensor)
#         x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
#         x = layers.Activation('relu')(x)
#
#         x = layers.Conv2D(filters2, kernel_size,
#                           padding='same',
#                           kernel_initializer='he_normal',
#                           name=conv_name_base + '2b')(x)
#         x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
#         x = layers.Activation('relu')(x)
#
#         x = layers.Conv2D(filters3, (1, 1),
#                           kernel_initializer='he_normal',
#                           name=conv_name_base + '2c')(x)
#         x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
#
#         x = layers.add([x, input_tensor])
#         x = layers.Activation('relu')(x)
#         return x
#
#     def conv_block(self,
#                    input_tensor,
#                    kernel_size,
#                    filters,
#                    stage,
#                    block,
#                    strides=(2, 2)):
#         """A block that has a conv layer at shortcut.
#
#         # Arguments
#             input_tensor: input tensor
#             kernel_size: default 3, the kernel size of
#                 middle conv layer at main path
#             filters: list of integers, the filters of 3 conv layer at main path
#             stage: integer, current stage label, used for generating layer names
#             block: 'a','b'..., current block label, used for generating layer names
#             strides: Strides for the first conv layer in the block.
#
#         # Returns
#             Output tensor for the block.
#
#         Note that from stage 3,
#         the first conv layer at main path is with strides=(2, 2)
#         And the shortcut should have strides=(2, 2) as well
#         """
#         filters1, filters2, filters3 = filters
#         # if backend.image_data_format() == 'channels_last':
#         #     bn_axis = 3
#         # else:
#         #     bn_axis = 1
#         bn_axis = 3
#         conv_name_base = 'res' + str(stage) + block + '_branch'
#         bn_name_base = 'bn' + str(stage) + block + '_branch'
#
#         x = layers.Conv2D(filters1, (1, 1), strides=strides,
#                           kernel_initializer='he_normal',
#                           name=conv_name_base + '2a')(input_tensor)
#         x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
#         x = layers.Activation('relu')(x)
#
#         x = layers.Conv2D(filters2, kernel_size, padding='same',
#                           kernel_initializer='he_normal',
#                           name=conv_name_base + '2b')(x)
#         x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
#         x = layers.Activation('relu')(x)
#
#         x = layers.Conv2D(filters3, (1, 1),
#                           kernel_initializer='he_normal',
#                           name=conv_name_base + '2c')(x)
#         x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
#
#         shortcut = layers.Conv2D(filters3, (1, 1), strides=strides,
#                                  kernel_initializer='he_normal',
#                                  name=conv_name_base + '1')(input_tensor)
#         shortcut = layers.BatchNormalization(
#             axis=bn_axis, name=bn_name_base + '1')(shortcut)
#
#         x = layers.add([x, shortcut])
#         x = layers.Activation('relu')(x)
#         return x
#
#     def dice_coefficient(self,
#                          y_true_cls,
#                          y_pred_cls,
#                          training_mask):
#         """
#         dice loss
#         :param y_true_cls:
#         :param y_pred_cls:
#         :param training_mask:
#         :return:
#         """
#         eps = 1e-5
#         intersection = tf.reduce_sum(y_true_cls * y_pred_cls * training_mask)
#         union = tf.reduce_sum(y_true_cls * training_mask) + \
#                 tf.reduce_sum(y_pred_cls * training_mask) + eps
#         loss = 1. - (2 * intersection / union)
#         tf.summary.scalar('classification_dice_loss', loss)
#         return loss
#
#     def unpool(self, inputs):
#         return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1] * 2, tf.shape(inputs)[2] * 2])
#
#     def mean_image_subtraction(self, images, means=[123.68, 116.78, 103.94]):
#         '''
#         image normalization
#         :param images:
#         :param means:
#         :return:
#         '''
#         num_channels = images.get_shape().as_list()[-1]
#         if len(means) != num_channels:
#             raise ValueError('len(means) must match the number of channels')
#         channels = tf.split(axis=3, num_or_size_splits=num_channels, value=images)
#         for i in range(num_channels):
#             channels[i] = channels[i] - tf.convert_to_tensor(means[i])
#         return tf.concat(axis=3, values=channels)
#
#     def set_input(self, images):
#         images = self.mean_image_subtraction(images)
#         self.img_input = layers.Input(shape=self.input_shape)
#
#     def get_outputs(self):
#
#         end_points = {}
#
#         bn_axis = 3
#
#         # http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006
#         x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(self.img_input)
#         x = layers.Conv2D(64, (7, 7),
#                           strides=(2, 2),
#                           padding='valid',
#                           kernel_initializer='he_normal',
#                           name='conv1')(x)
#         x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
#         x = layers.Activation('relu')(x)
#         x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
#         x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
#
#         x = self.conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
#         x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='b')
#         x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='c')
#
#         end_points["pool2"] = x
#
#         x = self.conv_block(x, 3, [128, 128, 512], stage=3, block='a')
#         x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='b')
#         x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='c')
#         x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='d')
#
#         end_points["pool3"] = x
#
#         x = self.conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
#         x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
#         x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
#         x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
#         x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
#         x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
#
#         end_points["pool4"] = x
#
#         x = self.conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
#         x = self.identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
#         x = self.identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
#
#         end_points["pool5"] = x
#
#         f = [end_points['pool5'], end_points['pool4'],
#              end_points['pool3'], end_points['pool2']]
#
#         for i in range(4):
#             print('Shape of f_{} {}'.format(i, f[i].shape))
#
#         g = [None, None, None, None]
#         h = [None, None, None, None]
#         num_outputs = [None, 128, 64, 32]
#
#         for i in range(4):
#             if i == 0:
#                 h[i] = f[i]
#             else:
#                 c1_1 = layers.Conv2D(filters=num_outputs[i], kernel_size=1)(tf.concat([g[i - 1], f[i]]), axis=-1)
#                 # slim.conv2d(tf.concat([g[i-1], f[i]], axis=-1), num_outputs[i], 1)
#                 h[i] = layers.Conv2D(filters=num_outputs[i], kernel_size=3)(c1_1)
#                 # slim.conv2d(c1_1, num_outputs[i], 3)
#             if i <= 2:
#                 g[i] = self.unpool(h[i])
#             else:
#                 g[i] = layers.Conv2D(filters=num_outputs[i], kernel_size=3)(h[i])
#                 # slim.conv2d(h[i], num_outputs[i], 3)
#             print('Shape of h_{} {}, g_{} {}'.format(i, h[i].shape, i, g[i].shape))
#
#         # here we use a slightly different way for regression part,
#         # we first use a sigmoid to limit the regression range, and also
#         # this is do with the angle map
#         F_score = layers.Conv2D(filters=1, kernel_size=1, activation=tf.nn.sigmoid)(g[3])
#         # slim.conv2d(g[3], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
#         # 4 channel of axis aligned bbox and 1 channel rotation angle
#         geo_map = layers.Conv2D(filters=4, kernel_size=1, activation=tf.nn.sigmoid)(g[3])
#         # slim.conv2d(g[3], 4, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) * FLAGS.text_scale
#         angle_map = layers.Conv2D(filters=1, kernel_size=1, activation=tf.nn.sigmoid)(g[3])
#         # (slim.conv2d(g[3], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) - 0.5) * np.pi/2 # angle is between [-45, 45]
#         F_geometry = tf.concat([geo_map, angle_map], axis=-1)
#
#         return F_score, F_geometry
#
#     def get_loss(self,
#                  y_true_cls,
#                  y_pred_cls,
#                  y_true_geo,
#                  y_pred_geo,
#                  training_mask):
#         """
#         define the loss used for training, contraning two part,
#         the first part we use dice loss instead of weighted logloss,
#         the second part is the iou loss defined in the paper
#         :param y_true_cls: ground truth of text
#         :param y_pred_cls: prediction os text
#         :param y_true_geo: ground truth of geometry
#         :param y_pred_geo: prediction of geometry
#         :param training_mask: mask used in training, to ignore some text annotated by ###
#         :return:
#         """
#
#         """
#         Section: EAST : 3.4.2 Loss for Geometries
#           p0 : d1                  p1 : d2
#            --------------------------
#           |                          |
#           |                          |
#           |                          |
#            --------------------------
#          p3 : d4                   p2 : d3
#
#          where d1,d2,d3 and d4 represents the distance from a pixel to the top, right, bottom and
#          left boundary of its corresponding rectangle, respectively.
#         """
#
#         classification_loss = self.dice_coefficient(y_true_cls, y_pred_cls, training_mask)
#
#         # scale classification loss to match the iou loss part
#         classification_loss *= 0.01
#
#         # p0 -> top, p1->right, p2->bottom, p3->left
#         p0_gt, p1_gt, p2_gt, p3_gt, theta_gt = tf.split(
#             value=y_true_geo, num_or_size_splits=5, axis=3)
#         p0_pred, p1_pred, p2_pred, p3_pred, theta_pred = tf.split(
#             value=y_pred_geo, num_or_size_splits=5, axis=3)
#
#         area_gt = (p0_gt + p2_gt) * (p1_gt + p3_gt)
#         area_pred = (p0_pred + p2_pred) * (p1_pred + p3_pred)
#
#         w_union = tf.minimum(p1_gt, p1_pred) + tf.minimum(p3_gt, p3_pred)
#         h_union = tf.minimum(p0_gt, p0_pred) + tf.minimum(p2_gt, p2_pred)
#         area_intersect = w_union * h_union
#
#         area_union = area_gt + area_pred - area_intersect
#
#         L_AABB = -tf.log((area_intersect + 1.0) / (area_union + 1.0))
#         L_theta = 1 - tf.cos(theta_pred - theta_gt)
#         L_g = L_AABB + 20 * L_theta
#
#         tf.summary.scalar('geometry_AABB', tf.reduce_mean(
#             L_AABB * y_true_cls * training_mask))
#         tf.summary.scalar('geometry_theta', tf.reduce_mean(
#             L_theta * y_true_cls * training_mask))
#
#         return tf.reduce_mean(L_g * y_true_cls * training_mask) + classification_loss