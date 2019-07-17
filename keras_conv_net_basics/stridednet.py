import tensorflow as tf

class StridedNet():
    """
    StridedNet has three important characteristics:

    It uses strided convolutions rather than pooling operations to reduce volume size
    The first CONV layer uses 7×7 filters but all other layers in the network use 3×3
    filters (similar to VGG)
    The MSRA/He et al. normal distribution algorithm is used to initialize all weights in the network

    We recommend batch normalization because it tends to stabilize training and make tuning
    hyperparameters easier. That said, it can double or triple your training time. Use it wisely.

    Dropout’s purpose is to help your network generalize and not overfit. Neurons from the current layer,
    with probability p, will randomly disconnect from neurons in the next layer so that the network has
    to rely on the existing connections. I highly recommend utilizing dropout.

    """
    def __init__(self, width, height, depth, classes, reg, init="he_normal"):
        """
        Creates Stridnet. The width , height , and depth  parameters affect the input volume shape.

        :param width: Image width in pixels.
        :param height: The image height in pixels.
        :param depth: The number of channels for the image.
        :param classes: The number of classes the model needs to predict.
        :param reg: Regularization method.
        :param init: The kernel initializer.
        """
        self.width = width
        self.height = height
        self.depth = depth
        self.classes = classes
        self.reg = reg
        self.initializer = init

    def build(self):
        # initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        model = tf.keras.models.Sequential()
        inputShape = (self.height, self.width, self.depth)
        chanDim = -1

        # # if we are using "channels first", update the input shape
        # # and channels dimension
        # if K.image_data_format() == "channels_first":
        #     inputShape = (self.depth, self.height, self.width)
        #     chanDim = 1

        # our first CONV layer will learn a total of 16 filters, each
        # Of which are 7x7 -- we'll then apply 2x2 strides to reduce
        # the spatial dimensions of the volume
        model.add(tf.keras.layers.Conv2D(16,
                                         (7, 7),
                                         strides=(2, 2),
                                         padding="valid",
                                         kernel_initializer=self.initializer,
                                         kernel_regularizer=self.reg,
                                         input_shape=inputShape)) #note input shape is provided

        # here we stack two CONV layers on top of each other where
        # each layerswill learn a total of 32 (3x3) filters
        model.add(tf.keras.layers.Conv2D(32, (3, 3),
                                         padding="same",
                                         kernel_initializer=self.initializer,
                                         kernel_regularizer=self.reg))
        model.add(tf.keras.layers.Activation("relu"))
        model.add(tf.keras.layers.BatchNormalization(axis=chanDim))
        model.add(tf.keras.layers.Conv2D(32, (3, 3),
                                         strides=(2, 2),
                                         padding="same",
                                         kernel_initializer=self.initializer,
                                         kernel_regularizer=self.reg))
        model.add(tf.keras.layers.Activation("relu"))
        model.add(tf.keras.layers.BatchNormalization(axis=chanDim))
        model.add(tf.keras.layers.Dropout(0.25))

        # stack two more CONV layers, keeping the size of each filter
        # as 3x3 but increasing to 64 total learned filters
        model.add(tf.keras.layers.Conv2D(64, (3, 3),
                                         padding="same",
                                         kernel_initializer=self.initializer,
                                         kernel_regularizer=self.reg))
        model.add(tf.keras.layers.Activation("relu"))
        model.add(tf.keras.layers.BatchNormalization(axis=chanDim))
        model.add(tf.keras.layers.Conv2D(64,
                                         (3, 3),
                                         strides=(2, 2),
                                         padding="same",
                                         kernel_initializer=self.initializer,
                                         kernel_regularizer=self.reg))
        model.add(tf.keras.layers.Activation("relu"))
        model.add(tf.keras.layers.BatchNormalization(axis=chanDim))
        model.add(tf.keras.layers.Dropout(0.25))

        # increase the number of filters again, this time to 128
        model.add(tf.keras.layers.Conv2D(128,
                                         (3, 3),
                                         padding="same",
                                         kernel_initializer=self.initializer,
                                         kernel_regularizer=self.reg))
        model.add(tf.keras.layers.Activation("relu"))
        model.add(tf.keras.layers.BatchNormalization(axis=chanDim))
        model.add(tf.keras.layers.Conv2D(128,
                                         (3, 3),
                                         strides=(2, 2),
                                         padding="same",
                                         kernel_initializer=self.initializer,
                                         kernel_regularizer=self.reg))
        model.add(tf.keras.layers.Activation("relu"))
        model.add(tf.keras.layers.BatchNormalization(axis=chanDim))
        model.add(tf.keras.layers.Dropout(0.25))

        # fully-connected layer
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(512, kernel_initializer=self.initializer,))
        model.add(tf.keras.layers.Activation("relu"))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.5))

        # softmax classifier
        model.add(tf.keras.layers.Dense(self.classes))
        model.add(tf.keras.layers.Activation("softmax"))

        # return the constructed network architecture
        return model