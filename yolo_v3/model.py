import keras.layers as KL
import keras.layers.advanced_activations as KLA
import keras.regularizers as KR

class BatchNorm(KL.BatchNormalization):
    """Extends the Keras BatchNormalization class to allow a central place
    to make changes if needed.

    Batch normalization has a negative effect on training if batches are small
    so this layer is often frozen (via setting in Config class) and functions
    as linear layer.
    """
    def call(self, inputs, training=None):
        """
        Note about training values:
            None: Train BN layers. This is the normal mode
            False: Freeze BN layers. Good when batch size is small
            True: (don't use). Set layer in training mode even when inferencing
        """
        return super(self.__class__, self).call(inputs, training=training)


def darknet_conv(x, filters, kernel_size, train_bn, block_name=None, idx=None, strides=(1, 1)):
    padding = 'valid' if strides == (2, 2) else 'same'
    conv_idx = str(idx) if idx is not None else ''
    name_kwargs = {'name': block_name + conv_idx} if block_name is not None else {}
    x = KL.Conv2D(filters, kernel_size,
                  kernel_regularizer=KR.l2(5e-4),
                  padding=padding,
                  use_bias=False, **name_kwargs)(x)

    x = BatchNorm(**name_kwargs)(x, training=train_bn)
    x = KLA.LeakyReLU(alpha=0.1)(x)

    return x


def darknet_resblock(input_tensor, filters, nb_conv, train_bn, block_idx):

    block_name = 'resblock' + str(block_idx)
    # Darknet uses left and top padding instead of 'same' mode
    x = KL.ZeroPadding2D(((1, 0), (1, 0)))(input_tensor)
    x = darknet_conv(x, filters, (3, 3), train_bn, block_name, strides=(2, 2))

    for i in range(nb_conv):
        x0 = darknet_conv(x, filters, (1, 1), train_bn, block_name, train_bn, i*2+1)
        y = darknet_conv(x0, filters, (3, 3), train_bn, block_name, train_bn, i*2+2)

        x = KL.Add()([x, y])

    return x


def darknet_graph(x, train_bn):
    '''Darknent graph having 52 Convolution2D layers'''
    x = darknet_conv(x, 32, (3, 3), train_bn)(x)
    C1 = x = darknet_resblock(x, 64, 1, train_bn, block_idx=1)
    C2 = x = darknet_resblock(x, 128, 2, train_bn, block_idx=2)
    C3 = x = darknet_resblock(x, 256, 8, train_bn, block_idx=3)
    C4 = x = darknet_resblock(x, 512, 8, train_bn, block_idx=4)
    C5 = x = darknet_resblock(x, 1024, 4, train_bn, block_idx=5)
    return [C1, C2, C3, C4, C5]


def feature_maps(x, num_filters, out_filters, train_bn):
    '''6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''

    x = darknet_conv(x, num_filters, (1, 1), train_bn)(x)
    x = darknet_conv(x, num_filters * 2, (3, 3), train_bn)(x)
    x = darknet_conv(x, num_filters, (1, 1), train_bn)(x)
    x = darknet_conv(x, num_filters * 2, (3, 3), train_bn)(x)
    x = darknet_conv(x, num_filters, (1, 1), train_bn)(x)

    x0 = darknet_conv(x, num_filters * 2, (3, 3), train_bn)(x)
    y = darknet_conv(x, out_filters, (1, 1), train_bn)(x0)

    return x, y


def yolo_backbone(config):
    input_image = KL.Input(
        shape=[None, None, 3], name="input_image")

    _, _, C3, C4, C5 = darknet_graph(input_image, train_bn=config.TRAIN_BN)

    x, P5 = feature_maps(C5, 512, config.NUM_ANCHORS * (config.NUM_CLASSES + 5))

    x = darknet_conv(x, 32, (3, 3), config.TRAIN_BN)(x)
    x = KL.UpSampling2D(2)(x)
    x = KL.Concatenate()([x, C4])
    x, P4 = feature_maps(x, 256, config.NUM_ANCHORS * (config.NUM_CLASSES + 5))

    x = darknet_conv(x, 128, (1, 1), config.TRAIN_BN)(x)
    x = KL.UpSampling2D(2)(x)
    x = KL.Concatenate()([x, C3])
    x, P3 = feature_maps(x, 128, config.NUM_ANCHORS * (config.NUM_CLASSES + 5))

    return [P3, P4, P5]
