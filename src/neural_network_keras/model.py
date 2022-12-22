import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def resblock_2d(x, n_blocks, out_d, stride):
    for i in range(n_blocks):
        res = x
        # Block 1 
        x = layers.Conv2D(out_d//4, 1, stride, padding='valid')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        # Block 2
        x = layers.Conv2D(out_d//4, 3, 1, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        # Block 3
        x = layers.Conv2D(out_d if i == n_blocks else out_d//4, 1, 1)(x)
        x = layers.BatchNormalization()(x)
        # Res Layer
        res = layers.Conv2D(out_d if i == n_blocks else out_d//4, 1, stride, padding='valid')(res)
        res = layers.BatchNormalization()(res)
        x = layers.Add()([x, res])
        x = layers.Activation("relu")(x)
    return x


def make_model_2d(input_shape, num_classes, resnet_type):
    inputs = keras.Input(shape=input_shape)
    n_layers = get_resnet_filters(resnet_type)
    # In layer
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, 7, 2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(3, strides=2)(x)

    # Entry block
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Resblocks
    x = resblock_2d(x, n_layers[0], 256, 1)
    x = resblock_2d(x, n_layers[1], 512, 2)
    x = resblock_2d(x, n_layers[2], 1024, 2)
    x = resblock_2d(x, n_layers[3], 2048, 2)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return keras.Model(inputs, outputs)


def get_resnet_filters(resnet_type):
    if resnet_type == 'resnet18':
        n_layers = [2, 2, 2, 2]
    elif resnet_type == 'resnet50':
        n_layers = [3, 4, 6, 3]
    elif resnet_type == 'resnet101':
        n_layers = [3, 4, 23, 3]
    elif resnet_type == 'resnet152':
        n_layers = [3, 8, 36, 3]
    return n_layers


