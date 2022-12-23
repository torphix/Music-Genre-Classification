import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def resblock_1D(x, n_blocks, out_d, stride):
    for i in range(n_blocks):
        res = x
        # Block 1 Downsample if layer 1
        x = layers.Conv1D(out_d//4, 1, stride if i==0 else 1, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        # Block 2
        x = layers.Conv1D(out_d//4, 3, 1, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        # Block 3
        x = layers.Conv1D(out_d, 1, 1)(x)
        x = layers.BatchNormalization()(x)
        if i == 0:
            res = layers.Conv1D(out_d, 1, stride, padding='valid')(res)
            res = layers.BatchNormalization()(res)
        x = layers.Add()([x, res])
        x = layers.Activation("relu")(x)
    return x


def make_model_1d(input_shape, num_classes, resnet_type):
    inputs = keras.Input(shape=input_shape)
    n_layers = get_resnet_filters(resnet_type)
    # In layer
    x = layers.Conv1D(128, 7, 2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(3, strides=2)(x)
    # Entry block
    x = layers.Conv1D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Resblocks
    x = resblock_1D(x, n_layers[0], 256, 1)
    x = resblock_1D(x, n_layers[1], 512, 2)
    x = resblock_1D(x, n_layers[2], 1024, 2)
    x = resblock_1D(x, n_layers[3], 2048, 2)
    
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(1024, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return keras.Model(inputs, outputs)


def resblock_2d(x, n_blocks, out_d, stride):
    for i in range(n_blocks):
        res = x
        # Block 1 Downsample if layer 1
        x = layers.Conv2D(out_d//4, 1, stride if i==0 else 1, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        # Block 2
        x = layers.Conv2D(out_d//4, 3, 1, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        # Block 3
        x = layers.Conv2D(out_d, 1, 1)(x)
        x = layers.BatchNormalization()(x)
        if i == 0:
            res = layers.Conv2D(out_d, 1, stride, padding='valid')(res)
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
    x = layers.Dense(1024, activation='relu')(x)
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




def load_pretrained():
    ResNet50_model = keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(256,256,3), classes=10)

    for layer in ResNet50_model.layers:
        layer.trainable=True

    x = layers.Flatten()(ResNet50_model.output)
    x = layers.Dense(256,activation='relu')(x)
    x = layers.Dense(10,activation='softmax')(x)
    model = keras.Model(inputs=ResNet50_model.input, outputs=x)
    return model


def load_model(data_type, resnet_type, ckpt_path=None, use_pretrained=False):
    if use_pretrained:
        print('Loading Pretrained Model')
        model = load_pretrained()
    else:
        if ckpt_path is not None and ckpt_path != '':
            print('Loading Model From CKPT')
            model = keras.models.load_model(ckpt_path)
        else:
            print(f'Training from scratch a {resnet_type} model')
            if data_type == 'mel':
                model = make_model_1d(input_shape=(128,130), num_classes=10, resnet_type=resnet_type)
            elif data_type == 'img':
                model = make_model_2d(input_shape=(256,256,3), num_classes=10, resnet_type=resnet_type)
    return model
    