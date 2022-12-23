import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


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




def load_pretrained(resnet_type):
    ResNet50_model = ResNet50(weights='imagenet', include_top=False, input_shape=(150,150,3), classes=6)

    for layers in ResNet50_model.layers:
        layers.trainable=True

    opt = SGD(lr=0.01,momentum=0.7)
    resnet50_x = Flatten()(ResNet50_model.output)
    resnet50_x = Dense(256,activation='relu')(resnet50_x)
    resnet50_x = Dense(6,activation='softmax')(resnet50_x)
    resnet50_x_final_model = Model(inputs=ResNet50_model.input, outputs=resnet50_x)
    resnet50_x_final_model.compile(loss = 'categorical_crossentropy', optimizer= opt, metrics=['acc'])

    number_of_epochs = 60
    resnet_filepath = 'resnet50'+'-saved-model-{epoch:02d}-val_acc-{val_acc:.2f}.hdf5'
    resnet_checkpoint = tf.keras.callbacks.ModelCheckpoint(resnet_filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    resnet_early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.05, patience=5, min_lr=0.000002)
    callbacklist = [resnet_checkpoint,resnet_early_stopping,reduce_lr]
    resnet50_history = resnet50_x_final_model.fit(train_generator, epochs = number_of_epochs ,validation_data = validation_generator,callbacks=callbacklist,verbose=1)