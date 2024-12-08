from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, AveragePooling2D, Input, Activation, Add, BatchNormalization
from keras.optimizers import Adam, SGD, Adadelta, RMSprop, Adagrad
from keras.models import Model

from constants import INPUT_SHAPE, NUM_CLASSES

AVAILABLE_OPTIMIZERS = {
    'sgd': SGD,
    'adagrad': Adagrad,
    'adadelta': Adadelta,
    'rmsprop': RMSprop,
    'adam': Adam,
}

def residual_block(x, filters, kernel_size=(3, 3), strides=(1, 1)):
    y = Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv2D(filters, kernel_size, padding='same')(y)
    y = BatchNormalization()(y)
    if strides != (1, 1):
        x = Conv2D(filters, (1, 1), strides=strides, padding='same')(x)
    out = Add()([x, y])
    out = Activation('relu')(out)
    return out

def get_model(model_name, dropout_rate, learning_rate, optimizer, augmentation, num_layers=3, extra_layers=None,
              num_residual_blocks=None):
    """
    Retrieve a model based on model_name with optional additional layers and residual blocks.

    Parameters:
    - model_name (str): Name of the model
    - dropout_rate (float): Dropout rate for the model
    - learning_rate (float): Learning rate for the optimizer
    - optimizer (str): Optimizer name
    - augmentation (bool): Whether data augmentation is used
    - num_layers (int): Number of layers to use in the model
    - extra_layers (int): Number of additional Dense layers to add
    - num_residual_blocks (int): Number of residual blocks to add

    Returns:
    - keras.Model: Compiled Keras model
    """
    MODELS_CONFIG = {
        'DNN1': Sequential([
            Flatten(input_shape=INPUT_SHAPE),
            Dense(units=128, activation='relu'),
            Dropout(rate=dropout_rate),
            Dense(NUM_CLASSES, activation='softmax'),
        ]),
        'DNN2': Sequential([
            Flatten(input_shape=INPUT_SHAPE),
            Dense(units=256, activation='relu'),
            Dropout(rate=dropout_rate),
            Dense(units=128, activation='relu'),
            Dropout(rate=dropout_rate),
            Dense(NUM_CLASSES, activation='softmax'),
        ]),
        'DNN3': Sequential([
            Flatten(input_shape=INPUT_SHAPE),
            Dense(units=256, activation='relu'),
            Dropout(rate=dropout_rate),
            Dense(units=128, activation='relu'),
            Dropout(rate=dropout_rate),
            Dense(units=64, activation='relu'),
            Dropout(rate=dropout_rate),
            Dense(NUM_CLASSES, activation='softmax'),
        ]),
        'CNN1': Sequential([
            Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=INPUT_SHAPE),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(units=128, activation='relu'),
            Dropout(rate=dropout_rate),
            Dense(NUM_CLASSES, activation='softmax'),
        ]),
        'CNN2': Sequential([
            Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=INPUT_SHAPE),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
            Flatten(),
            Dense(units=128, activation='relu'),
            Dropout(rate=dropout_rate),
            Dense(NUM_CLASSES, activation='softmax'),
        ]),
        'CNN3': Sequential([
            Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=INPUT_SHAPE),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(units=128, activation='relu'),
            Dropout(rate=dropout_rate),
            Dense(NUM_CLASSES, activation='softmax'),
        ]),
        'CNN4': Sequential([
            Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=INPUT_SHAPE),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
            Flatten(),
            Dense(units=128, activation='relu'),
            Dropout(rate=dropout_rate),
            Dense(NUM_CLASSES, activation='softmax'),
        ]),
        'CNN5': Sequential([
            Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=INPUT_SHAPE),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
            Flatten(),
            Dense(units=128, activation='relu'),
            Dropout(rate=dropout_rate),
            Dense(NUM_CLASSES, activation='softmax'),
        ]),
            # 'CNN5': Sequential([
            #     Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=INPUT_SHAPE),
            #     BatchNormalization(),
            #     MaxPooling2D(pool_size=(2, 2)),
            #     Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
            #     BatchNormalization(),
            #     MaxPooling2D(pool_size=(2, 2)),
            #     Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
            #     BatchNormalization(),
            #     MaxPooling2D(pool_size=(2, 2)),
            #     Flatten(),
            #     Dropout(rate=dropout_rate),
            #     Dense(NUM_CLASSES, activation='softmax'),
            # ]),

            # 'LeNet1': Sequential([
            #     Conv2D(filters=4, kernel_size=(5, 5), activation='sigmoid', input_shape=INPUT_SHAPE),
            #     AveragePooling2D(pool_size=(2, 2)),
            #     Conv2D(filters=12, kernel_size=(5, 5), activation='sigmoid'),
            #     AveragePooling2D(pool_size=(2, 2)),
            #     Flatten(),
            #     Dense(NUM_CLASSES, activation='softmax'),
            # ]),
            # 'ResNetSimple': Sequential([
            #     Input(shape=INPUT_SHAPE),
            #     Conv2D(32, (3, 3), activation='relu', padding='same'),
            #     BatchNormalization(),
            #     Conv2D(32, (3, 3), activation='relu', padding='same'),
            #     BatchNormalization(),
            #     MaxPooling2D(pool_size=(2, 2)),
            #
            #     resnet_block(Conv2D(32, (3, 3), activation='relu', padding='same')(Input(shape=(14, 14, 32))), 32, (3, 3)),
            #     resnet_block(Conv2D(32, (3, 3), activation='relu', padding='same')(Input(shape=(14, 14, 32))), 32, (3, 3)),
            #
            #     Conv2D(64, (3, 3), activation='relu', padding='same'),
            #     BatchNormalization(),
            #     MaxPooling2D(pool_size=(2, 2)),
            #     Flatten(),
            #     Dense(128, activation='relu'),
            #     Dropout(rate=dropout_rate),
            #     Dense(NUM_CLASSES, activation='softmax')
            # ]),
            # 'VGG16': Sequential([
            #     VGG16(include_top=False, input_shape=INPUT_SHAPE, pooling='avg', weights=None),
            #     Flatten(),
            #     Dense(units=256, activation='relu'),
            #     Dropout(rate=dropout_rate),
            #     Dense(NUM_CLASSES, activation='softmax'),
            # ]),

        }
    assert model_name in MODELS_CONFIG.keys(), f"Unknown model name: {model_name}, choose one of {MODELS_CONFIG.keys()}"
    model = MODELS_CONFIG[model_name]
    assert optimizer in AVAILABLE_OPTIMIZERS.keys(), f"Unknown optimizer: {optimizer}, choose one of {AVAILABLE_OPTIMIZERS.keys()}"
    optimizer_obj = AVAILABLE_OPTIMIZERS[optimizer](learning_rate=learning_rate)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer_obj,
        metrics=['accuracy']
    )
    return model


if __name__ == "__main__":
    get_model(model_name='CNN1', dropout_rate=0.02, learning_rate=0.01, optimizer='adam', augmentation=False)