from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.layers.normalization import BatchNormalization


def build_model_1(state_shape, nb_actions):
    model = Sequential(layers=[
        Conv2D(32, (3, 3), activation='relu', input_shape=state_shape),
        Conv2D(64, (3, 3), activation='relu'),
        Conv2D(128, (3, 3), activation='relu'),
        Flatten(),
        Dense(nb_actions, activation='linear')
    ])

    model.compile(optimizer=Adam(learning_rate=.0001), loss='mean_squared_error')
    return model


def build_model_2(state_shape, nb_actions):
    model = Sequential(layers=[
        Conv2D(32, (3, 3), activation='relu', input_shape=state_shape),
        Conv2D(64, (3, 3), activation='relu'),
        Conv2D(128, (3, 3), activation='relu'),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(nb_actions, activation='linear')
    ])

    model.compile(optimizer=Adam(learning_rate=.0001), loss='mean_squared_error')
    return model


def build_model_3(state_shape, nb_actions):
    model = Sequential(layers=[
        Conv2D(32, (3, 3), activation='relu', input_shape=state_shape),
        Conv2D(64, (3, 3), activation='relu'),
        Conv2D(128, (3, 3), activation='relu'),
        Conv2D(256, (3, 3), activation='relu'),
        Flatten(),
        Dense(nb_actions, activation='linear')
    ])

    model.compile(optimizer=Adam(learning_rate=.0001), loss='mean_squared_error')
    return model


def build_model_4(state_shape, nb_actions):
    model = Sequential(layers=[
        Conv2D(32, (8, 8), activation='relu', input_shape=state_shape),
        Conv2D(32, (5, 5), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(nb_actions, activation='linear')
    ])

    model.compile(optimizer=Adam(learning_rate=.001), loss='mean_squared_error')
    return model


def build_model_5(state_shape, nb_actions):
    model = Sequential(layers=[
        Conv2D(32, (8, 8), activation='relu', input_shape=state_shape),
        Conv2D(32, (5, 5), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(nb_actions, activation='linear')
    ])

    model.compile(optimizer=Adam(learning_rate=.0001), loss='mean_squared_error')
    return model


def build_model_6(state_shape, nb_actions):
    model = Sequential(layers=[
        Conv2D(32, (8, 8), input_shape=state_shape),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(32, (5, 5)),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(64, (3, 3)),
        BatchNormalization(),
        Activation('relu'),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(nb_actions, activation='linear')
    ])

    model.compile(optimizer=Adam(learning_rate=.0001), loss='mean_squared_error')
    return model