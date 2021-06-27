from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras import activations


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