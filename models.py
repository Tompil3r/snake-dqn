from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D
from tensorflow.keras.optimizers import Adam


def build_conv_model(state_shape, nb_actions, learning_rate):
    model = Sequential(layers=[
        Conv2D(64, (7, 7), activation='relu', input_shape=state_shape),
        Conv2D(64, (5, 5), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(nb_actions, activation='linear')
    ])

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
    return model