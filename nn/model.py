from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten


class Model():
    ''' static class used to build model
    '''

    def build_model(state_shape, nb_actions, name='model'):
        return Sequential(layers=[
            Flatten(input_shape=state_shape),
            Dense(units=32, activation='relu'),
            Dense(units=32, activation='relu'),
            Dense(units=64, activation='relu'),
            Dense(units=128, activation='relu'),
            Dense(units=nb_actions, activation='linear'),
        ], name=name)
