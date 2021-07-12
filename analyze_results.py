import pickle
import matplotlib.pyplot as plt
import numpy as np


log_path = 'training_log.pkl'

with open(log_path, 'rb') as file:
    history = pickle.load(file)

# print(np.count_nonzero(np.array(history['scores']) > 2))

figure, axis = plt.subplots(1, 4)
figure.set_size_inches(18.5, 5.5)
figure.tight_layout()

axis[0].plot(history['episodes'], history['rewards'])
axis[0].set_title('Rewards vs Episodes')
axis[0].set_xlabel('Episodes')
axis[0].set_ylabel('Rewards')

axis[1].plot(history['episodes'], history['steps'])
axis[1].set_title('Steps vs Episodes')
axis[1].set_xlabel('Episodes')
axis[1].set_ylabel('Steps')

axis[2].plot(history['episodes'], history['scores'])
axis[2].set_title('Scores vs Episodes')
axis[2].set_xlabel('Episodes')
axis[2].set_ylabel('Scores')

axis[3].scatter(history['validation']['episodes'], history['validation']['scores'])
axis[3].set_title('Val Mean Scores vs Episodes')
axis[3].set_xlabel('Episodes')
axis[3].set_ylabel('Mean Scores')

plt.show()

