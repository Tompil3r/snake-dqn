import pickle
import matplotlib.pyplot as plt


log_path = 'training_log.pkl'

with open(log_path, 'rb') as file:
    history = pickle.load(file)

figure, axis = plt.subplots(1, 2)

axis[0].set_title('Rewards')
axis[0].plot(history['episodes'], history['rewards'])

axis[1].set_title('Steps')
axis[1].plot(history['episodes'], history['steps'])

plt.show()

