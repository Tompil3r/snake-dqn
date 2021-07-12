import pickle
import matplotlib.pyplot as plt


log_path = 'training_log.pkl'

with open(log_path, 'rb') as file:
    history = pickle.load(file)

plt.plot(history['episodes'], history['rewards'])
plt.show()



