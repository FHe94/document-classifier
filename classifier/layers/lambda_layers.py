import tensorflow.keras
from tensorflow.keras import backend as K

sum_timesteps = tensorflow.keras.layers.Lambda(lambda x: K.sum(x, axis=1, keepdims=False))
reverse_timesteps = tensorflow.keras.layers.Lambda(lambda x: K.reverse(x, 1))