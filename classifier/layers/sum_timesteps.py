import tensorflow.keras as keras

sum_timesteps = keras.layers.Lambda(lambda x: keras.backend.sum(x, axis=1, keepdims=False))