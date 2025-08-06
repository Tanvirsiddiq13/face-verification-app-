# Import dependencies
import tensorflow as tf
from tensorflow.keras.layers import Layer

# Custom L1 Distance Layer
# Custom L1 Distance Layer
class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, inputs):
        input_embedding, validation_embedding = inputs
        return tf.math.abs(input_embedding - validation_embedding)
