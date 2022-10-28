import tensorflow as tf
import numpy as np


class dataprovider():
    """
    Dataprovider object, provides random training data
    """
    def __init__(self):
        nn_input_0 = np.random.normal(size=(378, 300))
        nn_input_1 = np.random.normal(size=(378, 300))
        labels = np.concatenate([np.ones((189, 1), dtype=int), np.zeros((189, 1), dtype=int)], axis=0)

        self.data_dict = dict(
            nn_input_0=nn_input_0,
            nn_input_1=nn_input_1,
            labels=labels
        )

    @property
    def training_data(self):
        """
        Function that converts the underlying dictionary into tensorflow input data for training.
        
        The @property allows us to access the data via "dataprovider.training_data", so without
        parentheses, which is more convenient.
        """
        return self.convert_to_tensorflow(self.data_dict)

    def convert_to_tensorflow(self, data_dict):
        """
        We need to convert our data to Tensorflow format.
        """
        tensorflow_data = dict()
        for key, data in data_dict.items():
            tensorflow_data[key] = tf.convert_to_tensor(data)
            
        return tensorflow_data