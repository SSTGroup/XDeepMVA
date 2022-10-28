import tensorflow as tf
from tensorflow.keras import layers

from utils import CCA, compute_l2

from model_fc import SeizureModel

class GRUEncoder(tf.keras.Model):
    def __init__(self, config, view_ind, **kwargs):
        super(GRUEncoder, self).__init__(name=f'GRUEncoder_view_{view_ind}', **kwargs)
        self.config = config

        self.rnn_layers = [
            layers.GRU(units=units, return_sequences=fs) 
                for (units,fs) in self.config
            ]

    def get_l2(self):
        return tf.math.reduce_sum([compute_l2(layer.trainable_variables[0]) for layer in self.rnn_layers])

    def call(self, inputs):
        x = inputs
        for layer in self.rnn_layers:
            x = layer(x)
        return x

class GRUDecoder(tf.keras.Model):
    def __init__(self, config, view_ind, **kwargs):
        super(GRUDecoder, self).__init__(name=f'GRUDecoder_view_{view_ind}', **kwargs)
        self.reconst_size = config[0]
        self.config = config[1:]

        self.rnn_layers = [
            layers.GRU(units=units, return_sequences=fs) 
                for (units,fs) in self.config
            ]

    def get_l2(self):
        return tf.math.reduce_sum([compute_l2(layer.trainable_variables[0]) for layer in self.rnn_layers])

    def call(self, inputs):
        x = inputs

        # First layer to get back to orig. dimension
        x = layers.RepeatVector(n=self.reconst_size)(x)

        for layer in self.rnn_layers:
            x = layer(x)

        return x


class GRUSeizureModel(SeizureModel):
    def __init__(self, encoder_config, decoder_config, name="SeizureModel", **kwargs):
        super(SeizureModel, self).__init__(name=name, **kwargs)
        self.encoder_v0 = GRUEncoder(config=encoder_config, view_ind=0)
        self.encoder_v1 = GRUEncoder(config=encoder_config, view_ind=1)
        
        self.decoder_v0 = GRUDecoder(config=decoder_config, view_ind=0)
        self.decoder_v1 = GRUDecoder(config=decoder_config, view_ind=1)

    def call(self, inputs, training=False, cca_reg=0):
        inp_view_0 = inputs['nn_input_0']
        inp_view_1 = inputs['nn_input_1']

        # Compute latent variables
        # Shape operations analouge to the CNN case
        latent_view_0 = tf.squeeze(self.encoder_v0(tf.expand_dims(inp_view_0, axis=2)))
        latent_view_1 = tf.squeeze(self.encoder_v1(tf.expand_dims(inp_view_1, axis=2)))
        
        # Reconstruct via decoder
        # Here, the decoder begins with a RepeatVector layer, thats why we dont need a
        # dimensionality expansion here
        reconst_view_0 = tf.squeeze(self.decoder_v0(latent_view_0))
        reconst_view_1 = tf.squeeze(self.decoder_v1(latent_view_1))

        if training == True:
            # During training we compute the CCA and save the latest transformations B1 and B2
            B1, B2, epsilon, omega, ccor = CCA(
                latent_view_0, 
                latent_view_1, 
                latent_view_1.shape[1], 
                rx=cca_reg, 
                ry=cca_reg
            )
            self.B1 = B1
            self.B2 = B2
        else:
            # During evaluation we use the latest transformations B1 and B2 to compute epsilon and omega
            m = latent_view_0.shape[0]
            v0_mean = tf.reduce_mean(latent_view_0, 0)
            v1_mean = tf.reduce_mean(latent_view_1, 0)
            v0_bar = tf.subtract(latent_view_0, tf.tile(tf.convert_to_tensor(v0_mean)[None], [m, 1]))
            v1_bar = tf.subtract(latent_view_1, tf.tile(tf.convert_to_tensor(v1_mean)[None], [m, 1]))
            epsilon = self.B1@tf.transpose(v0_bar)
            omega = self.B2@tf.transpose(v1_bar)
            diagonal = tf.linalg.diag_part(epsilon@tf.transpose(omega))
            ccor = diagonal / m
        
        return {
            'latent_view_0':latent_view_0, 
            'latent_view_1':latent_view_1, 
            'cca_view_0':tf.transpose(epsilon),
            'cca_view_1':tf.transpose(omega),
            'ccor':ccor,
            'reconst_view_0':reconst_view_0, 
            'reconst_view_1':reconst_view_1
        }
    