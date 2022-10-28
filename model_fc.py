import tensorflow as tf
from tensorflow.keras import layers

from utils import CCA, compute_l2


class Encoder(tf.keras.Model):
    def __init__(self, config, view_ind, **kwargs):
        super(Encoder, self).__init__(name=f'Encoder_view_{view_ind}', **kwargs)
        self.config = config
        self.view_index = view_ind

        self.dense_layers = [
            layers.Dense(
                dim,
                activation=activ,
            ) for (dim, activ) in self.config
        ]

    def get_l2(self):
        return tf.math.reduce_sum([compute_l2(layer.trainable_variables[0]) for layer in self.dense_layers])

    def call(self, inputs):
        x = inputs
        for layer in self.dense_layers:
            x = layer(x)

        return x


class Decoder(tf.keras.Model):
    def __init__(self, config, view_ind, **kwargs):
        super(Decoder, self).__init__(name=f'Decoder_view_{view_ind}', **kwargs)
        self.config = config
        self.view_index = view_ind

        self.dense_layers = [
            layers.Dense(
                dim,
                activation=activ,
            ) for (dim, activ) in self.config
        ]

    def get_l2(self):
        return tf.math.reduce_sum([compute_l2(layer.trainable_variables[0]) for layer in self.dense_layers])

    def call(self, inputs):
        x = inputs
        for layer in self.dense_layers:
            x = layer(x)

        return x
        

class SeizureModel(tf.keras.Model):
    def __init__(self, encoder_config, decoder_config, name="SeizureModel", **kwargs):
        super(SeizureModel, self).__init__(name=name, **kwargs)
        self.encoder_v0 = Encoder(config=encoder_config, view_ind=0)
        self.encoder_v1 = Encoder(config=encoder_config, view_ind=1)
        
        self.decoder_v0 = Decoder(config=decoder_config, view_ind=0)
        self.decoder_v1 = Decoder(config=decoder_config, view_ind=1)
        
    def get_l2(self):
        enc_0_l2 = self.encoder_v0.get_l2()
        enc_1_l2 = self.encoder_v1.get_l2()
        dec_0_l2 = self.decoder_v0.get_l2()
        dec_1_l2 = self.decoder_v1.get_l2()
        return tf.math.reduce_sum([enc_0_l2, enc_1_l2, dec_0_l2, dec_1_l2])
        
    def call(self, inputs, training=False, cca_reg=0):
        inp_view_0 = inputs['nn_input_0']
        inp_view_1 = inputs['nn_input_1']

        # Compute latent variables
        latent_view_0 = self.encoder_v0(inp_view_0)
        latent_view_1 = self.encoder_v1(inp_view_1)
        
        # Reconstruct via decoder
        reconst_view_0 = self.decoder_v0(latent_view_0)
        reconst_view_1 = self.decoder_v1(latent_view_1)

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
 