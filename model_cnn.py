import tensorflow as tf
from tensorflow.keras import layers

from utils import CCA, compute_l2

from model_fc import SeizureModel


class ConvEncoder(tf.keras.Model):
    def __init__(self, config, view_ind, **kwargs):
        super(ConvEncoder, self).__init__(name=f'ConvEncoder_view_{view_ind}', **kwargs)
        self.conv_layers = list()
        for layer_conf in config:
            # Check whether input config is valid and supported
            assert layer_conf['l_type'] in ['conv', 'maxpool']

            if layer_conf['l_type'] == 'conv':
                self.conv_layers.append(
                    layers.Conv1D(
                        filters=layer_conf['n_filters'],
                        kernel_size=layer_conf['k_size'],
                        strides=1,
                        padding='same',
                        activation=None
                    )
                )
            elif layer_conf['l_type'] == 'maxpool':
                self.conv_layers.append(
                    layers.MaxPool1D(
                        pool_size=layer_conf['pool_size'],
                        strides=None,
                        padding='valid',
                    )
                )
            else:
                raise NotImplementedError

    def get_l2(self):
        return tf.math.reduce_sum([compute_l2(layer.trainable_variables[0]) if len(layer.trainable_variables)>0 else 0 for layer in self.conv_layers])

    def call(self, inputs):
        x = inputs
        for layer in self.conv_layers:
            x = layer(x)

        return x
        
class ConvDecoder(tf.keras.Model):
    def __init__(self, config, view_ind, **kwargs):
        super(ConvDecoder, self).__init__(name=f'ConvDecoder_view_{view_ind}', **kwargs)
        self.conv_layers = list()
        for layer_conf in config:
            # Check whether input config is valid and supported
            assert layer_conf['l_type'] in ['conv', 'conv_transp']

            if layer_conf['l_type'] == 'conv':
                self.conv_layers.append(
                    layers.Conv1D(
                        filters=layer_conf['n_filters'],
                        kernel_size=layer_conf['k_size'],
                        strides=1,
                        padding='same',
                        activation=None
                    )
                )
            elif layer_conf['l_type'] == 'conv_transp':
                self.conv_layers.append(
                    layers.Conv1DTranspose(
                        filters=layer_conf['n_filters'],
                        kernel_size=layer_conf['k_size'],
                        strides=layer_conf['strides'],
                        padding='valid',
                        activation=None
                    )
                )
            else:
                raise NotImplementedError

    def get_l2(self):
        return tf.math.reduce_sum([compute_l2(layer.trainable_variables[0]) if len(layer.trainable_variables)>0 else 0 for layer in self.conv_layers])
        
    def call(self, inputs):
        x = inputs
        for layer in self.conv_layers:
            x = layer(x)

        return x


class ConvSeizureModel(SeizureModel):
    def __init__(self, encoder_config, decoder_config, name="SeizureModel", **kwargs):
        super(SeizureModel, self).__init__(name=name, **kwargs)
        self.encoder_v0 = ConvEncoder(config=encoder_config, view_ind=0)
        self.encoder_v1 = ConvEncoder(config=encoder_config, view_ind=1)
        
        self.decoder_v0 = ConvDecoder(config=decoder_config, view_ind=0)
        self.decoder_v1 = ConvDecoder(config=decoder_config, view_ind=1)

    def call(self, inputs, training=False, cca_reg=0):
        inp_view_0 = inputs['nn_input_0']
        inp_view_1 = inputs['nn_input_1']
        
        # Compute latent variables
        # 1D-Convlutions expect an input of shape (num_samples, num_timestamps, num_dimensions)
        # As our dimension is 1, we use tf.expand_dims to add the last dimension
        # We then use squeeze to remove the 1 in the last dimension
        latent_view_0 = tf.squeeze(self.encoder_v0(tf.expand_dims(inp_view_0, axis=2)))
        latent_view_1 = tf.squeeze(self.encoder_v1(tf.expand_dims(inp_view_1, axis=2)))

        # Reconstruct via decoder
        reconst_view_0 = tf.squeeze(self.decoder_v0(tf.expand_dims(latent_view_0, axis=2)))
        reconst_view_1 = tf.squeeze(self.decoder_v1(tf.expand_dims(latent_view_1, axis=2)))

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
