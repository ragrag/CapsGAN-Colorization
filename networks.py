import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Multiply




class Discriminator(object):
	def __init__(self, name, training=True):
		self.name = name
		self.training = training
		self.var_list = []

	def create(self, input_, kernel_size=None, seed=None, reuse_variables=None):
		
		output = input_

		with tf.variable_scope(self.name, reuse=reuse_variables):
			output = tf.layers.conv2d(inputs=output,filters=256, kernel_size=9, strides=1, padding='valid', name='conv1')
			output = tf.nn.leaky_relu(output)

			#ADDED Batch Norm
			output = tf.layers.batch_normalization(inputs=output,momentum=0.8, training=self.training)

			#Primary Capsules
			output = tf.layers.conv2d(inputs=output,filters=8 * 32, kernel_size=9, strides=2, padding='valid', name='primarycap_conv2')
			output = tf.reshape(output,[-1, 8], name='primarycap_reshape')
			
			# Squashing
			output = squash(output)
			output = tf.layers.batch_normalization(inputs=output,momentum=0.8, training=self.training)


			# Digit Capsules      
			output = tf.layers.flatten(inputs=output)
			uhat = tf.layers.dense(inputs=output,units=160, kernel_initializer='he_normal', bias_initializer='zeros', name='uhat_digitcaps')
			#1
			digi_caps = tf.nn.softmax(
				uhat,
				name='softmax_digitcaps1',
			)
			digi_caps = tf.layers.dense(inputs=digi_caps,units=160)
			output = Multiply()([uhat, digi_caps])
			s_j = tf.nn.leaky_relu(output)

			#2
			digi_caps = tf.nn.softmax(
			s_j,
			name='softmax_digitcaps2',
			)
			digi_caps = tf.layers.dense(inputs=digi_caps,units=160) 
			output = Multiply()([uhat, digi_caps])
			s_j = tf.nn.leaky_relu(output)

			#3
			digi_caps = tf.nn.softmax(
			s_j,
			name='softmax_digitcaps3',
			)
			digi_caps = tf.layers.dense(inputs=digi_caps,units=160)
			output = Multiply()([uhat, digi_caps])
			s_j = tf.nn.leaky_relu(output)

			#Sigmoid
			pred = tf.layers.dense(inputs=s_j,units=1, activation='sigmoid')
			
			self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
			return  pred


class Generator(object):
	def __init__(self, name, output_channels=3, training=True):
		self.name = name
		self.output_channels = output_channels
		self.training = training
		self.var_list = []

	def create(self, inputs, kernel_size=None, seed=None, reuse_variables=None):
		output = inputs
		with tf.variable_scope(self.name, reuse=reuse_variables):
			layers = []
			# encoder 
			output = conv2d_encode(inputs=output,name='conv0',kernel_size=kernel_size,filters=64, strides=1,activation=tf.nn.leaky_relu,seed=seed)
			layers.append(output)

			output = conv2d_encode(inputs=output,name='conv1',kernel_size=kernel_size,filters=128, strides=2,activation=tf.nn.leaky_relu,seed=seed)
			layers.append(output)

			output = conv2d_encode(inputs=output,name='conv2',kernel_size=kernel_size,filters=256, strides=2,activation=tf.nn.leaky_relu,seed=seed)
			layers.append(output)

			output = conv2d_encode(inputs=output,name='conv3',kernel_size=kernel_size,filters=512, strides=2,activation=tf.nn.leaky_relu,seed=seed)
			layers.append(output)

			output = conv2d_encode(inputs=output,name='conv4',kernel_size=kernel_size,filters=512, strides=2,activation=tf.nn.leaky_relu,seed=seed)
			layers.append(output)


			#decoder
			output = conv2d_decode(inputs=output,name='deconv0',kernel_size=kernel_size,filters=512, strides=2,activation=tf.nn.leaky_relu,seed=seed)
			keep_prob = 1.0 - 0.5 if self.training else 1.0
			output = tf.nn.dropout(output, keep_prob=keep_prob, name='dropout_' + 'deconv0', seed=seed)
			output = tf.concat([layers[len(layers) - 0 - 2], output], axis=3)
			

			output = conv2d_decode(inputs=output,name='deconv1',kernel_size=kernel_size,filters=256, strides=2,activation=tf.nn.leaky_relu,seed=seed)
			keep_prob = 1.0 - 0.5 if self.training else 1.0
			output = tf.nn.dropout(output, keep_prob=keep_prob, name='dropout_' + 'deconv1', seed=seed)
			output = tf.concat([layers[len(layers) - 1 - 2], output], axis=3)
			

			output = conv2d_decode(inputs=output,name='deconv2',kernel_size=kernel_size,filters=128, strides=2,activation=tf.nn.leaky_relu,seed=seed)
			output = tf.concat([layers[len(layers) - 2 - 2], output], axis=3)

			output = conv2d_decode(inputs=output,name='deconv3',kernel_size=kernel_size,filters=64, strides=2,activation=tf.nn.leaky_relu,seed=seed)
			output = tf.concat([layers[len(layers) - 3 - 2], output], axis=3)

			output = conv2d_encode(inputs=output,name='conv_last',
				filters=self.output_channels,  
				kernel_size=1,                  
				strides=1,                      
				bnorm=False,                    
				activation=tf.nn.tanh,          
				seed=seed
			)
			self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)

			return output


def conv2d_encode(inputs, filters, name, kernel_size=4, strides=2, bnorm=True, activation=None, seed=None):

    initializer=tf.variance_scaling_initializer(seed=seed)
    res = tf.layers.conv2d(
        name=name,
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        kernel_initializer=initializer)

    if bnorm:
        res = tf.layers.batch_normalization(inputs=res, name='bn_' + name, training=True)

    if activation is not None:
        res = activation(res)

    return res


    
def conv2d_decode(inputs, filters, name, kernel_size=4, strides=2, bnorm=True, activation=None, seed=None):
 
    initializer=tf.variance_scaling_initializer(seed=seed)
    res = tf.layers.conv2d_transpose(
        name=name,
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        kernel_initializer=initializer)

    if bnorm:
        res = tf.layers.batch_normalization(inputs=res, name='bn_' + name, training=True)

    if activation is not None:
        res = activation(res)

    return res

def squash(vectors, axis=-1):
    s_squared_norm = tf.keras.backend.sum(tf.keras.backend.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / tf.keras.backend.sqrt(s_squared_norm + tf.keras.backend.epsilon())
    return scale * vectors
