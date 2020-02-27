from __future__ import print_function

import os
import time
import numpy as np
import tensorflow as tf

from abc import abstractmethod
from networks import Generator, Discriminator
from dataset import LinnaeusDataset
from utility import stitch_images,  imsave, create_dir, Progbar, pixelwise_accuracy, preprocess, postprocess,COLORSPACE_RGB, COLORSPACE_LAB


class BaseModel:
	def __init__(self, sess, options):
		self.sess = sess
		self.options = options
		self.name = options.name
		self.global_step = tf.Variable(0, name='global_step', trainable=False)
		self.dataset_train = self.create_dataset(self.options.dataset, self.options.dataset_path,True)
		self.dataset_val = self.create_dataset(self.options.dataset, self.options.dataset_path, False)
		self.sample_generator = self.dataset_val.generator(options.sample_size, True)
		self.iteration = 0
		self.epoch = 0
		self.is_built = False

	def train(self):
		print("TRAINING START...")
		total = len(self.dataset_train)
		for epoch in range(self.options.epochs):
			lr_rate = self.sess.run(self.learning_rate)
			print('Training epoch: %d' % (epoch + 1) + " - learning rate: " + str(lr_rate))
			self.epoch = epoch + 1
			self.iteration = 0
			generator = self.dataset_train.generator(self.options.batch_size)
			progbar = Progbar(total, width=25, stateful_metrics=['epoch', 'iter', 'step'])
			for input_rgb in generator:
				feed_dic = {self.input_rgb: input_rgb}
				self.iteration = self.iteration + 1
				self.sess.run([self.dis_train], feed_dict=feed_dic)
				self.sess.run([self.gen_train, self.accuracy], feed_dict=feed_dic)
				self.sess.run([self.gen_train, self.accuracy], feed_dict=feed_dic)

				lossD, lossD_fake, lossD_real, lossG, lossG_l1, lossG_gan, acc, step = self.evaluate_outputs(feed_dic=feed_dic)

				progbar.add(len(input_rgb), values=[
					("epoch", epoch + 1),
					("iter", self.iteration),
					("step", step),
					("D loss", lossD),
					("D fake", lossD_fake),
					("D real", lossD_real),
					("G loss", lossG),
					("G L1", lossG_l1),
					("G gan", lossG_gan),
					("accuracy", acc)
				])


				if self.options.sample and step % self.options.sample_interval == 0:
					self.sample(show=False)

				if self.options.save and step % self.options.save_interval == 0:
					self.save()



	def sample(self, show=True):
		print("TESTING START...")
		input_rgb = next(self.sample_generator)
		feed_dic = {self.input_rgb: input_rgb}
		outputs_path = create_dir(self.options.checkpoints_path + '/output')
		step, rate = self.sess.run([self.global_step, self.learning_rate])
		fake_image, input_gray = self.sess.run([self.sampler, self.input_gray], feed_dict=feed_dic)
		fake_image = postprocess(tf.convert_to_tensor(fake_image), colorspace_in='LAB', colorspace_out=COLORSPACE_RGB)
		stitch_images(input_gray, input_rgb, fake_image.eval(),outputs_path,self.options.dataset + "_" + str(step).zfill(5))

	def build(self):
		print("BUILDING MODEL...")
		
		if self.is_built:
			return

		self.is_built = True

		generator_factory = self.create_generator()
		discriminator_factory = self.create_discriminator()
		smoothing = 1
		seed = self.options.seed
		kernel = 4

		self.input_rgb = tf.placeholder(tf.float32, shape=(None, None, None, 3), name='input_rgb')

		self.input_color = preprocess(self.input_rgb, colorspace_in=COLORSPACE_RGB, colorspace_out='LAB')

		self.input_gray = tf.image.rgb_to_grayscale(self.input_rgb)

		generator = generator_factory.create(self.input_gray, kernel, seed)
		discriminator_real = discriminator_factory.create(tf.concat([self.input_gray, self.input_color], 3), kernel, seed)
		discriminator_fake = discriminator_factory.create(tf.concat([self.input_gray, generator], 3), kernel, seed, reuse_variables=True)

		generator_ce = tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminator_fake, labels=tf.ones_like(discriminator_fake))
		discriminator_real_ce = tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminator_real, labels=tf.ones_like(discriminator_real) * smoothing)
		discriminator_real_ce = tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminator_fake, labels=tf.zeros_like(discriminator_fake))

		self.dis_loss_real = tf.reduce_mean(discriminator_real_ce)
		self.dis_loss_fake = tf.reduce_mean(discriminator_real_ce)
		self.dis_loss = tf.reduce_mean(discriminator_real_ce + discriminator_real_ce)

		self.gen_loss_gan = tf.reduce_mean(generator_ce)
		self.gen_loss_l1 = tf.reduce_mean(tf.abs(self.input_color - generator)) * 100.0
		self.gen_loss = self.gen_loss_gan + self.gen_loss_l1

		self.sampler = tf.identity(generator_factory.create(self.input_gray, kernel, seed, reuse_variables=True), name='output')
		self.accuracy = pixelwise_accuracy(self.input_color, generator, 'LAB', 2.0)
		self.learning_rate = tf.constant(self.options.lr)

		if self.options.lr_decay and self.options.lr_decay_rate > 0:
			self.learning_rate = tf.maximum(1e-6, tf.train.exponential_decay(
				learning_rate=self.options.lr,
				global_step=self.global_step,
				decay_steps=self.options.lr_decay_steps,
				decay_rate=self.options.lr_decay_rate))

		self.gen_train = tf.train.AdamOptimizer(
			learning_rate=self.learning_rate,
			beta1=0
		).minimize(self.gen_loss, var_list=generator_factory.var_list)

		self.dis_train = tf.train.AdamOptimizer(
			learning_rate=self.learning_rate / 10,
			beta1=0
		).minimize(self.dis_loss, var_list=discriminator_factory.var_list, global_step=self.global_step)

		self.saver = tf.train.Saver()

	def load(self):
		ckpt = tf.train.get_checkpoint_state(self.options.checkpoints_path)
		if ckpt is not None:
			print('loading model...\n')
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			self.saver.restore(self.sess, os.path.join(self.options.checkpoints_path, ckpt_name))
			return True

		return False

	def save(self):
		print('saving model...\n')
		self.saver.save(self.sess, os.path.join(self.options.checkpoints_path, 'CGAN_' + self.options.dataset), write_meta_graph=False)

	def evaluate_outputs(self, feed_dic):

		lossD_fake = self.dis_loss_fake.eval(feed_dict=feed_dic)
		lossD_real = self.dis_loss_real.eval(feed_dict=feed_dic)
		lossD = self.dis_loss.eval(feed_dict=feed_dic)

		lossG_l1 = self.gen_loss_l1.eval(feed_dict=feed_dic)
		lossG_gan = self.gen_loss_gan.eval(feed_dict=feed_dic)
		lossG = lossG_l1 + lossG_gan

		acc = self.accuracy.eval(feed_dict=feed_dic)
		step = self.sess.run(self.global_step)

		return lossD, lossD_fake, lossD_real, lossG, lossG_l1, lossG_gan, acc, step

	@abstractmethod
	def create_generator(self):
		raise NotImplementedError

	@abstractmethod
	def create_discriminator(self):
		raise NotImplementedError

	@abstractmethod
	def create_dataset(self,dataset_name, dpath, training):
		raise NotImplementedError


class LinnaeusModel(BaseModel):
	def __init__(self, sess, options):
		super(LinnaeusModel, self).__init__(sess, options)

	def create_generator(self):
		return Generator('gen', training=self.options.training)

	def create_discriminator(self):
		return Discriminator('dis',  training=self.options.training)

	def create_dataset(self,dataset_name, dpath,training=True):
		return LinnaeusDataset(
			dataset_name = dataset_name,
			path=dpath,
			training=training,
			augment=True)
