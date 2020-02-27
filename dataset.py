import os
import glob
import numpy as np
import tensorflow as tf
from imageio import imread
from abc import abstractmethod




class BaseDataset():
	def __init__(self, name, path, training=True, augment=True):
		self.name = name
		self.augment = augment and training
		self.training = training
		self.path = path
		self._data = []

	def __len__(self):
		return len(self.data)

	def __iter__(self):
		total = len(self)
		start = 0

		while start < total:
			item = self[start]
			start += 1
			yield item

		raise StopIteration

	def __getitem__(self, index):
		val = self.data[index]
		try:
			img = imread(val) if isinstance(val, str) else val

			# grayscale images
			if np.sum(img[:,:,0] - img[:,:,1]) == 0 and np.sum(img[:,:,0] - img[:,:,2]) == 0:
				return None

			if self.augment and np.random.binomial(1, 0.5) == 1:
				img = img[:, ::-1, :]

		except:
			img = None

		return img

	def generator(self, batch_size, recusrive=False):
		start = 0
		total = len(self)

		while True:
			while start < total:
				end = np.min([start + batch_size, total])
				items = []

				for ix in range(start, end):
					item = self[ix]
					if item is not None:
						items.append(item)

				start = end
				yield items

			if recusrive:
				start = 0

			else:
				raise StopIteration

	@property
	def data(self):
		if len(self._data) == 0:
			self._data = self.load()
			#np.random.shuffle(self._data)

		return self._data

	@abstractmethod
	def load(self):
		return []

class LinnaeusDataset(BaseDataset):
	def __init__(self, dataset_name, path, training=True, augment=True):
		super(LinnaeusDataset, self).__init__(dataset_name, path, training, augment)

	def load(self):
		if self.training:
			flist = os.path.join(self.path, 'train.flist')
			if os.path.exists(flist):
				os.unlink(flist)
				data = glob.glob(self.path +  '/train/**/*.jpg', recursive=True)
				np.savetxt(flist, data, fmt='%s')


			else:
				data = glob.glob(self.path + '/train/**/*.jpg', recursive=True)
				np.savetxt(flist, data, fmt='%s')

		else:
			flist = os.path.join(self.path, 'test.flist')
			if os.path.exists(flist):
				os.unlink(flist)
				data = np.array(glob.glob(self.path + '/test/**/*.jpg', recursive=True))
				data = np.sort(data)
				np.savetxt(flist, data, fmt='%s')

			else:
				data = np.array(glob.glob(self.path  + '/test/**/*.jpg', recursive=True))
				data = np.sort(data)
				np.savetxt(flist, data, fmt='%s')

		return data

