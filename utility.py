import os
import shutil, sys  
import time
import random
import numpy as np
from PIL import Image
import tensorflow as tf
from keras import backend as K
COLORSPACE_RGB = 'RGB'
COLORSPACE_LAB = 'LAB'
tf.nn.softmax_cross_entropy_with_logits_v2



def stitch_images(grayscale, original, pred,pathToSave,nameToSave):
	gap = 5
	width, height = original[0][:, :, 0].shape
	img_per_row = 1
	img = Image.new('RGB', (width * img_per_row * 3 + gap * (img_per_row - 1), height * int(len(original) / img_per_row)))

	grayscale = np.array(grayscale).squeeze()
	original = np.array(original)
	pred = np.array(pred)
	savePath = pathToSave+"/"+nameToSave

	if not os.path.exists(pathToSave):
			os.makedirs(savePath)
			os.makedirs(savePath+'/grayscale/')
			os.makedirs(savePath+'/real/')
			os.makedirs(savePath+'/prediction/')
	else:
			#shutil.rmtree(pathToSave)
			os.makedirs(savePath)
			os.makedirs(savePath+'/grayscale/')
			os.makedirs(savePath+'/real/')
			os.makedirs(savePath+'/prediction/')

	for ix in range(len(original)):
		xoffset = int(0 % img_per_row) * width * 3 + int(0 % img_per_row) * gap
		yoffset = int(0 / img_per_row) * height
		im1 = Image.fromarray(grayscale[ix])
		im2 = Image.fromarray(original[ix])
		im3 = Image.fromarray((pred[ix] * 255).astype(np.uint8))
		savePath = pathToSave+"/"+nameToSave
		print("Saving image ",str(ix+1))
		imsave(im1,os.path.join(savePath+'/grayscale/', str(ix+1)+"_gs_"+nameToSave+".png"))
		imsave(im2,os.path.join(savePath+'/real/', str(ix+1)+"_real_"+nameToSave+".png"))
		imsave(im3,os.path.join(savePath+'/prediction/', str(ix+1)+"_caps_"+nameToSave+".png"))
		

def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

    return dir



def moving_average(data, window_width):
    cumsum_vec = np.cumsum(np.insert(data, 0, 0))
    ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
    return ma_vec



def imsave(img, path):
    im = Image.fromarray(np.array(img).astype(np.uint8).squeeze())
    im.save(path)


class Progbar(object):


    def __init__(self, target, width=25, verbose=1, interval=0.05,
                 stateful_metrics=None):
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()

        self._dynamic_display = ((hasattr(sys.stdout, 'isatty') and
                                  sys.stdout.isatty()) or
                                 'ipykernel' in sys.modules or
                                 'posix' in sys.modules)
        self._total_width = 0
        self._seen_so_far = 0

        self._values = {}
        self._values_order = []
        self._start = time.time()
        self._last_update = 0

    def update(self, current, values=None):

        values = values or []
        for k, v in values:
            if k not in self._values_order:
                self._values_order.append(k)
            if k not in self.stateful_metrics:
                if k not in self._values:
                    self._values[k] = [v * (current - self._seen_so_far),
                                       current - self._seen_so_far]
                else:
                    self._values[k][0] += v * (current - self._seen_so_far)
                    self._values[k][1] += (current - self._seen_so_far)
            else:
                self._values[k] = v
        self._seen_so_far = current

        now = time.time()
        info = ' - %.0fs' % (now - self._start)
        if self.verbose == 1:
            if (now - self._last_update < self.interval and
                    self.target is not None and current < self.target):
                return

            prev_total_width = self._total_width
            if self._dynamic_display:
                sys.stdout.write('\b' * prev_total_width)
                sys.stdout.write('\r')
            else:
                sys.stdout.write('\n')

            if self.target is not None:
                numdigits = int(np.floor(np.log10(self.target))) + 1
                barstr = '%%%dd/%d [' % (numdigits, self.target)
                bar = barstr % current
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += ('=' * (prog_width - 1))
                    if current < self.target:
                        bar += '>'
                    else:
                        bar += '='
                bar += ('.' * (self.width - prog_width))
                bar += ']'
            else:
                bar = '%7d/Unknown' % current

            self._total_width = len(bar)
            sys.stdout.write(bar)

            if current:
                time_per_unit = (now - self._start) / current
            else:
                time_per_unit = 0
            if self.target is not None and current < self.target:
                eta = time_per_unit * (self.target - current)
                if eta > 3600:
                    eta_format = '%d:%02d:%02d' % (eta // 3600,
                                                   (eta % 3600) // 60,
                                                   eta % 60)
                elif eta > 60:
                    eta_format = '%d:%02d' % (eta // 60, eta % 60)
                else:
                    eta_format = '%ds' % eta

                info = ' - ETA: %s' % eta_format
            else:
                if time_per_unit >= 1:
                    info += ' %.0fs/step' % time_per_unit
                elif time_per_unit >= 1e-3:
                    info += ' %.0fms/step' % (time_per_unit * 1e3)
                else:
                    info += ' %.0fus/step' % (time_per_unit * 1e6)

            for k in self._values_order:
                info += ' - %s:' % k
                if isinstance(self._values[k], list):
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                else:
                    info += ' %s' % self._values[k]

            self._total_width += len(info)
            if prev_total_width > self._total_width:
                info += (' ' * (prev_total_width - self._total_width))

            if self.target is not None and current >= self.target:
                info += '\n'

            sys.stdout.write(info)
            sys.stdout.flush()

        elif self.verbose == 2:
            if self.target is None or current >= self.target:
                for k in self._values_order:
                    info += ' - %s:' % k
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                info += '\n'

                sys.stdout.write(info)
                sys.stdout.flush()

        self._last_update = now

    def add(self, n, values=None):
        self.update(self._seen_so_far + n, values)






def pixelwise_accuracy(img_real, img_fake, colorspace, thresh):

    img_real = postprocess(img_real, colorspace, COLORSPACE_LAB)
    img_fake = postprocess(img_fake, colorspace, COLORSPACE_LAB)

    diffL = tf.abs(tf.round(img_real[..., 0]) - tf.round(img_fake[..., 0]))
    diffA = tf.abs(tf.round(img_real[..., 1]) - tf.round(img_fake[..., 1]))
    diffB = tf.abs(tf.round(img_real[..., 2]) - tf.round(img_fake[..., 2]))

    predL = tf.cast(tf.less_equal(diffL, 1 * thresh), tf.float64)        
    predA = tf.cast(tf.less_equal(diffA, 2.2 * thresh), tf.float64)      
    predB = tf.cast(tf.less_equal(diffB, 2.2 * thresh), tf.float64)      

    
    pred = predL * predA * predB

    return tf.reduce_mean(pred)


def preprocess(img, colorspace_in, colorspace_out):
    if colorspace_out.upper() == COLORSPACE_RGB:
        if colorspace_in == COLORSPACE_LAB:
            img = lab_to_rgb(img)

        img = (img / 255.0) * 2 - 1

    elif colorspace_out.upper() == COLORSPACE_LAB:
        if colorspace_in == COLORSPACE_RGB:
            img = rgb_to_lab(img / 255.0)

        L_chan, a_chan, b_chan = tf.unstack(img, axis=3)
	
        img = tf.stack([L_chan / 50 - 1, a_chan / 110, b_chan / 110], axis=3)

    return img


def postprocess(img, colorspace_in, colorspace_out):
    if colorspace_in.upper() == COLORSPACE_RGB:
        img = (img + 1) / 2

        if colorspace_out == COLORSPACE_LAB:
            img = rgb_to_lab(img)

    elif colorspace_in.upper() == COLORSPACE_LAB:
        L_chan, a_chan, b_chan = tf.unstack(img, axis=3)

        img = tf.stack([(L_chan + 1) / 2 * 100, a_chan * 110, b_chan * 110], axis=3)

        if colorspace_out == COLORSPACE_RGB:
            img = lab_to_rgb(img)

    return img


def rgb_to_lab(srgb):
    
    with tf.name_scope("rgb_to_lab"):
        srgb_pixels = tf.reshape(srgb, [-1, 3])

        with tf.name_scope("srgb_to_xyz"):
            linear_mask = tf.cast(srgb_pixels <= 0.04045, dtype=tf.float32)
            exponential_mask = tf.cast(srgb_pixels > 0.04045, dtype=tf.float32)
            rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
            rgb_to_xyz = tf.constant([
                [0.412453, 0.212671, 0.019334], 
                [0.357580, 0.715160, 0.119193],  
                [0.180423, 0.072169, 0.950227],  
            ])
            xyz_pixels = tf.matmul(rgb_pixels, rgb_to_xyz)

       
        with tf.name_scope("xyz_to_cielab"):
            xyz_normalized_pixels = tf.multiply(xyz_pixels, [1 / 0.950456, 1.0, 1 / 1.088754])

            epsilon = 6 / 29
            linear_mask = tf.cast(xyz_normalized_pixels <= (epsilon**3), dtype=tf.float32)
            exponential_mask = tf.cast(xyz_normalized_pixels > (epsilon**3), dtype=tf.float32)
            fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon**2) + 4 / 29) * linear_mask + (xyz_normalized_pixels ** (1 / 3)) * exponential_mask

            fxfyfz_to_lab = tf.constant([
                [0.0, 500.0, 0.0],  
                [116.0, -500.0, 200.0], 
                [0.0, 0.0, -200.0], 
            ])
            lab_pixels = tf.matmul(fxfyfz_pixels, fxfyfz_to_lab) + tf.constant([-16.0, 0.0, 0.0])

        return tf.reshape(lab_pixels, tf.shape(srgb))


def lab_to_rgb(lab):
    with tf.name_scope("lab_to_rgb"):
        lab_pixels = tf.reshape(lab, [-1, 3])

        
        with tf.name_scope("cielab_to_xyz"):
            lab_to_fxfyfz = tf.constant([
                [1 / 116.0, 1 / 116.0, 1 / 116.0],  
                [1 / 500.0, 0.0, 0.0], 
                [0.0, 0.0, -1 / 200.0], 
            ])
            fxfyfz_pixels = tf.matmul(lab_pixels + tf.constant([16.0, 0.0, 0.0]), lab_to_fxfyfz)

            epsilon = 6 / 29
            linear_mask = tf.cast(fxfyfz_pixels <= epsilon, dtype=tf.float32)
            exponential_mask = tf.cast(fxfyfz_pixels > epsilon, dtype=tf.float32)
            xyz_pixels = (3 * epsilon**2 * (fxfyfz_pixels - 4 / 29)) * linear_mask + (fxfyfz_pixels ** 3) * exponential_mask
            
            xyz_pixels = tf.multiply(xyz_pixels, [0.950456, 1.0, 1.088754])

        with tf.name_scope("xyz_to_srgb"):
            xyz_to_rgb = tf.constant([
   
                [3.2404542, -0.9692660, 0.0556434], 
                [-1.5371385, 1.8760108, -0.2040259], 
                [-0.4985314, 0.0415560, 1.0572252],  
            ])
            rgb_pixels = tf.matmul(xyz_pixels, xyz_to_rgb)
            rgb_pixels = tf.clip_by_value(rgb_pixels, 0.0, 1.0)
            linear_mask = tf.cast(rgb_pixels <= 0.0031308, dtype=tf.float32)
            exponential_mask = tf.cast(rgb_pixels > 0.0031308, dtype=tf.float32)
            srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + ((rgb_pixels ** (1 / 2.4) * 1.055) - 0.055) * exponential_mask

        return tf.reshape(srgb_pixels, tf.shape(lab))
