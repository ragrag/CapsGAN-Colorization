import os
import random
import numpy as np
import tensorflow as tf
from models import  LinnaeusModel
from keras import backend as K
import easydict

def main(options):
    print("Starting...")
    
    tf.reset_default_graph()

    tf.set_random_seed(options.seed)
    np.random.seed(options.seed)
    random.seed(options.seed)

    with tf.Session() as sess:
        model = LinnaeusModel(sess, options)

        if not os.path.exists(options.checkpoints_path):
            os.makedirs(options.checkpoints_path)

        model.build()
        sess.run(tf.global_variables_initializer())
        model.load()


        if options.mode == 0:
            model.train()
        
        elif options.mode == 1:
            model.sample()





options = easydict.EasyDict({
    'batch':100,
    'seed':random.randint(0, 2**31 - 1), 
    'name':'CGAN', 
    'mode': 1,
    'dataset':'linnaeus5',
    'dataset_path':'./gdrive/My Drive/dataset', 
    'checkpoints_path':'./gdrive/My Drive/dataset/checkpoints',
    'batch_size':30,
    'epochs':201, 
    'lr':3e-4,
    'lr_decay':True, 
    'lr_decay_rate':0.1,
    'lr_decay_steps':5e5, 
    'gpu_ids':'0', 
    'save':True, 
    'save_interval':1200/30,
    'sample':True, 
    'sample_size':10,
    'sample_interval':1200/30,

    'training':0
})


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

options.training = options.mode != 1
options.dataset_path += ('/' + options.dataset)
options.checkpoints_path += ('/' + options.dataset)

main(options)