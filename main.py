import tensorflow as tf
import numpy as np
import os
import time
import argparse
import model
parser = argparse.ArgumentParser(description='NC network for PaviaU')
parser.add_argument('--result',dest='result',default='result')
parser.add_argument('--train_path',dest='train_path',default='../../../../Data')
parser.add_argument('--TFrecords',dest='TFrecords',default='TFrecords')
parser.add_argument('--checkpoint',dest='checkpoint',default='checkpoint')
parser.add_argument('--log',dest='log',default='log')

parser.add_argument('--use_pca',dest='use_pca',default=False)
parser.add_argument('--pca_dim',dest='pca_dim',default=30)

parser.add_argument('--learning_rate',dest='learning_rate',default=1e-3)
parser.add_argument('--momentum',dest='momentum',default=0.9)
parser.add_argument('--epoch_num',dest='epoch_num',default=100)
parser.add_argument('--batch_size',dest='batch_size',default=200)
parser.add_argument('--test_batch',dest='test_batch',default=10000)

parser.add_argument('--cube_size',dest='cube_size',default=5)
parser.add_argument('--windows_size',dest='windows_size',default=3)
parser.add_argument('--train_num',dest='train_num',default=200)


args = parser.parse_args()

def main(_):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    tf.reset_default_graph()
    with tf.Session(config=config) as sess:
        args.checkpoint = 'checkpoint'
        nc = model.NC(sess,args)
        nc.data_prepare()
        nc.train()
        nc.test()
        sess.close()

if __name__ == '__main__':
    tf.app.run()
