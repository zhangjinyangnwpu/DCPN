import tensorflow as tf
import numpy as np
import scipy.io as sio
import os
import random
import math
import time
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class NC(object):

    def __init__(self,sess,args,id=0):

        self.sess = sess
        self.id = str(id)
        self.train_path = args.train_path
        self.TFrecords = args.TFrecords
        self.result = args.result
        self.checkpoint = args.checkpoint
        self.log = args.log

        if not os.path.exists(self.TFrecords):
            os.mkdir(self.TFrecords)
        if not os.path.exists(self.result):
            os.mkdir(self.result)
        if not os.path.exists(self.checkpoint):
            os.mkdir(self.checkpoint)
        if not os.path.exists(self.log):
            os.mkdir(self.log)

        self.learning_rate = args.learning_rate
        self.epoch_num = args.epoch_num
        self.momentum = args.momentum
        self.batch_size = args.batch_size
        self.test_batch = args.test_batch
        self.cube_size = args.cube_size
        self.windows_size = args.windows_size
        self.train_num = args.train_num
        self.NUM_EPOCHS_PER_DECAY = 300
        self.LEARNING_RATE_DECAY_FACTOR = 0.1
        self.MOVING_AVERAGE_DECAY = 0.9999

        self.zero_pair = 3 # the rate for 0 class
        self.dim = 103
        self.num_class = 9

        self.use_pca = args.use_pca
        if self.use_pca:
            self.pca_dim = args.pca_dim
            self.dim = self.pca_dim
            self.id = self.id + '_pca'
        if self.use_pca:
            self.network = self.build_networkpca
        else:
            self.network = self.build_network

        self.images = tf.placeholder(
            tf.float32, shape=(None, self.cube_size * 2, self.cube_size, self.dim,1))
        self.labels = tf.placeholder(tf.int32, shape=(None, 1))


    def build_network(self,image,reuse=False,name='NC_network'):
        dim = 6
        if self.cube_size == 1:
            ks = [[1, 1, 1], [2, 1, 8], [1, 1, 3], [1, 1, 3], [1, 1, 3],
                  [1, 1, 3], [1, 1, 3],[1, 1, 3]]
            s = [[1, 1, 1], [1, 1, 3], [1, 1, 1], [1, 1, 2], [1, 1, 1],
                 [1, 1, 2], [1, 1, 1], [1, 1, 1]]
        if self.cube_size == 3:
            ks = [[1, 1, 1], [3, 1, 8], [1, 2, 3], [3, 1, 3], [2, 1, 3],
                  [1, 2, 3], [1, 1, 3],[1, 1, 3]]
            s = [[1, 1, 1], [1, 1, 3], [1, 1, 1], [1, 1, 2], [1, 1, 1],
                 [1, 1, 2],[1, 1, 1], [1, 1, 1]]
        if self.cube_size == 5:
            ks = [[1, 1, 1], [3, 1, 8], [3, 1, 3], [1, 3, 3], [3, 1, 3],
                  [3, 1, 3], [1, 3, 3],[2, 1, 3]]
            s = [[1, 1, 1], [1, 1, 3], [1, 1, 1], [1, 1, 2], [1, 1, 1],
                 [1, 1, 2], [1, 1, 1], [1, 1, 1]]
        with tf.variable_scope(name,reuse=reuse):
            with tf.variable_scope('conv0'):
                conv0 = tf.layers.conv3d(image,dim,ks[0],s[0])
                conv0 = tf.layers.batch_normalization(conv0, epsilon=1e-4, scale=True, momentum=0.9)
                conv0 = tf.nn.relu(conv0)
                print(conv0)
            with tf.variable_scope('conv1'):
                conv1 = tf.layers.conv3d(conv0,dim,ks[1],s[1])
                conv1 = tf.layers.batch_normalization(conv1, epsilon=1e-4, scale=True, momentum=0.9)
                conv1 = tf.nn.relu(conv1)
                print(conv1)
            with tf.variable_scope('conv2'):
                conv2 = tf.layers.conv3d(conv1,dim*2,ks[2],s[2])
                conv2 = tf.layers.batch_normalization(conv2, epsilon=1e-4, scale=True, momentum=0.9)
                conv2 = tf.nn.relu(conv2)
                print(conv2)
            with tf.variable_scope('conv3'):
                conv3 = tf.layers.conv3d(conv2,dim*4,ks[3],s[3])
                conv3 = tf.layers.batch_normalization(conv3, epsilon=1e-4, scale=True, momentum=0.9)
                conv3 = tf.nn.relu(conv3)
                print(conv3)
            with tf.variable_scope('conv4'):
                conv4 = tf.layers.conv3d(conv3,dim*8,ks[4],s[4])
                conv4 = tf.layers.batch_normalization(conv4, epsilon=1e-4, scale=True, momentum=0.9)
                conv4 = tf.nn.relu(conv4)
                print(conv4)
            with tf.variable_scope('conv5'):
                conv5 = tf.layers.conv3d(conv4,dim*8,ks[5],s[5])
                conv5 = tf.layers.batch_normalization(conv5, epsilon=1e-4, scale=True, momentum=0.9)
                conv5 = tf.nn.relu(conv5)
                print(conv5)
            with tf.variable_scope('conv6'):
                conv6 = tf.layers.conv3d(conv5,dim*16,ks[6],s[6])
                conv6 = tf.layers.batch_normalization(conv6, epsilon=1e-4, scale=True, momentum=0.9)
                conv6 = tf.nn.relu(conv6)
                print(conv6)
            with tf.variable_scope('conv7'):
                conv7 = tf.layers.conv3d(conv6,dim*16,ks[7],s[7])
                conv7 = tf.layers.batch_normalization(conv7, epsilon=1e-4, scale=True, momentum=0.9)
                conv7 = tf.nn.relu(conv7)
                print(conv7)
            with tf.variable_scope('conv8'):
                conv8 = tf.layers.conv3d(conv7,self.num_class+1,[1,1,1],[1,1,1])
                print(conv8)
            prediction = tf.layers.flatten(conv8)
            return prediction

    def build_networkpca(self,image,reuse=False,name='PCA_use'):
        def active_function(x,type):
            if type == 'relu':
                return tf.nn.relu(x)
            if type == 'lrelu':
                return tf.nn.leaky_relu(x,0.1)
            if type == 'swift':
                return tf.nn.sigmoid(x)*x
        dim = 10
        if self.cube_size == 1:
            ks = [[1, 1, 1], [2, 1, 3], [1, 1, 3], [1, 1, 3], [1, 1, 2],
                  [1, 1, 1]]
            s = [[1, 1, 1], [1, 1, 2], [1, 1, 2], [1, 1, 2], [1, 1, 1],
                 [1, 1, 1]]
        if self.cube_size == 3:
            ks = [[1, 1, 3], [3, 1, 3], [1, 2, 3], [3, 1, 3], [2, 2, 2],
                  [1, 1, 1]]
            s = [[1, 1, 1], [1, 1, 2], [1, 1, 2], [1, 1, 2], [1, 1, 1],
                 [1, 1, 1]]
        if self.cube_size == 5:
            ks = [[1, 1, 3], [3, 1, 3], [1, 3, 3], [3, 1, 3], [2, 1, 2],
                  [1, 1, 1]]
            s = [[1, 1, 1], [2, 1, 2], [1, 2, 2], [1, 2, 2], [1, 1, 1],
                 [1, 1, 1]]
        AF = 'relu'
        with tf.variable_scope(name,reuse=reuse):
            with tf.variable_scope('conv0'):
                conv0 = tf.layers.conv3d(image,dim,ks[0],s[0])
                conv0 = tf.layers.batch_normalization(conv0, epsilon=1e-4, scale=True, momentum=0.9)
                conv0 = active_function(conv0,AF)
                print(conv0)
            with tf.variable_scope('conv1'):
                conv1 = tf.layers.conv3d(conv0,dim,ks[1],s[1])
                conv1 = tf.layers.batch_normalization(conv1, epsilon=1e-4, scale=True, momentum=0.9)
                conv1 = active_function(conv1, AF)
                print(conv1)
            with tf.variable_scope('conv2'):
                conv2 = tf.layers.conv3d(conv1,dim*4,ks[2],s[2])
                conv2 = tf.layers.batch_normalization(conv2, epsilon=1e-4, scale=True, momentum=0.9)
                conv2 = active_function(conv2, AF)
                print(conv2)
            with tf.variable_scope('conv3'):
                conv3 = tf.layers.conv3d(conv2,dim*16,ks[3],s[3])
                conv3 = tf.layers.batch_normalization(conv3, epsilon=1e-4, scale=True, momentum=0.9)
                conv3 = active_function(conv3, AF)
                print(conv3)
            with tf.variable_scope('conv4'):
                conv4 = tf.layers.conv3d(conv3,dim*16,ks[4],s[4])
                conv4 = tf.layers.batch_normalization(conv4, epsilon=1e-4, scale=True, momentum=0.9)
                conv4 = active_function(conv4, AF)
                print(conv4)
            with tf.variable_scope('conv5'):
                conv5 = tf.layers.conv3d(conv4,self.num_class+1,[1,1,1],[1,1,1])
                print(conv5)
            prediction = tf.layers.flatten(conv5)
            return prediction

    def data_prepare(self):

        data = sio.loadmat(os.path.join(self.train_path, 'PaviaU.mat'))
        data_gt = sio.loadmat(os.path.join(self.train_path, 'PaviaU_gt.mat'))

        im = data['paviaU']
        imGIS = data_gt['paviaU_gt']
        plt.pcolor(imGIS, cmap='jet')
        if not os.path.exists(os.path.join(self.result, self.id)):
            os.mkdir(os.path.join(self.result, self.id))
        plt.savefig(os.path.join(self.result, self.id, 'groundtrouth.png'), format='png')
        plt.close()
        self.num_class = np.max(imGIS)

        if self.use_pca:
            x,y,k = im.shape
            im_t = im.reshape(x*y,k)
            pca = PCA(n_components=self.pca_dim)
            im = pca.fit_transform(im_t)
            im = im.reshape(x,y,self.pca_dim)

        im = (im - float(np.min(im)))
        im = im / np.max(im)


        lable_pos = {}  # per class's pos
        for i in range(1, self.num_class + 1):
            lable_pos[i] = []
        for row in range(imGIS.shape[0]):
            for col in range(imGIS.shape[1]):
                for t in range(1, self.num_class + 1):
                    if imGIS[row, col] == 0: continue
                    if imGIS[row, col] == t:
                        lable_pos[t].append([row, col])
                        continue
        t = time.time()
        random.seed(t)

        if not os.path.exists(self.result):
            os.mkdir(self.result)

        f = open(os.path.join(self.result,self.id,'seed_pos.txt'),'w')
        f.write('seed:'+str(t)+'\n')
        for i in range(1, self.num_class + 1):
            train_indices = random.sample(lable_pos[i], self.train_num)
            f.write(str(train_indices))
            for k in range(len(train_indices)):
                imGIS[train_indices[k][0], train_indices[k][1]] = i + self.num_class + 1
        f.close()


        self.shape = im.shape
        self.dim = im.shape[2]
        trainclass = {}
        testclass = {}
        for i in range(1, self.num_class + 1):
            trainclass[i] = []
            testclass[i] = []
        train_data_pos = []
        for i in range(imGIS.shape[0]):
            for j in range(imGIS.shape[1]):
                if imGIS[i, j] > 0 and imGIS[i, j] <= self.num_class:
                    testclass[imGIS[i, j]].append([[i, j], im[i, j]])
                elif imGIS[i, j] > self.num_class + 1 and imGIS[i, j] <= self.num_class * 2 + 1:
                    trainclass[imGIS[i, j] - self.num_class - 1].append([[i, j], im[i, j]])
                    # train_data_pos.append([[i, j], imGIS[i, j] - self.num_class - 1])

        def neighbor_add(row, col, labels, w_size=3, flag=True):  # 给出 row，col和标签，返回w_size大小的cube，flag=True表示为训练样本
            t = w_size // 2
            cube = np.zeros(shape=[w_size, w_size, im.shape[2]])
            for i in range(-t, t + 1):
                for j in range(-t, t + 1):
                    if i + row < 0 or i + row >= im.shape[0] or j + col < 0 or j + col >= im.shape[1]:
                        # s = random.sample(trainclass[j],3)
                        if flag == True:
                            s = random.sample(trainclass[labels], 1)
                            cube[i + t, j + t] = s[0][1]
                        else:
                            s = random.sample(testclass[labels], 1)
                            cube[i + t, j + t] = s[0][1]
                    else:
                        cube[i + t, j + t] = im[i + row, j + col]
            return cube

        filename = os.path.join(self.TFrecords,
                                'paviaU_traindata_' + self.id + '_cube_' + str(self.cube_size) +
                                '_windows_size_' + str(self.windows_size) +
                                '_train_num_' + str(self.train_num) + '.tfrecords')

        def _int64_feature(value):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        writer = tf.python_io.TFRecordWriter(filename)

        train_data = {}
        for i in range(self.num_class + 1):
            train_data[i] = []
        index = 0
        for i in range(1, self.num_class + 1):
            n = len(trainclass[i])
            for j in range(n):
                for k in range(n):
                    if j == k: continue
                    row1, col1 = trainclass[i][j][0]
                    row2, col2 = trainclass[i][k][0]
                    cube1 = neighbor_add(row1, col1, i, w_size=self.cube_size, flag=True)
                    cube2 = neighbor_add(row2, col2, i, w_size=self.cube_size, flag=True)
                    cube = np.vstack((cube1, cube2)).astype(np.float32)
                    label = np.array(np.array(i).astype(np.int64))
                    # label = i
                    cube = cube.tostring()
                    example = tf.train.Example(features=(tf.train.Features(feature={
                        'train_data': _bytes_feature(cube),
                        'train_label': _int64_feature(label)
                    })))
                    index += 1
                    # print("pre for train %d"%index)
                    writer.write(example.SerializeToString())

        for i in range(1, self.num_class + 1):
            for j in range(1, self.num_class + 1):
                if i == j:
                    continue
                n = len(trainclass[i])
                for k in range(n):
                    s = random.sample(trainclass[j], self.zero_pair)
                    for p in range(self.zero_pair):
                        row1, col1 = s[p][0]
                        row2, col2 = trainclass[i][k][0]
                        cube1 = neighbor_add(row1, col1, i, w_size=self.cube_size, flag=True)
                        cube2 = neighbor_add(row2, col2, j, w_size=self.cube_size, flag=True)
                        cube = np.vstack((cube1, cube2)).astype(np.float32)
                        # print('cube write:',cube.dtype)
                        cube = cube.tostring()
                        label = np.array(np.array(0).astype(np.int64))
                        # label = 0
                        example = tf.train.Example(features=(tf.train.Features(feature={
                            'train_data': _bytes_feature(cube),
                            'train_label': _int64_feature(label)
                        })))
                        index += 1
                        # print("pre for train %d" % index)
                        writer.write(example.SerializeToString())
        self.total_train_num = index
        writer.close()

        test_num_pixel = 0
        test_num = 0
        # filename_test = os.path.join(self.TFrecords,'test_data.tfrecords')
        filename_test = os.path.join(self.TFrecords,
                                'paviaU_testdata_'+self.id+'_cube_' + str(self.cube_size) +
                                '_windows_size_' + str(self.windows_size) +
                                '.tfrecords')
        writer_test_data = tf.python_io.TFRecordWriter(filename_test)

        test_data_label = list()
        test_data_num_width = list()
        windows_size = self.windows_size//2
        for i in range(1, len(testclass) + 1):
            for j in range(len(testclass[i])):
                row, col = testclass[i][j][0]
                index = 0
                for m in range(-windows_size, windows_size + 1):
                    for n in range(-windows_size, windows_size + 1):
                        if m == 0 and n == 0:
                            continue
                        r = row + m
                        c = col + n
                        if r >= 0 and r < imGIS.shape[0] and c >= 0 and c < imGIS.shape[1]:
                            cube1 = neighbor_add(row, col, i, w_size=self.cube_size, flag=False)
                            cube2 = neighbor_add(r, c, i, w_size=self.cube_size, flag=False)
                            cube = np.vstack((cube1, cube2)).astype(np.float32)
                            # print('cube write:',cube.dtype)
                            cube = cube.tostring()
                            example = tf.train.Example(features=(tf.train.Features(feature={
                                'test_data': _bytes_feature(cube),
                            })))
                            writer_test_data.write(example.SerializeToString())
                            index += 1
                            test_num_pixel += 1
                            # print("pre for test pixels %d" % test_num_pixel)

                test_data_label.append(i)
                test_data_num_width.append(index)
                test_num += 1
                # print("pre for test pixel%d" % test_num)
                index = 0
        self.test_num_pixel = test_num_pixel
        self.test_num = test_num
        writer_test_data.close()
        sio.savemat(os.path.join(self.result,self.id,'data_num_information.mat'),{
            'test_num_pixel':test_num_pixel,
            'test_num':test_num,
            'total_train_num':self.total_train_num,
            'test_data_label':test_data_label,
            'test_data_num_width':test_data_num_width})

    def load(self, checkpoint_dir):
        print("Loading model ...")
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False


    def dataset_input(self,filename,type):
        dataset = tf.data.TFRecordDataset([filename])

        def parser1(record):
            keys_to_features = {
                'train_data': tf.FixedLenFeature([], tf.string),
                'train_label': tf.FixedLenFeature([], tf.int64),
            }
            features = tf.parse_single_example(record, features=keys_to_features)
            train_data = tf.decode_raw(features['train_data'], tf.float32)
            train_label = tf.cast(features['train_label'], tf.int32)
            shape = [self.cube_size * 2, self.cube_size, self.dim, 1]
            train_data = tf.reshape(train_data, shape)
            train_label = tf.reshape(train_label, [1])
            return train_data, train_label

        def parser2(record):
            keys_to_features = {
                'test_data': tf.FixedLenFeature([], tf.string)
            }
            features = tf.parse_single_example(record, features=keys_to_features)
            test_data = tf.decode_raw(features['test_data'], tf.float32)
            shape = [self.cube_size * 2, self.cube_size, self.dim, 1]
            test_data = tf.reshape(test_data, shape)
            return test_data
        if type == 0:
            dataset = dataset.map(parser1,num_parallel_calls=96)
            dataset = dataset.shuffle(buffer_size=500000)
            dataset = dataset.batch(self.batch_size)
            dataset = dataset.repeat(self.epoch_num)
            iterator = dataset.make_one_shot_iterator()
            features = iterator.get_next()
            return features
        if type ==1:
            dataset = dataset.map(parser2)
            dataset = dataset.batch(self.test_batch)
            iterator = dataset.make_one_shot_iterator()
            features = iterator.get_next()
            return features

    def train(self):
        def loss(logpros, labels):
            labels = tf.reshape(labels, [self.batch_size])
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(_sentinel=None, labels=labels,
                                                                           logits=logpros, name='Xentropy')
            loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
            tf.add_to_collection('losses', loss)
            return tf.add_n(tf.get_collection('losses'), name='total_loss')

        def _add_loss_summaries(total_loss):
            loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
            losses = tf.get_collection('losses')
            loss_averages_op = loss_averages.apply(losses + [total_loss])
            return loss_averages_op

        def trainop(total_loss, global_step):
            info = sio.loadmat(os.path.join(self.result,self.id,'data_num_information.mat'))
            num_examples = info['total_train_num']
            num_batches_per_epoch = num_examples / self.batch_size
            decay_steps = int(num_batches_per_epoch * self.NUM_EPOCHS_PER_DECAY)
            # Decay the learning rate exponentially based on the number of steps.
            lr = tf.train.exponential_decay(self.learning_rate,
                                            global_step,
                                            decay_steps,
                                            self.LEARNING_RATE_DECAY_FACTOR,
                                            staircase=True)
            loss_averages_op = _add_loss_summaries(total_loss)

            with tf.control_dependencies([loss_averages_op]):
                opt = tf.train.AdamOptimizer(lr)
                grads = opt.compute_gradients(total_loss)

            # Apply gradients.
            apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

            # Track the moving averages of all trainable variables.
            variable_averages = tf.train.ExponentialMovingAverage(self.MOVING_AVERAGE_DECAY, global_step)
            variables_averages_op = variable_averages.apply(tf.trainable_variables())

            with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
                train_op = tf.no_op(name='train')
            return train_op

        global_step = tf.Variable(0, trainable=False)
        with tf.variable_scope('inference'):
            logits = self.network(self.images,reuse=False)
        loss_ = loss(logits, self.labels)
        train_op = trainop(loss_, global_step)
        saver = tf.train.Saver(tf.global_variables(),max_to_keep=100)
        filename = os.path.join(self.TFrecords,
                                'paviaU_traindata_'+self.id+'_cube_' + str(self.cube_size) +
                                '_windows_size_' + str(self.windows_size) +
                                '_train_num_' + str(self.train_num) + '.tfrecords')
        dataset = self.dataset_input(filename, 0)
        init = tf.global_variables_initializer()
        self.sess.run(init)

        model_name = os.path.join(self.checkpoint, 'NC.model')
        print('strating...\n')
        info = sio.loadmat(os.path.join(self.result, self.id,'data_num_information.mat'))
        self.num_examples = info['total_train_num']
        iterate_num = int(self.num_examples*self.epoch_num/self.batch_size)
        print(iterate_num)
        # iterate_num = 2001
        print(iterate_num)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess,coord=coord)
        loss_list = []
        start = time.time()
        for step in range(iterate_num):
            image,label = self.sess.run(dataset)
            # print(label)
            if image.shape[0]!=self.batch_size:
                continue
            self.sess.run(train_op, feed_dict={self.images: image, self.labels: label})
            if step % 100 == 0:
                loss = self.sess.run([loss_], feed_dict={self.images: image,
                                                       self.labels: label})
                print(step,":",loss)
                loss_list.append(loss)
            if step % 10000 == 0 and step != 0:
                saver.save(self.sess, model_name, global_step=step)
                print('saved')
                # self.test()
        sio.savemat(os.path.join(self.result,self.id,'loss_list.mat'),{'loss_list':loss_list})
        times = (time.time()-start)/3600
        sio.savemat(os.path.join(self.result,self.id,'train_time_cost.mat'),{'time':times})
        coord.request_stop()
        coord.join(threads)

    def test(self):
        with tf.variable_scope('inference'):
            logpros = self.network(self.images,reuse=tf.AUTO_REUSE)
            y_conv = tf.nn.softmax(logpros)
            y_ = tf.slice(y_conv, [0, 1], [-1, self.num_class])

        if self.load(self.checkpoint):
            print('load successful...')
        else:
            print('load fail!!!')
            return
        filename_test = os.path.join(self.TFrecords,
                                     'paviaU_testdata_' + self.id + '_cube_' + str(self.cube_size) +
                                     '_windows_size_' + str(self.windows_size) +
                                     '.tfrecords')
        dataset = self.dataset_input(filename_test, 1)
        info = sio.loadmat(os.path.join(self.result,self.id,'data_num_information.mat'))
        test_data_label = info['test_data_label'][0]
        test_data_num_width = info['test_data_num_width'][0]
        test_num_pixel = info['test_num_pixel'][0][0]
        print('test_num_pixel:',test_num_pixel)
        test_num = info['test_num'][0][0]
        print('test_num',test_num)
        prediction_list = list()
        print(test_num)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
        start = time.time()
        try:
            while True:
                test_images = self.sess.run(dataset)
                pre = self.sess.run(tf.argmax(y_, 1), feed_dict={self.images: test_images})
                prediction_list.append(pre)
        except tf.errors.OutOfRangeError:
            print("end!")

        coord.request_stop()
        coord.join(threads)

        pre_index = int(0)
        matrix = np.zeros((self.num_class, self.num_class))
        t = list()
        for prediction in prediction_list:
            for i in prediction:
                t.append(i)
        prediction = []
        for i in range(test_num):
            predictions = t[pre_index:pre_index + int(test_data_num_width[i])]
            pre_index += int(test_data_num_width[i])
            pre_label = np.argmax(np.bincount(predictions))
            prediction.append(pre_label)
            matrix[pre_label, test_data_label[i]-1] += 1
        times = (time.time() - start)
        sio.savemat(os.path.join(self.result, self.id, 'test_time_cost.mat'), {'time': times})
        print(max(prediction),min(prediction))
        ac_list = []
        for i in range(len(matrix)):
            ac = matrix[i, i] / sum(matrix[:, i])
            ac_list.append(ac)
            print('(',matrix[i, i],'/',sum(matrix[:, i]),')',ac)
        print(np.int_(matrix))
        print('total right num:',np.sum(np.trace(matrix)))
        accuracy = np.sum(np.trace(matrix)) / np.sum(matrix)
        print('accuracy:',accuracy)
        # 计算kappa值
        kk = 0
        for i in range(matrix.shape[0]):
            kk += np.sum(matrix[i]) * np.sum(matrix[:, i])
        pe = kk / (np.sum(matrix) * np.sum(matrix))
        pa = np.trace(matrix) / np.sum(matrix)
        kappa = (pa - pe) / (1 - pe)
        print('kappa:',kappa)
        if not os.path.exists(os.path.join(self.result,self.id)):
            os.mkdir(os.path.join(self.result,self.id))
        sio.savemat(os.path.join(self.result,self.id,'matrix_kappa_'+self.id+'.mat'),{
            'matrix':matrix,
            'kappa':kappa,
            'accuracy':accuracy,
            'ac_list':ac_list
        })
