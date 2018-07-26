import tensorflow as tf
from opt_utils import CGOptimizer, BFGSOptimizer, ACGOptimizer


class Mnist:
    """
    load mnist, provide mini-batch method.
    """
    def __str__(self):
        return 'Mnist'

    def __init__(self, dir_path, one_hot):
        """
        :param dir_path: path of cifar-10 data sets
        :param one_hot: trans labels into one-hot form
        """
        from tensorflow.examples.tutorials.mnist import input_data as mnist_data

        if one_hot:
            self._data = mnist_data.read_data_sets(dir_path, one_hot=True, reshape=False, validation_size=0)
        else:
            self._data = mnist_data.read_data_sets(dir_path, one_hot=False, reshape=False, validation_size=0)

        self.num_train_images = self._data.train.images.shape[0]
        self.images_shape = self._data.train.images.shape[1:]
        self.num_cls = 10

        self.test_images = self._data.test.images
        self.test_labels = self._data.test.labels

    def next_batch(self, batch_size):
        """
        :param batch_size: number of samples in a mini-batch
        :return: images and labels of samples
        """
        return self._data.train.next_batch(batch_size)


class MnistFrame:
    """
    neural network frame for mnist data sets
    """
    def __init__(self, batch_size):
        self.batch_size = batch_size

        with tf.name_scope('input'):
            self.x = tf.placeholder(tf.float32, [self.batch_size, 28, 28, 1], name='x')
            self.y = tf.placeholder(tf.float32, [self.batch_size, 10], name='y')

        with tf.name_scope('conv_1'):
            init_value = tf.truncated_normal([3, 3, 1, 4], stddev=0.1, dtype=tf.float32)
            K = tf.Variable(initial_value=init_value, name='K')
            conv = tf.nn.conv2d(self.x, K, [1, 1, 1, 1], 'SAME', name='conv')
            relu = tf.nn.relu(conv, 'relu')
            pool = tf.nn.max_pool(relu, [1, 2, 2, 1], [1, 1, 1, 1], 'SAME', name='pool')
        with tf.name_scope('conv_2'):
            init_value = tf.truncated_normal([3, 3, 4, 8], stddev=0.1, dtype=tf.float32)
            K = tf.Variable(initial_value=init_value, name='K')
            conv = tf.nn.conv2d(pool, K, [1, 1, 1, 1], 'SAME', name='conv')
            relu = tf.nn.relu(conv, 'relu')
            pool = tf.nn.max_pool(relu, [1, 2, 2, 1], [1, 1, 1, 1], 'SAME', name='pool')
        with tf.name_scope('global_avg_pool'):
            x = tf.reduce_mean(pool, axis=[1, 2])
        with tf.name_scope('full_connect'):
            input_shape = x.get_shape().as_list()
            init_value = tf.truncated_normal([input_shape[1], 10], stddev=0.1, dtype=tf.float32)
            W = tf.Variable(initial_value=init_value, name='W')
            fc = tf.matmul(x, W, name='fc')
            y_ = tf.nn.softmax(fc, name='softmax')
        with tf.name_scope('loss'):
            self.loss = -tf.reduce_sum(tf.reduce_mean(tf.multiply(self.y, tf.log(y_)), axis=0))

        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name='alpha')
        self.train_cg = CGOptimizer(self.lr).minimize(self.loss)
        self.train_bfgs = BFGSOptimizer(self.lr).minimize(self.loss)

        self.acg = ACGOptimizer(self.loss, self.lr)
        self.train_acg = self.acg.auto_minimize()

    def train(self):
        data = Mnist(dir_path='./datasets/mnist', one_hot=True)

        with tf.Session() as sess:
            x, y = data.next_batch(8)
            tf.global_variables_initializer().run(feed_dict={self.x: x, self.y: y})

            x, y = data.next_batch(8)  # TODO: you can move this line into for-loop.

            for i in range(10):
                loss, train_step = sess.run([self.loss, self.train_cg],
                                            feed_dict={self.x: x, self.y: y, self.lr: 0.1})
                print('update variables by cg in step %d: loss = %f' % (i, loss))

            for i in range(10):
                loss, train_step = sess.run([self.loss, self.train_bfgs],
                                            feed_dict={self.x: x, self.y: y, self.lr: 0.1})
                print('update variables by bfgs in step %d: loss = %f' % (i, loss))

            for i in range(10):
                lr = self.acg.get_step_length(sess, feed_dict={self.x: x, self.y: y})
                loss, train_step = sess.run([self.loss, self.train_acg],
                                            feed_dict={self.x: x, self.y: y, self.lr: lr})
                print('update variables by acg in step %d: loss = %f, lr = %f' % (i, loss, lr))


MnistFrame(8).train()
