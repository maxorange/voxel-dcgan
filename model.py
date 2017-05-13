import tensorflow as tf
import config
import util
from ops import *

class Model(object):

    def __init__(self, vars):
        self.saver = tf.train.Saver(vars)

    def session(self, sess):
        if sess is not None:
            self.sess = sess
        else:
            config_proto = tf.ConfigProto()
            config_proto.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config_proto)

    def initialize(self):
        self.sess.run(tf.global_variables_initializer())

    def save(self, path):
        self.saver.save(self.sess, path)

    def restore(self, path):
        self.saver.restore(self.sess, path)

    def close(self):
        self.sess.close()

class GAN(Model):

    def __init__(self, batch_size, nz, nvx, sess=None):
        self.session(sess)
        opt = tf.train.AdamOptimizer(2e-4, 0.5)
        tower_gradsG = []
        tower_gradsD = []
        self.lossesG = []
        self.lossesD = []
        self.x_g_list = []
        self.train = tf.placeholder(tf.bool)
        self.netG = Generator()
        self.netD = Discriminator()

        self.build_model(batch_size, nz, nvx, 0)
        gradsG = opt.compute_gradients(self.lossesG[-1], var_list=self.varsG)
        gradsD = opt.compute_gradients(self.lossesD[-1], var_list=self.varsD)
        tower_gradsG.append(gradsG)
        tower_gradsD.append(gradsD)

        # multi-GPU mode
        # gpus = ['/gpu:0', '/gpu:1']
        # n_gpu = len(gpus)
        # for i, gpu in enumerate(gpus):
        #     with tf.device(gpu):
        #         self.build_model(batch_size/n_gpu, nz, nvx, i)
        #         gradsG = opt.compute_gradients(self.lossesG[-1], var_list=self.varsG)
        #         gradsD = opt.compute_gradients(self.lossesD[-1], var_list=self.varsD)
        #         tower_gradsG.append(gradsG)
        #         tower_gradsD.append(gradsD)

        self.optG = opt.apply_gradients(average_gradients(tower_gradsG))
        self.optD = opt.apply_gradients(average_gradients(tower_gradsD))
        self.lossG = tf.reduce_mean(self.lossesG)
        self.lossD = tf.reduce_mean(self.lossesD)
        self.x_g = tf.concat(self.x_g_list, 0)

        if sess is None:
            self.initialize()

        variables_to_save = self.varsG + self.varsD + tf.moving_average_variables()
        super(GAN, self).__init__(variables_to_save)

    def build_model(self, batch_size, nz, nvx, gpu_idx):
        reuse = False if gpu_idx == 0 else True
        z = tf.placeholder(tf.float32, [batch_size, nz], 'z'+str(gpu_idx))
        x = tf.placeholder(tf.float32, [batch_size, nvx, nvx, nvx, 1], 'x'+str(gpu_idx))

        # generator
        x_g = self.netG(z, self.train, reuse=reuse)
        self.x_g_list.append(x_g)

        # discriminator
        d_g = self.netD(x_g, self.train, reuse=reuse)
        d_r = self.netD(x, self.train, reuse=True)

        if gpu_idx == 0:
            t_vars = tf.trainable_variables()
            self.varsG = [var for var in t_vars if var.name.startswith('G')]
            self.varsD = [var for var in t_vars if var.name.startswith('D')]

        # generator loss
        lossG_adv = tf.reduce_mean(sigmoid_kl_with_logits(d_g, 0.8))
        weight_decayG = tf.add_n([tf.nn.l2_loss(var) for var in self.varsG])
        self.lossesG.append(lossG_adv + 5e-4*weight_decayG)

        # discriminator loss
        lossD_real = tf.reduce_mean(sigmoid_kl_with_logits(d_r, 0.8))
        lossD_fake = tf.reduce_mean(sigmoid_ce_with_logits(d_g, tf.zeros_like(d_g)))
        weight_decayD = tf.add_n([tf.nn.l2_loss(var) for var in self.varsD])
        self.lossesD.append(lossD_real + lossD_fake + 5e-4*weight_decayD)

    def optimize(self, z, x):
        fd = {'z0:0':z, 'x0:0':x, self.train:True}
        # fd = {'z0:0':z[0], 'z1:0':z[1], 'x0:0':x[0], 'x1:0':x[1], self.train:True} # multi-GPU mode
        self.sess.run(self.optD, feed_dict=fd)
        self.sess.run(self.optG, feed_dict=fd)

    def get_errors(self, z, x):
        fd = {'z0:0':z, 'x0:0':x, self.train:False}
        # fd = {'z0:0':z[0], 'z1:0':z[1], 'x0:0':x[0], 'x1:0':x[1], self.train:False} # multi-GPU mode
        lossD = self.sess.run(self.lossD, feed_dict=fd)
        lossG = self.sess.run(self.lossG, feed_dict=fd)
        return lossD, lossG

    def generate(self, z):
        x_g = self.sess.run(self.x_g, feed_dict={'z0:0':z, self.train:False})
        # x_g = self.sess.run(self.x_g, feed_dict={'z0:0':z[0], 'z1:0':z[1], self.train:False}) # multi-GPU mode
        return x_g[:, :, :, :, 0]

class GANTest(Model):

    def __init__(self, batch_size, nz, sess=None):
        self.session(sess)
        self.batch_size = batch_size
        self.nz = nz
        self.train = tf.placeholder(tf.bool)
        self.netG = Generator()
        self.build_model()

        if sess is None:
            self.initialize()

        variables_to_save = self.varsG + tf.moving_average_variables()
        super(GANTest, self).__init__(variables_to_save)

    def build_model(self):
        z = tf.placeholder(tf.float32, [self.batch_size, self.nz], 'z')
        self.x_g = self.netG(z, self.train)
        self.varsG = [var for var in tf.trainable_variables() if var.name.startswith('G')]

    def generate(self, z):
        x_g = self.sess.run(self.x_g, feed_dict={'z:0':z, self.train:False})
        return x_g[0, :, :, :, 0] > 0.9

class Generator(object):

    def __call__(self, z, train, nf=32, name="G", reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            batch_size, nz = z.get_shape().as_list()

            u = linear(z, [nz, 4*4*4*nf*8], 'h1')
            h = tf.nn.relu(batch_norm(u, train, 'bn1'))

            h = tf.reshape(h, [batch_size, 4, 4, 4, nf*8])

            u = deconv3d(h, [4, 4, 4, nf*4, nf*8], [batch_size, 8, 8, 8, nf*4], 'h2')
            h = tf.nn.relu(batch_norm(u, train, 'bn2'))

            u = deconv3d(h, [4, 4, 4, nf*2, nf*4], [batch_size, 16, 16, 16, nf*2], 'h3')
            h = tf.nn.relu(batch_norm(u, train, 'bn3'))

            u = deconv3d(h, [4, 4, 4, nf, nf*2], [batch_size, 32, 32, 32, nf], 'h4')
            h = tf.nn.relu(batch_norm(u, train, 'bn4'))

            u = deconv3d(h, [4, 4, 4, 1, nf], [batch_size, 32, 32, 32, 1], 'h5', bias=True, stride=1)
            return tf.nn.sigmoid(u)

class Discriminator(object):

    def __call__(self, x, train, nf=32, name="D", reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            batch_size = x.get_shape().as_list()[0]

            x *= binary_mask(x.get_shape())

            u = conv3d(x, [4, 4, 4, 1, nf], 'h1', bias=True, stride=1)
            h = lrelu(u)

            u = conv3d(h, [4, 4, 4, nf, nf*2], 'h2')
            h = lrelu(batch_norm(u, train, 'bn2'))

            u = conv3d(h, [4, 4, 4, nf*2, nf*4], 'h3')
            h = lrelu(batch_norm(u, train, 'bn3'))

            u = conv3d(h, [4, 4, 4, nf*4, nf*8], 'h4')
            h = lrelu(batch_norm(u, train, 'bn4'))

            h = tf.reshape(h, [batch_size, -1])
            h = minibatch_discrimination(h, 300, 50, 'md1')

            ni = h.get_shape().as_list()[1]
            return linear(h, [ni, 1], 'h5', bias=True)
