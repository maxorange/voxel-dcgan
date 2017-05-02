import tensorflow as tf
import numpy as np
import model
import util
import config

batch_size = 10
z_size = 100

z = tf.placeholder(tf.float32, [batch_size, z_size])
train = tf.placeholder(tf.bool)

G = model.Generator(z_size)
x = G(z)

vars_G = [v for v in tf.trainable_variables() if 'g_' in v.name]

saver = tf.train.Saver()

config_proto = tf.ConfigProto()
config_proto.gpu_options.allow_growth = True

with tf.Session(config=config_proto) as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, config.params_path)

    batch_z = np.random.uniform(-1, 1, [batch_size, z_size]).astype(np.float32)
    data = sess.run(x, feed_dict={z:batch_z, train:False})

    for i, v in enumerate(data):
        util.save_binvox(v.reshape([32, 32, 32]), "out/model-{0}.binvox".format(i))
