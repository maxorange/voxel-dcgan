import tensorflow as tf
import numpy as np

def weight_variable(shape):
    return tf.get_variable('W', shape, initializer=tf.random_normal_initializer(0., 0.02))

def bias_variable(shape):
    return tf.get_variable('b', shape, initializer=tf.constant_initializer(0.))

def keep_prob(dropout, train):
    return tf.cond(train, lambda: tf.constant(dropout), lambda: tf.constant(1.))

def softmax_ce_with_logits(logits, labels):
    return tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)

def sigmoid_ce_with_logits(logits, labels):
    return tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)

def sigmoid_kl_with_logits(logits, targets):
    assert isinstance(targets, float)
    if targets in [0., 1.]:
        entropy = 0.
    else:
        entropy = - targets*tf.log(targets) - (1. - targets)*tf.log(1. - targets)
    return sigmoid_ce_with_logits(logits, tf.ones_like(logits)*targets) - entropy

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def linear(x, shape, name, bias=False):
    with tf.variable_scope(name):
        W = weight_variable(shape)
        h = tf.matmul(x, W)
        if bias:
            b = bias_variable([shape[-1]])
            h = h + b
        return h

def conv2d(x, shape, name, bias=False, stride=2):
    with tf.variable_scope(name):
        W = weight_variable(shape)
        h = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')
        if bias:
            b = bias_variable([shape[-1]])
            h = h + b
        return h

def deconv2d(x, shape, output_shape, name, bias=False, stride=2):
    with tf.variable_scope(name):
        W = weight_variable(shape)
        h = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding='SAME')
        if bias:
            b = bias_variable([shape[-2]])
            h = h + b
        return h

def conv3d(x, shape, name, bias=False, stride=2, padding='SAME'):
    with tf.variable_scope(name):
        W = weight_variable(shape)
        h = tf.nn.conv3d(x, W, strides=[1, stride, stride, stride, 1], padding=padding)
        if bias:
            b = bias_variable([shape[-1]])
            h = h + b
        return h

def deconv3d(x, shape, output_shape, name, bias=False, stride=2, padding='SAME'):
    with tf.variable_scope(name):
        W = weight_variable(shape)
        h = tf.nn.conv3d_transpose(x, W, output_shape, strides=[1, stride, stride, stride, 1], padding=padding)
        if bias:
            b = bias_variable([shape[-2]])
            h = h + b
        return h

def phase_shift_3d(x, r):
    batch_size, d, h, w, c = x.get_shape().as_list()
    x = tf.reshape(x, (batch_size, d, h, w, r, r, r))
    for ns in [d, h, w]:
        x = tf.split(x, ns, 1)
        x = tf.concat([tf.squeeze(v, 1) for v in x], 3)
    return tf.reshape(x, (batch_size, d*r, h*r, w*r, 1))

def subpixel_conv3d(x, r, out_channels):
    x = tf.split(x, out_channels, 4)
    x = tf.concat([phase_shift_3d(v, r) for v in x], 4)
    return x

def pixel_shuffler_3d(x, r, k, out_channels, name):
    in_channels = x.get_shape.as_list()[4]
    with tf.variable_scope(name):
        u = conv3d(x, [k, k, k, in_channels, out_channels*pow(r, 3)], 'conv', bias=True, stride=1)
        h = subpixel_conv3d(u, r, out_channels)
        return h

def minibatch_discrimination(x, n_kernels, dim_per_kernel, name):
    with tf.variable_scope(name):
        batch_size, nf = x.get_shape().as_list()
        h = linear(x, [nf, n_kernels*dim_per_kernel], 'h1')
        activation = tf.reshape(h, (batch_size, n_kernels, dim_per_kernel))

        big = tf.eye(batch_size)
        big = tf.expand_dims(big, 1)

        abs_dif = tf.reduce_sum(tf.abs(tf.expand_dims(activation, 3) - tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)), 2)
        mask = 1. - big
        masked = tf.exp(-abs_dif) * mask

        def half(tens, second):
            m, n, _ = tens.get_shape().as_list()
            return tf.slice(tens, [0, 0, second*(batch_size/2)], [m, n, batch_size/2])

        f1 = tf.reduce_sum(half(masked, 0), 2) / tf.reduce_sum(half(mask, 0))
        f2 = tf.reduce_sum(half(masked, 1), 2) / tf.reduce_sum(half(mask, 1))
        return tf.concat([x, f1, f2], 1)

def batch_norm(x, train, name, decay=0.99, epsilon=1e-5):
    shape = x.get_shape().as_list()
    with tf.variable_scope(name):
        beta = tf.get_variable('beta', [shape[-1]], initializer=tf.constant_initializer(0.))
        gamma = tf.get_variable('gamma', [shape[-1]], initializer=tf.random_normal_initializer(1., 0.02))
        pop_mean = tf.get_variable('pop_mean', [shape[-1]], initializer=tf.constant_initializer(0.), trainable=False)
        pop_var = tf.get_variable('pop_var', [shape[-1]], initializer=tf.constant_initializer(1.), trainable=False)

        if pop_mean not in tf.moving_average_variables():
            tf.add_to_collection(tf.GraphKeys.MOVING_AVERAGE_VARIABLES, pop_mean)
            tf.add_to_collection(tf.GraphKeys.MOVING_AVERAGE_VARIABLES, pop_var)

        def func1():
            # execute at training time
            batch_mean, batch_var = tf.nn.moments(x, range(len(shape) - 1))
            update_mean = tf.assign_sub(pop_mean, (1 - decay)*(pop_mean - batch_mean))
            update_var = tf.assign_sub(pop_var, (1 - decay)*(pop_var - batch_var))
            with tf.control_dependencies([update_mean, update_var]):
                return tf.nn.batch_normalization(x, batch_mean, batch_var, beta, gamma, epsilon)

        def func2():
            # execute at test time
            return tf.nn.batch_normalization(x, pop_mean, pop_var, beta, gamma, epsilon)

        return tf.cond(train, func1, func2)

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        var = grad_and_vars[0][1]
        grad_and_var = (grad, var)
        average_grads.append(grad_and_var)
    return average_grads

def binary_mask(shape):
    samples = tf.random_uniform(shape, minval=0.0, maxval=1.0)
    mask = tf.less_equal(samples, 0.7)
    return tf.cast(mask, tf.float32)
