import tensorflow as tf
import numpy as np

class Model:

    def __init__(self, conv_size, hidden_size, n_classes):
        self.compiled = False
        self.conv_layers = []
        self.hidden_layers = []
        self.params = []
        temp = 3
        for size in conv_size:
            self.conv_layers.append(ConvLayer(
                [5, 5],
                [1, 1, 1, 1],
                temp,
                size))
            temp = size
        temp = 8450
        for size in hidden_size:
            self.hidden_layers.append(HiddenLayer(temp, size))
            temp = size
        self.hidden_layers.append(HiddenLayer(temp, n_classes, lambda x: x))
        self.n_classes = n_classes

    def compile(self, H, W):
        self.compiled = True
        self.tfX = tf.placeholder(tf.float32, [None, H, W, 3])
        self.tfy = tf.placeholder(tf.int32, [None,])
        self.sess = tf.Session()
        Z = self.tfX
        for layer in self.conv_layers:
            Z = layer.forward(Z)
        Z = tf.layers.Flatten()(Z)
        for layer in self.hidden_layers:
            Z = layer.forward(Z)
        self.logits = Z
        # costa = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        #     logits=Z, labels=self.tfy))
        costa = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=Z, labels=self.tfy))
        self.train_op = tf.train.RMSPropOptimizer(0.0001).minimize(costa)
        self.predict_op = tf.nn.softmax(Z)
        self.cost = costa
        init = tf.global_variables_initializer()
        self.sess.run(init)
        for layer in self.hidden_layers:
            self.params += layer.params
        for layer in self.conv_layers:
            self.params += layer.params
        self.saver = tf.train.Saver(self.params)
        self.losses = []

    def fit(self, X, y):
        if not self.compiled:
            raise RuntimeError("Compile the model first.")
            return
        _, loss, tpreds = self.sess.run([self.train_op, self.cost, self.predict_op], {self.tfX:X, self.tfy:y})
        acc = calc_acc(tpreds, y)
        print("Loss: %.2f Acc: %.2f"%(loss, acc))
        self.losses.append(loss)
        return self.losses


def calc_acc(p, t):
    correct = 0
    for i in range(len(t)):
        if np.argmax(p[i], axis=0) == np.argmax(t[i], axis=0):
            correct += 1
    return correct/len(t) * 100


class HiddenLayer:

    def __init__(self, D, M, f=tf.nn.relu):
        self.W = tf.Variable(tf.random_normal([D, M]))
        self.b = tf.Variable(tf.random_normal([1, M]))
        self.f = f
        self.params = [self.W, self.b]

    def forward(self, X):
        out = tf.matmul(X, self.W) + self.b
        return self.f(out)

class ConvLayer:

    def __init__(self, filter_size, strides, in_channels, out_channels, bias=True):
        self.Wc = tf.Variable(init_filter(
            [*filter_size, in_channels, out_channels]))
        self.b = tf.Variable(tf.random_normal([out_channels,]))
        self.strides = strides
        self.bias = bias
        self.params = [self.Wc, self.b]

    def forward(self, X):
        out = tf.nn.conv2d(X, self.Wc, self.strides, 'SAME')
        if self.bias:
            out = tf.nn.bias_add(out, self.b)
        out = tf.nn.max_pool(out, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
        return tf.nn.relu(out)

def init_filter(shape):
    w = np.random.randn(*shape) * np.sqrt(2.0 / np.prod(shape[:-1]))
    return w.astype(np.float32)