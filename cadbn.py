#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
 Deep Belief Nets (DBN)

 References :
   - Y. Bengio, P. Lamblin, D. Popovici, H. Larochelle: Greedy Layer-Wise
   Training of Deep Networks, Advances in Neural Information Processing
   Systems 19, 2007


   - DeepLearningTutorials
   https://github.com/lisa-lab/DeepLearningTutorials


'''

import sys
import numpy
from carbm import *

numpy.seterr(all='ignore')

def sigmoid(x):
    return 1. / (1 + numpy.exp(-x))

def softmax(x):
    e = numpy.exp(x - numpy.max(x))  # prevent overflow
    if e.ndim == 1:
        return e / numpy.sum(e, axis=0)
    else:
        return e / numpy.array([numpy.sum(e, axis=1)]).T  # ndim = 2


class DDBN(object):
    def __init__(self, input=None, label=None,\
                 n_ins=2, hidden_layer_sizes=[3, 3], n_outs=2,\
                 numpy_rng=None):

        self.x = input
        self.y = label

        self.sigmoid_layers = []
        self.rbm_layers = []
        self.n_layers = len(hidden_layer_sizes)  # = len(self.rbm_layers)

        if numpy_rng is None:
            numpy_rng = numpy.random.RandomState(1234)


        assert self.n_layers > 0


        # construct multi-layer
        for i in xrange(self.n_layers):
            # layer_size
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layer_sizes[i - 1]

            # layer_input
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].sample_h_given_v()

            # construct rbm_layer
            rbm_layer = DRBM(input=layer_input,
                            n_visible=input_size)
            self.rbm_layers.append(rbm_layer)

            hidden_layer_sizes[i] = rbm_layer.n_hidden

            # construct sigmoid_layer
            sigmoid_layer = HiddenLayer(input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layer_sizes[i],
                                        W=rbm_layer.W,
                                        b=rbm_layer.hbias,
                                        numpy_rng=numpy_rng,
                                        activation=sigmoid)
            self.sigmoid_layers.append(sigmoid_layer)



        # layer for output using Logistic Regression
        self.log_layer = LogisticRegression(input=self.sigmoid_layers[-1].sample_h_given_v(),
                                            label=self.y,
                                            n_in=hidden_layer_sizes[-1],
                                            n_out=n_outs)

        # finetune cost: the negative log likelihood of the logistic regression layer
        self.finetune_cost = self.log_layer.negative_log_likelihood()



    def pretrain(self, k=1, epochs=100):
        # pre-train layer-wise
        for i in xrange(self.n_layers):
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[i-1].sample_h_given_v(layer_input)
            rbm = self.rbm_layers[i]

            for epoch in xrange(epochs):
                rbm.contrastive_divergence(k=k, input=layer_input)
                #cost = rbm.get_reconstruction_cross_entropy()
                #print >> sys.stderr, \
                #        'Pre-training layer %d, epoch %d, cost ' %(i, epoch), cost

    # def pretrain(self, lr=0.1, k=1, epochs=100):
    #     # pre-train layer-wise
    #     for i in xrange(self.n_layers):
    #         rbm = self.rbm_layers[i]

    #         for epoch in xrange(epochs):
    #             layer_input = self.x
    #             for j in xrange(i):
    #                 layer_input = self.sigmoid_layers[j].sample_h_given_v(layer_input)

    #             rbm.contrastive_divergence(lr=lr, k=k, input=layer_input)
    #             # cost = rbm.get_reconstruction_cross_entropy()
    #             # print >> sys.stderr, \
    #             #        'Pre-training layer %d, epoch %d, cost ' %(i, epoch), cost


    def finetune(self, lr=0.1, epochs=100):
        layer_input = self.sigmoid_layers[-1].sample_h_given_v(self.x)

        # train log_layer
        epoch = 0
        done_looping = False
        while (epoch < epochs) and (not done_looping):
            self.log_layer.train(lr=lr, input=layer_input, output=self.y)
            # self.finetune_cost = self.log_layer.negative_log_likelihood()
            # print >> sys.stderr, 'Training epoch %d, cost is ' % epoch, self.finetune_cost

            lr *= 0.95
            epoch += 1

    def oblivion(self, input, output, lr=0.1, epochs=100):
        self.x = input
        self.y = output
        layer_input = self.sigmoid_layers[-1].sample_h_given_v()

        # train log_layer
        epoch = 0
        done_looping = False
        while (epoch < epochs) and (not done_looping):
            self.log_layer.train(lr=-lr, input=layer_input, output=output)
            # self.finetune_cost = self.log_layer.negative_log_likelihood()
            # print >> sys.stderr, 'Training epoch %d, cost is ' % epoch, self.finetune_cost

            lr *= 0.95
            epoch += 1

    def predict(self, x):
        layer_input = x

        for i in xrange(self.n_layers):
            sigmoid_layer = self.sigmoid_layers[i]
            # rbm_layer = self.rbm_layers[i]
            layer_input = sigmoid_layer.output(input=layer_input)

        out = self.log_layer.predict(layer_input)

        if out[0] != out[0]:
            print "nan!!"
            print x
            print layer_input
            print numpy.dot(layer_input, self.log_layer.W)
            print self.log_layer.W
            print self.log_layer.b

        return out

    def synthesis(self, network):
        self.rbm_layers[0].synthesis(network.rbm_layers[0])
        self.sigmoid_layers[0].W = self.rbm_layers[0].W
        self.sigmoid_layers[0].b = self.rbm_layers[0].hbias
        self.log_layer.W = numpy.r_[self.log_layer.W,network.log_layer.W]
        # self.log_layer.b = numpy.r_[self.log_layer.b,network.log_layer.b]

    def reset_data(self, input=None, output=None):
        if input is not None:
            self.x = numpy.array(input)
            #self.sigmoid_layers[0].x = input
            #self.log_layer.x = numpy.array(input)
        if output is not None:
            self.y = numpy.array(output)
            self.log_layer.y = numpy.array(output)

class HiddenLayer(object):
    def __init__(self, input, n_in, n_out,\
                 W=None, b=None, numpy_rng=None, activation=numpy.tanh):

        if numpy_rng is None:
            numpy_rng = numpy.random.RandomState(1234)

        if W is None:
            a = 1. / n_in
            initial_W = numpy.array(numpy_rng.uniform(  # initialize W uniformly
                low=-a,
                high=a,
                size=(n_in, n_out)))

            W = initial_W

        if b is None:
            b = numpy.zeros(n_out)  # initialize bias 0


        self.numpy_rng = numpy_rng
        self.input = input
        self.W = W
        self.b = b

        self.activation = activation

    def output(self, input=None):
        if input is not None:
            self.input = input

        linear_output = numpy.dot(self.input, self.W) + self.b

        return (linear_output if self.activation is None
                else self.activation(linear_output))

    def sample_h_given_v(self, input=None):
        if input is not None:
            self.input = input


        v_mean = self.output()
        h_sample = self.numpy_rng.binomial(size=v_mean.shape,
                                           n=1,
                                           p=v_mean)
        return h_sample


class LogisticRegression(object):
    def __init__(self, input, label, n_in, n_out):
        self.x = input
        self.y = label
        self.W = numpy.zeros((n_in, n_out))  # initialize W 0
        self.b = numpy.zeros(n_out)          # initialize bias 0

    def train(self, lr=0.1, input=None, output=None, L2_reg=0.00) :

        if input is not None:
            self.x = input
        if output is not None:
            self.y = output

        p_y_given_x = softmax(numpy.dot(self.x, self.W) + self.b)
        d_y = self.y - p_y_given_x

        self.W += lr * numpy.dot(self.x.T, d_y) - lr * L2_reg * self.W
        self.b += lr * numpy.mean(d_y, axis=0)

    def negative_log_likelihood(self):
        sigmoid_activation = softmax(numpy.dot(self.x, self.W) + self.b)

        cross_entropy = - numpy.mean(
            numpy.sum(self.y * numpy.log(sigmoid_activation) +
            (1 - self.y) * numpy.log(1 - sigmoid_activation),
                      axis=1))

        return cross_entropy

    def predict(self, x):
        return softmax(numpy.dot(x, self.W) + self.b)


def test_dbn(pretrain_lr=0.1, pretraining_epochs=1000, k=1, \
             finetune_lr=0.1, finetune_epochs=200):

    x = numpy.array(([0]*50 + [1]*50,
                    [0]*30 + [1]*40 + [0]*30,
                    [1]*50 + [0]*50))
    y = numpy.identity(len(x))

    rng = numpy.random.RandomState(123)

    # construct DBN
    dbn = DBN(input=x, label=y, n_ins=len(x[0]), hidden_layer_sizes=[1, 1], n_outs=len(y[0]), numpy_rng=rng)

    # pre-training (TrainUnsupervisedDBN)
    dbn.pretrain(lr=pretrain_lr, k=1, epochs=pretraining_epochs)

    # fine-tuning (DBNSupervisedFineTuning)
    dbn.finetune(lr=finetune_lr, epochs=finetune_epochs)


    # test
    print dbn.predict(x)


if __name__ == "__main__":
    test_dbn()